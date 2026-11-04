"""
Долгосрочная семантическая память корневого агента (stage 5 плана контекст-оптимизации).

Что даёт: агент вспоминает устойчивые факты о пользователе из прошлых диалогов
(роль, предпочтения по формату ответов, повторяющиеся темы, правки стиля работы),
даже если они вытеснены из контекста суммаризацией или диалог начался заново.

Механика:
- Store: AsyncSqliteStore — переживает рестарт сервиса; факты лежат в namespace
  ("memories", user_id) и ищутся по смыслу (sqlite-vec + embedding-сервис);
- MemoryInjectionMiddleware:
  * awrap_model_call — перед каждым вызовом модели ищет релевантные факты по
    последней реплике пользователя и дописывает их в system prompt (view-level,
    история сообщений не меняется);
  * aafter_agent — после прогона агента фоново (asyncio.create_task, без задержки
    ответа) извлекает из последнего обмена новые факты (LLM-экстрактор) и пишет
    их в store;
- отказоустойчивость: недоступные embedding-сервис/store/LLM не блокируют диалог —
  ошибки памяти логируются, агент продолжает работать без неё.
"""

import asyncio
import hashlib
import json
import os
import re

from embedding_service.src.client import embedding_client
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.store.sqlite.aio import AsyncSqliteStore

from shared.utils.logger import get_logger

from .prompts import memory_extraction_prompt, memory_injection_prompt

logger = get_logger("agent_service.core.memory")

# Namespace store'а: воспоминания группируются по пользователю. user_id берётся
# из configurable (api.py), поэтому один и тот же namespace работает и для одного
# дефолтного менеджера, и для будущей авторизации.
MEMORY_NAMESPACE = "memories"

# Внутренние пределы экстрактора: локальный Qwen, окно 16k — на промпт оставляем запас
_MAX_FACTS_PER_TURN = 8
_MIN_FACT_CHARS = 3
_MAX_FACT_CHARS = 500
_EXCERPT_MAX_CHARS = 8000


def _current_user_id() -> str:
    """user_id из configurable запроса (api.py), с фолбэками на thread_id и DEFAULT_USER_ID.

    get_config() читает contextvar текущего запуска графа; вне запуска
    (фоновая задача уже стартована с захваченным user_id) не вызывается.
    """
    try:
        configurable = get_config().get("configurable") or {}
    except Exception:
        configurable = {}
    default = os.getenv("DEFAULT_USER_ID", "manager_default")
    return str(configurable.get("user_id") or configurable.get("thread_id") or default)


class EmbeddingServiceEmbeddings(Embeddings):
    """Мост между LangGraph Store и embedding-сервисом (FRIDA).

    AsyncSqliteStore вызывает aembed_documents и при aput (индексация фактов),
    и при asearch (эмбеддинг поискового запроса) — различить "search_document" и
    "search_query" на этом уровне невозможно, поэтому для обеих сторон используем
    симметричную instruct-задачу "paraphrase" ("Retrieve semantically similar text").
    """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await embedding_client.get_embeddings(texts, task="paraphrase")

    async def aembed_query(self, text: str) -> list[float]:
        return (await embedding_client.get_embeddings([text], task="paraphrase"))[0]

    # Абстрактные sync-методы Embeddings: единственный потребитель — AsyncSqliteStore,
    # который работает через aembed_*. Оставляем их нереализованными явно.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Синхронный путь не используется: store вызывает aembed_*")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("Синхронный путь не используется: store вызывает aembed_*")


def build_memory_store(conn) -> AsyncSqliteStore:
    """Store на готовом aiosqlite-соединении с семантическим индексом.

    Соединение должно быть открыто с isolation_level=None (autocommit) —
    store сам управляет транзакциями, иначе ловим «transaction within a transaction».
    Расширение sqlite-vec (векторный поиск) — зависимость langgraph-checkpoint-sqlite
    и загружается в conn внутри store.setup().
    """
    dims = int(os.getenv("MEMORY_EMBEDDING_DIMS", "1536"))  # размерность FRIDA
    return AsyncSqliteStore(
        conn,
        index={"embed": EmbeddingServiceEmbeddings(), "dims": dims, "fields": ["text"]},
    )


def _message_text(content) -> str:
    """Текст сообщения: строка или список parts (так отдаёт content LLM-ответов)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = (part.get("text", "") for part in content if isinstance(part, dict))
        return "\n".join(filter(None, (part.strip() for part in parts)))
    return ""


def _last_user_text(messages) -> str | None:
    """Текст последней реплики пользователя — запрос для семантического поиска."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = _message_text(message.content)
            return text or None
    return None


def _last_exchange(messages) -> list | None:
    """Последний обмен: от последней реплики пользователя до финального ответа ассистента."""
    start = None
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = i
    if start is None:
        return None
    excerpt = messages[start:]
    # Извлекаем только из завершённого обмена: после вопроса пользователя должен быть ответ
    if not any(isinstance(m, AIMessage) and _message_text(m.content) for m in excerpt[1:]):
        return None
    return excerpt


def _format_excerpt(messages) -> str:
    """Диалог для экстрактора: только реплики пользователя и ассистента, без тулов."""
    lines = []
    for message in messages:
        text = _message_text(message.content)
        if not text:
            continue
        role = "Пользователь" if isinstance(message, HumanMessage) else "Ассистент"
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)[-_EXCERPT_MAX_CHARS:]


def _parse_facts(raw: str | None) -> list[str]:
    """Разбор ответа экстрактора: JSON-массив строк (терпим ```-заборы и мусор вокруг)."""
    if not raw:
        return []
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    facts: list[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        fact = item.strip()
        if _MIN_FACT_CHARS < len(fact) <= _MAX_FACT_CHARS and fact not in facts:
            facts.append(fact)
    return facts[:_MAX_FACTS_PER_TURN]


class MemoryInjectionMiddleware(AgentMiddleware):
    """Долгосрочная семантическая память: инъекция фактов + фоновое извлечение.

    Store берётся из runtime (create_agent(store=...)), экстрактор — отдельный
    LLM-клиент (конструктор), чтобы тесты могли подставить фейк.
    """

    def __init__(self, extract_llm, *, top_k: int | None = None, extraction_enabled: bool | None = None):
        self.extract_llm = extract_llm
        self.top_k = top_k if top_k is not None else int(os.getenv("MEMORY_TOP_K", "5"))
        if extraction_enabled is not None:
            self.extraction_enabled = extraction_enabled
        else:
            self.extraction_enabled = os.getenv("MEMORY_EXTRACTION_ENABLED", "true").lower() in ("1", "true", "yes")
        # Сильные ссылки на фоновые задачи: asyncio хранит задачи слабо, без этого GC может убить их
        self._background_tasks: set[asyncio.Task] = set()

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        """Дописывает релевантные факты из store в system prompt (view-level)."""
        store = getattr(request.runtime, "store", None) if request.runtime is not None else None
        query = _last_user_text(request.messages) if store is not None else None
        if query is None:
            return await handler(request)
        try:
            items = await store.asearch((MEMORY_NAMESPACE, _current_user_id()), query=query, limit=self.top_k)
        except Exception:
            logger.warning("Память: поиск фактов не удался, продолжаем без памяти", exc_info=True)
            return await handler(request)
        if not items:
            return await handler(request)
        facts = "\n".join(f"- {item.value.get('text', '').strip()}" for item in items)
        logger.debug(f"Память: инъекция {len(items)} фактов")
        base = request.system_message.content if request.system_message is not None else ""
        system_message = SystemMessage(content=f"{base}\n\n{memory_injection_prompt.format(facts=facts)}")
        return await handler(request.override(system_message=system_message))

    async def aafter_agent(self, state, runtime):
        """Запускает фоновое извлечение фактов из последнего обмена (ответ уже не ждёт)."""
        store = getattr(runtime, "store", None) if runtime is not None else None
        if store is None or not self.extraction_enabled or self.extract_llm is None:
            return None
        excerpt = _last_exchange(list(state.get("messages") or []))
        if excerpt is None:
            return None
        # user_id захватываем до create_task: contextvar запуска графа в фоновой задаче уже недоступен
        task = asyncio.create_task(self._extract_and_store(store, _current_user_id(), excerpt))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return None

    async def _extract_and_store(self, store, user_id: str, excerpt: list) -> None:
        """Извлекает факты из обмена LLM-экстрактором и пишет их в store (идемпотентно)."""
        try:
            prompt = memory_extraction_prompt.format(messages=_format_excerpt(excerpt))
            response = await self.extract_llm.ainvoke(prompt)
            facts = _parse_facts(_message_text(response.content))
            if not facts:
                return
            namespace = (MEMORY_NAMESPACE, user_id)
            for fact in facts:
                # Ключ — хэш текста: повторное извлечение того же факта перезапишет запись, а не задублирует
                key = hashlib.sha1(fact.encode("utf-8")).hexdigest()[:16]
                await store.aput(namespace, key, {"text": fact}, index=["text"])
            logger.info(f"Память: сохранено фактов: {len(facts)} (user={user_id})")
        except Exception:
            logger.warning("Память: извлечение фактов не удалось", exc_info=True)
