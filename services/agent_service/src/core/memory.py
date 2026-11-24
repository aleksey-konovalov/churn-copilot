"""
Долгосрочная память корневого агента (stage 5 плана контекст-оптимизации).

Что даёт: агент вспоминает устойчивые предпочтения пользователя из прошлых
диалогов (стиль и детальность ответов, роль, паттерны работы, правки), даже
если они вытеснены из контекста суммаризацией или диалог начался заново.
В память попадает только то, что полезно независимо от темы и клиента;
данные конкретных клиентов (вероятности, причины, планы) туда не пишутся.

Механика:
- Store: AsyncSqliteStore без векторного индекса (простое KV-хранилище) —
  переживает рестарт сервиса; факты лежат в namespace ("memories", user_id);
- MemoryInjectionMiddleware:
  * awrap_model_call — перед каждым вызовом модели подставляет последние
    top_k фактов пользователя в system prompt (view-level, история сообщений
    не меняется). Без семантического поиска по реплике: факты — про
    пользователя, а не про клиентов, поэтому релевантны в любом диалоге;
  * aafter_agent — после прогона агента фоново (asyncio.create_task, без задержки
    ответа) извлекает из последнего обмена новые факты (LLM-экстрактор) и пишет
    их в store;
- отказоустойчивость: недоступные store/LLM не блокируют диалог — ошибки памяти
  логируются, агент продолжает работать без неё.
"""

import asyncio
import hashlib
import json
import os
import re

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
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

# Сколько последних записей сканируем из store перед сортировкой по свежести
# (де-факто верхняя граница «глубины» памяти: старое за пределами скана не всплывает)
_FACTS_SCAN_LIMIT = 100


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


def build_memory_store(conn) -> AsyncSqliteStore:
    """Store на готовом aiosqlite-соединении, без векторного индекса.

    Семантический поиск не нужен: в памяти только предпочтения и паттерны работы
    пользователя — они уместны в любом диалоге, поэтому инъекция просто берёт
    последние top_k фактов. Без индекса store не требует embedding-сервис и не
    грузит sqlite-vec. Соединение должно быть открыто с isolation_level=None
    (autocommit) — store сам управляет транзакциями, иначе ловим
    «transaction within a transaction».
    """
    return AsyncSqliteStore(conn)


def _message_text(content) -> str:
    """Текст сообщения: строка или список parts (так отдаёт content LLM-ответов)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = (part.get("text", "") for part in content if isinstance(part, dict))
        return "\n".join(filter(None, (part.strip() for part in parts)))
    return ""


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
        """Дописывает последние факты о пользователе в system prompt (view-level)."""
        store = getattr(request.runtime, "store", None) if request.runtime is not None else None
        if store is None:
            return await handler(request)
        try:
            items = await store.asearch((MEMORY_NAMESPACE, _current_user_id()), limit=_FACTS_SCAN_LIMIT)
        except Exception:
            logger.warning("Память: чтение фактов не удалось, продолжаем без памяти", exc_info=True)
            return await handler(request)
        # Свежие факты важнее: сортировка по updated_at (перезаписанный факт поднимается наверх)
        items = sorted(items, key=lambda item: item.updated_at, reverse=True)[: self.top_k]
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
                await store.aput(namespace, key, {"text": fact})
            logger.info(f"Память: сохранено фактов: {len(facts)} (user={user_id})")
        except Exception:
            logger.warning("Память: извлечение фактов не удалось", exc_info=True)
