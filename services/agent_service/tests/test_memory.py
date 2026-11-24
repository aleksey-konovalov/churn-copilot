"""Тесты долгосрочной памяти (core/memory.py) и стека контекст-оптимизации (stage 3+5).

Запуск из корня репозитория:
    python -m pytest services/agent_service/tests/test_memory.py -q

Сеть не нужна: embedding-сервис не используется (память без векторного индекса),
LLM — фейки; store/агент собираются напрямую через create_agent.
"""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.store.memory import InMemoryStore
from pydantic import Field

SERVICE_ROOT = Path(__file__).resolve().parents[1]  # services/agent_service
REPO_ROOT = SERVICE_ROOT.parents[1]  # корень репозитория

for path in (str(SERVICE_ROOT / "src"), str(REPO_ROOT / "services"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("COLLECTION_NAME", "test_collection")
os.environ.setdefault("ACTIVE_LLM_PROFILE", "local_qwen_agent")

from core.agent import _memory_middleware, _root_middleware  # noqa: E402
from core.memory import (  # noqa: E402
    MemoryInjectionMiddleware,
    _last_exchange,
    _parse_facts,
    build_memory_store,
)
from core.prompts import root_agent_prompt  # noqa: E402


class EchoModel(BaseChatModel):
    """Фейк-модель: пишет все полученные сообщения в sink и отвечает 'ok'."""

    sink: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.sink.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    @property
    def _llm_type(self) -> str:
        return "echo"


class FakeExtractor:
    """Фейк LLM-экстрактора: возвращает фиксированный JSON-массив фактов."""

    def __init__(self, content: str):
        self.content = content

    async def ainvoke(self, prompt, **kwargs):
        return AIMessage(content=self.content)


def _memory_store() -> InMemoryStore:
    """InMemoryStore без индекса — как боевой store памяти (инъекция берёт последние факты)."""
    return InMemoryStore()


# === stage 3: стек middleware корневого агента ===


def test_root_middleware_order_and_defaults(monkeypatch):
    # OPENAI_API_KEY из локального .env включил бы fallback и сломал бы порядок — тест про стек без него
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CONTEXT_EDITING_TRIGGER_TOKENS", raising=False)
    stack = _root_middleware()
    types = [type(m).__name__ for m in stack]
    # Порядок важен: ContextEditing чистит tool-результаты ДО порога суммаризации
    assert types == ["ContextEditingMiddleware", "SummarizationMiddleware", "ToolCallLimitMiddleware"]
    edit = stack[0].edits[0]
    assert edit.trigger == 8000
    assert edit.keep == 1
    assert edit.clear_tool_inputs is True
    assert isinstance(stack[1], SummarizationMiddleware)


def test_root_middleware_env_overrides(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONTEXT_EDITING_TRIGGER_TOKENS", "4000")
    monkeypatch.setenv("CONTEXT_EDITING_KEEP_TOOL_RESULTS", "3")
    stack = _root_middleware()
    edit = stack[0].edits[0]
    assert edit.trigger == 4000
    assert edit.keep == 3


def test_root_middleware_context_editing_disabled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONTEXT_EDITING_TRIGGER_TOKENS", "0")
    types = [type(m).__name__ for m in _root_middleware()]
    assert types == ["SummarizationMiddleware", "ToolCallLimitMiddleware"]


# === stage 5: проводка памяти в build_root_agent ===


def test_memory_middleware_disabled_without_store():
    assert _memory_middleware(None) is None


def test_memory_middleware_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    assert _memory_middleware(_memory_store()) is None


def test_memory_middleware_created_with_store():
    middleware = _memory_middleware(_memory_store())
    assert isinstance(middleware, MemoryInjectionMiddleware)


def test_build_memory_store_roundtrip(tmp_path):
    """Store на реальном SQLite: setup() создаёт схему, put/search работают.

    Store без векторного индекса — sqlite-vec и embedding-сервис не нужны.
    """

    async def main():
        import aiosqlite

        # isolation_level=None обязателен: store сам управляет транзакциями
        conn = await aiosqlite.connect(tmp_path / "memory.db", isolation_level=None)
        try:
            store = build_memory_store(conn)
            await store.setup()
            await store.aput(("memories", "u1"), "f1", {"text": "факт"})
            return await store.asearch(("memories", "u1"), limit=5)
        finally:
            await conn.close()

    items = asyncio.run(main())
    assert [i.value["text"] for i in items] == ["факт"]


# === stage 5: инъекция фактов в system prompt ===


def test_injection_adds_relevant_facts():
    async def main():
        store = _memory_store()
        await store.aput(("memories", "u1"), "f1", {"text": "Пользователь предпочитает краткие ответы"})
        echo = EchoModel(sink=[])
        agent = create_agent(
            model=echo,
            tools=[],
            system_prompt=root_agent_prompt,
            middleware=[MemoryInjectionMiddleware(extract_llm=None, extraction_enabled=False)],
            store=store,
        )
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": "как дела?"}]},
            config={"configurable": {"thread_id": "t1", "user_id": "u1"}},
        )
        return echo.sink

    seen = asyncio.run(main())
    system_texts = [m.content for m in seen[0] if isinstance(m, SystemMessage)]
    assert any("Пользователь предпочитает краткие ответы" in t for t in system_texts)
    # основной system prompt сохранён, блок памяти дописан в конец
    assert any(t.startswith("Ты — дружелюбный корпоративный ассистент") for t in system_texts)


def test_injection_no_facts_keeps_system_prompt():
    async def main():
        store = _memory_store()  # пустой store
        echo = EchoModel(sink=[])
        agent = create_agent(
            model=echo,
            tools=[],
            system_prompt="СИСТЕМНЫЙ ПРОМПТ",
            middleware=[MemoryInjectionMiddleware(extract_llm=None, extraction_enabled=False)],
            store=store,
        )
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": "привет"}]},
            config={"configurable": {"thread_id": "t1", "user_id": "u1"}},
        )
        return echo.sink

    seen = asyncio.run(main())
    system_texts = [m.content for m in seen[0] if isinstance(m, SystemMessage)]
    assert system_texts == ["СИСТЕМНЫЙ ПРОМПТ"]


def test_middleware_passthrough_without_runtime():
    """Нет runtime/store — запрос уходит в модель без изменений (агент без памяти)."""
    request = ModelRequest(
        model=None,
        messages=[HumanMessage("привет")],
        system_message=SystemMessage(content="СИСТЕМНЫЙ ПРОМПТ"),
        tool_choice=None,
        tools=[],
        response_format=None,
        state={},
        runtime=None,
    )
    seen = []

    async def handler(req):
        seen.append(req.system_message)
        return AIMessage("ok")

    middleware = MemoryInjectionMiddleware(extract_llm=None, extraction_enabled=False)
    asyncio.run(middleware.awrap_model_call(request, handler))
    assert len(seen) == 1
    assert seen[0].content == "СИСТЕМНЫЙ ПРОМПТ"


def test_injection_takes_latest_facts_up_to_top_k():
    """Инъекция не зависит от темы вопроса: берутся свежие факты, старые вытесняются."""
    seen = []

    async def handler(req):
        seen.append(req.system_message)
        return AIMessage("ok")

    async def main():
        store = _memory_store()
        middleware = MemoryInjectionMiddleware(extract_llm=None, extraction_enabled=False, top_k=2)
        runtime = type("R", (), {"store": store})()
        # Прямой вызов вне запуска графа: configurable недоступен, user_id = manager_default
        namespace = ("memories", "manager_default")
        await store.aput(namespace, "old", {"text": "СТАРЫЙ ФАКТ"})
        await asyncio.sleep(0.002)  # разные updated_at: сортировка по свежести должна быть устойчивой
        await store.aput(namespace, "mid", {"text": "СРЕДНИЙ ФАКТ"})
        await asyncio.sleep(0.002)
        await store.aput(namespace, "new", {"text": "НОВЫЙ ФАКТ"})
        request = ModelRequest(
            model=None,
            messages=[HumanMessage("разбор клиента 71")],  # вопрос про клиента — факты всё равно подставляются
            system_message=SystemMessage(content="СИСТЕМНЫЙ ПРОМПТ"),
            tool_choice=None,
            tools=[],
            response_format=None,
            state={},
            runtime=runtime,
        )
        await middleware.awrap_model_call(request, handler)
        return seen[0].content

    text = asyncio.run(main())
    assert "СИСТЕМНЫЙ ПРОМПТ" in text
    assert "НОВЫЙ ФАКТ" in text and "СРЕДНИЙ ФАКТ" in text
    assert "СТАРЫЙ ФАКТ" not in text


# === stage 5: извлечение фактов ===


def test_aafter_agent_extracts_and_stores_in_background():
    async def main():
        store = _memory_store()
        middleware = MemoryInjectionMiddleware(extract_llm=FakeExtractor('["Пользователь работает с VIP-клиентами"]'))
        state = {"messages": [HumanMessage("расскажи про клиента 12"), AIMessage("отчёт готов")]}
        runtime = type("R", (), {"store": store})()
        await middleware.aafter_agent(state, runtime)
        assert middleware._background_tasks, "фоновая задача извлечения должна быть создана"
        await asyncio.gather(*list(middleware._background_tasks))
        return store

    store = asyncio.run(main())
    items = asyncio.run(store.asearch(("memories", "manager_default"), limit=5))
    assert [i.value["text"] for i in items] == ["Пользователь работает с VIP-клиентами"]


def test_aafter_agent_skips_when_disabled_or_incomplete():
    async def main():
        store = _memory_store()
        middleware = MemoryInjectionMiddleware(extract_llm=FakeExtractor("[]"))
        runtime = type("R", (), {"store": store})()

        # извлечение выключено
        middleware.extraction_enabled = False
        state = {"messages": [HumanMessage("вопрос"), AIMessage("ответ")]}
        await middleware.aafter_agent(state, runtime)
        first = len(middleware._background_tasks)

        # нет ответа ассистента — незавершённый обмен
        middleware.extraction_enabled = True
        await middleware.aafter_agent({"messages": [HumanMessage("вопрос")]}, runtime)
        return first, len(middleware._background_tasks)

    first, second = asyncio.run(main())
    assert first == 0 and second == 0


def test_aafter_agent_without_store_is_noop():
    async def main():
        middleware = MemoryInjectionMiddleware(extract_llm=FakeExtractor('["факт"]'))
        return await middleware.aafter_agent({"messages": [HumanMessage("в"), AIMessage("о")]}, runtime=None)

    assert asyncio.run(main()) is None


# === Юнит-тесты хелперов ===


def test_parse_facts_variants():
    # чистый JSON
    assert _parse_facts('["факт один", "факт два"]') == ["факт один", "факт два"]
    # ```-заборы и текст вокруг массива
    assert _parse_facts('```json\n["факт"]\n```') == ["факт"]
    assert _parse_facts('Ответ: ["факт"] конец.') == ["факт"]
    # не-строки, дубликаты, короткий/длинный мусор
    raw = json.dumps(["факт", "факт", "ок", 5, None, "x" * 501])
    parsed = _parse_facts(raw)
    assert parsed == ["факт"]
    # мусор без массива
    assert _parse_facts("никакого json") == []
    assert _parse_facts("") == []
    assert _parse_facts(None) == []


def test_last_exchange():
    messages = [
        HumanMessage("первый вопрос"),
        AIMessage("первый ответ"),
        HumanMessage("второй вопрос"),
        AIMessage([{"type": "text", "text": "финальный ответ"}]),
        HumanMessage("третий вопрос"),  # незавершённый обмен без ответа
    ]
    exchange = _last_exchange(messages)
    assert exchange is None  # последний обмен не завершён — извлекать нечего

    completed = messages[:-1]
    exchange = _last_exchange(completed)
    assert [type(m).__name__ for m in exchange] == ["HumanMessage", "AIMessage"]
    assert _last_exchange([AIMessage("без вопросов")]) is None
    assert _last_exchange([]) is None


def test_fact_key_stable_and_dedup():
    """Идемпотентность записи: тот же факт перезаписывает ту же запись store'а."""
    fact = "Пользователь просил отвечать без Markdown"
    key = hashlib.sha1(fact.encode("utf-8")).hexdigest()[:16]

    async def main():
        store = _memory_store()
        namespace = ("memories", "u1")
        await store.aput(namespace, key, {"text": fact})
        await store.aput(namespace, key, {"text": fact})
        return await store.asearch(namespace, limit=10)

    items = asyncio.run(main())
    assert len(items) == 1
    assert items[0].key == key
