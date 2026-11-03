"""
Сборка агентов: корневой диалоговый + суб-агент экспертизы по удержанию.

Краткосрочная память корневого агента — persistent checkpointer (SQLite).
Инициализация — init_root_agent() при старте сервиса (вызывается из main.py).
"""
import os
from pathlib import Path

import aiosqlite
import httpx
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from shared.utils.logger import get_logger
from shared.utils.text import sanitize_llm_text

from .llm_client import get_llm_client, get_optional_llm_client, llm
from .prompts import (
    churn_analisys_and_retention_agent_prompt,
    churn_dialogue_summary_prompt,
    root_agent_prompt,
)
from .tools import get_churn_info

logger = get_logger("agent_service.core.agent")

# Профили LLM из llm_config.json
SUMMARIZER_PROFILE = os.getenv("SUMMARIZER_LLM_PROFILE", "local_qwen_summarizer")
FALLBACK_PROFILE = os.getenv("FALLBACK_LLM_PROFILE", "summarizer")


def _fallback_middleware() -> list:
    """ModelFallbackMiddleware, если у fallback-профиля задан API-ключ, иначе — пустой список."""
    fallback_llm = get_optional_llm_client(FALLBACK_PROFILE)
    if fallback_llm is None:
        logger.info(f"Fallback-модель недоступна (нет ключа для профиля '{FALLBACK_PROFILE}') — контур отключён")
        return []
    return [ModelFallbackMiddleware(fallback_llm)]


# =========  Суб-агент для получения экспертизы по корпоративным политикам и маркетинговым предложениям по удержанию.  =========

churn_analisys_and_retention_agent = create_agent(
    llm,
    tools=[get_churn_info],
    system_prompt=churn_analisys_and_retention_agent_prompt,
    middleware=_fallback_middleware() + [
        # Сетевые сбои тулов (prediction-сервис, embedding, Qdrant) и 5xx: 2 ретрая,
        # затем ошибка уходит модели текстом
        ToolRetryMiddleware(
            max_retries=2,
            tools=["get_churn_info"],
            retry_on=(httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError),
            on_failure="continue",
        ),
        # Страховка от зацикливания инструмента в рамках одного запуска
        ToolCallLimitMiddleware(tool_name="get_churn_info", run_limit=3, exit_behavior="end"),
    ],
)

# Тул, в рамках которого будет работать суб-агент churn_analisys_and_retention_agent
@tool(
    "get_churn_analisys_and_retention",
    description="Используй для получения экспертизы по корпоративным политикам и маркетинговым предложениям по удержанию. \
        Не используй для обычного диалога или приветствий."
)
async def get_churn_analisys_and_retention(query: str):
    logger.info("Задача передана агенту churn_analisys_and_retention_agent")
    result = await churn_analisys_and_retention_agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    # Суб-агент эфемерный (без checkpointer): его внутренняя история не сохраняется,
    # наружу отдаём только финальный ответ
    return f"Результат по твоему запросу {query}: \n{sanitize_llm_text(result['messages'][-1].content)}"


# ==========   Основной агент: диалог с пользователем и раздача задач суб-агентам   ==========

def _root_middleware() -> list:
    """Стек middleware корневого агента: fallback, суммаризация истории, лимит вызовов тулов."""
    middleware = _fallback_middleware()

    # Порог задается в токенах и должен быть заметно меньше MAX_MODEL_LEN vLLM
    # (при 16384 — 10000, при 8192 — ~5000)
    trigger_tokens = int(os.getenv("SUMMARIZATION_TRIGGER_TOKENS", 10000))
    keep_messages = int(os.getenv("SUMMARIZATION_KEEP_MESSAGES", 8))
    middleware.append(
        SummarizationMiddleware(
            model=get_llm_client(SUMMARIZER_PROFILE),
            trigger=("tokens", trigger_tokens),
            keep=("messages", keep_messages),
            summary_prompt=churn_dialogue_summary_prompt,
        )
    )

    middleware.append(
        ToolCallLimitMiddleware(
            tool_name="get_churn_analisys_and_retention", run_limit=5, exit_behavior="end"
        )
    )
    return middleware


def build_root_agent(checkpointer, model=None):
    """Корневой агент с persistent-памятью. model — точка расширения для тестов."""
    return create_agent(
        model or llm,
        tools=[get_churn_analisys_and_retention],
        system_prompt=root_agent_prompt,
        middleware=_root_middleware(),
        checkpointer=checkpointer,
    )


# =========  Жизненный цикл корневого агента (lifespan в main.py)  =========

root_agent = None
_checkpointer_conn = None
_db_path = None


async def init_root_agent() -> None:
    """Открывает SQLite checkpointer и собирает root_agent.

    Вызывается один раз при старте сервиса: соединение aiosqlite должно жить
    в том же event loop, который обслуживает запросы.
    """
    global root_agent, _checkpointer_conn, _db_path

    db_path = os.getenv("CHECKPOINT_DB_PATH") or str(
        Path(__file__).resolve().parents[1] / "checkpoints.db"
    )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _checkpointer_conn = await aiosqlite.connect(db_path)
    _db_path = db_path
    root_agent = build_root_agent(AsyncSqliteSaver(_checkpointer_conn))
    logger.info(f"root_agent готов (checkpointer: {db_path})")


async def shutdown_root_agent() -> None:
    """Закрывает соединение checkpointer'а при остановке сервиса."""
    global root_agent, _checkpointer_conn, _db_path
    if _checkpointer_conn is not None:
        await _checkpointer_conn.close()
        logger.info("Соединение checkpointer'а закрыто")
    _checkpointer_conn = None
    _db_path = None
    root_agent = None


def get_root_agent():
    """Доступ к корневому агенту после init_root_agent()."""
    return root_agent


def get_checkpoint_db_path() -> str | None:
    """Путь к SQLite-файлу checkpointer'а: там же живет служебная таблица потоков API-слоя."""
    return _db_path
