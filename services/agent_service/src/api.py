"""
REST/SSE API для веб-фронтенда (frontend_service).

Состояние диалогов:
- история сообщений — checkpointer LangGraph (SQLite, core.agent);
- служебная таблица thread_meta (в том же SQLite-файле) хранит заголовки,
  привязку к клиенту и время обновления чатов — это список диалогов
  для левой панели фронтенда. Треды без записи (служебные разборы) не видны.

Единый протокол SSE-стрима для чата и разборов клиентов:
  {"type": "tool",  "phase": "start" | "end", "name": str, "ok": bool?}
  {"type": "token", "text": str}
  {"type": "data",  "client_id", "churn_probability", "reasons": [...]}   — только /analyze
  {"type": "done"}  /  {"type": "error", "message": str}

Выгрузка клиентов живёт в SQLite (clients_store): CSV перегоняется туда при
старте сервиса, /api/clients и /api/clients/{id} читают из БД.
"""
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from shared.constants import BINARY_FEATURES, FEATURE_MAP, FEATURE_VALUE_MAPS
from shared.utils.http import create_async_client
from shared.utils.logger import get_logger
from shared.utils.text import sanitize_llm_text

import clients_store as store
from core.agent import get_checkpoint_db_path, get_root_agent
from core.callbacks import TokenUsageCallback

logger = get_logger("agent_service.api")

router = APIRouter(prefix="/api")


# =========  SSE-утилиты  =========

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_response(generator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _user_id() -> str:
    return os.getenv("DEFAULT_USER_ID", "manager_default")


def _content_text(content) -> str:
    """Достаёт текст из content-поля сообщения (str или список parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text"):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return ""


# =========  Служебная таблица диалогов (thread_meta)  =========

_meta_conn: aiosqlite.Connection | None = None


async def init_thread_meta() -> None:
    """Открывает соединение к БД checkpointer'а и создаёт таблицу заголовков чатов."""
    global _meta_conn
    db_path = get_checkpoint_db_path()
    if not db_path:
        return
    _meta_conn = await aiosqlite.connect(db_path)
    await _meta_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_meta (
            thread_id  TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            client_id  TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    await _meta_conn.commit()
    logger.info(f"thread_meta готова ({db_path})")


async def close_thread_meta() -> None:
    global _meta_conn
    if _meta_conn is not None:
        await _meta_conn.close()
    _meta_conn = None


def _short_title(text: str) -> str:
    title = (text or "").strip().replace("\n", " ") or "Диалог"
    return title[:57] + "…" if len(title) > 60 else title


async def _touch_thread(thread_id: str, title: str | None = None, client_id: str | None = None) -> None:
    """Создаёт запись о чате (первое сообщение) или продлевает updated_at."""
    if _meta_conn is None:
        raise HTTPException(503, "Хранилище диалогов ещё инициализируется")
    now = datetime.now(timezone.utc).isoformat()
    cur = await _meta_conn.execute("SELECT thread_id FROM thread_meta WHERE thread_id = ?", (thread_id,))
    row = await cur.fetchone()
    if row is None:
        await _meta_conn.execute(
            "INSERT INTO thread_meta (thread_id, title, client_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, _short_title(title), client_id, now, now),
        )
    else:
        await _meta_conn.execute("UPDATE thread_meta SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
    await _meta_conn.commit()


# =========  Модели запросов  =========

class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str = Field(min_length=1)


class SeedRequest(BaseModel):
    """Создание чата с уже готовым разбором клиента в памяти агента."""
    title: str
    analysis_text: str
    client_id: str | None = None
    thread_id: str | None = None


class AnalyzeRequest(BaseModel):
    client_id: int


# =========  Общий стрим агента (чат)  =========

async def _stream_agent(agent, message_text: str, thread_id: str, text_sink: list | None = None):
    """Стримит работу корневого агента в SSE: старт/финиш тулов и токены ответов.

    Текст последнего сообщения при text_sink накапливается списком (для кэша разборов).
    """
    config = {
        "configurable": {"thread_id": thread_id, "user_id": _user_id()},
        "metadata": {"thread_id": thread_id},
        "callbacks": [TokenUsageCallback()],
    }
    started_tools: dict[str, str] = {}  # tool_call_id -> name
    ended_tools: set[str] = set()
    try:
        async for mode, payload in agent.astream(
            {"messages": [HumanMessage(message_text)]},
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk = payload[0]
                # create_agent в зависимости от middleware-стека отдаёт либо AIMessageChunk
                # (потоковый режим), либо готовый AIMessage одним событием — обрабатываем оба
                if not isinstance(chunk, AIMessage):
                    continue
                # Старт тулов: модель начала генерировать tool call
                for tc in getattr(chunk, "tool_call_chunks", None) or []:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    if tc_id and name and tc_id not in started_tools:
                        started_tools[tc_id] = name
                        yield _sse({"type": "tool", "phase": "start", "name": name})
                if not isinstance(chunk, AIMessageChunk):
                    # Готовое сообщение: tool_calls уже собраны
                    for tc in chunk.tool_calls or []:
                        tc_id, name = tc.get("id"), tc.get("name")
                        if tc_id and name and tc_id not in started_tools:
                            started_tools[tc_id] = name
                            yield _sse({"type": "tool", "phase": "start", "name": name})
                text = _content_text(chunk.content)
                if text:
                    if text_sink is not None:
                        text_sink.append(text)
                    yield _sse({"type": "token", "text": text})
            elif mode == "updates":
                deltas = payload.values() if isinstance(payload, dict) else []
                for delta in deltas:
                    if not isinstance(delta, dict):
                        continue
                    for msg in delta.get("messages", []) or []:
                        if (
                            isinstance(msg, ToolMessage)
                            and msg.tool_call_id in started_tools
                            and msg.tool_call_id not in ended_tools
                        ):
                            ended_tools.add(msg.tool_call_id)
                            yield _sse({
                                "type": "tool",
                                "phase": "end",
                                "name": msg.name or started_tools[msg.tool_call_id],
                                "ok": getattr(msg, "status", "success") == "success",
                            })
        yield _sse({"type": "done"})
    except Exception as exc:
        logger.exception(f"Ошибка стрима агента (thread={thread_id})")
        yield _sse({"type": "error", "message": str(exc)})


@router.post("/chat")
async def chat(body: ChatRequest):
    """Сообщение в чат: SSE-стрим ответа. thread_id генерируется клиентом или здесь."""
    agent = get_root_agent()
    if agent is None:
        raise HTTPException(503, "Агент ещё инициализируется, повторите запрос через пару секунд")
    thread_id = body.thread_id or str(uuid.uuid4())
    if len(thread_id) > 128:
        raise HTTPException(422, "Слишком длинный thread_id")
    await _touch_thread(thread_id, title=body.message)
    return _sse_response(_stream_agent(agent, body.message, thread_id))


# =========  Список и история диалогов  =========

@router.get("/threads")
async def list_threads():
    """Диалоги для левой панели: только чаты с метаданными (служебные разборы скрыты)."""
    if _meta_conn is None:
        raise HTTPException(503, "Хранилище диалогов ещё инициализируется")
    cur = await _meta_conn.execute(
        "SELECT thread_id, title, client_id, updated_at FROM thread_meta ORDER BY updated_at DESC"
    )
    rows = [dict(zip(["thread_id", "title", "client_id", "updated_at"], r)) for r in await cur.fetchall()]
    return {"threads": rows}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """История диалога из checkpointer'а: только текстовые user/assistant сообщения."""
    agent = get_root_agent()
    if agent is None:
        raise HTTPException(503, "Агент ещё инициализируется")
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    values = getattr(state, "values", None) or {}
    messages = []
    for msg in values.get("messages", []):
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": _content_text(msg.content)})
        elif isinstance(msg, AIMessage):
            text = _content_text(msg.content)
            if text.strip():
                messages.append({"role": "assistant", "content": text})
    return {"thread_id": thread_id, "messages": messages}


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Удаление чата: запись о заголовке + история сообщений из checkpointer'а."""
    if _meta_conn is None:
        raise HTTPException(503, "Хранилище диалогов ещё инициализируется")
    cur = await _meta_conn.execute("DELETE FROM thread_meta WHERE thread_id = ?", (thread_id,))
    await _meta_conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, f"Чат {thread_id} не найден")
    # История в checkpointer'е: без неё чат не «воскреснет» при повторном thread_id.
    # Не чинится — чат уже пропал из списка, ошибка записи не должна ломать ответ.
    agent = get_root_agent()
    checkpointer = getattr(agent, "checkpointer", None)
    if checkpointer is not None and hasattr(checkpointer, "adelete_thread"):
        try:
            await checkpointer.adelete_thread(thread_id)
        except Exception:
            logger.exception(f"Не удалось удалить историю чата {thread_id} из checkpointer'а")
    return {"deleted": thread_id}


@router.post("/threads/seed")
async def seed_thread(body: SeedRequest):
    """Создаёт чат, в памяти агента которого уже лежит готовый разбор клиента.

    Разбор записывается как сообщение ассистента (update_state) — при переходе
    из нижней панели агент «помнит», что менеджер уже видел, и история в UI
    начинается с этого разбора.
    """
    agent = get_root_agent()
    if agent is None:
        raise HTTPException(503, "Агент ещё инициализируется")
    thread_id = body.thread_id or str(uuid.uuid4())
    await _touch_thread(thread_id, title=body.title, client_id=body.client_id)
    config = {"configurable": {"thread_id": thread_id, "user_id": _user_id()}}
    last_error: Exception | None = None
    # Имя узла модели зависит от версии create_agent: пробуем известные варианты
    for as_node in ("model", "agent", None):
        try:
            await agent.aupdate_state(config, {"messages": [AIMessage(content=body.analysis_text)]}, as_node=as_node)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            logger.warning(f"aupdate_state (as_node={as_node}) не сработал: {exc}")
    if last_error is not None:
        raise HTTPException(500, f"Не удалось записать контекст разбора в память агента: {last_error}")
    return {"thread_id": thread_id}


# =========  Разбор клиента одной кнопкой (нижняя панель)  =========

def _format_value(feature: str, value) -> str:
    if feature in BINARY_FEATURES:
        try:
            return "Да" if int(value) == 1 else "Нет"
        except (TypeError, ValueError):
            return str(value)
    value_map = FEATURE_VALUE_MAPS.get(feature)
    if value_map:
        return value_map.get(str(value), str(value))
    return str(value)


def _labeled_reasons(prediction: dict) -> list[dict]:
    """SHAP-факторы с человекопонятными подписями из FEATURE_MAP."""
    reasons = []
    for reason in prediction.get("top_reasons", []):
        feature = reason.get("feature", "")
        reasons.append({
            "feature": feature,
            "label": FEATURE_MAP.get(feature, feature.replace("_", " ")),
            "value": _format_value(feature, reason.get("feature_value")),
            "pct": reason.get("risk_contribution_pct", 0.0),
        })
    return reasons


async def _fetch_prediction(client_id: int) -> dict | None:
    """Прогноз напрямую от churn_pred_service — для карточки клиента. None при недоступности."""
    base_url = os.getenv("CHURN_PRED_SERVICE_URL", "http://localhost:3010")
    try:
        async with create_async_client(timeout=5.0) as http:
            response = await http.get(f"{base_url}/predict", params={"client_id": client_id})
        if response.status_code == 200:
            return response.json()
        logger.warning(f"prediction-сервис вернул {response.status_code} для клиента {client_id}")
    except Exception:
        logger.warning(f"prediction-сервис недоступен для клиента {client_id}")
    return None


# Результаты разборов в памяти процесса: повторный клик по клиенту — мгновенно
_analyze_cache: dict[str, dict] = {}


async def _replay_analysis(item: dict):
    """Повтор выдачи закэшированного разбора: те же события, без обращения к LLM."""
    if item.get("data"):
        yield _sse({"type": "data", **item["data"]})
    yield _sse({"type": "token", "text": item["text"]})
    yield _sse({"type": "done"})


async def _run_analysis(agent, client_id: int):
    """Полный разбор: структурированные данные + текст агентом (тулы, RAG и т.д.)."""
    prediction = await _fetch_prediction(client_id)
    data = None
    if prediction:
        data = {
            "client_id": client_id,
            "churn_probability": prediction.get("churn_probability"),
            "reasons": _labeled_reasons(prediction),
        }
        yield _sse({"type": "data", **data})
    else:
        yield _sse({"type": "data", "client_id": client_id, "churn_probability": None, "reasons": []})

    prompt = (
        f"Клиент с ID {client_id}. Дай краткий разбор клиента: вероятность оттока, основные причины "
        "и конкретный план удержания с опорой на корпоративные политики."
    )
    thread_id = f"analysis-{client_id}-{uuid.uuid4().hex[:8]}"
    parts: list[str] = []
    async for event in _stream_agent(agent, prompt, thread_id, text_sink=parts):
        yield event
    if parts:
        _analyze_cache[str(client_id)] = {"text": sanitize_llm_text("".join(parts)), "data": data}


@router.post("/analyze")
async def analyze(body: AnalyzeRequest):
    """Разбор клиента одной кнопкой: SSE-стрим в нижнюю панель."""
    agent = get_root_agent()
    if agent is None:
        raise HTTPException(503, "Агент ещё инициализируется, повторите запрос через пару секунд")
    cached = _analyze_cache.get(str(body.client_id))
    if cached:
        return _sse_response(_replay_analysis(cached))
    return _sse_response(_run_analysis(agent, body.client_id))


# =========  Список клиентов (SQLite-выгрузка + риски)  =========

async def _fetch_risks(client_ids: list[int]) -> dict[int, float | None]:
    """Риски пачкой параллельно: локальный predict-сервис отвечает миллисекундами."""
    base_url = os.getenv("CHURN_PRED_SERVICE_URL", "http://localhost:3010")
    semaphore = asyncio.Semaphore(8)
    risks: dict[int, float | None] = {}

    async def fetch(client_id: int) -> None:
        async with semaphore:
            prediction = await _fetch_prediction(client_id)
            risks[client_id] = prediction.get("churn_probability") if prediction else None

    # _fetch_prediction открывает свой клиент на вызов; локальный predict отвечает быстро
    await asyncio.gather(*(fetch(cid) for cid in client_ids))
    return risks


def _display_client(row: dict) -> dict:
    """Строка для UI: категориальные значения в русском отображении."""
    value_map = FEATURE_VALUE_MAPS.get("Geography", {})
    return {**row, "geography": value_map.get(row["geography"], row["geography"])}


@router.get("/clients")
async def list_clients(limit: int = 50, offset: int = 0, q: str | None = None):
    """Страница выгрузки клиентов с вероятностями оттока (None, если predict-сервис недоступен).

    q — поиск по фамилии или ID (в БД clients_store).
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    try:
        rows = await store.fetch_clients(limit=limit, offset=offset, search=q)
        total = await store.count_clients(search=q)
    except store.StoreNotReady:
        raise HTTPException(503, "Хранилище клиентов ещё инициализируется")
    risks = await _fetch_risks([row["client_id"] for row in rows])
    clients = [
        {**_display_client(row), "churn_probability": risks.get(row["client_id"])}
        for row in rows
    ]
    return {"total": total, "clients": clients}


@router.get("/clients/{client_id}")
async def get_client(client_id: int):
    """Один клиент по id (карточка, deep-link после seed) + свежий риск из predict-сервиса."""
    try:
        row = await store.fetch_client(client_id)
    except store.StoreNotReady:
        raise HTTPException(503, "Хранилище клиентов ещё инициализируется")
    if row is None:
        raise HTTPException(404, f"Клиент {client_id} не найден в выгрузке")
    prediction = await _fetch_prediction(client_id)
    row["churn_probability"] = prediction.get("churn_probability") if prediction else None
    return _display_client(row)


@router.get("/health")
async def health():
    return {"status": "ok", "agent_ready": get_root_agent() is not None}
