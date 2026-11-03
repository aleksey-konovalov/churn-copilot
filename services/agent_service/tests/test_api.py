"""Тесты REST API-слоя agent_service (frontend_service его проксирует).

Запуск из корня репозитория:
    python -m pytest services/agent_service/tests -q

LLM, prediction-сервис и Qdrant не нужны: агент не инициализируется,
недоступный predict-сервис даёт churn_probability=None.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

SERVICE_ROOT = Path(__file__).resolve().parents[1]      # services/agent_service
REPO_ROOT = SERVICE_ROOT.parents[1]                     # корень репозитория

for path in (str(SERVICE_ROOT / "src"), str(REPO_ROOT / "services"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("COLLECTION_NAME", "test_collection")
os.environ.setdefault("ACTIVE_LLM_PROFILE", "local_qwen_agent")

from api import _short_title, _stream_agent  # noqa: E402
from clients_store import close_clients_store, init_clients_store  # noqa: E402
from main import app  # noqa: E402


# ---------- чистые функции ----------

def test_short_title_truncates():
    assert _short_title("Привет") == "Привет"
    long = "к" * 100
    title = _short_title(long)
    assert len(title) == 58
    assert title.endswith("…")


def test_short_title_multiline_and_empty():
    assert _short_title("первая строка\nвторая") == "первая строка вторая"
    assert _short_title("") == "Диалог"
    assert _short_title("   ") == "Диалог"


# ---------- SSE-стрим агента ----------

class _FakeAgent:
    """Отдаёт заранее заданные события astream(stream_mode=["messages", "updates"])."""

    def __init__(self, events):
        self._events = events

    async def astream(self, input, config=None, stream_mode=None):
        for event in self._events:
            yield event


def _parse_sse(chunks):
    return [json.loads(chunk[len("data: "):]) for chunk in chunks]


def test_stream_agent_handles_full_ai_message():
    """create_agent с middleware отдаёт готовые AIMessage (не chunk): тулы и текст должны попасть в SSE."""
    events = [
        ("messages", (AIMessage(content="", tool_calls=[
            {"id": "t1", "name": "get_churn_info", "args": {"client_id": 1}, "type": "tool_call"}
        ]), {"langgraph_step": "1"})),
        ("updates", {"tools": {"messages": [ToolMessage(content="ok", tool_call_id="t1", name="get_churn_info")]}}),
        ("messages", (AIMessage(content="Готовый разбор клиента."), {"langgraph_step": "2"})),
    ]

    async def run():
        return [chunk async for chunk in _stream_agent(_FakeAgent(events), "разбери клиента 1", "t-1")]

    parsed = _parse_sse(asyncio.run(run()))
    assert parsed[0] == {"type": "tool", "phase": "start", "name": "get_churn_info"}
    assert parsed[1] == {"type": "tool", "phase": "end", "name": "get_churn_info", "ok": True}
    assert parsed[2] == {"type": "token", "text": "Готовый разбор клиента."}
    assert parsed[-1] == {"type": "done"}


def test_stream_agent_reports_errors():
    class _BrokenAgent:
        async def astream(self, input, config=None, stream_mode=None):
            raise RuntimeError("vLLM недоступен")
            yield  # pragma: no cover — делает функцию генератором

    async def run():
        return [chunk async for chunk in _stream_agent(_BrokenAgent(), "привет", "t-2")]

    parsed = _parse_sse(asyncio.run(run()))
    assert parsed == [{"type": "error", "message": "vLLM недоступен"}]


# ---------- эндпоинты (агент и хранилище клиентов не инициализированы) ----------

@pytest.fixture(scope="module")
def client():
    # TestClient без контекст-менеджера: lifespan не запускается, агент остаётся неинициализированным
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["agent_ready"] is False


def test_threads_requires_init(client):
    assert client.get("/api/threads").status_code == 503
    assert client.get("/api/threads/abc").status_code == 503


def test_chat_requires_agent(client):
    resp = client.post("/api/chat", json={"message": "привет"})
    assert resp.status_code == 503


def test_analyze_requires_agent(client):
    resp = client.post("/api/analyze", json={"client_id": 1})
    assert resp.status_code == 503


def test_clients_requires_store(client):
    assert client.get("/api/clients").status_code == 503
    assert client.get("/api/clients/1").status_code == 503


# ---------- эндпоинты клиентов поверх инициализированного SQLite-стора ----------

CSV_CONTENT = (
    "id,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,"
    "NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n"
    "15000,15782454.0,Ivanov,586,France,Male,52,8,0.0,2,1,1,103608.06\n"
    "15001,15776631.0,Petrova,601,Spain,Female,41,3,75000.5,1,0,0,45000\n"
)


@pytest.fixture()
def clients_store(tmp_path, monkeypatch):
    """Инициализирует clients_store на временных CSV/БД; агент по-прежнему не нужен."""
    csv_path = tmp_path / "clients.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")
    monkeypatch.setenv("CLIENTS_CSV_PATH", str(csv_path))
    monkeypatch.setenv("CLIENTS_DB_PATH", str(tmp_path / "clients.db"))

    asyncio.run(init_clients_store())
    yield
    asyncio.run(close_clients_store())


def test_clients_list_from_db(client, clients_store):
    """Выгрузка из БД: пейджинг и справочные id; риск приходит, если predict-сервис запущен."""
    resp = client.get("/api/clients?limit=1&offset=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["clients"]) == 1
    row = data["clients"][0]
    assert row["client_id"] == 1
    assert row["csv_id"] == 15001
    assert row["surname"] == "Petrova"
    assert row["geography"] == "Spain"
    assert row["age"] == 41
    assert row["balance"] == pytest.approx(75000.5)
    # риск зависит от окружения: None (predict недоступен) или число (локальный make up)
    assert row["churn_probability"] is None or isinstance(row["churn_probability"], float)


def test_clients_search_q(client, clients_store):
    resp = client.get("/api/clients", params={"q": "Petrova"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["clients"][0]["surname"] == "Petrova"


def test_clients_get_by_id(client, clients_store):
    found = client.get("/api/clients/0")
    assert found.status_code == 200
    assert found.json()["surname"] == "Ivanov"

    missing = client.get("/api/clients/999")
    assert missing.status_code == 404


def test_clients_limit_bounds(client, clients_store):
    resp = client.get("/api/clients?limit=0")
    assert resp.status_code == 200
    assert len(resp.json()["clients"]) == 1  # limit 0 -> приведён к 1
