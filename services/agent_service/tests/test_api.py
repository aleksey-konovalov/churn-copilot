"""Тесты REST API-слоя agent_service (frontend_service его проксирует).

Запуск из корня репозитория:
    python -m pytest services/agent_service/tests -q

LLM, prediction-сервис и Qdrant не нужны: агент не инициализируется,
недоступный predict-сервис даёт churn_probability=None.
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parents[1]      # services/agent_service
REPO_ROOT = SERVICE_ROOT.parents[1]                     # корень репозитория

for path in (str(SERVICE_ROOT / "src"), str(REPO_ROOT / "services"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("COLLECTION_NAME", "test_collection")
os.environ.setdefault("ACTIVE_LLM_PROFILE", "local_qwen_agent")

from api import _short_title  # noqa: E402
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


# ---------- эндпоинты (агент не инициализирован, predict-сервис недоступен) ----------

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


def test_clients_without_agent(client, tmp_path, monkeypatch):
    """Выгрузка клиентов не зависит от агента; недоступный predict даёт риски None."""
    csv_path = tmp_path / "clients.csv"
    csv_path.write_text(
        "id,Surname,Geography,Gender,Age,Tenure,Balance,NumOfProducts,"
        "HasCrCard,IsActiveMember,CreditScore,EstimatedSalary\n"
        "15000,Ivanov,France,Male,52,8,0.0,2,1,1,586,103608.06\n"
        "15001,Petrova,Spain,Female,41,3,75000.5,1,0,0,601,45000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("api._clients_cache", {"path": None, "mtime": None, "rows": []})
    monkeypatch.setenv("CLIENTS_CSV_PATH", str(csv_path))

    resp = client.get("/api/clients?limit=1&offset=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["clients"]) == 1
    row = data["clients"][0]
    assert row["client_id"] == 1
    assert row["surname"] == "Petrova"
    assert row["geography"] == "Spain"
    assert row["age"] == 41
    assert row["balance"] == pytest.approx(75000.5)
    # churn_pred_service не запущен — риск недоступен, но эндпоинт работает
    assert row["churn_probability"] is None


def test_clients_limit_bounds(client, tmp_path, monkeypatch):
    csv_path = tmp_path / "clients.csv"
    csv_path.write_text(
        "id,Surname\n1,A\n2,B\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("api._clients_cache", {"path": None, "mtime": None, "rows": []})
    monkeypatch.setenv("CLIENTS_CSV_PATH", str(csv_path))

    resp = client.get("/api/clients?limit=0")
    assert resp.status_code == 200
    assert len(resp.json()["clients"]) == 1  # limit 0 -> приведён к 1
