"""Тесты SQLite-хранилища клиентов (clients_store): перегонка CSV, идемпотентность, поиск.

Запуск из корня репозитория:
    python -m pytest services/agent_service/tests -q

LLM и prediction-сервис не нужны.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]      # services/agent_service
REPO_ROOT = SERVICE_ROOT.parents[1]                     # корень репозитория

for path in (str(SERVICE_ROOT / "src"), str(REPO_ROOT / "services"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import clients_store as store  # noqa: E402


CSV_CONTENT = (
    "id,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,"
    "NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n"
    "15000,15782454.0,Ivanov,586,France,Male,52,8,0.0,2,1,1,103608.06\n"
    "15001,15776631.0,Petrova,601,Spain,Female,41,3,75000.5,1,0,0,45000\n"
    "15002,15600512.0,Smirnov,502,France,Male,29,2,0.0,1,1,0,78000\n"
)


@pytest.fixture()
def store_paths(tmp_path, monkeypatch):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(CSV_CONTENT, encoding="utf-8")
    monkeypatch.setenv("CLIENTS_CSV_PATH", str(csv_path))
    monkeypatch.setenv("CLIENTS_DB_PATH", str(tmp_path / "clients.db"))
    return csv_path


def test_ingest_maps_columns_and_ids(store_paths):
    """client_id = номер строки CSV; id/CustomerId сохраняются как справочные."""

    async def scenario():
        await store.init_clients_store()
        try:
            row = await store.fetch_client(0)
            assert row["client_id"] == 0
            assert row["csv_id"] == 15000
            assert row["customer_id"] == "15782454"  # float-запись из CSV приведена к int-строке
            assert row["surname"] == "Ivanov"
            assert row["geography"] == "France"
            assert row["age"] == 52
            assert row["balance"] == 0.0
            assert row["estimated_salary"] == pytest.approx(103608.06)
            assert row["is_active_member"] == 1

            assert await store.count_clients() == 3
            page = await store.fetch_clients(limit=2, offset=2)
            assert [r["client_id"] for r in page] == [2]
        finally:
            await store.close_clients_store()

    asyncio.run(scenario())


def test_reingest_only_when_source_changes(store_paths, monkeypatch):
    """Повторный init без изменений CSV не перезаливает таблицу; новый файл — перезаливает."""
    calls = {"reingest": 0}
    original_reingest = store._reingest

    async def counting_reingest(csv_path, fingerprint):
        calls["reingest"] += 1
        await original_reingest(csv_path, fingerprint)

    monkeypatch.setattr(store, "_reingest", counting_reingest)

    async def scenario():
        await store.init_clients_store()
        assert await store.count_clients() == 3
        await store.close_clients_store()

        await store.init_clients_store()
        assert await store.count_clients() == 3  # без второй перезаливки
        await store.close_clients_store()

        csv_path = store_paths
        csv_path.write_text(CSV_CONTENT + "15003,15593969.0,Novak,704,Germany,Female,35,5,12000.0,2,1,1,50000\n", encoding="utf-8")
        os.utime(csv_path)  # гарантированно новое mtime

        await store.init_clients_store()
        assert await store.count_clients() == 4
        row = await store.fetch_client(3)
        assert row["surname"] == "Novak"
        await store.close_clients_store()

    asyncio.run(scenario())
    assert calls["reingest"] == 2  # первичная заливка + обновлённый файл, но не повтор без изменений


def test_search_by_surname(store_paths):
    async def scenario():
        await store.init_clients_store()
        try:
            rows = await store.fetch_clients(limit=50, offset=0, search="Petrov")
            assert [r["surname"] for r in rows] == ["Petrova"]
            assert await store.count_clients(search="Petrov") == 1
            assert await store.count_clients(search="петров") == 0  # Latin-данные: кириллица не матчится
            assert await store.count_clients(search="") == 3        # пустой запрос — без фильтра
        finally:
            await store.close_clients_store()

    asyncio.run(scenario())


def test_search_by_numeric_ids(store_paths):
    """Цифровой запрос: точное совпадение по client_id / csv_id / CustomerId + фамилия-подстрока."""
    async def scenario():
        await store.init_clients_store()
        try:
            assert (await store.fetch_client(1))["surname"] == "Petrova"
            assert [r["surname"] for r in await store.fetch_clients(limit=10, offset=0, search="1")] == ["Petrova"]
            assert [r["surname"] for r in await store.fetch_clients(limit=10, offset=0, search="15000")] == ["Ivanov"]
            assert [r["surname"] for r in await store.fetch_clients(limit=10, offset=0, search="15600512")] == ["Smirnov"]
            assert await store.count_clients(search="150") == 0  # частичный ID не матчится
        finally:
            await store.close_clients_store()

    asyncio.run(scenario())


def test_missing_csv_is_empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIENTS_CSV_PATH", str(tmp_path / "absent.csv"))
    monkeypatch.setenv("CLIENTS_DB_PATH", str(tmp_path / "clients.db"))

    async def scenario():
        await store.init_clients_store()  # не падает — просто пустое хранилище
        try:
            assert await store.count_clients() == 0
            assert await store.fetch_clients(limit=10, offset=0) == []
            assert await store.fetch_client(0) is None
        finally:
            await store.close_clients_store()

    asyncio.run(scenario())


def test_store_not_ready_without_init(store_paths):
    async def scenario():
        with pytest.raises(store.StoreNotReady):
            await store.fetch_clients(limit=10, offset=0)

    asyncio.run(scenario())
