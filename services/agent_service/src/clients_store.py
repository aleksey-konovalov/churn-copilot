"""
SQLite-хранилище клиентов для экрана «Клиенты» веб-фронтенда.

CSV-выгрузка (data_science/data/raw/test.csv) перегоняется в SQLite при старте
сервиса: файл перечитывается только если изменились mtime/размер, поэтому
обычный рестарт не трогает данные. client_id = номер строки CSV (0..N-1) — тот
же контракт, что у churn_pred_service (/predict?client_id=N читает строку N).
Колонки id и CustomerId из CSV сохраняются как справочные.
"""

import asyncio
import csv
import os
from pathlib import Path

import aiosqlite

from shared.utils.logger import get_logger

logger = get_logger("agent_service.clients_store")


class StoreNotReady(RuntimeError):
    """Хранилище ещё не инициализировано (init_clients_store не вызван)."""


_clients_conn: aiosqlite.Connection | None = None

_COLUMNS = [
    "client_id",
    "csv_id",
    "customer_id",
    "surname",
    "credit_score",
    "geography",
    "gender",
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "has_cr_card",
    "is_active_member",
    "estimated_salary",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id        INTEGER PRIMARY KEY,  -- номер строки CSV (контракт churn_pred)
    csv_id           INTEGER,              -- справочное: колонка id из CSV
    customer_id      TEXT NOT NULL DEFAULT '',  -- справочное: колонка CustomerId из CSV
    surname          TEXT NOT NULL DEFAULT '',
    credit_score     INTEGER,
    geography        TEXT NOT NULL DEFAULT '',
    gender           TEXT,
    age              INTEGER,
    tenure           INTEGER,
    balance          REAL,
    num_of_products  INTEGER,
    has_cr_card      INTEGER,
    is_active_member INTEGER,
    estimated_salary REAL
);
CREATE INDEX IF NOT EXISTS idx_clients_surname ON clients(surname);
CREATE TABLE IF NOT EXISTS clients_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def clients_db_path() -> str:
    return os.getenv("CLIENTS_DB_PATH") or str(Path(__file__).resolve().parents[1] / "clients.db")


def clients_csv_path() -> str:
    override = os.getenv("CLIENTS_CSV_PATH")
    if override:
        return override
    # Дефолт для локального запуска: ищем data_science вверх по дереву от src/
    # (в контейнере путь всегда задаётся через CLIENTS_CSV_PATH)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data_science" / "data" / "raw" / "test.csv"
        if candidate.is_file():
            return str(candidate)
    return str(here.parents[1] / "data_science" / "data" / "raw" / "test.csv")


def _to_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_customer_id(value) -> str:
    """CustomerId в CSV записан как float ('15782454.0') — убираем дробную часть."""
    text = (value or "").strip()
    try:
        num = float(text)
    except ValueError:
        return text
    return str(int(num)) if num.is_integer() else text


def _parse_csv_row(index: int, raw: dict) -> tuple:
    return (
        index,
        _to_int(raw.get("id")),
        _clean_customer_id(raw.get("CustomerId")),
        raw.get("Surname") or "",
        _to_int(raw.get("CreditScore")),
        raw.get("Geography") or "",
        raw.get("Gender") or "",
        _to_int(raw.get("Age")),
        _to_int(raw.get("Tenure")),
        _to_float(raw.get("Balance")),
        _to_int(raw.get("NumOfProducts")),
        _to_int(raw.get("HasCrCard")),
        _to_int(raw.get("IsActiveMember")),
        _to_float(raw.get("EstimatedSalary")),
    )


def _read_csv_rows(path: str) -> list[tuple]:
    with open(path, newline="", encoding="utf-8") as file:
        return [_parse_csv_row(index, raw) for index, raw in enumerate(csv.DictReader(file))]


def _source_fingerprint(path: str) -> tuple[str, str, str] | None:
    """(path, mtime_ns, size) выгрузки или None, если файла нет."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (path, str(stat.st_mtime_ns), str(stat.st_size))


async def _stored_fingerprint() -> tuple[str, str, str] | None:
    cur = await _clients_conn.execute(
        "SELECT key, value FROM clients_meta WHERE key IN ('source_path', 'source_mtime', 'source_size')"
    )
    meta = dict(await cur.fetchall())
    if len(meta) != 3:
        return None
    return (meta["source_path"], meta["source_mtime"], meta["source_size"])


async def _reingest(csv_path: str, fingerprint: tuple[str, str, str]) -> None:
    rows = await asyncio.to_thread(_read_csv_rows, csv_path)
    placeholders = ", ".join("?" * len(_COLUMNS))
    # DELETE открывает неявную транзакцию: полная перезаливка атомарна по commit
    await _clients_conn.execute("DELETE FROM clients")
    await _clients_conn.executemany(f"INSERT INTO clients ({', '.join(_COLUMNS)}) VALUES ({placeholders})", rows)
    await _clients_conn.executemany(
        "INSERT INTO clients_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        list(zip(("source_path", "source_mtime", "source_size"), fingerprint, strict=False)),
    )
    await _clients_conn.commit()
    logger.info(f"Выгрузка клиентов перегнана в БД: {len(rows)} строк ({csv_path})")


async def init_clients_store() -> None:
    """Открывает БД клиентов и при необходимости перегоняет в неё CSV-выгрузку."""
    global _clients_conn
    db_path = clients_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _clients_conn = await aiosqlite.connect(db_path)
    await _clients_conn.execute("PRAGMA journal_mode=WAL")
    await _clients_conn.executescript(_SCHEMA)

    csv_path = clients_csv_path()
    fingerprint = _source_fingerprint(csv_path)
    if fingerprint is None:
        logger.warning(f"CSV-выгрузка клиентов не найдена ({csv_path}): экран «Клиенты» будет пуст")
        return
    if fingerprint != await _stored_fingerprint():
        await _reingest(csv_path, fingerprint)
    else:
        logger.info(f"Выгрузка клиентов актуальна: {await count_clients()} строк ({csv_path})")


async def close_clients_store() -> None:
    global _clients_conn
    if _clients_conn is not None:
        await _clients_conn.close()
    _clients_conn = None


def _require_conn() -> aiosqlite.Connection:
    if _clients_conn is None:
        raise StoreNotReady()
    return _clients_conn


def _search_clause(search: str) -> tuple[str, list]:
    """WHERE-условие поиска: фамилия (подстрока) + ID, если запрос цифровой.

    Цифровой запрос ищет точное совпадение по client_id и csv_id и точное —
    по CustomerId (префикс по 8-значному id дал бы тысячи ложных срабатываний).
    """
    like = f"%{search}%"
    if search.isdigit() and len(search) <= 10:
        return (
            "(surname LIKE ? OR client_id = ? OR csv_id = ? OR customer_id = ?)",
            [like, int(search), int(search), search],
        )
    return ("surname LIKE ?", [like])


def _row_to_dict(row: tuple) -> dict:
    return dict(zip(_COLUMNS, row, strict=False))


async def count_clients(search: str | None = None) -> int:
    conn = _require_conn()
    sql = "SELECT COUNT(*) FROM clients"
    params: list = []
    if search:
        clause, params = _search_clause(search)
        sql += f" WHERE {clause}"
    cur = await conn.execute(sql, params)
    return (await cur.fetchone())[0]


async def stored_fingerprint() -> tuple[str, str, str] | None:
    """(path, mtime_ns, size) выгрузки из meta; None, если данных ещё нет.

    Публичный аксессор для инвалидации внешних кэшей, зависящих от выгрузки
    (карта рисков в api.py).
    """
    return await _stored_fingerprint()


async def fetch_clients(limit: int, offset: int, search: str | None = None) -> list[dict]:
    """Страница клиентов, отсортированная по client_id (опционально поиск по фамилии/ID)."""
    conn = _require_conn()
    sql = f"SELECT {', '.join(_COLUMNS)} FROM clients"
    params: list = []
    if search:
        clause, params = _search_clause(search)
        sql += f" WHERE {clause}"
    sql += " ORDER BY client_id LIMIT ? OFFSET ?"
    params += [limit, offset]
    cur = await conn.execute(sql, params)
    return [_row_to_dict(row) for row in await cur.fetchall()]


async def fetch_client(client_id: int) -> dict | None:
    conn = _require_conn()
    cur = await conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM clients WHERE client_id = ?", (client_id,))
    row = await cur.fetchone()
    return _row_to_dict(row) if row else None
