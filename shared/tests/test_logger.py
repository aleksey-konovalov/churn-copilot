"""Тесты настройки логирования и сквозного request_id.

Запуск из корня репозитория:
    python -m pytest shared/tests -q
"""
import json
import logging

from shared.utils.logger import get_logger, setup_logging
from shared.utils.request_id import ensure_request_id, request_id_var


def test_console_format_contains_request_id(capsys) -> None:
    setup_logging("test_service")
    token = request_id_var.set("req-1")
    try:
        logging.getLogger("module.x").info("hello %s", "world")
    finally:
        request_id_var.reset(token)
    out = capsys.readouterr().out
    # Сервис в консольном формате не выводится: он и так виден в префиксе
    # docker compose logs; machine-readable имя сервиса — в json-формате
    assert "[req-1]" in out
    assert "hello world" in out


def test_json_format(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging("test_service")
    logging.getLogger("module.y").warning("warm")
    entry = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert entry["service"] == "test_service"
    assert entry["level"] == "WARNING"
    assert entry["message"] == "warm"
    assert "request_id" in entry


def test_get_logger_named() -> None:
    assert get_logger("some.name").name == "some.name"


def test_ensure_request_id_accepts_valid() -> None:
    assert ensure_request_id("abc-123") == "abc-123"


def test_ensure_request_id_rejects_garbage() -> None:
    # Пробелы/переносы отбрасываются: заголовок попадает в логи, его формату нельзя доверять
    rid = ensure_request_id("bad id\nwith spaces")
    assert rid and " " not in rid
    assert len(rid) == 32  # uuid4().hex
