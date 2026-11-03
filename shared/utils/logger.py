"""
Единая настройка логирования сервисов.

Принцип (12-factor): сервис пишет только в stdout — хранением и ротацией
занимается docker (logging driver + max-size/max-file в docker-compose.yml),
сборщики логов подписываются на тот же поток. Файловые хендлеры внутри
контейнеров не используются: docker logs их не видит, а writable-слой
контейнера эфемерен.

Каждый сервис вызывает setup_logging() ровно один раз в точке входа; остальной
код только объявляет именованный логгер через get_logger(__name__ или имя).

ENV:
  LOG_LEVEL   — уровень root-логгера: имя (DEBUG/INFO/...) или число (по умолчанию INFO);
  LOG_FORMAT  — console (по умолчанию) | json — машинный формат для сборщиков логов
                (Loki/ELK и т.п.).

Сквозной request_id: HTTP-middleware сервиса кладёт его в request_id_var
(см. shared.utils.request_id), ContextFilter добавляет в каждую запись лога.
Вне HTTP-запроса — "-".
"""

import json
import logging
import logging.config
import os

from shared.utils.request_id import request_id_var

__all__ = [
    "ContextFilter",
    "JsonFormatter",
    "get_logger",
    "request_id_var",
    "setup_logging",
]


def get_logger(name: str = "") -> logging.Logger:
    """Именованный логгер; конфигурацию делает setup_logging() в точке входа."""
    return logging.getLogger(name)


class ContextFilter(logging.Filter):
    """Добавляет в каждую запись имя сервиса и request_id."""

    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        # Присваивание динамических атрибутов mypy разрешает; в JSON-форматере
        # ниже для чтения нужны ignore
        record.service = self.service
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Минимальный JSON-форматер без внешних зависимостей."""

    def format(self, record: logging.LogRecord) -> str:
        # service/request_id добавлены ContextFilter'ом, для mypy это не поля LogRecord
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": record.service,  # type: ignore[attr-defined]
            "logger": record.name,
            "request_id": record.request_id,  # type: ignore[attr-defined]
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def _level_from_env() -> int:
    raw = os.getenv("LOG_LEVEL", "INFO")
    return int(raw) if raw.isdigit() else getattr(logging, raw.upper(), logging.INFO)


def setup_logging(service: str) -> None:
    """Настраивает logging процесса. Вызывать один раз в точке входа сервиса."""
    level = _level_from_env()
    use_json = os.getenv("LOG_FORMAT", "console").strip().lower() == "json"
    formatter = (
        {"()": JsonFormatter}
        if use_json
        else {
            "format": "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s - %(message)s",
            "datefmt": "%H:%M:%S",
        }
    )
    logging.config.dictConfig(
        {
            "version": 1,
            # Логгеры, созданные импортами до этого вызова (disable_existing_loggers
            # по умолчанию их глушит), остаются рабочими — они всплывают в root
            "disable_existing_loggers": False,
            "filters": {"ctx": {"()": ContextFilter, "service": service}},
            "formatters": {"std": formatter},
            "handlers": {
                "out": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "std",
                    "filters": ["ctx"],
                },
            },
            "root": {"handlers": ["out"], "level": level},
            # uvicorn по умолчанию ставит своим логгерам отдельные хендлеры с другим
            # форматом — перецепляем их на общий хендлер, чтобы в docker compose logs
            # всё выглядело единообразно
            "loggers": {
                "uvicorn": {"handlers": ["out"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["out"], "level": level, "propagate": False},
                "uvicorn.access": {"handlers": ["out"], "level": level, "propagate": False},
            },
        }
    )
