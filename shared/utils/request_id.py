"""Сквозной request_id.

frontend_service (точка входа браузера) генерирует id на запросе к /api/* (или
принимает клиентский) и проксирует его заголовком X-Request-ID в agent_service;
HTTP-middleware каждого сервиса кладёт его в request_id_var, общий httpx-клиент
из shared.utils.http автоматически добавляет заголовок в исходящие вызовы,
ContextFilter в shared.utils.logger — в каждую запись лога. Один разбор клиента
трассируется через все сервисы одной строкой поиска в логах.
"""

import contextvars
import uuid

# "-" вне HTTP-запроса: фоновые задачи и скрипты логируются без id
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_MAX_REQUEST_ID_LEN = 128


def new_request_id() -> str:
    """Новый идентификатор запроса (uuid4.hex — компактнее в логах)."""
    return uuid.uuid4().hex


def current_request_id() -> str:
    """Актуальный request_id контекста."""
    return request_id_var.get()


def ensure_request_id(incoming: str | None) -> str:
    """Принять клиентский request_id или сгенерировать свой.

    Значения с пробелами/управляющими символами и слишком длинные отбрасываются:
    заголовок без проверки попадает в логи, его формату нельзя доверять.
    """
    if incoming:
        candidate = incoming.strip()
        if (
            candidate
            and candidate.isprintable()
            and len(candidate) <= _MAX_REQUEST_ID_LEN
            and not any(ch.isspace() for ch in candidate)
        ):
            return candidate
    return new_request_id()
