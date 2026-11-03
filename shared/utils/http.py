"""Общий httpx-клиент для межсервисных вызовов.

Клиенты чужих сервисов создаются фабрикой и закрываются в lifespan сервиса.
Транспорт автоматически пробрасывает X-Request-ID из request_id_var в каждый
исходящий запрос — сквозная трассировка не требует передавать заголовок руками.
"""

from typing import Any

import httpx

from shared.utils.request_id import current_request_id


class RequestIDTransport(httpx.AsyncBaseTransport):
    """Обёртка транспорта: добавляет X-Request-ID в каждый исходящий запрос."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport | None = None) -> None:
        self._wrapped = wrapped if wrapped is not None else httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.headers["x-request-id"] = current_request_id()
        return await self._wrapped.handle_async_request(request)


def create_async_client(*, timeout: float | httpx.Timeout | None, **kwargs: Any) -> httpx.AsyncClient:
    """Фабрика клиента соседнего сервиса с автопробросом X-Request-ID.

    timeout — секунды или httpx.Timeout; остальные параметры передаются
    в httpx.AsyncClient как есть.
    """
    kwargs.setdefault("transport", RequestIDTransport())
    return httpx.AsyncClient(timeout=timeout, **kwargs)
