"""
Точка входа Frontend Service: статика веб-интерфейса + обратный прокси /api -> agent_service.

Браузер общается только с этим сервисом: клиент не знает адресов внутренних
сервисов и не упирается в CORS. SSE-ответы agent_service проксируются
потоково, без буферизации.
"""
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from shared.utils.http import create_async_client
from shared.utils.logger import get_logger, setup_logging
from shared.utils.request_id import ensure_request_id, request_id_var

SERVICE_NAME = os.getenv("SERVICE_NAME", "frontend_service")
setup_logging(SERVICE_NAME)
logger = get_logger(name="frontend_service.server")

AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:7860").rstrip("/")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

_client = create_async_client(base_url=AGENT_SERVICE_URL, timeout=httpx.Timeout(None))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Frontend Service готов: статика {WEB_DIR}, прокси /api -> {AGENT_SERVICE_URL}")
    yield
    await _client.aclose()
    logger.info("HTTP-клиент прокси закрыт")


app = FastAPI(title="Churn Copilot Frontend Service", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
) -> Response:
    """Сквозной request_id: frontend — точка входа, генерирует id на каждый запрос."""
    rid = ensure_request_id(request.headers.get("x-request-id"))
    request_id_var.set(rid)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


# Заголовки hop-by-hop и длины при стриминге прокси не переносит
_SKIP_HEADERS = {"content-length", "transfer-encoding", "connection"}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_api(path: str, request: Request):
    """Прозрачный потоковый прокси /api/* на agent_service."""
    url = f"/api/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in ("content-type", "accept", "authorization")
    }
    # Трассировка: agent_service залогирует тот же request_id в своих строках
    headers["x-request-id"] = request_id_var.get()
    try:
        upstream_response = await _client.send(
            _client.build_request(
                request.method, url, content=await request.body(), headers=headers
            ),
            stream=True,
        )
    except httpx.HTTPError:
        logger.exception(f"agent_service недоступен: {request.method} {url}")
        raise HTTPException(502, "Agent Service недоступен")
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers={k: v for k, v in upstream_response.headers.items() if k.lower() not in _SKIP_HEADERS},
        background=BackgroundTask(upstream_response.aclose),
    )


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    port = int(os.getenv("FRONTEND_PORT", "3050"))
    logger.info(f"Запуск Frontend Service UI на http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
