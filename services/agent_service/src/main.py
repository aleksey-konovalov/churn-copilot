"""
Точка входа Agent Service: REST/SSE API агента для веб-фронтенда (frontend_service).

Lifespan управляет состоянием сервиса: при старте открывает SQLite checkpointer,
служебную таблицу диалогов (thread_meta) и хранилище клиентов (clients_store с
перегонкой CSV-выгрузки), при остановке закрывает соединения (checkpointer,
embedding-сервис, Qdrant).
"""
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

from shared.utils.logger import get_logger, setup_logging
from shared.utils.request_id import ensure_request_id, request_id_var

from api import close_thread_meta, init_thread_meta, router
from clients_store import close_clients_store, init_clients_store
from core.agent import init_root_agent, shutdown_root_agent
from core.rag import close_qdrant
from embedding_service.src.client import embedding_client

SERVICE_NAME = os.getenv("SERVICE_NAME", "agent_service")
setup_logging(SERVICE_NAME)
logger = get_logger(name="agent_service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_root_agent()
    await init_thread_meta()
    await init_clients_store()
    logger.info("Agent Service готов к работе")
    yield
    await shutdown_root_agent()
    await close_thread_meta()
    await close_clients_store()
    # Закрываем фоновые HTTP-соединения RAG-контура (созданы при импорте модулей)
    await embedding_client.close()
    await close_qdrant()
    logger.info("Соединения embedding-сервиса и Qdrant закрыты")


app = FastAPI(title="Churn Copilot Agent Service", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
) -> Response:
    """Сквозной request_id: принимает клиентский (от frontend_service) или свой."""
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

if __name__ == "__main__":
    logger.info("Запуск Agent Service API на http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
