"""
Точка входа Agent Service: REST/SSE API агента для веб-фронтенда (frontend_service).

Lifespan управляет состоянием сервиса: при старте открывает SQLite checkpointer,
служебную таблицу диалогов (thread_meta) и хранилище клиентов (clients_store с
перегонкой CSV-выгрузки), при остановке закрывает соединения (checkpointer,
embedding-сервис, Qdrant).
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from shared.utils.logger import get_logger

from api import close_thread_meta, init_thread_meta, router
from clients_store import close_clients_store, init_clients_store
from core.agent import init_root_agent, shutdown_root_agent
from core.rag import close_qdrant
from embedding_service.src.client import embedding_client

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

if __name__ == "__main__":
    logger.info("Запуск Agent Service API на http://0.0.0.0:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
