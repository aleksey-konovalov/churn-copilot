import os
from typing import Literal

import httpx

from shared.utils.http import create_async_client
from shared.utils.logger import get_logger

logger = get_logger("embedding_service.client")

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding_service:3020").rstrip("/")


class EmbeddingClient:
    """Асинхронный клиент для запросов к embedding-сервису с переиспользованием соединений."""

    def __init__(self, base_url: str = EMBEDDING_SERVICE_URL):
        self.base_url = base_url
        # httpx.AsyncClient переиспользует соединения (Connection Pooling);
        # транспорт прокидывает X-Request-ID текущего запроса agent_service
        self.client = create_async_client(timeout=60.0)

    async def get_embeddings(
        self, texts: list[str], task: Literal["search_document", "search_query", "paraphrase"]
    ) -> list[list[float]]:
        """Получение эмбеддингов из сервиса."""
        logger.debug(f"Запрос эмбеддингов для {len(texts)} текстов с task: {task}")
        try:
            response = await self.client.post(f"{self.base_url}/embeddings", json={"texts": texts, "task": task})
            response.raise_for_status()
            data = response.json()
            logger.debug("Успешно получены эмбеддинги")
            return data["embeddings"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Embedding service error: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка в EmbeddingClient: {str(e)}", exc_info=True)
            raise

    async def health_check(self) -> bool:
        """Проверка доступности сервиса."""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Обязательно вызывать при завершении работы приложения (в app.shutdown())"""
        await self.client.aclose()


# Глобальный экземпляр для импорта
embedding_client = EmbeddingClient()
