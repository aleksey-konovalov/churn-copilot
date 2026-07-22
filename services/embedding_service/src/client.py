import os
import httpx
from typing import List, Literal

EMBEDDING_SERVICE_URL = os.getenv('EMBEDDING_SERVICE_URL', 'http://embedding_service:3020').rstrip('/')

class EmbeddingClient:
    """Асинхронный клиент для запросов к embedding-сервису с переиспользованием соединений."""
    
    def __init__(self, base_url: str = EMBEDDING_SERVICE_URL):
        self.base_url = base_url
        # httpx.AsyncClient переиспользует соединения (Connection Pooling)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    
    async def get_embeddings(
        self, 
        texts: List[str], 
        task: Literal['search_document', 'search_marker']
    ) -> List[List[float]]:
        """Получение эмбеддингов из сервиса."""
        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                json={"texts": texts, "task": task}
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except httpx.HTTPStatusError as e:
            # Логируем тело ответа для удобства дебага
            print(f"Embedding service error: {e.response.text}")
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

