import os

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.utils.logger import get_logger

logger = get_logger("rag_indexing_service.qdrant_client")


QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "policies")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))
QDRANT_SPACE_NAME = os.getenv("QDRANT_SPACE_NAME", "text_vector")


async def ensure_collection(client: AsyncQdrantClient) -> None:
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    distance = Distance.COSINE

    if COLLECTION_NAME not in names:
        logger.info(f"[Qdrant] Creating collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, distance={distance})")
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            # Добавляем указание имени пространства для каждого вектора
            vectors_config={QDRANT_SPACE_NAME: VectorParams(size=VECTOR_SIZE, distance=distance)},
        )
    else:
        logger.info(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists.")


async def batch_upsert(client: AsyncQdrantClient, points: list[PointStruct], batch_size: int = 64) -> None:
    """Пакетная загрузка точек в Qdrant."""
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"[Qdrant] Upserted batch {i // batch_size + 1} ({len(batch)} points).")


def make_point(idx: int, vector: list[float], payload: dict) -> PointStruct:
    # Указываем имя пространства для конкретного вектора
    return PointStruct(id=idx, vector={QDRANT_SPACE_NAME: vector}, payload=payload)
