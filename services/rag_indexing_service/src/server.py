import asyncio
import os
import re
import sys

sys.path.insert(0, "/app/embedding_client")
from client import EmbeddingClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from qdrant_wrapper import batch_upsert, ensure_collection, make_point

from shared.utils.logger import get_logger, setup_logging

setup_logging(os.getenv("SERVICE_NAME", "rag_indexing_service"))
logger = get_logger("rag_indexing_service.server")


# Конфигурация
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", "/data/rag")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "policies")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))
EMBEDDING_TASK = "search_document"


async def wait_for_embedding_service(client: EmbeddingClient, timeout: int = 120):
    """Ждет, пока embedding_service загрузит модель и станет доступен."""
    logger.info("Waiting for embedding service to become ready...")
    for _ in range(timeout // 5):
        if await client.health_check():
            logger.info("Embedding service is ready.")
            return
        await asyncio.sleep(5)
    raise RuntimeError("Embedding service did not become ready in time.")


async def load_and_chunk_file(filepath: str, filename: str) -> list[dict]:
    """
    Загружает файл и разрезает на чанки по '--...'.
    Сохраняет заголовок (# Политика №...) как отдельное поле, добавляет название файла.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Разделитель: строка, состоящая только из тире (3 и более) + возможные пробелы
    raw_chunks = re.split(r"\n-{3,}\n", content.strip())

    chunks: list[dict] = []
    for raw in raw_chunks:
        text = raw.strip()
        if not text:
            continue

        # Извлекаем заголовок (первая строка, начинающаяся с #)
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else ""
        # В payload попадает весь текст чанка (с заголовком)
        chunks.append(
            {
                "title": title,
                "content": text,
                "source_file": filename,
            }
        )
    return chunks


async def reset_collection(qdrant: AsyncQdrantClient) -> None:
    """Пересоздает коллекцию, чтобы в базе не оставались чанки политик,
    удаленные из исходных файлов при переиндексации."""
    collections = await qdrant.get_collections()
    if COLLECTION_NAME in [c.name for c in collections.collections]:
        logger.info(f"[Qdrant] Dropping existing collection '{COLLECTION_NAME}' for full reindex")
        await qdrant.delete_collection(COLLECTION_NAME)
    await ensure_collection(qdrant)


async def main():
    logger.info(f"Starting RAG indexing from '{RAG_DATA_DIR}'")

    if not os.path.isdir(RAG_DATA_DIR):
        raise RuntimeError(f"Directory '{RAG_DATA_DIR}' does not exist.")

    qdrant = AsyncQdrantClient(url=QDRANT_URL)
    embedding_client = EmbeddingClient()
    try:
        # Чтение и чанкинг всех txt-файлов ДО пересоздания коллекции:
        # при пустом корпусе базу не трогаем
        all_chunks: list[dict] = []
        for filename in sorted(os.listdir(RAG_DATA_DIR)):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(RAG_DATA_DIR, filename)
            chunks = await load_and_chunk_file(filepath, filename)
            logger.info(f"Loaded {len(chunks)} chunks from '{filename}'.")
            all_chunks.extend(chunks)

        logger.info(f"Total chunks to index: {len(all_chunks)}")
        if not all_chunks:
            logger.warning("No chunks found. Exit.")
            return

        # Инициализируем клиент эмбеддингов и дожидаемся готовности сервиса
        await wait_for_embedding_service(embedding_client)

        # Полная переиндексация: чистая коллекция без «хвостов» от прошлого корпуса
        await reset_collection(qdrant)

        # Эмбеддинг и загрузка батчами
        points: list[PointStruct] = []
        idx = 0

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i : i + BATCH_SIZE]
            texts = [chunk["content"] for chunk in batch]
            embeddings = await embedding_client.get_embeddings(texts, task=EMBEDDING_TASK)
            for chunk, vector in zip(batch, embeddings, strict=False):
                payload = {
                    "title": chunk["title"],
                    "content": chunk["content"],
                    "source_file": chunk["source_file"],
                    "retention": True,
                }
                points.append(make_point(idx=idx, vector=vector, payload=payload))
                idx += 1
            logger.info(f"Embedded batch {i // BATCH_SIZE + 1}/{(len(all_chunks) - 1) // BATCH_SIZE + 1}")

        # Загрузка в Qdrant
        await batch_upsert(qdrant, points, batch_size=BATCH_SIZE)

        logger.info(f"[RAG Indexing] Successfully indexed {len(points)} chunks to Qdrant '{COLLECTION_NAME}'.")
    finally:
        await embedding_client.close()
        await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
