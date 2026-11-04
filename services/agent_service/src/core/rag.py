import asyncio
import os
import time

from dotenv import find_dotenv, load_dotenv
from embedding_service.src.client import embedding_client
from qdrant_client import AsyncQdrantClient

from shared.utils.logger import get_logger

# Модуль может быть импортирован напрямую (тесты, скрипты) в обход llm_client,
# поэтому .env грузим здесь тоже: load_dotenv не перезаписывает уже заданные переменные
load_dotenv(find_dotenv())

logger = get_logger("agent_service.core.rag")

# Конфигурация
COLLECTION = os.getenv("COLLECTION_NAME")
QDRANT_SPACE_NAME = os.getenv("QDRANT_SPACE_NAME", "text_vector")
DEFAULT_CHUNKS_LIMIT = int(os.getenv("DEFAULT_CHUNKS_LIMIT", 5))
MAX_TOTAL_CHUNKS = int(os.getenv("MAX_TOTAL_CHUNKS", 30))
MIN_ABSOLUTE_SCORE = float(os.getenv("MIN_ABSOLUTE_SCORE", 0.4))
VOTE_BOOST = float(os.getenv("VOTE_BOOST", 0.05))

client = AsyncQdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333)),  # штатный порт Qdrant
)


async def close_qdrant() -> None:
    """Закрывает соединение с Qdrant. Вызывается в lifespan сервиса."""
    await client.close()


async def search_qdrant(query_variants: list[str], limit: int = MAX_TOTAL_CHUNKS) -> str | None:
    """Векторный поиск по Qdrant с голосованием вариантов запроса и адаптивным порогом.

    Принимает список вариантов запроса (по одному на причину оттока): чанк,
    найденный несколькими вариантами, получает буст VOTE_BOOST.

    Возвращает отформатированный контекст для LLM или None, если ничего не найдено.
    Ошибки инфраструктуры (embedding-сервис, Qdrant) пробрасываются выше:
    httpx-исключения ретраит ToolRetryMiddleware, остальные обрабатывает вызывающий код.
    """
    if not query_variants:
        return None
    if not COLLECTION:
        # Fail-fast: без коллекции поиск бессмыслен, молча искать некуда
        raise RuntimeError("COLLECTION_NAME не задана — проверьте env-конфигурацию agent_service")

    logger.info(f"🔧 [search_qdrant] Вариантов запроса: {len(query_variants)}")

    start_search_time = time.time()

    # Увеличиваем лимит в зависимости от количества запросов
    fetch_limit = min(DEFAULT_CHUNKS_LIMIT * len(query_variants), 50)
    embeddings = await embedding_client.get_embeddings(query_variants, "search_query")

    # Не глотаем исключения по вариантам: все варианты идут в одну коллекцию,
    # сбой одного означает системный сбой, частичный результат был бы неполным
    query_results = await asyncio.gather(
        *[
            client.query_points(
                collection_name=COLLECTION,
                query=emb,
                using=QDRANT_SPACE_NAME,
                limit=fetch_limit,
                with_payload=True,
            )
            for emb in embeddings
        ]
    )

    search_time = round(time.time() - start_search_time, 2)
    logger.info(f"Поиск выполнен ({search_time} сек)")

    # Агрегация: дедупликация по id точки, голосование за перекрытие вариантов
    unique_results = {}
    for response in query_results:
        for point in response.points:
            existing = unique_results.get(point.id)
            if existing is None:
                unique_results[point.id] = {"point": point, "max_score": point.score, "hits": 1}
            else:
                if point.score > existing["max_score"]:
                    existing["max_score"] = point.score
                    existing["point"] = point
                existing["hits"] += 1

    if not unique_results:
        logger.warning("Qdrant не вернул ни одной точки")
        return None

    top_score = max(res["max_score"] for res in unique_results.values())
    score_threshold = max(top_score * 0.8, MIN_ABSOLUTE_SCORE)

    final_chunks = [
        {
            "point": res["point"],
            "score": res["max_score"] + (res["hits"] - 1) * VOTE_BOOST,
            "hits": res["hits"],
        }
        for res in unique_results.values()
        if res["max_score"] >= score_threshold
    ]

    if not final_chunks:
        logger.warning(f"Все {len(unique_results)} кандидатов отсечены порогом {score_threshold:.3f}")
        return None

    final_chunks.sort(key=lambda x: x["score"], reverse=True)
    final_chunks = final_chunks[:limit]  # Ограничение по лимиту

    # Форматирование
    context = ""
    for i, item in enumerate(final_chunks):
        payload = item["point"].payload or {}
        # 'content' — актуальная схема индексации; 'text' — обратная совместимость
        # со старыми коллекциями
        doc_text = payload.get("content") or payload.get("text") or "Текст не найден"
        title = payload.get("title")
        title_suffix = f" — {title}" if title else ""
        context += (
            f"[Документ {i + 1}]{title_suffix} (score: {item['score']:.4f}, совпадений: {item['hits']}):\n"
            f"{doc_text}\n\n"
        )

    return context.strip()
