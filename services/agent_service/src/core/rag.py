import os
import time
import asyncio
from qdrant_client import AsyncQdrantClient

from embedding_service import embedding_client
from shared.utils.logger import app_logger

# Конфигурация
COLLECTION = os.getenv('COLLECTION_NAME')
QDRANT_SPACE_NAME = os.getenv('QDRANT_SPACE_NAME', 'text_vector')
DEFAULT_CHUNKS_LIMIT = int(os.getenv('DEFAULT_CHUNKS_LIMIT', 5))
MAX_TOTAL_CHUNKS = int(os.getenv('MAX_TOTAL_CHUNKS', 30))
MIN_ABSOLUTE_SCORE = float(os.getenv('MIN_ABSOLUTE_SCORE', 0.4))
VOTE_BOOST = float(os.getenv('VOTE_BOOST', 0.05))

client = AsyncQdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 3030))
)


async def search_qdrant(query_variants: list[str], limit: int = MAX_TOTAL_CHUNKS) -> str:
    """
    Базовая логика поиска по Qdrant с учетом голосования и адаптивного порога.
    Возвращает отформатированный контекст для LLM.
    """
    app_logger.info(f"🔧 [search_qdrant] Query variants: {query_variants}")

    try:
        start_search_time = time.time()
        
        # Увеличиваем лимит в зависимости от количества запросов
        fetch_limit = min(DEFAULT_CHUNKS_LIMIT * len(query_variants), 50)
        embeddings = await embedding_client.get_embeddings(query_variants, "search_query")
        
        search_tasks = [(QDRANT_SPACE_NAME, emb) for emb in embeddings]

        async def query_space(space_name: str, embedding):
            try:
                response = await client.query_points(
                    collection_name=COLLECTION,
                    query=embedding,
                    using=space_name, 
                    limit=fetch_limit, 
                    with_payload=True
                )
                return space_name, response.points, None
            except Exception as e:
                return space_name, None, e

        tasks = [query_space(space_name, emb) for space_name, emb in search_tasks]
        query_results = await asyncio.gather(*tasks)

        search_time = round(time.time() - start_search_time, 2)
        app_logger.info(f"Поиск выполнен ({search_time} сек)")
        
        # Агрегация результатов
        unique_results = {}
        for space_name, points, exc in query_results:
            if exc:
                app_logger.error(f"Ошибка при поиске в пространстве {space_name}: {exc}")
                continue
            if not points:
                continue
            for point in points:
                item_id = point.id
                if item_id not in unique_results:
                    unique_results[item_id] = {'point': point, 'max_score': point.score, 'hits': 1}
                else:
                    existing = unique_results[item_id]
                    if point.score > existing['max_score']:
                        existing['max_score'] = point.score
                        existing['point'] = point
                    existing['hits'] += 1
        
        if not unique_results:
            return "По вашему запросу ничего не найдено в базе знаний."

        top_score = max(res['max_score'] for res in unique_results.values())
        score_threshold = max(top_score - top_score * 0.80, MIN_ABSOLUTE_SCORE)
        
        final_chunks = []
        for res in unique_results.values():
            if res['max_score'] >= score_threshold:
                composite_score = res['max_score'] + (res['hits'] - 1) * VOTE_BOOST
                final_chunks.append({
                    'point': res['point'], 'score': composite_score,
                    'original_score': res['max_score'], 'hits': res['hits']
                })
        
        final_chunks.sort(key=lambda x: x['score'], reverse=True)
        final_chunks = final_chunks[:MAX_TOTAL_CHUNKS - limit + MAX_TOTAL_CHUNKS] # Ограничение
        
        # Форматирование
        context = ""
        for i, item in enumerate(final_chunks):
            doc_text = item['point'].payload.get('text', 'Текст не найден')
            score = item['score']
            hits = item['hits']
            context += f"[Документ {i+1}] (score: {score:.4f}, совпадений: {hits}):\n{doc_text}\n\n"

        return context

    except Exception as e:
        app_logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
        return f"Ошибка извлечения из базы знаний: {str(e)}"