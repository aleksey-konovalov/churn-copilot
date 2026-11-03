import os
import httpx
from langchain.tools import tool

from shared.utils.logger import get_logger
from shared.utils.validators import parse_and_validate_client_id
from .formatters import generate_report, build_complex_query_from_reasons
from .rag import search_qdrant

logger = get_logger("agent_service.core.tools")

# Конфигурация
CHURN_PROB_TRASHOLD = float(os.getenv('CHURN_PROB_TRASHOLD', 0.5))


@tool(
    "get_churn_info",
    description="Используй для получения информации о клиенте, вероятности его оттока и рекомендаций по удержанию."
)
async def get_churn_info(client_id: str):
    """
    Используй для получения информации о клиенте, вероятности его оттока и рекомендаций по удержанию.

    Args:
        client_id: ID клиента (целое число)
    Returns:
        Информация о клиенте и вероятности его оттока.
    """

    logger.info(f"Вызван get_churn_info с client_id: {client_id}")
    CHURN_PRED_SERVICE_URL = os.getenv("CHURN_PRED_SERVICE_URL", "http://localhost:3010")
    url = f"{CHURN_PRED_SERVICE_URL}/predict"

    try:
        client_id = parse_and_validate_client_id(client_id)
    except ValueError as e:
        logger.warning(f"Невалидный client_id '{client_id}': {e}")
        return f"Ошибка: {e}"

    # Сетевые ошибки не глотаем: их ретраит ToolRetryMiddleware,
    # после исчерпания попыток модель получит сообщение об ошибке (on_failure='continue')
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"client_id": client_id}, timeout=5.0)
    except (httpx.TimeoutException, httpx.ConnectError):
        logger.warning(f"Сетевая ошибка при обращении к prediction-сервису (client_id={client_id})")
        raise
    except Exception:
        logger.exception(f"Непредвиденная ошибка при обращении к prediction-сервису (client_id={client_id})")
        return "Ошибка: непредвиденная ошибка при обращении к prediction-сервису."

    logger.debug(f"Ответ от prediction-сервиса: статус {response.status_code}")

    if response.status_code == 200:
        churn_pred_data = response.json()
        report = generate_report(churn_pred_data)

        # Если риск оттока высок, тянем политики удержания автоматически
        churn_prob = churn_pred_data.get("churn_probability", 0.0)

        if churn_prob >= CHURN_PROB_TRASHOLD:
            logger.info(f"Высокий риск оттока ({churn_prob}). Формируем RAG запрос...")
            # Вариант запроса на каждую причину оттока: включает голосование в search_qdrant
            rag_queries = build_complex_query_from_reasons(churn_pred_data)
            logger.debug(f"Сформированы RAG запросы ({len(rag_queries)} шт.): {rag_queries}")

            # search_qdrant возвращает контекст или None (ничего не найдено);
            # сетевые сбои пробрасываются — их ретраит ToolRetryMiddleware,
            # прочие ошибки инфраструктуры превращаем в безопасный ответ модели
            try:
                policies_context = await search_qdrant(rag_queries)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError):
                logger.warning(f"Сетевая ошибка при RAG-поиске политик (client_id={client_id})")
                raise
            except Exception:
                logger.exception(f"Ошибка RAG-поиска политик (client_id={client_id})")
                return "Ошибка: база знаний политик временно недоступна."

            if policies_context:
                logger.debug(f"Получен контекст политик (длина: {len(policies_context)} символов)")
                report += f"\n\n--- НАЙДЕННЫЕ КОРПОРАТИВНЫЕ ПОЛИТИКИ И РЕКОМЕНДАЦИИ ПО УДЕРЖАНИЮ ---\n{policies_context}"
            else:
                logger.warning("RAG ничего не нашел по запросу удержания")
        else:
            logger.info(f"Риск оттока низкий ({churn_prob}), RAG поиск не выполняется.")

        logger.debug(f"Сформирован report (длина: {len(report)} символов)")
        return report

    # Если 404 - значит такого ID не существует в нашей выборке
    elif response.status_code == 404:
        error_detail = response.json().get("detail", "Клиент не найден")
        logger.warning(f"Клиент с ID {client_id} не найден (404)")
        return f"Ошибка: {error_detail}"
    elif response.is_server_error:
        # 5xx — временный сбой: HTTPStatusError уйдет в ToolRetryMiddleware на ретрай,
        # после исчерпания попыток модель получит сообщение об ошибке
        logger.warning(f"prediction-сервис вернул {response.status_code} — передаем в ретрай")
        response.raise_for_status()
    else:
        logger.error(f"prediction-сервис вернул статус {response.status_code}")
        return f"Ошибка: prediction-сервис вернул статус {response.status_code}"
