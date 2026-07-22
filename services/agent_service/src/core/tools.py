import os
import json
import time
import httpx
import asyncio
from langchain.tools import tool

from shared.utils.logger import app_logger, get_logger
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
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"client_id": client_id}, timeout=5.0)
            
            logger.debug(f"Ответ от prediction-сервиса: статус {response.status_code}")
            
            if response.status_code == 200:
                churn_pred_data = response.json()
                report = generate_report(churn_pred_data)
                
                # Если риск оттока высок, тянем политики удержания автоматически
                churn_prob = churn_pred_data.get("churn_probability", 0.0)
                
                if churn_prob >= CHURN_PROB_TRASHOLD:
                    logger.info(f"Высокий риск оттока ({churn_prob}). Формируем RAG запрос...")
                    rag_query = build_complex_query_from_reasons(churn_pred_data)
                    logger.debug(f"Сформирован RAG запрос: {rag_query}")
                    
                    policies_context = await search_qdrant([rag_query])
                    logger.debug(f"Получен контекст политик (длина: {len(policies_context)} символов)")
                    
                    if "ничего не найдено" not in policies_context.lower():
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
            else:
                logger.error(f"prediction-сервис вернул статус {response.status_code}")
                return f"Ошибка: prediction-сервис вернул статус {response.status_code}"
                   
    except httpx.ConnectError:
        logger.error("Не удалось подключиться к prediction-сервису", exc_info=True)
        return "Ошибка: Не удалось подключиться к prediction-сервису. Убедитесь, что Docker-контейнер сервиса запущен."
    except httpx.TimeoutException:
        logger.error("Таймаут при обращении к prediction-сервису", exc_info=True)
        return "Ошибка: Превышено время ожидания ответа от prediction-сервиса."
    except Exception as e:
        logger.error(f"Непредвиденная ошибка в get_churn_info: {str(e)}", exc_info=True)
        return f"Непредвиденная ошибка при обращении к prediction-сервису: {str(e)}"
    