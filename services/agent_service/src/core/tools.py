import os
import json
import asyncio
import requests
from langchain.tools import tool

from shared.utils.validators import parse_and_validate_client_id
from .formatters import generate_report

@tool(
    "get_churn_prediction",
    description="Используй для получения информации о клиенте и вероятности его оттока."
)
async def get_churn_prediction(client_id: str):    
    """    
    Используй для получения информации о клиенте и вероятности его оттока.

    Args:
        client_id: ID клиента (целое число)
    Returns:
        Информация о клиенте и вероятности его оттока.
    """

    CHURN_PRED_SERVICE_URL = os.getenv("CHURN_PRED_SERVICE_URL", "http://localhost:3010")
    url = f"{CHURN_PRED_SERVICE_URL}/predict"

    try:
        client_id = parse_and_validate_client_id(client_id)

        # Делаем GET запрос к нашему FastAPI сервису
        response = requests.get(url, params={"client_id": client_id}, timeout=5)
        
        # Если статус-код 200 - отдаем JSON строкой
        if response.status_code == 200:
            churn_pred_data = response.json()
            
        # Если 404 - значит такого ID не существует в нашей выборке
        elif response.status_code == 404:
            error_detail = response.json().get("detail", "Клиент не найден")
            return f"Ошибка: {error_detail}"
                       
    except requests.exceptions.ConnectionError:
        return "Ошибка: Не удалось подключиться к prediction-сервису. Убедитесь, что Docker-контейнер сервиса запущен."
    except requests.exceptions.Timeout:
        return "Ошибка: Превышено время ожидания ответа от prediction-сервиса."
    except Exception as e:
        return f"Непредвиденная ошибка при обращении к prediction-сервису: {str(e)}"

    return generate_report(churn_pred_data)

