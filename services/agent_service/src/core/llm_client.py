import os
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Загружаем переменные окружения из .env
load_dotenv(find_dotenv())

# Путь к конфигу. По дефолту лежит в той же папке, что и этот скрипт
CONFIG_PATH = Path(__file__).parent / "llm_config.json"

def load_config() -> dict:
    """Чтение JSON конфига."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_llm_client(profile_name: str = None) -> ChatOpenAI:
    """
    Инициализация LLM на основе профиля из llm_config.json.
    Если profile_name не передан, берет его из ACTIVE_LLM_PROFILE.
    """
    if profile_name is None:
        profile_name = os.getenv("ACTIVE_LLM_PROFILE")

    if not profile_name:
        raise ValueError("Не задано имя профиля. Укажите ACTIVE_LLM_PROFILE в .env или передайте в функцию.")

    config = load_config()

    if profile_name not in config:
        raise ValueError(f"Профиль '{profile_name}' не найден в llm_config.json")

    profile_config = config[profile_name]

    # Извлекаем API ключ.
    # Если в конфиге "api_key_env": "OPENAI_API_KEY", он возьмет ключ из os.environ.
    # Если "api_key_env": "any" — переменной "any" в окружении нет, и os.getenv вернет саму строку "any".
    api_key_env_name = profile_config.get("api_key_env")
    api_key = os.getenv(api_key_env_name, api_key_env_name)

    # Приводим типы к float (на случай, если в JSON они записаны как строки)
    temperature = float(profile_config.get("temperature", 0.2))
    top_p = float(profile_config.get("top_p", 0.9))

    # Дополнительные параметры провайдера (например, chat_template_kwargs для vLLM)
    extra_body = profile_config.get("extra_body")

    # Инициализация клиента
    # ENV-оверрайд перенаправляет LLM-трафик для всех профилей (например, из
    # docker-сети: LLM_BASE_URL=http://vllm:8901/v1, см. docker-compose.yml)
    llm = ChatOpenAI(
        model=profile_config["model"],
        base_url=os.getenv("LLM_BASE_URL") or profile_config["base_url"],
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        extra_body=extra_body,
    )

    return llm

def get_optional_llm_client(profile_name: str) -> ChatOpenAI | None:
    """
    Клиент из профиля или None, если профиль не существует или его API-ключ не задан.

    Используется для опциональных контуров (например, ModelFallbackMiddleware):
    сервис должен подниматься без ключа, просто без fallback-модели.
    Профили с "api_key_env": "any" считаются настроенными всегда (ключ не нужен).
    """
    profile_config = load_config().get(profile_name)
    if not profile_config:
        return None

    api_key_env_name = profile_config.get("api_key_env")
    if api_key_env_name and api_key_env_name != "any" and not os.getenv(api_key_env_name):
        return None

    return get_llm_client(profile_name)

# Глобальная инициализация для импорта в других файлах
# В .env должна быть определена переменная ACTIVE_LLM_PROFILE
CURRENT_PROFILE = os.getenv("ACTIVE_LLM_PROFILE")

if CURRENT_PROFILE:
    llm = get_llm_client(CURRENT_PROFILE)
else:
    raise ValueError("env-переменная ACTIVE_LLM_PROFILE не определена")  # Или можно сделать llm = None
