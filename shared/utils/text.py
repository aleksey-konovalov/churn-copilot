"""Утилиты очистки текста, приходящего от LLM."""

import re

# <think>...</think> либо незакрытый <think> до конца строки
# (обрыв генерации по лимиту токенов)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def sanitize_llm_text(text) -> str:
    """
    Вычищает утёкший reasoning-блок (<think>) из ответа модели.

    Defense-in-depth: основная защита — на уровне провайдера
    (vLLM --reasoning-parser qwen3, API-модели отдают reasoning отдельным
    полем). Санитайзер страхует случаи, когда модель отдаётся без парсинга
    (агрегаторы, локальный запуск без флага).
    """
    if not isinstance(text, str):
        text = str(text)
    return _THINK_RE.sub("", text).strip()
