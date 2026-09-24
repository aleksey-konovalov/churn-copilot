"""Наблюдаемость: логирование расхода токенов LLM для калибровки порогов контекста."""
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from shared.utils.logger import get_logger

logger = get_logger("agent_service.core.token_usage")


class TokenUsageCallback(BaseCallbackHandler):
    """
    Логирует input/output токены каждого завершённого вызова LLM.

    Подключается через config={"callbacks": [...], "metadata": {"thread_id": ...}}.
    По логам калибруются пороги суммаризации (SUMMARIZATION_TRIGGER_TOKENS):
    input_tokens на ход должны выходить на плато, а не расти линейно.
    """

    def on_llm_end(self, response: LLMResult, *, metadata: dict | None = None, **kwargs) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is None:
            # Часть провайдеров отдаёт usage на уровне сообщения, а не в llm_output
            generations = response.generations or []
            if generations and generations[0]:
                msg_usage = getattr(generations[0][0].message, "usage_metadata", None) or {}
                prompt_tokens = msg_usage.get("input_tokens")
                completion_tokens = msg_usage.get("output_tokens")

        if prompt_tokens is None:
            return  # провайдер не отдал usage — логировать нечего

        thread_id = (metadata or {}).get("thread_id", "-")
        model = (response.llm_output or {}).get("model_name", "-")
        logger.info(
            f"LLM usage: thread={thread_id} model={model} "
            f"input_tokens={prompt_tokens} output_tokens={completion_tokens}"
        )
