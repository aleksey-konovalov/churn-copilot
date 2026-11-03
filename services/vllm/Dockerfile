# Пин версии, с которой собран и проверен стек (tool calling, REASONING_PARSER
# qwen3 меняются между релизами vLLM): v0.25.1
FROM vllm/vllm-openai:v0.25.1

# Убедимся, что скрипт запуска доступен
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

EXPOSE 8901

ENTRYPOINT ["/app/entrypoint.sh"]

