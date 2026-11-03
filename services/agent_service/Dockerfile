# Agent Service: REST/SSE API агента (LangGraph + SQLite checkpointer).
# Сборка из корня репозитория: docker build -f services/agent_service/Dockerfile .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/agent_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Общие пакеты: shared-utils и лёгкий HTTP-клиент embedding-сервиса (у него нет своих зависимостей)
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared
COPY services/embedding_service/pyproject.toml services/embedding_service/__init__.py /app/services/embedding_service/
COPY services/embedding_service/src/client.py /app/services/embedding_service/src/client.py
RUN pip install --no-cache-dir /app/services/embedding_service

COPY services/agent_service/src /app/src

# Точка монтирования volume'а: checkpointer, thread_meta, БД клиентов, CSV-выгрузка
RUN mkdir -p /data

WORKDIR /app/src

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
