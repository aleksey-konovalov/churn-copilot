FROM python:3.10-slim

WORKDIR /app

COPY ./services/rag_indexing_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем shared-utils
COPY ./shared /app/shared
RUN pip install --no-cache-dir /app/shared

# Копируем клиент embedding-сервиса
COPY ./services/embedding_service/src/client.py /app/embedding_client/client.py

# Переименовываем qdrant_client.py в qdrant_wrapper.py, чтобы не конфликтовать с библиотекой qdrant_client
COPY ./services/rag_indexing_service/src/qdrant_client.py /app/src/qdrant_wrapper.py
COPY ./services/rag_indexing_service/src/server.py /app/src/server.py

WORKDIR /app/src
CMD ["python", "server.py"]
