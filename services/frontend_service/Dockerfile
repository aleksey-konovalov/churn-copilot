# Frontend Service: статика веб-интерфейса + потоковый прокси /api -> agent_service.
# Сборка из корня репозитория: docker build -f services/frontend_service/Dockerfile .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/frontend_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Общий пакет shared-utils (логгер)
COPY shared /app/shared
RUN pip install --no-cache-dir /app/shared

COPY services/frontend_service/src /app/src
COPY services/frontend_service/web /app/web

WORKDIR /app/src

EXPOSE 3050

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3050"]
