FROM python:3.10-slim

ENV APP_HOME=/app
ENV APP_USER=user
ENV PYTHONPATH=$APP_HOME/src
ENV PROJECT_DIR=$APP_HOME
ENV PYTHONUNBUFFERED=1

WORKDIR $APP_HOME

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание пользователя
RUN groupadd -r $APP_USER && \
    useradd -r -g $APP_USER -d $APP_HOME -s /sbin/nologin -c "Docker image user" $APP_USER && \
    chown $APP_USER:$APP_USER $APP_HOME

RUN pip install --upgrade pip

# Установка Python зависимостей
COPY --chown=$APP_USER:$APP_USER services/embedding_service/requirements.txt .
RUN --mount=type=cache,target=/tmp/pip-cache,sharing=locked \
    pip install --cache-dir /tmp/pip-cache -r requirements.txt

# Устанавливаем shared-utils
COPY ./shared /app/shared
RUN pip install --no-cache-dir /app/shared

# Копирование кода
COPY --chown=$APP_USER:$APP_USER services/embedding_service/src/ ./src/

# Заранее создаем папку для кэша и даем на нее права пользователю,
# чтобы смонтированный Docker volume (app-cache) имел правильные владельца.
RUN mkdir -p $APP_HOME/.cache/huggingface && \
    chown -R $APP_USER:$APP_USER $APP_HOME/.cache

USER $APP_USER

EXPOSE 3020

# Запуск через uvicorn напрямую для нативной поддержки сигналов ОС и reloading
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "3020"]
