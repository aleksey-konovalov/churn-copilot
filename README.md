# ChurnCoPilot

> **Соединяем ML-предсказания и бизнес-действия через объяснимый ИИ.**

Гибридная система для анализа оттока клиентов банка: **CatBoost** даёт точный прогноз и локальную интерпретацию через **TreeSHAP**, **LLM-агент** переводит результат в понятный бизнес-язык и подбирает план удержания по **корпоративным политикам (RAG)**.

---

## Зачем это нужно

Классический ML-сервис отвечает на вопрос «уйдёт ли клиент?», но не объясняет *почему* и *что делать*. ChurnCoPilot закрывает полный цикл:

1. Модель считает вероятность оттока и топ-5 факторов риска (SHAP).
2. Агент получает структурированный отчёт через tool calling — без доступа к «сырым» данным и без галлюцинаций.
3. При высоком риске автоматически подтягиваются релевантные политики удержания из векторной БД.

Архитектурный принцип: **ML отвечает за точность и объяснимость, LLM — за коммуникацию и синтез рекомендаций**. Это осознанный антипаттерн «LLM угадывает отток по промпту».

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Менеджер  →  Web UI (:3050)  →  agent_service API  →  root_agent                │
│  (frontend_service:             (REST/SSE, LangGraph, checkpointer)             │
│   статика + прокси /api)                │                                       │
│                     get_churn_analysis_and_retention (sub-agent)                │
│                                         │                                       │
│                                 get_churn_info (tool)                           │
│                    ┌────────────────────┼────────────────────┐                  │
│                    ▼                    ▼                    ▼                  │
│           churn_pred           embedding_service      vLLM (:3040)              │
│           CatBoost + SHAP               │              Qwen3.6                  │
│                                         ▼                                       │
│                                    RAG-поиск                                    │
│                                    Qdrant                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

| Сервис | Порт в compose-сети | На хосте | Назначение |
|--------|------|------|------------|
| `frontend_service` | 3050 | **3050** | Веб-интерфейс: клиенты + чаты (статика + прокси /api) |
| `churn_pred_service` | 3010 | — | Inference CatBoost + TreeSHAP |
| `embedding_service` | 3020 | — | Эмбеддинги для RAG (FRIDA / Giga) |
| `qdrant` | 6333 | — | Векторное хранилище политик |
| `vllm` | 8901 | **3040** | Локальный LLM (OpenAI-compatible API) |
| `agent_service` | 7860 | — | REST/SSE API агента (LangGraph, checkpointer, SQLite-выгрузка клиентов) |
| `rag_indexing_service` | — | — | Batch-индексация (Docker profile) |
| `dev-tools` | — | — | Скачивание ассетов и снапшотов (Docker profile) |

**Multi-agent design:** корневой агент ведёт диалог и делегирует экспертизу суб-агенту удержания. Каждый уровень имеет узкий набор tools и жёсткие промпты (запрет на выдумывание ID, политик, профилей).

---

## Структура репозитория

```
churn_copilot/
├── docker-compose.yml          # Оркестрация микросервисов
├── data/
│   └── rag/                    # Корпоративные политики (.txt, gitignored)
├── data_science/
│   ├── artifacts/              # best_model.cbm — продакшен-артефакт
│   ├── data/
│   │   ├── raw/                # train_raw.csv, test.csv (Kaggle)
│   │   └── processed/          # train.csv / val.csv (stratified split)
│   ├── notebooks/
│   │   ├── 00_data_split.ipynb
│   │   ├── 01_eda.ipynb
│   │   └── 02_model_pipeline.ipynb
│   └── src/
│       └── feature_engineering.py   # Единый pipeline train ↔ serve
├── infra/qdrant/               # Persistence векторной БД
├── services/
│   ├── agent_service/          # LangGraph + REST/SSE API
│   ├── frontend_service/       # Веб-интерфейс: статика + прокси /api
│   ├── churn_pred_service/     # FastAPI inference
│   ├── embedding_service/      # FastAPI embeddings
│   ├── rag_indexing_service/   # One-shot indexer
│   └── vllm/                   # Qwen3.6 через vLLM
└── shared/                     # Общий Python-пакет (constants, logger, validators)
```

---

## Быстрый старт

### Требования

- Docker + docker-compose, **NVIDIA Container Toolkit** (GPU для vLLM и embeddings)
- Python 3.10+ (для тестов и разработки; рантайм сервисов — Docker, локальный запуск вне compose не поддерживается)
- Датасет Kaggle в `data_science/data/raw/`
- Обученная модель `data_science/artifacts/best_model.cbm`
- Политики удержания в `data/rag/*.txt` (для RAG)

### Подготовка ассетов (модель, датасеты, Qdrant)

Чтобы не коммитить ассеты (веса моделей, датасеты, снапшоты Qdrant), проект содержит Makefile для автоматизации их скачивания из Google Drive.

**0. Конфигурация скачивания:**
```bash
cp shared/.env.example shared/.env
```
Заполните `GOOGLE_DRIVE_*` file ID (файл gitignored). Те же переменные нужны целям
`restore-qdrant` / `save-snapshot` (имя коллекции — `QDRANT_COLLECTION`).

**1. Скачивание файлов:**
```bash
make download-assets
```
Скрипт прочитает конфигурацию из `shared/.env` (`GOOGLE_DRIVE_MODEL_ID`, `GOOGLE_DRIVE_TRAIN_ID` и др.), скачает файлы и разложит их по нужным директориям (`data_science/`, `infra/qdrant/`). Существующие файлы пропускаются.

**2. Восстановление Qdrant из снапшота:**
После скачивания снапшота разверните коллекцию в Qdrant:
```bash
make restore-qdrant
```
Команда (при необходимости) запускает скачивание ассетов, поднимает контейнер Qdrant, копирует `snapshot.snapshot` внутрь контейнера через `docker cp`, инициирует восстановление коллекции через REST API и корректно останавливает контейнер, если он не был запущен до этого.

**3. Сохранение снапшота Qdrant:**
Если вы обновили векторную базу (добавили новые политики) и хотите создать резервную копию:
```bash
make save-snapshot
```
Команда запросит создание снапшота у работающего сервиса Qdrant, достанет файл из контейнера и сохранит его локально в `infra/qdrant/snapshots/{COLLECTION_NAME}/snapshot.snapshot`. После этого файл можно загрузить на Google Drive и обновить `GOOGLE_DRIVE_QDRANT_SNAPSHOT_ID` в `shared/.env`.

### 1. Инфраструктура:

```bash
docker compose up -d churn_pred_service embedding_service vllm qdrant
```

### 2. Индексация политик (при обновлении документов для RAG, один раз)

```bash
docker compose --profile rag_indexing up rag_indexing_service
```

> Убедитесь, что `COLLECTION_NAME` в `.env` агента совпадает с `QDRANT_COLLECTION` в compose (`policies`).

### 3. Запуск фронта с агентом (Docker)

```bash
make up   # поднимает всё, кроме vLLM: churn_pred, embedding, qdrant, agent_service, frontend_service
```

Веб-интерфейс: **http://localhost:3050** (единственный пользовательский порт — фронт проксирует 
REST/SSE API агента, который наружу не публикуется). Состояние агента (checkpointer + БД клиентов) 
живёт в docker volume `agent-data`, так что чаты и выгрузка переживают `make down` / `make up`. 
Перезаливка кода в образы — тот же `make up` (--build).

vLLM — самый тяжёлый контейнер (GPU, 27B AWQ). Когда он нужен:

```bash
docker compose up -d vllm   # агент найдёт его по имени: http://vllm:8901/v1 (LLM_BASE_URL в compose)
```

Если vLLM развёрнут на хосте, а не в compose — задайте `LLM_BASE_URL` / `LLM_MODEL` в `.env`
корня репозитория (см. `.env.example`), compose подставит их в agent_service.

**Про порты:** наружу публикуются только `3050` (UI) и `3040` (vLLM — для локальных клиентов LLM и отладки). 
Остальные сервисы доступны только по внутренней сети compose: меньше путаницы и конфликтов портов 
с другими приложениями. Отладка внутрь — `docker compose exec`, REST qdrant из целей Makefile 
(`restore-qdrant`, `save-snapshot`) ходит через одноразовый curl-контейнер в сети compose.

Раздел «Клиенты» (открывается по умолчанию) читает выгрузку из SQLite (CSV `data_science/data/raw/test.csv` 
перегоняется в БД при старте agent_service, client_id = номер строки CSV): поиск по фамилии или ID и 
фильтры по рискам — на сервере (риски всей выгрузки считаются одним batch-вызовом churn_pred_service 
и кэшируются до смены CSV), листалка страниц внизу таблицы. Кнопка «Анализ оттока» открывает справа панель с двумя 
подвкладками: «Анализ оттока» (риск, факторы SHAP, стрим разбора агента) и «Карточка клиента» 
(атрибуты). Оттуда «Продолжить в чате» — агент уже будет помнить разбор. Ответы агента рендерятся 
как markdown — и в чате, и в панели анализа. Чаты удаляются из списка кнопкой «×» (второй клик 
подтверждает удаление вместе с историей).

### Запуск сервисов без Makefile

Цели `run-agent` / `run-frontend` убраны: сервисы ходят друг к другу и к зависимостям по внутренней 
сети compose, с хоста она недоступна, поэтому локальный запуск вне Docker не поддерживается. 
При необходимости оба сервиса можно поднять вручную (`uvicorn main:app` / `uvicorn server:app`), 
но полноценно всё работает только через `make up`.

---

## Тесты и линтинг

Модульные тесты `shared/` (логгер) и `agent_service/` (REST API, SQLite-стор клиентов, RAG).
Внешние зависимости не нужны: LLM, prediction-сервис и Qdrant заглушены, всё бегает на обычном Python:

```bash
python -m pytest    # из корня репозитория; набор тестов — testpaths в pyproject.toml
```

Линтинг — ruff (конфиг в корневом `pyproject.toml`, line-length 120, target py310 — как базовый образ сервисов):

```bash
ruff check . && ruff format --check .
```

Оба шага гоняются в CI (GitHub Actions) на каждый push в `main` и PR.

---

## Data Science: от данных до модели

### Источник данных

[Bank Churn Competition (Kaggle)](https://www.kaggle.com/competitions/bank-churn-competition-by-ipii-hs-ex-mts/data) — 15 000 записей train, CC BY 4.0.  
Train разбит на `train.csv` (12 000) / `val.csv` (3 000) стратифицированно (`random_state=29`), доля оттока **20.12%** сохранена. Сами записи не модифицировались.

### EDA → Feature Engineering

Признаки проектировались **от бизнес-гипотез, подтверждённых в EDA**, а не «наугад»:

| Наблюдение | Инженерный признак |
|------------|-------------------|
| Германия ~40% оттока vs ~20% в других странах | `is_germany` |
| Нелинейная зависимость от числа продуктов; 3+ → >97% оттока | `is_single_product`, `is_3plus_products` |
| Скачок оттока после 45 лет | `is_age_over_45` |
| Бимодальный баланс; нулевой баланс — минимальный риск | `is_zero_balance` |
| Высокий баланс + неактивность → 40% оттока | `is_high_balance_inactive` |

Категориальные признаки (`Geography`, `Gender`, `HasCrCard`, `IsActiveMember`) обрабатываются нативно в CatBoost — без ручного one-hot.

### Выбор модели и экспериментальный дизайн

**CatBoost** выбран осознанно:
- нативная работа с категориями;
- GPU-обучение;
- встроенный **TreeSHAP** для production-explainability;
- сильные результаты на tabular-данных.

**Валидация — многоуровневая:**

| Этап | Метод | Результат |
|------|-------|-----------|
| Baseline | 5-fold StratifiedKFold | ROC AUC **0.9335 ± 0.0049** |
| HPO | Optuna TPE, 100 trials × 5-fold CV | Best CV ROC AUC **0.9346** |
| Hold-out | `val.csv` (3 000 строк) | ROC AUC **0.9325**, Accuracy **0.8883**, F1(churn) **0.7305** |

**Работа с дисбалансом:** `auto_class_weights: SqrtBalanced` в лучшем trial.  
**Анализ порога:** F1 максимален при 0.60 (0.7409 vs 0.7305 @ 0.5), но для бизнеса выбран порог **0.5** — выше recall, меньше пропущенных «уходящих». ~79% предсказаний имеют confidence ≥ 80%.

**Production inference:** `client_id` = индекс строки в `test.csv` (0…9999). API возвращает вероятность и топ-5 **положительных** SHAP-факторов с нормализованной долей риска.

---

## Инженерные решения:

### 1. Единый feature pipeline (train/serve parity)

`data_science/src/feature_engineering.py` импортируется и в ноутбуках, и в `churn_pred_service` через volume mount + `PYTHONPATH`. Это устраняет классическую проблему **train/serve skew**.

### 2. Микросервисная декомпозиция

Каждый компонент — отдельный контейнер с чёткой ответственностью: prediction, embeddings, vector store, LLM. Независимое масштабирование и замена компонентов (например, OpenAI вместо vLLM) без переписывания агента.

### 3. Tool Calling вместо RAG-over-CSV

LLM не видит датасет напрямую. Доступ только через `get_churn_info` → HTTP к prediction-сервису. Промпты запрещают галлюцинации ID и политик.

### 4. Условный RAG

Поиск политик включается только при `churn_probability ≥ CHURN_PROB_THRESHOLD` (default 0.5). RAG-запрос **динамически строится из SHAP top reasons** через `FEATURE_MAP` — retrieval привязан к конкретному профилю риска клиента.

### 5. Продвинутый vector search

- адаптивный score threshold (относительно top-1);
- multi-query voting с `VOTE_BOOST` за повторные попадания;
- composite scoring для ранжирования чанков.

### 6. Pluggable LLM и embeddings

`llm_config.json` — переключение профилей (`local_qwen_agent` / `summarizer`) без изменения кода.  
Embedding service: strategy pattern (`FridaEmbedder` / `GigaEmbedder`) с task-specific prefixes.

### 7. Shared infrastructure

Пакет `shared/` — единый логгер (stdout-only по 12-factor, сквозной request_id через `X-Request-ID`, `LOG_FORMAT=json` для сборщиков; ротацию делает docker logging driver), `FEATURE_MAP` для human-readable отчётов, валидаторы `client_id`.

### 8. Docker / MLOps практики

- прикладной стек полностью в контейнерах: `make up` поднимает всё, кроме vLLM (`--build` — правки кода попадают в образы без ручных шагов);
- выгрузка клиентов живёт в SQLite (`clients_store` в agent_service): идемпотентная перегонка CSV при старте (по mtime/size файла), состояние — в volume `agent-data`;
- hot-reload модели и данных через volume mount;
- Docker profile `rag_indexing` для batch-job (`restart: "no"`);
- HF cache volumes для embedding/vLLM;
- Qdrant persistence в `infra/qdrant/`;
- env-based конфигурация vLLM без rebuild;
- `LLM_BASE_URL` env-оверрайд: один llm_config.json работает и локально (localhost:3040), и в compose-сети (vllm:8901).

---

## Пример диалога

> «Почему может уйти клиент 42?»

Агент → sub-agent → `get_churn_info(42)` → CatBoost + SHAP → (если риск ≥ 0.5) RAG → Markdown-ответ с причинами и планом удержания.

### API prediction-сервиса

Порт наружу не публикуется: из сети compose — `http://churn_pred_service:3010`, с хоста — 
через одноразовый контейнер:

```bash
docker run --rm --network churn_copilot_default curlimages/curl \
  "http://churn_pred_service:3010/predict?client_id=42"
```

```json
{
  "client_id": 42,
  "churn_probability": 0.7823,
  "top_reasons": [
    {
      "feature": "is_germany",
      "feature_value": 1,
      "risk_contribution_pct": 34.2
    }
  ]
}
```

---

## Стек технологий

| Категория | Технологии |
|-----------|------------|
| ML | CatBoost, scikit-learn, Optuna, pandas |
| Explainability | TreeSHAP (CatBoost native) |
| Agent | LangChain 1.4, LangGraph, Tool Calling |
| LLM | vLLM + Qwen3.6-27B-AWQ; fallback OpenAI |
| RAG | Qdrant, FRIDA embeddings |
| API / UI | FastAPI, vanilla JS SPA, httpx |
| Infra | Docker Compose, NVIDIA GPU |

---

## Лицензии

| Компонент | Лицензия |
|-----------|----------|
| Код проекта | [Apache 2.0](LICENSE) |
| Датасет Kaggle | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

---

**ChurnCoPilot** — пример осознанного сочетания классического ML, explainable AI и LLM-оркестрации: точность там, где нужна математика; язык и политики — там, где нужен диалог с бизнесом.
