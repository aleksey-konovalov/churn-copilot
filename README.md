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
┌──────────────────────────────────────────────────────────────────┐
│  Менеджер  →  Gradio UI (:7860)  →  root_agent (LangGraph)       │
│                                         │                        │
│                    get_churn_analisys_and_retention (sub-agent)  │
│                                         │                        │
│                              get_churn_info (tool)               │
│                    ┌────────────────────┼────────────────────┐   │
│                    ▼                    ▼                    ▼   │
│           churn_pred :3010     embedding_service      vLLM :3040 │
│           CatBoost + SHAP               │            Qwen3.6     │
│                                         ▼                        │
│                                    RAG-поиск                     │
│                                   Qdrant :3030                   │
└──────────────────────────────────────────────────────────────────┘
```

| Сервис | Порт | Назначение |
|--------|------|------------|
| `churn_pred_service` | 3010 | Inference CatBoost + TreeSHAP |
| `embedding_service` | 3020 | Эмбеддинги для RAG (FRIDA / Giga) |
| `qdrant` | 3030 | Векторное хранилище политик |
| `vllm` | 3040 | Локальный LLM (OpenAI-compatible API) |
| `agent_service` | 7860 | Gradio-чат (запуск локально) |
| `rag_indexing_service` | — | Batch-индексация (Docker profile) |

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
│   │   └── 02_model_pipeline.ipynb
│   └── src/
│       └── feature_engineering.py   # Единый pipeline train ↔ serve
├── infra/qdrant/               # Persistence векторной БД
├── services/
│   ├── agent_service/          # LangGraph + Gradio UI
│   ├── churn_pred_service/     # FastAPI inference
│   ├── embedding_service/      # FastAPI embeddings
│   ├── rag_indexing_service/   # One-shot indexer
│   └── vllm/                   # Qwen3.6 через vLLM
└── shared/                     # Общий Python-пакет (constants, logger, validators)
```

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

Поиск политик включается только при `churn_probability ≥ CHURN_PROB_TRASHOLD` (default 0.5). RAG-запрос **динамически строится из SHAP top reasons** через `FEATURE_MAP` — retrieval привязан к конкретному профилю риска клиента.

### 5. Продвинутый vector search

- адаптивный score threshold (относительно top-1);
- multi-query voting с `VOTE_BOOST` за повторные попадания;
- composite scoring для ранжирования чанков.

### 6. Pluggable LLM и embeddings

`llm_config.json` — переключение профилей (`local_qwen_agent` / `summarizer`) без изменения кода.  
Embedding service: strategy pattern (`FridaEmbedder` / `GigaEmbedder`) с task-specific prefixes.

### 7. Shared infrastructure

Пакет `shared/` — единый логгер (colorlog, rotation, env-driven level), `FEATURE_MAP` для human-readable отчётов, валидаторы `client_id`.

### 8. Docker / MLOps практики

- hot-reload модели и данных через volume mount;
- Docker profile `rag_indexing` для batch-job (`restart: "no"`);
- HF cache volumes для embedding/vLLM;
- Qdrant persistence в `infra/qdrant/`;
- env-based конфигурация vLLM без rebuild.

---

## Быстрый старт

### Требования

- Docker + docker-compose, **NVIDIA Container Toolkit** (GPU для vLLM и embeddings)
- Python 3.10 (для `agent_service`)
- Датасет Kaggle в `data_science/data/raw/`
- Обученная модель `data_science/artifacts/best_model.cbm`
- Политики удержания в `data/rag/*.txt` (для RAG)

### 1. Инфраструктура

```bash
docker compose up -d churn_pred_service embedding_service vllm qdrant
```

### 2. Индексация политик (один раз)

```bash
docker compose --profile rag_indexing up rag_indexing_service
```

> Убедитесь, что `COLLECTION_NAME` в `.env` агента совпадает с `QDRANT_COLLECTION` в compose (`policies`).

### 3. Агент

```bash
cd services/agent_service
cp env.example .env
pip install -r requirements.txt gradio
python src/main.py   # → http://localhost:7860
```

### Пример диалога

> «Почему может уйти клиент 42?»

Агент → sub-agent → `get_churn_info(42)` → CatBoost + SHAP → (если риск ≥ 0.5) RAG → Markdown-ответ с причинами и планом удержания.

### API prediction-сервиса

```http
GET http://localhost:3010/predict?client_id=42
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
| Agent | LangChain 1.0, LangGraph, Tool Calling |
| LLM | vLLM + Qwen3.6-27B-AWQ; fallback OpenAI |
| RAG | Qdrant, FRIDA embeddings |
| API / UI | FastAPI, Gradio, httpx |
| Infra | Docker Compose, NVIDIA GPU |

---

## Лицензии

| Компонент | Лицензия |
|-----------|----------|
| Код проекта | [Apache 2.0](LICENSE) |
| Датасет Kaggle | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

---

**ChurnCoPilot** — пример осознанного сочетания классического ML, explainable AI и LLM-оркестрации: точность там, где нужна математика; язык и политики — там, где нужен диалог с бизнесом.
