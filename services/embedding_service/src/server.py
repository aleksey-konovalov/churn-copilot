import asyncio
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Literal

import torch
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from shared.utils.logger import get_logger, setup_logging
from shared.utils.request_id import ensure_request_id, request_id_var

warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*deprecated.*", category=FutureWarning)
load_dotenv(find_dotenv())

# После load_dotenv: LOG_LEVEL/LOG_FORMAT могут прийти из локального .env
setup_logging(os.getenv("SERVICE_NAME", "embedding_service"))
logger = get_logger("embedding_service.server")


# Конфигурация
DEVICE = os.getenv("DEVICE", "cuda")
EMBEDDER_TYPE = os.getenv("EMBEDDER_TYPE", "giga").lower()  # 'giga' | 'frida'
MODEL_PATH = os.getenv("MODEL_PATH", "iMiW/Giga-Embeddings-instruct-4bit-nf4")
TOKENIZER_MAX_LENGTH = int(os.getenv("TOKENIZER_MAX_LENGTH", 512))
EMBEDDING_PORT = int(os.getenv("EMBEDDING_PORT", 3020))


# -----------------   Базовые классы и реализации моделей   -----------------


class BaseEmbedder:
    """Базовый интерфейс для embedding-моделей"""

    def load(self):
        pass

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        pass


class GigaEmbedder(BaseEmbedder):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"[Giga-Embeddings] Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print(f"[Giga-Embeddings] Loading model on {DEVICE}...")
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.eval()
        if DEVICE == "cuda":
            self.model.cuda()

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        # Логика формирования текста в зависимости от задачи
        if task == "search_query":
            task_description = "Given the search query, retrieve relevant paragraph with the answer"
            input_texts = [f"Instruct: {task_description}\nQuery: {text}" for text in texts]
        elif task == "paraphrase":
            task_description = "Retrieve semantically similar text"
            input_texts = [f"Instruct: {task_description}\nQuery: {text}" for text in texts]
        else:
            # Для search_document (индексация) инструкция не добавляется
            input_texts = texts

        batch_dict = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            embeddings = self.model(**batch_dict, return_embeddings=True, use_cache=False)
        return embeddings.cpu().numpy().tolist()


class FridaEmbedder(BaseEmbedder):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        print(f"[FRIDA] Loading model from {self.model_path} on {DEVICE}...")
        self.model = SentenceTransformer(self.model_path, device=DEVICE)

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        # Префиксы
        prefix_map = {
            "search_document": "search_document",
            "search_query": "search_query",
            "paraphrase": "paraphrase",
        }

        prefix = prefix_map.get(task, "paraphrase")
        prefixed_texts = [f"{prefix}: {text}" for text in texts]
        embeddings = self.model.encode(prefixed_texts, convert_to_tensor=False)
        return embeddings.tolist()


EMBEDDERS = {
    "giga": GigaEmbedder,
    "frida": FridaEmbedder,
}


# -----------------------     FastAPI Application      -----------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if EMBEDDER_TYPE not in EMBEDDERS:
        logger.error(f"Unknown EMBEDDER_TYPE={EMBEDDER_TYPE!r}, expected one of {list(EMBEDDERS)}")
        raise RuntimeError(f"Unknown EMBEDDER_TYPE={EMBEDDER_TYPE!r}, expected one of {list(EMBEDDERS)}")
    embedder = EMBEDDERS[EMBEDDER_TYPE](MODEL_PATH)
    embedder.load()
    app.state.embedder = embedder
    logger.info(f"[Embedding Service] {EMBEDDER_TYPE}:{MODEL_PATH} successfully loaded.")
    yield
    logger.info("[Embedding Service] Shutting down.")


app = FastAPI(title="Embedding Service", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
) -> Response:
    """Сквозной request_id: принимает клиентский (от agent_service) или свой."""
    rid = ensure_request_id(request.headers.get("x-request-id"))
    request_id_var.set(rid)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


class EmbeddingRequest(BaseModel):
    texts: list[str]
    task: Literal["search_document", "search_query", "paraphrase"]


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


@app.get("/health")
async def health_check():
    return {"status": "ok", "embedder": EMBEDDER_TYPE, "model": MODEL_PATH, "device": DEVICE}


@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    try:
        embedder: BaseEmbedder = app.state.embedder
        embeddings = await asyncio.to_thread(embedder.embed, request.texts, request.task)
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддингов: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=EMBEDDING_PORT)
