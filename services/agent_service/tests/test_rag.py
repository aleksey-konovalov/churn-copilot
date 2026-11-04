"""Юнит-тесты RAG-поиска agent_service.

Запуск из корня репозитория:
    python -m pytest services/agent_service/tests -q

Qdrant и embedding-сервис мокаются: сеть не нужна.
"""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]  # services/agent_service
REPO_ROOT = SERVICE_ROOT.parents[1]  # корень репозитория

for path in (str(SERVICE_ROOT / "src"), str(REPO_ROOT / "services"), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# COLLECTION_NAME читается при импорте модуля
os.environ.setdefault("COLLECTION_NAME", "test_collection")
os.environ.setdefault("MIN_ABSOLUTE_SCORE", "0.4")
os.environ.setdefault("VOTE_BOOST", "0.05")

from core import rag  # noqa: E402


def make_point(point_id, score, payload):
    return SimpleNamespace(id=point_id, score=score, payload=payload)


def make_response(points):
    return SimpleNamespace(points=points)


class FakeQdrant:
    """Подменяет rag.client: на каждый вариант запроса возвращает свою выдачу."""

    def __init__(self, results_per_variant, error=None):
        self.results_per_variant = results_per_variant
        self.error = error
        self.calls = []

    async def query_points(self, collection_name, query, using, limit, with_payload):
        self.calls.append({"limit": limit, "using": using, "collection": collection_name})
        if self.error is not None:
            raise self.error
        return make_response(self.results_per_variant[len(self.calls) - 1])


class FakeEmbedder:
    async def get_embeddings(self, texts, task):
        assert task == "search_query"
        return [[0.1] * 4 for _ in texts]


def patch_rag(monkeypatch, qdrant_fake):
    monkeypatch.setattr(rag, "client", qdrant_fake)
    monkeypatch.setattr(rag, "embedding_client", FakeEmbedder())


POLICY_PAYLOAD = {"content": "Текст политики", "title": "Политика №1", "source_file": "policies.txt"}


def test_vote_boost_raises_multi_variant_hits(monkeypatch):
    """Чанк, найденный обоими вариантами, получает буст и пометку о совпадениях."""
    qdrant = FakeQdrant(
        [
            [make_point(1, 0.60, POLICY_PAYLOAD), make_point(2, 0.55, POLICY_PAYLOAD)],
            [make_point(1, 0.62, POLICY_PAYLOAD)],
        ]
    )
    patch_rag(monkeypatch, qdrant)

    context = asyncio.run(rag.search_qdrant(["запрос A", "запрос B"]))

    # Порог: max(0.62*0.8, 0.4) = 0.496 — оба чанка проходят
    assert context is not None
    assert "Текст политики" in context
    # Документ 1 (id=1) найден дважды: score 0.62 + VOTE_BOOST, и он первый в выдаче
    assert "совпадений: 2" in context
    first_block = context.split("[Документ 2]")[0]
    assert "0.6700" in first_block  # 0.62 + 0.05


def test_content_key_is_used_for_text(monkeypatch):
    """Текст чанка берется из payload['content'] (актуальная схема индексации)."""
    qdrant = FakeQdrant([[make_point(1, 0.7, POLICY_PAYLOAD)]])
    patch_rag(monkeypatch, qdrant)

    context = asyncio.run(rag.search_qdrant(["запрос"]))

    assert "Текст политики" in context
    assert "Текст не найден" not in context
    assert "Политика №1" in context  # заголовок в шапке документа


def test_text_key_fallback_for_legacy_payload(monkeypatch):
    """Обратная совместимость: старые коллекции с ключом 'text' тоже читаются."""
    legacy = {"text": "Старый формат чанка"}
    qdrant = FakeQdrant([[make_point(1, 0.7, legacy)]])
    patch_rag(monkeypatch, qdrant)

    context = asyncio.run(rag.search_qdrant(["запрос"]))

    assert "Старый формат чанка" in context


def test_all_candidates_below_threshold_returns_none(monkeypatch):
    """Если порог отсек всех кандидатов — None, а не пустая строка/ошибка."""
    qdrant = FakeQdrant([[make_point(1, 0.35, POLICY_PAYLOAD)]])
    patch_rag(monkeypatch, qdrant)

    # top_score=0.35 < MIN_ABSOLUTE_SCORE=0.4 → отсекаются все
    assert asyncio.run(rag.search_qdrant(["запрос"])) is None


def test_no_points_returns_none(monkeypatch):
    qdrant = FakeQdrant([[], []])
    patch_rag(monkeypatch, qdrant)

    assert asyncio.run(rag.search_qdrant(["запрос A", "запрос B"])) is None


def test_empty_variants_returns_none(monkeypatch):
    patch_rag(monkeypatch, FakeQdrant([]))
    assert asyncio.run(rag.search_qdrant([])) is None


def test_network_error_propagates_for_retry(monkeypatch):
    """Сетевые сбои пробрасываются — их ретраит ToolRetryMiddleware."""
    qdrant = FakeQdrant([], error=httpx.ConnectError("connection refused"))
    patch_rag(monkeypatch, qdrant)

    with pytest.raises(httpx.ConnectError):
        asyncio.run(rag.search_qdrant(["запрос"]))


def test_fetch_limit_grows_with_variants(monkeypatch):
    """Лимит выдачи на вариант = DEFAULT_CHUNKS_LIMIT * число вариантов."""
    qdrant = FakeQdrant([[], [], []])
    patch_rag(monkeypatch, qdrant)

    asyncio.run(rag.search_qdrant(["A", "B", "C"]))

    assert len(qdrant.calls) == 3
    assert all(call["limit"] == 15 for call in qdrant.calls)  # 5 * 3
    assert all(call["collection"] == "test_collection" for call in qdrant.calls)


def test_missing_collection_name_fails_fast(monkeypatch):
    """Без COLLECTION_NAME поиск падает сразу с внятной ошибкой."""
    patch_rag(monkeypatch, FakeQdrant([[make_point(1, 0.7, POLICY_PAYLOAD)]]))
    monkeypatch.setattr(rag, "COLLECTION", None)

    with pytest.raises(RuntimeError, match="COLLECTION_NAME"):
        asyncio.run(rag.search_qdrant(["запрос"]))


def test_build_complex_query_returns_list_per_reason():
    from core.formatters import build_complex_query_from_reasons

    data = {
        "top_reasons": [
            {"feature": "is_age_over_45", "feature_value": 1, "risk_contribution_pct": 40.0},
            {"feature": "is_zero_balance", "feature_value": 1, "risk_contribution_pct": 30.0},
        ]
    }

    queries = build_complex_query_from_reasons(data)

    # Список вариантов (не склеенная строка) — включает голосование в search_qdrant
    assert isinstance(queries, list)
    assert len(queries) == 2
    assert queries[0] == "Возраст клиента более 45 лет: Да"
    assert queries[1] == "У клиента нулевой баланс: Да"
