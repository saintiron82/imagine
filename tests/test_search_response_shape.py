"""Phase A: search endpoint returns confidence + empty mode."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeSearcher:
    """Stand-in that lets us steer top-1 raw scores."""
    def __init__(self, results):
        self._results = results

    def search(self, *args, **kwargs):
        return self._results


def _client(results):
    from backend.server.routers import search as search_router
    from backend.server import deps

    fake = _FakeSearcher(results)
    search_router._get_searcher = lambda: fake  # type: ignore[assignment]

    app = FastAPI()
    app.include_router(search_router.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": 1, "username": "u"}
    return TestClient(app)


def test_search_returns_confidence_field_high():
    results = [{
        "id": 1, "file_path": "a.png",
        "vector_score": 0.7, "text_vec_score": 0.4,
        "rrf_score": 0.5,
    }]
    client = _client(results)
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "high"
    assert body["top1_raw_score"] == pytest.approx(0.7)
    assert len(body["results"]) == 1


def test_search_returns_empty_mode_when_top1_below_low_threshold():
    results = [{
        "id": 1, "file_path": "a.png",
        "vector_score": 0.05, "text_vec_score": 0.05,
        "rrf_score": 0.01,
    }]
    client = _client(results)
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    body = resp.json()
    assert body["confidence"] == "empty"
    assert body["results"] == []
    assert body["count"] == 0
    assert "empty_reason" in body


def test_search_returns_low_when_only_fts_hits():
    results = [{
        "id": 1, "file_path": "a.png",
        "vector_score": 0.05, "text_vec_score": 0.05,
        "text_score": 4.5,
        "rrf_score": 0.01,
    }]
    client = _client(results)
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    body = resp.json()
    assert body["confidence"] == "low"
    assert len(body["results"]) == 1


def test_search_returns_empty_when_no_results():
    client = _client([])
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    body = resp.json()
    assert body["confidence"] == "empty"
    assert body["results"] == []
    assert body["top1_raw_score"] == 0.0
