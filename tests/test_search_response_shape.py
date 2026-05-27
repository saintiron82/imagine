"""Phase A: search endpoint returns confidence + empty mode."""
from __future__ import annotations

import sys
import types

# Preserve pytest's captured stdout/stderr before importing the search router.
# backend.api_search (transitive import) rewraps sys.stdout at import time for
# UTF-8 IPC, which breaks pytest's fd-level capture if we don't snapshot first.
_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Stub the optional pyjwt dependency so importing the auth deps tree
# doesn't fail if pyjwt isn't installed at test time.
sys.modules.setdefault(
    "jwt",
    types.SimpleNamespace(
        ExpiredSignatureError=Exception,
        InvalidTokenError=Exception,
        decode=lambda *args, **kwargs: {},
        encode=lambda *args, **kwargs: "",
    ),
)

# Trigger the search router import now so backend.api_search's stdout rewrap
# happens up-front, then restore pytest's capture handles before any test runs.
# Keep a reference to the wrapper api_search created so its underlying buffer
# isn't garbage-collected (which would close pytest's capture tempfile).
from backend.server.routers import search as _search_router_preload  # noqa: F401
_API_SEARCH_STDOUT_KEEPALIVE = sys.stdout  # keep wrapper alive
_API_SEARCH_STDIN_KEEPALIVE = sys.stdin
sys.stdout = _ORIG_STDOUT
sys.stderr = _ORIG_STDERR


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
