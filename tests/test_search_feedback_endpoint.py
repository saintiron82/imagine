"""Phase D: POST /api/v1/search/feedback persists irrelevant labels."""
from __future__ import annotations

import sqlite3
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps  # noqa: E402


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        """CREATE TABLE search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            label TEXT NOT NULL CHECK (label IN ('irrelevant')),
            user_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    return types.SimpleNamespace(conn=conn)


def _client(db):
    from backend.server.routers import search_feedback

    app = FastAPI()
    app.include_router(search_feedback.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": 7, "username": "u"}
    return TestClient(app)


def test_feedback_persists_irrelevant_row():
    db = _db()
    client = _client(db)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "x", "file_id": 42, "label": "irrelevant"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("success") is True

    row = db.conn.execute(
        "SELECT query, file_id, label, user_id FROM search_feedback"
    ).fetchone()
    assert row == ("x", 42, "irrelevant", 7)


def test_feedback_rejects_unknown_label():
    db = _db()
    client = _client(db)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "x", "file_id": 1, "label": "love-it"},
    )
    assert resp.status_code in (400, 422)


def test_feedback_rejects_empty_query():
    db = _db()
    client = _client(db)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "", "file_id": 1, "label": "irrelevant"},
    )
    assert resp.status_code in (400, 422)


def test_feedback_requires_authentication(monkeypatch):
    from backend.server.routers import search_feedback as sf
    from fastapi import HTTPException

    db = _db()

    def deny():
        raise HTTPException(status_code=401, detail="auth required")

    app = FastAPI()
    app.include_router(sf.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = deny

    client = TestClient(app)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "x", "file_id": 1, "label": "irrelevant"},
    )
    assert resp.status_code == 401


def test_feedback_default_label_is_irrelevant():
    """Even without explicit label, irrelevant is the only supported value."""
    db = _db()
    client = _client(db)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "x", "file_id": 99},  # no label
    )
    assert resp.status_code == 200, resp.text
    row = db.conn.execute(
        "SELECT label FROM search_feedback WHERE file_id = 99"
    ).fetchone()
    assert row[0] == "irrelevant"
