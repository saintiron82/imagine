"""Sprint 2 γ2: admin search-feedback summary."""
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
    from backend.server.routers import feedback_dashboard

    app = FastAPI()
    app.include_router(feedback_dashboard.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    return TestClient(app)


def test_summary_empty_when_no_feedback():
    db = _db()
    client = _client(db)
    resp = client.get("/api/v1/admin/search-feedback/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_30d"] == 0
    assert body["top_files"] == []
    assert body["top_queries"] == []


def test_summary_groups_top_files_and_queries():
    db = _db()
    rows = [
        ("cat",  10, "irrelevant"),
        ("cat",  10, "irrelevant"),
        ("cat",  11, "irrelevant"),
        ("dog",  10, "irrelevant"),
        ("bird", 22, "irrelevant"),
    ]
    db.conn.executemany(
        "INSERT INTO search_feedback (query, file_id, label, user_id) VALUES (?, ?, ?, 1)",
        rows,
    )
    db.conn.commit()
    client = _client(db)
    body = client.get("/api/v1/admin/search-feedback/summary").json()
    assert body["total_30d"] == 5
    # file 10: 3 hits — should be top
    assert body["top_files"][0] == {"file_id": 10, "count": 3}
    # query 'cat': 3 — should be top
    assert body["top_queries"][0] == {"query": "cat", "count": 3}


def test_summary_caps_top_files_at_20_and_queries_at_10():
    db = _db()
    # 25 distinct files, each with 1 row
    db.conn.executemany(
        "INSERT INTO search_feedback (query, file_id, label, user_id) VALUES (?, ?, 'irrelevant', 1)",
        [(f"q{i}", i, ) for i in range(25)] if False else [(f"q{i}", 100 + i) for i in range(25)],
    )
    # 15 distinct queries, each with 1 row
    db.conn.executemany(
        "INSERT INTO search_feedback (query, file_id, label, user_id) VALUES (?, ?, 'irrelevant', 1)",
        [(f"query-{i}", 999) for i in range(15)],
    )
    db.conn.commit()
    client = _client(db)
    body = client.get("/api/v1/admin/search-feedback/summary").json()
    assert len(body["top_files"]) <= 20
    assert len(body["top_queries"]) <= 10


def test_summary_requires_admin():
    from backend.server.routers import feedback_dashboard
    from fastapi import HTTPException

    db = _db()
    app = FastAPI()
    app.include_router(feedback_dashboard.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db

    def deny():
        raise HTTPException(status_code=403, detail="admin only")

    app.dependency_overrides[deps.require_admin] = deny
    test_client = TestClient(app)
    resp = test_client.get("/api/v1/admin/search-feedback/summary")
    assert resp.status_code == 403
