"""Durable worker block: a blocked worker stays blocked across reconnects.

Covers the 2026-06-11 control-plane fixes:
- connect rejects a worker whose (user, worker_name) has a blocked session
- the worker's own graceful disconnect must NOT lift an admin block
- admin unblock flips the session to offline, after which connect succeeds
"""

import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps
from backend.server.routers import workers


class FakeAnalysisManager:
    def __init__(self):
        self.reclaimed = []

    def reclaim_worker_tasks(self, session_id):
        self.reclaimed.append(session_id)
        return 0


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""
        CREATE TABLE worker_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            worker_name TEXT NOT NULL,
            hostname TEXT,
            origin TEXT,
            launcher TEXT,
            batch_capacity INTEGER DEFAULT 5,
            status TEXT DEFAULT 'online',
            pending_command TEXT,
            processing_mode_override TEXT,
            batch_capacity_override INTEGER,
            resources_json TEXT,
            connected_at TEXT,
            last_heartbeat TEXT,
            disconnected_at TEXT
        )
    """)
    conn.execute("CREATE TABLE analysis_jobs (id INTEGER PRIMARY KEY, status TEXT)")
    conn.execute("""
        CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            analysis_job_id INTEGER,
            parse_status TEXT, mc_status TEXT, vv_status TEXT, mv_status TEXT
        )
    """)
    conn.commit()
    return SimpleNamespace(conn=conn)


def _client(db, monkeypatch, user=None, admin=None):
    app = FastAPI()
    app.include_router(workers.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    if user is not None:
        app.dependency_overrides[deps.get_current_user] = lambda: user
    if admin is not None:
        app.dependency_overrides[deps.require_admin] = lambda: admin

    import backend.server.queue.analysis_manager as mgr_module
    manager = FakeAnalysisManager()
    monkeypatch.setattr(mgr_module, "AnalysisJobManager", lambda _db: manager)
    monkeypatch.setattr(workers.audit_log, "record", lambda *a, **k: None)
    return TestClient(app)


USER = {"id": 1, "username": "owner"}
ADMIN = {"id": 9, "username": "admin"}
CONNECT = {"worker_name": "gpu-box", "hostname": "h", "batch_capacity": 5}


def _connect(client):
    return client.post("/api/v1/workers/connect", json=CONNECT)


def test_blocked_worker_cannot_reconnect(monkeypatch):
    db = _db()
    client = _client(db, monkeypatch, user=USER, admin=ADMIN)

    sid = _connect(client).json()["session_id"]
    db.conn.execute(
        "UPDATE worker_sessions SET status='blocked', pending_command='block' WHERE id=?",
        (sid,),
    )
    db.conn.commit()

    resp = _connect(client)
    assert resp.status_code == 403
    assert "blocked" in resp.json()["detail"].lower()


def test_worker_disconnect_preserves_blocked_status(monkeypatch):
    db = _db()
    client = _client(db, monkeypatch, user=USER, admin=ADMIN)

    sid = _connect(client).json()["session_id"]
    db.conn.execute("UPDATE worker_sessions SET status='blocked' WHERE id=?", (sid,))
    db.conn.commit()

    # Worker receives the block command and gracefully disconnects itself —
    # this must NOT overwrite 'blocked' with 'offline'.
    resp = client.post("/api/v1/workers/disconnect", json={"session_id": sid})
    assert resp.status_code == 200

    row = db.conn.execute(
        "SELECT status FROM worker_sessions WHERE id=?", (sid,)
    ).fetchone()
    assert row[0] == "blocked"


def test_unblock_allows_reconnect(monkeypatch):
    db = _db()
    client = _client(db, monkeypatch, user=USER, admin=ADMIN)

    sid = _connect(client).json()["session_id"]
    db.conn.execute("UPDATE worker_sessions SET status='blocked' WHERE id=?", (sid,))
    db.conn.commit()
    assert _connect(client).status_code == 403

    resp = client.post(f"/api/v1/admin/workers/{sid}/unblock")
    assert resp.status_code == 200
    row = db.conn.execute(
        "SELECT status FROM worker_sessions WHERE id=?", (sid,)
    ).fetchone()
    assert row[0] == "offline"

    assert _connect(client).status_code == 200


def test_unblock_404_when_not_blocked(monkeypatch):
    db = _db()
    client = _client(db, monkeypatch, user=USER, admin=ADMIN)
    sid = _connect(client).json()["session_id"]

    resp = client.post(f"/api/v1/admin/workers/{sid}/unblock")
    assert resp.status_code == 404
