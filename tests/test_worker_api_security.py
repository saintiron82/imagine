import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps
from backend.server.routers import analysis, workers


class FakeScheduler:
    def __init__(self, phase="mc", count=1):
        self.phase = phase
        self.count = count
        self.calls = []

    def assign(self, worker_id):
        self.calls.append(worker_id)
        return {"phase": self.phase, "count": self.count}


class FakeAnalysisManager:
    def __init__(self):
        self.claims = []
        self.started = []
        self.completed = []
        self.reclaimed = []

    def claim_tasks(self, phase, worker_id, count):
        self.claims.append((phase, worker_id, count))
        return [{"task_id": 100, "file_id": 500, "phase": phase}]

    def start_task_phase(self, task_id, phase):
        self.started.append((task_id, phase))

    def complete_task_phase(self, task_id, phase, success, error_message=None, elapsed_s=None):
        self.completed.append((task_id, phase, success, error_message, elapsed_s))

    def reclaim_worker_tasks(self, session_id):
        self.reclaimed.append(session_id)
        return 3


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""
        CREATE TABLE worker_sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            disconnected_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            mc_status TEXT DEFAULT 'pending',
            mc_assigned_to INTEGER,
            vv_status TEXT DEFAULT 'pending',
            vv_assigned_to INTEGER,
            mv_status TEXT DEFAULT 'pending',
            mv_assigned_to INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO worker_sessions (id, user_id, status) VALUES (10, 1, 'online')"
    )
    conn.execute(
        "INSERT INTO worker_sessions (id, user_id, status) VALUES (20, 2, 'online')"
    )
    conn.execute("""
        INSERT INTO file_tasks
            (id, file_id, mc_status, mc_assigned_to, vv_status, vv_assigned_to)
        VALUES
            (100, 500, 'assigned', 10, 'assigned', 20)
    """)
    conn.commit()
    return SimpleNamespace(conn=conn)


def _client(db, user, monkeypatch, manager=None, scheduler=None):
    app = FastAPI()
    app.include_router(analysis.router)
    app.include_router(workers.router, prefix="/api/v1")
    if scheduler is not None:
        app.state.scheduler = scheduler

    app.dependency_overrides[deps.get_current_user] = lambda: user
    app.dependency_overrides[deps.get_db_safe] = lambda: db

    if manager is not None:
        monkeypatch.setattr(analysis, "_get_manager", lambda _db: manager)

        import backend.server.queue.analysis_manager as analysis_manager_module

        monkeypatch.setattr(
            analysis_manager_module,
            "AnalysisJobManager",
            lambda _db: manager,
        )

    return TestClient(app)


def test_claim_rejects_worker_session_owned_by_another_user(monkeypatch):
    db = _db()
    scheduler = FakeScheduler()
    manager = FakeAnalysisManager()
    client = _client(db, {"id": 1, "username": "alice"}, monkeypatch, manager, scheduler)

    resp = client.post("/api/v1/tasks/claim", json={"worker_id": 20})

    assert resp.status_code == 403
    assert scheduler.calls == []
    assert manager.claims == []


def test_claim_uses_only_current_users_worker_session(monkeypatch):
    db = _db()
    scheduler = FakeScheduler(phase="mc", count=1)
    manager = FakeAnalysisManager()
    client = _client(db, {"id": 1, "username": "alice"}, monkeypatch, manager, scheduler)

    resp = client.post("/api/v1/tasks/claim", json={"worker_id": 10})

    assert resp.status_code == 200
    assert resp.json()["count"] == 1
    assert scheduler.calls == [10]
    assert manager.claims == [("mc", 10, 1)]


def test_start_and_complete_reject_task_assigned_to_another_users_worker(monkeypatch):
    db = _db()
    manager = FakeAnalysisManager()
    client = _client(db, {"id": 2, "username": "bob"}, monkeypatch, manager)

    start_resp = client.post("/api/v1/tasks/start", json={"task_id": 100, "phase": "mc"})
    complete_resp = client.post(
        "/api/v1/tasks/complete",
        json={"task_id": 100, "phase": "mc", "success": True},
    )

    assert start_resp.status_code == 403
    assert complete_resp.status_code == 403
    assert manager.started == []
    assert manager.completed == []


def test_owner_can_start_and_complete_assigned_task(monkeypatch):
    db = _db()
    manager = FakeAnalysisManager()
    client = _client(db, {"id": 1, "username": "alice"}, monkeypatch, manager)

    start_resp = client.post("/api/v1/tasks/start", json={"task_id": 100, "phase": "mc"})
    complete_resp = client.post(
        "/api/v1/tasks/complete",
        json={"task_id": 100, "phase": "mc", "success": True, "elapsed_s": 1.5},
    )

    assert start_resp.status_code == 200
    assert complete_resp.status_code == 200
    assert manager.started == [(100, "mc")]
    assert manager.completed == [(100, "mc", True, None, 1.5)]


def test_disconnect_rejects_worker_session_owned_by_another_user(monkeypatch):
    db = _db()
    manager = FakeAnalysisManager()
    client = _client(db, {"id": 1, "username": "alice"}, monkeypatch, manager)

    resp = client.post("/api/v1/workers/disconnect", json={"session_id": 20})

    assert resp.status_code == 403
    assert manager.reclaimed == []


def test_disconnect_reclaims_only_current_users_worker_session(monkeypatch):
    db = _db()
    manager = FakeAnalysisManager()
    client = _client(db, {"id": 1, "username": "alice"}, monkeypatch, manager)

    resp = client.post("/api/v1/workers/disconnect", json={"session_id": 10})

    assert resp.status_code == 200
    assert resp.json()["reclaimed"] == 3
    assert manager.reclaimed == [10]
