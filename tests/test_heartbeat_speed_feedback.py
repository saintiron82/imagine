"""Heartbeat → scheduler speed feedback (2026-06-11).

Workers report measured per-phase throughput in every heartbeat; the
handler must feed it into scheduler.update_speed (EMA) — this is what
warms cold-start workers past COLD_START_BATCH and keeps benchmark
speeds current. Also pins the slimmed heartbeat response contract:
command + batch_hint only (no per-beat mode re-derivation).
"""

import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps
from backend.server.routers import workers


class RecordingScheduler:
    def __init__(self):
        self.speed_updates = []

    def update_speed(self, session_id, phase, fpm):
        self.speed_updates.append((session_id, phase, fpm))


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""
        CREATE TABLE worker_sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            status TEXT DEFAULT 'online',
            pending_command TEXT,
            batch_capacity INTEGER DEFAULT 5,
            processing_mode_override TEXT,
            batch_capacity_override INTEGER,
            jobs_completed INTEGER DEFAULT 0,
            jobs_failed INTEGER DEFAULT 0,
            current_job_id INTEGER,
            current_file TEXT,
            current_phase TEXT,
            resources_json TEXT,
            phase_job_count INTEGER DEFAULT 0,
            last_heartbeat TEXT
        )
    """)
    conn.execute(
        "INSERT INTO worker_sessions (id, user_id, status) VALUES (10, 1, 'online')"
    )
    conn.commit()
    return SimpleNamespace(conn=conn)


def _client(db, scheduler=None):
    app = FastAPI()
    app.include_router(workers.router, prefix="/api/v1")
    if scheduler is not None:
        app.state.scheduler = scheduler
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": 1, "username": "w"}
    return TestClient(app)


def test_phase_throughput_feeds_scheduler():
    db = _db()
    scheduler = RecordingScheduler()
    client = _client(db, scheduler)

    resp = client.post("/api/v1/workers/heartbeat", json={
        "session_id": 10,
        "jobs_completed": 5,
        "phase_throughput": {"mc": 7.5, "vv": 80.0, "mv": 0},
    })
    assert resp.status_code == 200
    # mv=0 must be skipped (no measurement)
    assert scheduler.speed_updates == [(10, "mc", 7.5), (10, "vv", 80.0)]

    # And it must be persisted for the UI speed cells
    row = db.conn.execute(
        "SELECT resources_json FROM worker_sessions WHERE id=10"
    ).fetchone()
    assert '"phase_throughput"' in row[0]


def test_heartbeat_without_throughput_skips_scheduler():
    db = _db()
    scheduler = RecordingScheduler()
    client = _client(db, scheduler)

    resp = client.post("/api/v1/workers/heartbeat", json={"session_id": 10})
    assert resp.status_code == 200
    assert scheduler.speed_updates == []


def test_heartbeat_response_contract_and_command_delivery():
    db = _db()
    db.conn.execute("UPDATE worker_sessions SET pending_command='stop' WHERE id=10")
    db.conn.commit()
    client = _client(db, RecordingScheduler())

    data = client.post(
        "/api/v1/workers/heartbeat", json={"session_id": 10}
    ).json()

    assert data["command"] == "stop"
    assert "batch_hint" in data
    # Per-beat mode re-derivation was removed — claim is authoritative
    assert "processing_mode" not in data

    # Command is consumed (cleared) on delivery
    row = db.conn.execute(
        "SELECT pending_command FROM worker_sessions WHERE id=10"
    ).fetchone()
    assert row[0] is None


def test_heartbeat_throttle_reduces_batch_hint():
    db = _db()
    client = _client(db, RecordingScheduler())

    data = client.post("/api/v1/workers/heartbeat", json={
        "session_id": 10,
        "throttle_level": "danger",
    }).json()
    assert data["batch_hint"] == 1

    data = client.post("/api/v1/workers/heartbeat", json={
        "session_id": 10,
        "throttle_level": "critical",
    }).json()
    assert data["batch_hint"] == 0
