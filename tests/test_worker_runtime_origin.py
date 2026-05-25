import sqlite3
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.modules.setdefault(
    "jwt",
    types.SimpleNamespace(
        ExpiredSignatureError=Exception,
        InvalidTokenError=Exception,
        decode=lambda *args, **kwargs: {},
        encode=lambda *args, **kwargs: "",
    ),
)

from backend.server import deps
from backend.server.routers import workers


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""
        CREATE TABLE worker_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            worker_name TEXT NOT NULL,
            hostname TEXT,
            origin TEXT DEFAULT 'headless',
            launcher TEXT DEFAULT 'cli',
            status TEXT DEFAULT 'online',
            batch_capacity INTEGER DEFAULT 5,
            jobs_completed INTEGER DEFAULT 0,
            jobs_failed INTEGER DEFAULT 0,
            current_job_id INTEGER,
            current_file TEXT,
            current_phase TEXT,
            resources_json TEXT DEFAULT NULL,
            pending_command TEXT DEFAULT NULL,
            processing_mode_override TEXT DEFAULT NULL,
            batch_capacity_override INTEGER DEFAULT NULL,
            assigned_mode TEXT DEFAULT NULL,
            phase_job_count INTEGER DEFAULT 0,
            connected_at TEXT DEFAULT (datetime('now')),
            last_heartbeat TEXT DEFAULT (datetime('now')),
            disconnected_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE analysis_jobs (
            id INTEGER PRIMARY KEY,
            status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            analysis_job_id INTEGER,
            parse_status TEXT DEFAULT 'pending',
            mc_status TEXT DEFAULT 'pending',
            vv_status TEXT DEFAULT 'pending',
            mv_status TEXT DEFAULT 'pending'
        )
    """)
    conn.commit()
    return types.SimpleNamespace(conn=conn)


def _client(db, user):
    app = FastAPI()
    app.include_router(workers.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = lambda: user
    app.dependency_overrides[deps.require_admin] = lambda: user
    return TestClient(app)


def test_worker_connect_persists_origin_and_launcher():
    db = _db()
    client = _client(db, {"id": 1, "username": "worker-user", "role": "user"})

    response = client.post(
        "/api/v1/workers/connect",
        json={
            "worker_name": "cloud-a100",
            "hostname": "gpu-node-1",
            "batch_capacity": 7,
            "origin": "headless",
            "launcher": "cloud",
            "resources": {"os": "Linux", "gpu_type": "cuda"},
        },
    )

    assert response.status_code == 200
    row = db.conn.execute(
        "SELECT worker_name, origin, launcher, resources_json FROM worker_sessions"
    ).fetchone()
    assert row[0] == "cloud-a100"
    assert row[1] == "headless"
    assert row[2] == "cloud"
    assert '"os": "Linux"' in row[3]


def test_user_worker_list_returns_origin_and_launcher():
    db = _db()
    db.conn.execute(
        """INSERT INTO worker_sessions
           (user_id, worker_name, hostname, origin, launcher, status)
           VALUES (1, 'client-mac', 'macbook', 'client-launched', 'electron', 'online')"""
    )
    db.conn.commit()
    client = _client(db, {"id": 1, "username": "owner", "role": "user"})

    response = client.get("/api/v1/workers/my")

    assert response.status_code == 200
    worker = response.json()["workers"][0]
    assert worker["origin"] == "client-launched"
    assert worker["launcher"] == "electron"


def test_http_transport_connect_sends_origin_and_launcher():
    from backend.worker.transport import HttpTransport

    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return types.SimpleNamespace(
            status_code=200,
            json=lambda: {"session_id": 123, "processing_mode": "mc"},
        )

    transport = HttpTransport("http://server", fake_request)
    result = transport.connect(
        worker_name="client-worker",
        hostname="host",
        batch_capacity=3,
        resources={"os": "Darwin"},
        origin="client-launched",
        launcher="electron",
    )

    assert result["session_id"] == 123
    payload = calls[0][2]["json"]
    assert payload["origin"] == "client-launched"
    assert payload["launcher"] == "electron"


def test_worker_daemon_defaults_to_headless_cli_origin():
    from backend.worker.worker_daemon import WorkerDaemon

    daemon = WorkerDaemon(origin="headless", launcher="cli")

    assert daemon.origin == "headless"
    assert daemon.launcher == "cli"
