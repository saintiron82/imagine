import sqlite3
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps
from backend.server.routers import workers


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            firebase_uid TEXT,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            last_login_at TEXT,
            quota_files_per_day INTEGER DEFAULT 1000,
            quota_search_per_min INTEGER DEFAULT 60
        )
    """)
    conn.execute("""
        CREATE TABLE refresh_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            revoked INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        """INSERT INTO users (username, email, password_hash, role, is_active)
           VALUES ('admin', 'admin@example.com', '', 'admin', 1)"""
    )
    conn.commit()
    return types.SimpleNamespace(conn=conn)


def _client(db):
    app = FastAPI()
    app.include_router(workers.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1,
        "username": "admin",
        "role": "admin",
    }
    return TestClient(app, base_url="https://imagine.test")


def test_linux_bootstrap_script_is_public_and_runs_headless_cli():
    client = _client(_db())

    response = client.get("/api/v1/workers/bootstrap/linux.sh")

    assert response.status_code == 200
    assert "text/x-shellscript" in response.headers["content-type"]
    assert "IMAGINE_SERVER_URL" in response.text
    assert "IMAGINE_WORKER_ACCESS_TOKEN" in response.text
    assert "python -m backend.worker.cli --launcher" in response.text


def test_admin_can_issue_headless_worker_command(monkeypatch):
    db = _db()
    client = _client(db)

    response = client.post(
        "/api/v1/admin/workers/headless-command",
        json={
            "worker_name": "cloud-a100-1",
            "launcher": "cloud",
            "expires_minutes": 120,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["worker_name"] == "cloud-a100-1"
    assert body["launcher"] == "cloud"
    assert body["server_url"] == "https://imagine.test"
    assert body["bootstrap_url"] == "https://imagine.test/api/v1/workers/bootstrap/linux.sh"
    assert "curl -fsSL" in body["linux_command"]
    assert "IMAGINE_WORKER_ACCESS_TOKEN=" in body["linux_command"]
    assert body["access_token"]
    assert body["refresh_token"]

    user_row = db.conn.execute(
        "SELECT username, role, is_active FROM users WHERE username = ?",
        ("worker:cloud-a100-1",),
    ).fetchone()
    assert user_row == ("worker:cloud-a100-1", "user", 1)

    token_count = db.conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]
    assert token_count == 1
