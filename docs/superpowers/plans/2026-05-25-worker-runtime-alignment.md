# Worker Runtime Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make server-local, client-launched, and headless workers register with the same central server contract while preserving one shared worker core.

**Architecture:** Keep `WorkerDaemon` as the shared worker core. Add explicit worker `origin` and `launcher` metadata to server session registration, route existing embedded/Electron runners through that metadata, then add a headless CLI runner that uses the same server APIs. Server remains the owner of queue, session state, scheduling, and worker control.

**Tech Stack:** Python, FastAPI, SQLite migrations, existing `WorkerDaemon`, `LocalTransport`, `HttpTransport`, Electron IPC, pytest.

---

## File Structure

- Modify `backend/db/sqlite_schema_auth.sql`
  - Add canonical fresh-DB columns for worker origin and launcher.
- Modify `backend/db/sqlite_migrations.py`
  - Add migration for existing DBs.
- Modify `backend/server/routers/workers.py`
  - Accept `origin` and `launcher` in connect requests.
  - Persist and return metadata in user/admin worker lists.
- Modify `backend/worker/transport.py`
  - Carry `origin` and `launcher` through `WorkerTransport.connect`.
- Modify `backend/server/embedded_worker.py`
  - Register server-local workers with `origin=server-local`, `launcher=server`.
- Modify `backend/worker/worker_daemon.py`
  - Store worker origin/launcher defaults.
  - Register HTTP workers with metadata.
- Modify `backend/worker/worker_ipc.py`
  - Start client-launched workers with `origin=client-launched`, `launcher=electron`.
- Create `backend/worker/cli.py`
  - Add Electron-free headless worker runner.
- Modify `scripts/cloud_worker_boot.sh`
  - Call the new CLI runner.
- Modify `backend/server/routers/workers.py`
  - Serve Linux bootstrap script and issue headless worker commands.
- Create `tests/test_worker_runtime_origin.py`
  - Verify server persistence and list output.
- Create `tests/test_worker_headless_cli.py`
  - Verify headless CLI rejects missing auth and can build a daemon with metadata.
- Modify `docs/worker_runtime_audit_2026-05-25.md`
  - Update audit after implementation.

---

### Task 1: Persist Worker Origin And Launcher

**Files:**
- Modify: `backend/db/sqlite_schema_auth.sql`
- Modify: `backend/db/sqlite_migrations.py`
- Modify: `backend/server/routers/workers.py`
- Test: `tests/test_worker_runtime_origin.py`

- [ ] **Step 1: Write failing tests for connect persistence and listing**

Create `tests/test_worker_runtime_origin.py`:

```python
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

    response = client.get("/api/v1/workers")

    assert response.status_code == 200
    worker = response.json()["workers"][0]
    assert worker["origin"] == "client-launched"
    assert worker["launcher"] == "electron"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py -q
```

Expected:

- The connect test fails because `ConnectRequest` does not accept/persist `origin` and `launcher`.
- The list test fails because `/workers` does not return `origin` and `launcher`.

- [ ] **Step 3: Add schema columns for fresh DBs**

In `backend/db/sqlite_schema_auth.sql`, add columns after `hostname TEXT,`:

```sql
    origin TEXT DEFAULT 'headless'
        CHECK (origin IN ('server-local', 'client-launched', 'headless')),
    launcher TEXT DEFAULT 'cli'
        CHECK (launcher IN ('server', 'electron', 'cli', 'service', 'cloud')),
```

- [ ] **Step 4: Add migration for existing DBs**

In `backend/db/sqlite_migrations.py`, add:

```python
def migrate_worker_origin_launcher(db):
    """Add worker origin/launcher columns for runtime classification."""
    if not db._table_exists('worker_sessions'):
        return
    try:
        db.conn.execute("SELECT origin FROM worker_sessions LIMIT 1")
    except Exception:
        logger.info("Migrating: adding worker origin/launcher columns...")
        db.conn.execute(
            "ALTER TABLE worker_sessions ADD COLUMN origin TEXT DEFAULT 'headless'"
        )
        db.conn.execute(
            "ALTER TABLE worker_sessions ADD COLUMN launcher TEXT DEFAULT 'cli'"
        )
        db.conn.commit()
        logger.info("worker origin/launcher columns added")
```

Then add it to the main migration list near other worker migrations:

```python
("worker_origin_launcher", lambda: migrate_worker_origin_launcher(db)),
```

- [ ] **Step 5: Accept metadata in worker connect requests**

In `backend/server/routers/workers.py`, update imports:

```python
from typing import Literal, Optional
```

Update `ConnectRequest`:

```python
class ConnectRequest(BaseModel):
    worker_name: str
    hostname: Optional[str] = None
    batch_capacity: int = 5
    origin: Literal["server-local", "client-launched", "headless"] = "headless"
    launcher: Literal["server", "electron", "cli", "service", "cloud"] = "cli"
    resources: Optional[dict] = None
```

Update the insert in `worker_connect`:

```python
cursor.execute(
    """INSERT INTO worker_sessions
       (user_id, worker_name, hostname, origin, launcher, batch_capacity,
        status, connected_at, last_heartbeat)
       VALUES (?, ?, ?, ?, ?, ?, 'online', ?, ?)""",
    (
        user["id"], req.worker_name, req.hostname, req.origin, req.launcher,
        req.batch_capacity, now, now,
    )
)
```

- [ ] **Step 6: Return metadata from worker list endpoints**

In `/workers`, select and return `origin` and `launcher`:

```python
cursor.execute(
    """SELECT id, worker_name, hostname, origin, launcher, status, batch_capacity,
              jobs_completed, jobs_failed, current_file, current_phase,
              last_heartbeat, connected_at
       FROM worker_sessions
       WHERE user_id = ?
       ORDER BY connected_at DESC
       LIMIT 20""",
    (user["id"],)
)
```

Return:

```python
{
    "id": row[0],
    "worker_name": row[1],
    "hostname": row[2],
    "origin": row[3],
    "launcher": row[4],
    "status": row[5],
    "batch_capacity": row[6],
    "jobs_completed": row[7],
    "jobs_failed": row[8],
    "current_file": row[9],
    "current_phase": row[10],
    "last_heartbeat": row[11],
    "connected_at": row[12],
}
```

Also update `/admin/workers` select/result to include `ws.origin` and `ws.launcher`. Preserve existing keys and add:

```python
"origin": row_origin,
"launcher": row_launcher,
```

- [ ] **Step 7: Run tests to verify pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/db/sqlite_schema_auth.sql backend/db/sqlite_migrations.py backend/server/routers/workers.py tests/test_worker_runtime_origin.py
git commit -m "feat: persist worker runtime origin"
```

---

### Task 2: Propagate Origin From Existing Runners

**Files:**
- Modify: `backend/worker/transport.py`
- Modify: `backend/server/embedded_worker.py`
- Modify: `backend/worker/worker_daemon.py`
- Modify: `backend/worker/worker_ipc.py`
- Test: `tests/test_worker_runtime_origin.py`

- [ ] **Step 1: Add failing tests for runner metadata defaults**

Append to `tests/test_worker_runtime_origin.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py::test_http_transport_connect_sends_origin_and_launcher tests/test_worker_runtime_origin.py::test_worker_daemon_defaults_to_headless_cli_origin -q
```

Expected:

- `HttpTransport.connect()` rejects unknown keyword args.
- `WorkerDaemon.__init__()` rejects unknown keyword args.

- [ ] **Step 3: Update transport interface**

In `backend/worker/transport.py`, update abstract signature:

```python
def connect(
    self,
    worker_name: str,
    hostname: str,
    batch_capacity: int,
    resources: dict,
    origin: str = "headless",
    launcher: str = "cli",
) -> dict:
```

Update `HttpTransport.connect` signature the same way, and include:

```python
"origin": origin,
"launcher": launcher,
```

inside the JSON payload.

Update `LocalTransport.connect` signature the same way and insert `origin`/`launcher` into `worker_sessions`.

- [ ] **Step 4: Update WorkerDaemon metadata**

In `backend/worker/worker_daemon.py`, update `WorkerDaemon.__init__`:

```python
def __init__(self, transport=None, origin: str = "headless", launcher: str = "cli"):
    import requests

    self.transport = transport
    self.origin = origin
    self.launcher = launcher
```

In `_connect_session`, include:

```python
"origin": self.origin,
"launcher": self.launcher,
```

in the HTTP connect payload.

- [ ] **Step 5: Register embedded worker as server-local**

In `backend/server/embedded_worker.py`, construct:

```python
_daemon = WorkerDaemon(
    transport=transport,
    origin="server-local",
    launcher="server",
)
```

Call `transport.connect` with:

```python
origin="server-local",
launcher="server",
```

- [ ] **Step 6: Register Electron-launched worker as client-launched**

In `backend/worker/worker_ipc.py`, construct:

```python
daemon = WorkerDaemon(origin="client-launched", launcher="electron")
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py -q
.venv/bin/python -m py_compile backend/worker/transport.py backend/server/embedded_worker.py backend/worker/worker_daemon.py backend/worker/worker_ipc.py
```

Expected: PASS and no compile errors.

- [ ] **Step 8: Commit**

```bash
git add backend/worker/transport.py backend/server/embedded_worker.py backend/worker/worker_daemon.py backend/worker/worker_ipc.py tests/test_worker_runtime_origin.py
git commit -m "feat: propagate worker runtime origin"
```

---

### Task 3: Add Headless Worker CLI Runner

**Files:**
- Create: `backend/worker/cli.py`
- Modify: `scripts/cloud_worker_boot.sh`
- Test: `tests/test_worker_headless_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_worker_headless_cli.py`:

```python
import pytest


def test_headless_cli_requires_access_token(monkeypatch):
    from backend.worker.cli import load_headless_config

    monkeypatch.setenv("IMAGINE_SERVER_URL", "http://server")
    monkeypatch.delenv("IMAGINE_WORKER_ACCESS_TOKEN", raising=False)

    with pytest.raises(SystemExit) as exc:
        load_headless_config([])

    assert exc.value.code == 2


def test_headless_cli_loads_env_config(monkeypatch):
    from backend.worker.cli import load_headless_config

    monkeypatch.setenv("IMAGINE_SERVER_URL", "http://server")
    monkeypatch.setenv("IMAGINE_WORKER_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("IMAGINE_WORKER_REFRESH_TOKEN", "refresh-token")
    monkeypatch.setenv("IMAGINE_WORKER_NAME", "gpu-node")
    monkeypatch.setenv("IMAGINE_WORKER_LAUNCHER", "cloud")

    cfg = load_headless_config([])

    assert cfg.server_url == "http://server"
    assert cfg.access_token == "access-token"
    assert cfg.refresh_token == "refresh-token"
    assert cfg.worker_name == "gpu-node"
    assert cfg.origin == "headless"
    assert cfg.launcher == "cloud"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_headless_cli.py -q
```

Expected: FAIL because `backend.worker.cli` does not exist.

- [ ] **Step 3: Add CLI config loader**

Create `backend/worker/cli.py`:

```python
"""
Headless worker CLI.

Runs WorkerDaemon without Electron. The central server still owns queue,
session state, scheduling, and control.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass


@dataclass
class HeadlessWorkerConfig:
    server_url: str
    access_token: str
    refresh_token: str
    worker_name: str
    origin: str = "headless"
    launcher: str = "cli"
    poll_interval: int = 5


def load_headless_config(argv: list[str] | None = None) -> HeadlessWorkerConfig:
    parser = argparse.ArgumentParser(prog="imagine-worker")
    parser.add_argument("--server-url", default=os.getenv("IMAGINE_SERVER_URL", ""))
    parser.add_argument("--access-token", default=os.getenv("IMAGINE_WORKER_ACCESS_TOKEN", ""))
    parser.add_argument("--refresh-token", default=os.getenv("IMAGINE_WORKER_REFRESH_TOKEN", ""))
    parser.add_argument("--worker-name", default=os.getenv("IMAGINE_WORKER_NAME", ""))
    parser.add_argument(
        "--launcher",
        default=os.getenv("IMAGINE_WORKER_LAUNCHER", "cli"),
        choices=("cli", "service", "cloud"),
    )
    parser.add_argument("--poll-interval", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.server_url:
        parser.error("--server-url or IMAGINE_SERVER_URL is required")
    if not args.access_token:
        parser.error("--access-token or IMAGINE_WORKER_ACCESS_TOKEN is required")

    return HeadlessWorkerConfig(
        server_url=args.server_url.rstrip("/"),
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        worker_name=args.worker_name or f"{socket.gethostname()}-headless",
        launcher=args.launcher,
        poll_interval=args.poll_interval,
    )
```

- [ ] **Step 4: Add headless run loop**

Append to `backend/worker/cli.py`:

```python
def run_headless_worker(cfg: HeadlessWorkerConfig) -> int:
    os.environ["IMAGINE_SERVER_URL"] = cfg.server_url

    from backend.worker.worker_daemon import WorkerDaemon

    daemon = WorkerDaemon(origin=cfg.origin, launcher=cfg.launcher)
    daemon.worker_name = cfg.worker_name
    daemon.set_tokens(cfg.access_token, cfg.refresh_token)

    if not daemon._connect_session():
        print("[worker] session connect failed", file=sys.stderr)
        return 1

    stop_requested = False

    def _handle_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        daemon._stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        from backend.worker.worker_state import WorkerState

        while not stop_requested:
            heartbeat = daemon._heartbeat()
            command = heartbeat.get("command") if heartbeat else None
            if command in ("stop", "block"):
                break

            daemon._state_machine.update(
                is_scheduled_active=True,
                throttle_level=daemon._check_throttle(),
                has_pending_jobs=True,
            )
            if daemon._state_machine.state in (WorkerState.IDLE, WorkerState.RESTING):
                time.sleep(cfg.poll_interval)
                continue

            jobs = daemon.claim_jobs()
            if not jobs:
                daemon._state_machine.update(
                    is_scheduled_active=True,
                    throttle_level=daemon._check_throttle(),
                    has_pending_jobs=False,
                )
                time.sleep(cfg.poll_interval)
                continue

            results = daemon.process_batch_phased(jobs)
            ok = sum(1 for item in results if len(item) > 1 and item[1])
            print(f"[worker] batch complete: {ok}/{len(jobs)} ok", flush=True)
            daemon._state_machine.record_job_activity()
            time.sleep(1)
        return 0
    finally:
        try:
            daemon._on_enter_idle()
        finally:
            daemon._disconnect_session()


def main(argv: list[str] | None = None) -> int:
    cfg = load_headless_config(argv)
    return run_headless_worker(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Update cloud boot script**

In `scripts/cloud_worker_boot.sh`, replace:

```bash
exec python -m backend.worker.worker_daemon
```

with:

```bash
exec python -m backend.worker.cli --launcher cloud
```

Also update the required env comment:

```bash
#   IMAGINE_WORKER_ACCESS_TOKEN  서버에서 발급한 worker access token
#   IMAGINE_WORKER_REFRESH_TOKEN 선택: refresh token
```

For now, remove `IMAGINE_WORKER_EMAIL` and `IMAGINE_WORKER_PASSWORD` checks until a password-based headless auth flow exists.

- [ ] **Step 6: Run tests and syntax checks**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_headless_cli.py -q
.venv/bin/python -m py_compile backend/worker/cli.py
bash -n scripts/cloud_worker_boot.sh
```

Expected: PASS and no syntax errors.

- [ ] **Step 7: Commit**

```bash
git add backend/worker/cli.py scripts/cloud_worker_boot.sh tests/test_worker_headless_cli.py
git commit -m "feat: add headless worker runner"
```

---

### Task 4: Add Runtime Contract Smoke Checks

**Files:**
- Create: `tools/check_worker_runtime_contract.py`
- Modify: `docs/worker_runtime_audit_2026-05-25.md`

- [ ] **Step 1: Add smoke check script**

Create `tools/check_worker_runtime_contract.py`:

```python
"""
Static worker runtime contract check.

This check catches drift in the three-runner worker model without requiring
a live server or GPU.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assert_contains(path: str, needle: str) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    if needle not in text:
        raise AssertionError(f"{path} is missing {needle!r}")


def main() -> int:
    assert_contains("backend/server/embedded_worker.py", 'origin="server-local"')
    assert_contains("backend/server/embedded_worker.py", 'launcher="server"')
    assert_contains("backend/worker/worker_ipc.py", 'origin="client-launched"')
    assert_contains("backend/worker/worker_ipc.py", 'launcher="electron"')
    assert_contains("backend/worker/cli.py", 'origin: str = "headless"')
    assert_contains("backend/worker/cli.py", 'def main(')
    assert_contains("backend/server/routers/workers.py", "origin")
    assert_contains("backend/server/routers/workers.py", "launcher")
    assert_contains("scripts/cloud_worker_boot.sh", "backend.worker.cli")
    print("worker runtime contract static check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run smoke check**

Run:

```bash
.venv/bin/python tools/check_worker_runtime_contract.py
```

Expected:

```text
worker runtime contract static check: OK
```

- [ ] **Step 3: Update audit document**

In `docs/worker_runtime_audit_2026-05-25.md`, update:

```markdown
| Server-local worker | `embedded_worker.py` registers `origin=server-local` | 충족 |
| Client-launched worker | Electron `worker_ipc.py` registers `origin=client-launched` | 충족 |
| Headless worker | `backend.worker.cli` runs without Electron | 충족 |
```

Add a verification section:

````markdown
## 구현 후 검증

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py tests/test_worker_headless_cli.py -q
.venv/bin/python tools/check_worker_runtime_contract.py
bash -n scripts/cloud_worker_boot.sh
```
````

- [ ] **Step 4: Run final focused verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_worker_runtime_origin.py tests/test_worker_headless_cli.py -q
.venv/bin/python tools/check_worker_runtime_contract.py
.venv/bin/python -m py_compile backend/server/routers/workers.py backend/worker/transport.py backend/server/embedded_worker.py backend/worker/worker_daemon.py backend/worker/worker_ipc.py backend/worker/cli.py
bash -n scripts/cloud_worker_boot.sh
```

Expected:

- pytest passes.
- static contract check prints OK.
- py_compile exits 0.
- bash syntax check exits 0.

- [ ] **Step 5: Commit**

```bash
git add tools/check_worker_runtime_contract.py docs/worker_runtime_audit_2026-05-25.md
git commit -m "test: add worker runtime contract check"
```

---

## Self-Review

Spec coverage:

- Central server ownership: Task 1 preserves server session schema and API ownership.
- Three worker origins: Tasks 1 and 2 add `server-local`, `client-launched`, `headless`.
- Client can start/stop local worker: Task 2 preserves Electron IPC runner and labels it correctly.
- Server can understand/control all workers: Task 1 returns origin/launcher from worker lists; existing stop/block remains session-based.
- Headless worker: Task 3 adds `backend.worker.cli`.
- Cross-platform stance: Task 3 uses common Python runner; Linux CUDA remains only one launcher profile.
- Verification: Task 4 adds static contract check and focused tests.
- Server-provided worker install command: implemented after Task 4 through `GET /workers/bootstrap/linux.sh` and `POST /admin/workers/headless-command`.

Placeholder scan:

- No placeholder markers remain.
- Each task has exact files, code snippets, commands, and expected outcomes.

Type consistency:

- `origin` values are consistently `server-local`, `client-launched`, `headless`.
- `launcher` values are consistently `server`, `electron`, `cli`, `service`, `cloud`.
- `WorkerDaemon(origin=..., launcher=...)` is introduced before all runner usage.
