"""
Local worker supervisor — runs the server machine's worker as a separate
OS process, identical to an external worker.

The worker is the same WorkerDaemon + HttpTransport used by external
workers; it connects to this server over localhost HTTP. The server only
spawns and supervises the process — placement is still the scheduler's job.
This replaces the old in-process embedded worker (LocalTransport).

Public API kept compatible with the old embedded_worker module:
  start_worker() / stop_worker() / get_status()
"""

import json
import logging
import os
import secrets
import socket
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

WORKER_USERNAME = "local-worker"

_proc: subprocess.Popen | None = None
_start_lock = threading.Lock()
_last_error: str | None = None


def _worker_name() -> str:
    return f"{socket.gethostname()}-local"


def _ensure_worker_user(db) -> tuple[int, str, str]:
    """Get or create the dedicated system user the local worker runs as.

    get_current_user() requires a live row in `users`, so the worker
    authenticates as this account (role: user — worker endpoints only
    need an authenticated session).
    """
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT id, username, role, is_active FROM users WHERE username = ?",
        (WORKER_USERNAME,),
    )
    row = cursor.fetchone()
    if row:
        if not row[3]:
            cursor.execute("UPDATE users SET is_active = 1 WHERE id = ?", (row[0],))
        db.conn.commit()
        return row[0], row[1], row[2]

    # password_hash is NOT NULL; store an unusable value (no password login)
    cursor.execute(
        "INSERT INTO users (username, password_hash, role, is_active) VALUES (?, ?, 'user', 1)",
        (WORKER_USERNAME, "!" + secrets.token_hex(32)),
    )
    db.conn.commit()
    user_id = cursor.lastrowid
    logger.info(f"Created system user '{WORKER_USERNAME}' (id={user_id})")
    return user_id, WORKER_USERNAME, "user"


def _mint_tokens(db, user_id: int, username: str, role: str) -> tuple[str, str]:
    """Issue access + refresh tokens so the worker can use the normal
    /auth/refresh rotation like any external worker."""
    from backend.server.auth.jwt import (
        create_access_token, create_refresh_token,
        hash_refresh_token, get_refresh_token_expiry,
    )

    access = create_access_token(user_id, username, role)
    refresh = create_refresh_token()
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
        (user_id, hash_refresh_token(refresh), get_refresh_token_expiry().isoformat()),
    )
    db.conn.commit()
    return access, refresh


def _server_port() -> int:
    """Resolve the port this server is listening on."""
    env_port = os.environ.get("IMAGINE_SELF_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    try:
        from backend.utils.config import get_config
        return int(get_config().get("server.port", 8000) or 8000)
    except Exception:
        return 8000


def _build_command() -> list[str]:
    """Worker launch command — frozen build uses the bundled CLI binary,
    dev uses `python -m backend.worker.cli`."""
    args = [
        "--server-url", f"http://127.0.0.1:{_server_port()}",
        "--worker-name", _worker_name(),
        "--launcher", "server",
    ]
    if getattr(sys, "frozen", False):
        return [sys.executable, "worker", *args]
    return [sys.executable, "-m", "backend.worker.cli", *args]


def _pump_output(proc: subprocess.Popen):
    """Forward worker stdout/stderr lines into the server log."""
    try:
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                logger.info(f"[local-worker] {line}")
    except Exception:
        pass


def start_worker() -> dict:
    """Spawn the local worker process if not already running."""
    global _proc, _last_error

    if not _start_lock.acquire(blocking=False):
        return {"success": False, "error": "Start in progress"}
    try:
        if _proc is not None and _proc.poll() is None:
            return {"success": False, "error": "Worker already running"}

        from backend.server.deps import get_db
        db = get_db()
        user_id, username, role = _ensure_worker_user(db)
        access, refresh = _mint_tokens(db, user_id, username, role)

        project_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        env["IMAGINE_WORKER_ACCESS_TOKEN"] = access
        env["IMAGINE_WORKER_REFRESH_TOKEN"] = refresh

        _proc = subprocess.Popen(
            _build_command(),
            cwd=str(project_root),
            env=env,
            stdin=subprocess.PIPE,   # worker's parent watchdog exits on EOF
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=_pump_output, args=(_proc,), daemon=True).start()
        _last_error = None
        logger.info(f"Local worker spawned (pid={_proc.pid}, name={_worker_name()})")
        return {"success": True, "pid": _proc.pid}
    except Exception as e:
        _last_error = str(e)
        logger.error(f"Local worker start failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        _start_lock.release()


def stop_worker() -> dict:
    """Terminate the local worker process (SIGTERM → kill)."""
    global _proc
    if _proc is None or _proc.poll() is not None:
        _proc = None
        return {"success": True, "message": "Not running"}

    jobs_completed = (_session_row() or {}).get("jobs_completed", 0)
    _proc.terminate()
    try:
        _proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Local worker did not exit on SIGTERM — killing")
        _proc.kill()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    logger.info("Local worker stopped")
    _proc = None
    return {"success": True, "jobs_completed": jobs_completed}


def _session_row() -> dict | None:
    """Latest online session for the local worker (heartbeat-backed)."""
    try:
        from backend.server.deps import get_db
        db = get_db()
        cursor = db.conn.cursor()
        cursor.execute(
            """SELECT jobs_completed, current_phase, current_file,
                      batch_capacity, resources_json
               FROM worker_sessions
               WHERE worker_name = ? AND status = 'online'
               ORDER BY last_heartbeat DESC LIMIT 1""",
            (_worker_name(),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        resources = {}
        if row[4]:
            try:
                resources = json.loads(row[4])
            except (TypeError, ValueError):
                pass
        return {
            "jobs_completed": row[0] or 0,
            "current_phase": row[1],
            "current_file": row[2],
            "batch_capacity": row[3] or 0,
            "phase_counts": resources.get("phase_counts"),
            "batch_throughput": resources.get("batch_throughput", 0.0),
            "phase_throughput": resources.get("phase_throughput", {}),
        }
    except Exception:
        return None


def get_status() -> dict:
    """Process liveness + session metrics (same data external workers report)."""
    running = _proc is not None and _proc.poll() is None
    result = {
        "running": running,
        "status": "running" if running else "idle",
        "jobs_completed": 0,
        "last_error": _last_error,
        "current_phase": None,
        "current_file": None,
        "phase_counts": None,
        "batch_throughput": 0.0,
        "batch_capacity": 0,
        "phase_throughput": {},
    }
    if not running:
        return result
    session = _session_row()
    if session:
        result.update(session)
    return result
