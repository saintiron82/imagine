"""
Worker IPC — Electron <-> Python worker bridge via stdin/stdout JSON protocol.

Allows Electron's main process to control a worker daemon through JSON messages:
  {"cmd": "start", "server_url": "...", "username": "...", "password": "..."}
  {"cmd": "stop"}
  {"cmd": "status"}

Worker sends back events:
  {"event": "status", "status": "running|idle|error", "jobs": [...]}
  {"event": "log", "message": "...", "type": "info|error|success|warning"}
  {"event": "job_done", "job_id": ..., "file_path": "...", "file_name": "..."}
  {"event": "stats", "pending": ..., "completed": ..., ...}

Usage:
    python -m backend.worker.worker_ipc
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("WorkerIPC")


def _emit(event_data: dict):
    """Send JSON event to stdout (for Electron to read)."""
    try:
        line = json.dumps(event_data, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _emit_log(message: str, log_type: str = "info"):
    _emit({"event": "log", "message": message, "type": log_type})


def _emit_status(status: str, jobs=None):
    _emit({"event": "status", "status": status, "jobs": jobs or []})


class WorkerIPCController:
    """Controls a WorkerDaemon via IPC messages."""

    def __init__(self):
        self._daemon = None
        self._thread = None
        self._running = False

    def start(self, server_url: str, username: str = "", password: str = "",
               access_token: str = "", refresh_token: str = ""):
        """Start the worker loop in a background thread.

        Supports two auth modes:
        - Token mode (access_token + refresh_token): reuse existing session from Electron
        - Credential mode (username + password): login independently
        """
        if self._running:
            _emit_log("Worker already running", "warning")
            return

        # Override config for this session
        import os
        os.environ["IMAGINE_SERVER_URL"] = server_url

        # Store auth info for the worker thread
        self._auth_mode = "token" if access_token else "credentials"
        self._access_token = access_token
        self._refresh_token = refresh_token

        if not access_token:
            os.environ["IMAGINE_WORKER_USERNAME"] = username
            os.environ["IMAGINE_WORKER_PASSWORD"] = password

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        _emit_status("running")
        _emit_log("Worker started", "success")

    def stop(self):
        """Stop the worker loop."""
        self._running = False
        _emit_status("idle")
        _emit_log("Worker stopped", "info")

    def get_status(self):
        """Report current status."""
        status = "running" if self._running else "idle"
        _emit_status(status)

    def _worker_loop(self):
        """Main worker loop — claim → process → repeat."""
        try:
            from backend.worker.worker_daemon import WorkerDaemon

            daemon = WorkerDaemon()
            self._daemon = daemon

            # Authenticate — token mode (from Electron session) or credentials mode
            _emit_log("Authenticating with server...", "info")
            if self._auth_mode == "token":
                if not daemon.set_tokens(self._access_token, self._refresh_token):
                    _emit_log("Token injection failed", "error")
                    _emit_status("error")
                    self._running = False
                    return
            elif not daemon.login():
                _emit_log("Authentication failed", "error")
                _emit_status("error")
                self._running = False
                return

            _emit_log("Authenticated successfully", "success")
            poll_interval = 5

            while self._running:
                # Claim jobs
                jobs = daemon.claim_jobs()

                if not jobs:
                    _emit_status("running", [])
                    time.sleep(poll_interval)
                    continue

                _emit_log(f"Claimed {len(jobs)} jobs", "info")

                # Process each job
                for job in jobs:
                    if not self._running:
                        break

                    _emit_status("running", [job])
                    file_name = job.get("file_path", "").rsplit("/", 1)[-1]
                    _emit_log(f"Processing: {file_name}", "info")

                    success = daemon.process_job(job)

                    _emit({
                        "event": "job_done",
                        "job_id": job.get("job_id"),
                        "file_path": job.get("file_path"),
                        "file_name": file_name,
                        "success": success,
                    })

                    if not success:
                        _emit_log(f"Failed: {file_name}", "error")

                # Brief pause between batches
                time.sleep(1)

        except Exception as e:
            _emit_log(f"Worker error: {e}", "error")
            _emit_status("error")
        finally:
            self._running = False
            _emit_status("idle")


def main():
    """IPC main loop — read JSON commands from stdin, dispatch to controller."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Logs go to stderr, events go to stdout
    )

    controller = WorkerIPCController()
    _emit({"event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            _emit_log(f"Invalid JSON: {line}", "error")
            continue

        action = cmd.get("cmd")

        if action == "start":
            controller.start(
                server_url=cmd.get("server_url", "http://localhost:8000"),
                username=cmd.get("username", ""),
                password=cmd.get("password", ""),
                access_token=cmd.get("access_token", ""),
                refresh_token=cmd.get("refresh_token", ""),
            )
        elif action == "stop":
            controller.stop()
        elif action == "status":
            controller.get_status()
        elif action == "exit":
            controller.stop()
            break
        else:
            _emit_log(f"Unknown command: {action}", "warning")


if __name__ == "__main__":
    main()
