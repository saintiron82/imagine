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

    def update_tokens(self, access_token: str, refresh_token: str):
        """Update JWT tokens mid-session (forwarded from browser refresh)."""
        if self._daemon:
            self._daemon.set_tokens(access_token, refresh_token)
            _emit_log("Tokens refreshed from app session", "info")

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
        """Main worker loop — claim → batch process by phase → repeat."""
        try:
            from backend.worker.worker_daemon import WorkerDaemon
            from backend.worker.config import get_heartbeat_interval

            daemon = WorkerDaemon()
            self._daemon = daemon

            # Authenticate — token mode (from Electron session) or credentials mode
            at_preview = (self._access_token[:20] + "...") if self._access_token else "(none)"
            rt_preview = (self._refresh_token[:16] + "...") if self._refresh_token else "(none)"
            _emit_log(f"Auth mode: {self._auth_mode}, access={at_preview}, refresh={rt_preview}", "info")
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

            # Register worker session with server
            daemon._connect_session()
            if daemon.session_id:
                _emit_log(f"Session registered (id={daemon.session_id})", "info")

            poll_interval = 5
            heartbeat_interval = get_heartbeat_interval()
            last_heartbeat = time.time()

            while self._running:
                # Periodic heartbeat
                if time.time() - last_heartbeat >= heartbeat_interval:
                    hb = daemon._heartbeat()
                    last_heartbeat = time.time()
                    cmd = hb.get("command")
                    if cmd in ("stop", "block"):
                        _emit_log(f"Server command: {cmd}", "warning")
                        self._running = False
                        break

                # Claim jobs
                jobs = daemon.claim_jobs()

                if not jobs:
                    _emit_status("running", [])
                    time.sleep(poll_interval)
                    continue

                _emit_log(f"Claimed {len(jobs)} jobs — batch processing", "info")
                _emit({"event": "batch_start", "batch_size": len(jobs)})

                # Batch progress callback — relay events to Electron
                def _batch_progress_cb(event_type, data):
                    evt = {"event": f"batch_{event_type}", **data}
                    _emit(evt)
                    if event_type == "phase_start":
                        _emit_log(f"Phase {data.get('phase', '?')} — {data.get('count', 0)} files", "info")
                    elif event_type == "file_done":
                        bs = data.get('batch_size', 1)
                        bs_tag = f" [x{bs}]" if bs > 1 else ""
                        _emit_log(f"  [{data.get('phase', '?')}] {data.get('index', 0)}/{data.get('count', 0)} {data.get('file_name', '')}{bs_tag}", "info")
                    elif event_type == "phase_complete":
                        fpm = data.get('files_per_min', 0)
                        elapsed = data.get('elapsed_s', 0)
                        _emit_log(f"Phase {data.get('phase', '?')} done — {elapsed}s ({fpm:.1f}/min)", "success")
                    elif event_type == "batch_complete":
                        fpm = data.get('files_per_min', 0)
                        elapsed = data.get('elapsed_s', 0)
                        _emit_log(f"Batch complete — {data.get('count', 0)} files in {elapsed}s ({fpm:.1f}/min)", "success")

                # Phase-level batch processing
                results = daemon.process_batch_phased(
                    jobs, progress_callback=_batch_progress_cb
                )

                # Emit individual job_done events (for compatibility)
                for job_id, success in results:
                    job_info = next(
                        (j for j in jobs if j.get("job_id") == job_id), {}
                    )
                    file_name = job_info.get("file_path", "").rsplit("/", 1)[-1]
                    _emit({
                        "event": "job_done",
                        "job_id": job_id,
                        "file_path": job_info.get("file_path"),
                        "file_name": file_name,
                        "success": success,
                    })
                    if not success:
                        _emit_log(f"Failed: {file_name}", "error")

                # Brief pause between batches
                time.sleep(1)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _emit_log(f"Worker error: {e}", "error")
            _emit_log(f"Traceback: {tb}", "error")
            _emit_status("error")
            logger.error(f"Worker loop crashed: {e}\n{tb}")
        finally:
            # Disconnect session from server
            if self._daemon:
                self._daemon._disconnect_session()
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
        elif action == "update_tokens":
            controller.update_tokens(
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
