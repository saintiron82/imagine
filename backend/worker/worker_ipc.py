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
    python -u -m backend.worker.worker_ipc
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

from backend.utils.win32_stdin import make_stdin_reader, STDIN_CLOSED

logger = logging.getLogger("WorkerIPC")

# Thread-safe stdout lock — prevents interleaved JSON lines from main/bg threads
_stdout_lock = threading.Lock()


def _emit(event_data: dict):
    """Send JSON event to stdout (for Electron to read)."""
    try:
        line = json.dumps(event_data, ensure_ascii=False)
        with _stdout_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    except Exception as e:
        # Last resort: write to stderr so Electron can at least see something
        sys.stderr.write(f"[IPC _emit ERROR] {e} | data={event_data}\n")
        sys.stderr.flush()


def _emit_log(message: str, log_type: str = "info"):
    _emit({"event": "log", "message": message, "type": log_type})
    # Also echo to stderr for Electron console visibility
    sys.stderr.write(f"[IPC:{log_type}] {message}\n")
    sys.stderr.flush()


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
        """Start the worker loop in a background thread."""
        if self._running:
            _emit_log("Worker already running", "warning")
            return

        _emit_log(f"[START] server_url={server_url}", "info")
        _emit_log(f"[START] username={'yes' if username else 'no'}, password={'yes' if password else 'no'}", "info")
        _emit_log(f"[START] access_token={'yes(' + access_token[:16] + '...)' if access_token else 'no'}, refresh_token={'yes' if refresh_token else 'no'}", "info")

        # Override config for this session
        import os
        os.environ["IMAGINE_SERVER_URL"] = server_url
        _emit_log(f"[START] Set IMAGINE_SERVER_URL={server_url}", "info")

        # Store auth info for the worker thread
        self._auth_mode = "token" if access_token else "credentials"
        self._access_token = access_token
        self._refresh_token = refresh_token

        # Always set credentials if provided (fallback for token refresh failure)
        if username:
            os.environ["IMAGINE_WORKER_USERNAME"] = username
            _emit_log(f"[START] Set IMAGINE_WORKER_USERNAME={username}", "info")
        if password:
            os.environ["IMAGINE_WORKER_PASSWORD"] = "***set***"
            os.environ["IMAGINE_WORKER_PASSWORD"] = password

        _emit_log(f"[START] Auth mode: {self._auth_mode}", "info")

        self._running = True
        # Reset stop signal from previous session
        if self._daemon:
            self._daemon._stop_requested = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        _emit_status("running")
        _emit_log("Worker thread launched", "success")

    def update_tokens(self, access_token: str, refresh_token: str):
        """Update JWT tokens mid-session (forwarded from browser refresh)."""
        if self._daemon:
            self._daemon.set_tokens(access_token, refresh_token)
            _emit_log("Tokens refreshed from app session", "info")

    def stop(self):
        """Stop the worker loop — also signals daemon to abort current batch."""
        self._running = False
        if self._daemon:
            self._daemon._stop_requested = True
        _emit_status("idle")
        _emit_log("Worker stopped", "info")

    def get_status(self):
        """Report current status."""
        status = "running" if self._running else "idle"
        _emit_status(status)

    def _worker_loop(self):
        """Main worker loop — claim → batch process by phase → repeat."""
        try:
            _emit_log("[THREAD] Worker thread started", "info")

            _emit_log("[IMPORT] Importing WorkerDaemon (numpy=lazy)...", "info")
            from backend.worker.worker_daemon import WorkerDaemon
            from backend.worker.config import get_heartbeat_interval
            _emit_log("[IMPORT] OK — all imports complete", "info")

            _emit_log("[THREAD] Creating WorkerDaemon...", "info")
            daemon = WorkerDaemon()
            self._daemon = daemon
            _emit_log(f"[THREAD] Daemon created, server_url={daemon.server_url}", "info")

            # ── Authentication ──
            import os
            username_env = os.getenv("IMAGINE_WORKER_USERNAME", "")
            password_env = os.getenv("IMAGINE_WORKER_PASSWORD", "")
            _emit_log(f"[AUTH] Env check: IMAGINE_WORKER_USERNAME={'yes' if username_env else 'no'}, IMAGINE_WORKER_PASSWORD={'yes' if password_env else 'no'}", "info")
            _emit_log(f"[AUTH] access_token={'yes' if self._access_token else 'no'}", "info")

            has_creds = bool(username_env and password_env)

            if has_creds:
                _emit_log(f"[AUTH] Mode: independent login (username={username_env})", "info")
                _emit_log(f"[AUTH] Calling daemon.login() -> POST {daemon.server_url}/api/v1/auth/login ...", "info")
                login_ok = daemon.login()
                _emit_log(f"[AUTH] daemon.login() returned: {login_ok}", "info" if login_ok else "error")

                if not login_ok:
                    if self._access_token:
                        _emit_log("[AUTH] Login failed, trying shared token fallback...", "warning")
                        token_ok = daemon.set_tokens(self._access_token, self._refresh_token)
                        _emit_log(f"[AUTH] set_tokens() returned: {token_ok}", "info" if token_ok else "error")
                        if not token_ok:
                            _emit_log("[AUTH] FAILED: both login and token injection failed", "error")
                            _emit_status("error")
                            self._running = False
                            return
                    else:
                        _emit_log("[AUTH] FAILED: login failed, no tokens for fallback", "error")
                        _emit_status("error")
                        self._running = False
                        return

            elif self._access_token:
                at_preview = (self._access_token[:20] + "...") if self._access_token else "(none)"
                _emit_log(f"[AUTH] Mode: shared token (access={at_preview})", "info")
                token_ok = daemon.set_tokens(self._access_token, self._refresh_token)
                _emit_log(f"[AUTH] set_tokens() returned: {token_ok}", "info" if token_ok else "error")
                if not token_ok:
                    _emit_log("[AUTH] FAILED: token injection failed", "error")
                    _emit_status("error")
                    self._running = False
                    return
            else:
                _emit_log("[AUTH] FAILED: no credentials AND no tokens available!", "error")
                _emit_status("error")
                self._running = False
                return

            _emit_log("[AUTH] Authentication successful!", "success")

            # ── Network test: quick server ping ──
            _emit_log(f"[NET] Testing connectivity to {daemon.server_url}...", "info")
            try:
                import requests
                test_resp = daemon.session.get(f"{daemon.server_url}/api/v1/health", timeout=10)
                _emit_log(f"[NET] Health check: {test_resp.status_code} {test_resp.text[:200]}", "info")
            except Exception as net_err:
                _emit_log(f"[NET] Health check FAILED: {net_err}", "error")

            # ── Register worker session ──
            _emit_log("[SESSION] Connecting worker session...", "info")
            daemon._connect_session()
            if daemon.session_id:
                _emit_log(f"[SESSION] Registered (id={daemon.session_id}, mode={daemon.processing_mode})", "success")
                # Notify UI of processing mode so phase pills can be dimmed
                _emit({"event": "processing_mode", "mode": daemon.processing_mode})
            else:
                _emit_log("[SESSION] Warning: session not registered (no session_id)", "warning")

            # ── Main claim loop ──
            poll_interval = 5
            heartbeat_interval = get_heartbeat_interval()
            last_heartbeat = time.time()
            _emit_log(f"[LOOP] Starting job claim loop (poll={poll_interval}s, heartbeat={heartbeat_interval}s)", "info")

            loop_count = 0
            while self._running:
                loop_count += 1

                # Periodic heartbeat
                if time.time() - last_heartbeat >= heartbeat_interval:
                    _emit_log(f"[HEARTBEAT] Sending heartbeat (loop #{loop_count})...", "info")
                    old_mode = daemon.processing_mode
                    hb = daemon._heartbeat()
                    last_heartbeat = time.time()
                    cmd = hb.get("command")
                    _emit_log(f"[HEARTBEAT] Response: {hb}", "info")
                    # Relay processing mode changes from server
                    if daemon.processing_mode != old_mode:
                        _emit({"event": "processing_mode", "mode": daemon.processing_mode})
                    if cmd in ("stop", "block"):
                        _emit_log(f"[HEARTBEAT] Server command: {cmd}", "warning")
                        self._running = False
                        break

                # Claim jobs
                if loop_count <= 3 or loop_count % 10 == 0:
                    _emit_log(f"[CLAIM] Requesting jobs (loop #{loop_count})...", "info")
                jobs = daemon.claim_jobs()

                if not jobs:
                    if loop_count <= 3 or loop_count % 10 == 0:
                        _emit_log(f"[CLAIM] No jobs available, waiting {poll_interval}s...", "info")
                    _emit_status("running", [])
                    time.sleep(poll_interval)
                    continue

                _emit_log(f"[CLAIM] Got {len(jobs)} jobs — batch processing", "info")
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
                    elif event_type == "file_error":
                        _emit_log(f"[ERROR] {data.get('file_name', '?')}: {data.get('error', 'unknown')}", "error")
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
            _emit_log(f"[CRASH] Worker error: {e}", "error")
            _emit_log(f"[CRASH] Traceback: {tb}", "error")
            _emit_status("error")
            logger.error(f"Worker loop crashed: {e}\n{tb}")
        finally:
            # Disconnect session from server
            _emit_log("[SHUTDOWN] Disconnecting session...", "info")
            if self._daemon:
                self._daemon._disconnect_session()
            self._running = False
            _emit_status("idle")
            _emit_log("[SHUTDOWN] Worker loop ended", "info")


def main():
    """IPC main loop — read JSON commands from stdin, dispatch to controller."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Logs go to stderr, events go to stdout
    )

    # Pre-import heavy modules in main thread to avoid DLL loading deadlock
    # on Windows when background threads try to import numpy/torch/etc.
    # (Windows LoadLibrary + Python import lock = hang in background threads)
    sys.stderr.write("[IPC] Pre-loading heavy modules in main thread...\n")
    sys.stderr.flush()
    try:
        import numpy
        sys.stderr.write(f"[IPC] numpy {numpy.__version__} OK\n")
    except Exception as e:
        sys.stderr.write(f"[IPC] numpy failed: {e}\n")
    try:
        import psd_tools
        sys.stderr.write(f"[IPC] psd_tools OK\n")
    except Exception as e:
        sys.stderr.write(f"[IPC] psd_tools failed: {e}\n")
    try:
        from PIL import Image
        sys.stderr.write(f"[IPC] PIL OK\n")
    except Exception as e:
        sys.stderr.write(f"[IPC] PIL failed: {e}\n")
    try:
        import torch
        sys.stderr.write(f"[IPC] torch {torch.__version__} (CUDA={torch.cuda.is_available()}) OK\n")
    except Exception as e:
        sys.stderr.write(f"[IPC] torch failed: {e}\n")
    sys.stderr.write("[IPC] Pre-loading done\n")
    sys.stderr.flush()

    controller = WorkerIPCController()
    _emit({"event": "ready"})

    # Platform-aware stdin reader:
    # - Windows: Win32 PeekNamedPipe (non-blocking, avoids CRT I/O lock deadlock)
    # - Unix: blocking iter(sys.stdin) (safe, no lock contention)
    # See backend/utils/win32_stdin.py for details.
    _read_line = make_stdin_reader()

    while True:
        line = _read_line()

        if line is STDIN_CLOSED:
            controller.stop()
            break
        if line is None:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            _emit_log(f"Invalid JSON: {line}", "error")
            continue

        action = cmd.get("cmd")
        _emit_log(f"[IPC] Received command: {action}", "info")

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
