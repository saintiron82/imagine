"""
Embedded worker — runs WorkerDaemon in a background thread inside the FastAPI server.

Allows admin users to start/stop the pipeline worker from the web UI,
without needing a separate terminal or Electron app.
"""

import gc
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────

_worker_thread: threading.Thread = None
_worker_daemon = None
_shutdown_flag = False
_status = "idle"  # idle | running | stopping | error
_last_error = None
_jobs_completed = 0


def start_worker(server_url: str, access_token: str, refresh_token: str = "") -> dict:
    """Start the embedded worker in a background thread."""
    global _worker_thread, _worker_daemon, _shutdown_flag, _status, _last_error, _jobs_completed

    if _worker_thread and _worker_thread.is_alive():
        return {"success": False, "error": "Worker already running"}

    _shutdown_flag = False
    _status = "running"
    _last_error = None
    _jobs_completed = 0

    try:
        from backend.worker.worker_daemon import WorkerDaemon

        _worker_daemon = WorkerDaemon()

        # Override server URL to point to ourselves (loopback)
        _worker_daemon.server_url = server_url
        _worker_daemon.uploader.server_url = server_url

        if not _worker_daemon.set_tokens(access_token, refresh_token):
            _status = "error"
            return {"success": False, "error": "Failed to set auth tokens"}

    except Exception as e:
        _status = "error"
        _last_error = str(e)
        logger.error(f"Failed to initialize embedded worker: {e}")
        return {"success": False, "error": str(e)}

    def _run_loop():
        global _shutdown_flag, _status, _last_error, _jobs_completed

        logger.info("Embedded worker started")
        consecutive_empty = 0

        while not _shutdown_flag:
            try:
                jobs = _worker_daemon.claim_jobs()

                if not jobs:
                    consecutive_empty += 1
                    wait = min(5 * consecutive_empty, 60)
                    for _ in range(wait):
                        if _shutdown_flag:
                            break
                        time.sleep(1)
                    continue

                consecutive_empty = 0

                for job in jobs:
                    if _shutdown_flag:
                        break
                    success = _worker_daemon.process_job(job)
                    if success:
                        _jobs_completed += 1

                # Cleanup GPU memory between batches
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except ImportError:
                    pass

            except Exception as e:
                logger.error(f"Embedded worker loop error: {e}", exc_info=True)
                _last_error = str(e)
                time.sleep(5)

        _status = "idle"
        logger.info(f"Embedded worker stopped (completed {_jobs_completed} jobs)")

    _worker_thread = threading.Thread(target=_run_loop, daemon=True, name="embedded-worker")
    _worker_thread.start()
    logger.info(f"Embedded worker thread started (server={server_url})")
    return {"success": True}


def stop_worker() -> dict:
    """Stop the embedded worker gracefully."""
    global _shutdown_flag, _worker_thread, _status

    if not _worker_thread or not _worker_thread.is_alive():
        _status = "idle"
        return {"success": True, "message": "Worker was not running"}

    _status = "stopping"
    _shutdown_flag = True
    logger.info("Stopping embedded worker...")

    # Wait for thread to finish (with timeout)
    _worker_thread.join(timeout=60)

    if _worker_thread.is_alive():
        logger.warning("Embedded worker did not stop within timeout")
        return {"success": False, "error": "Worker did not stop within 60s"}

    _worker_thread = None
    _status = "idle"
    return {"success": True, "jobs_completed": _jobs_completed}


def get_status() -> dict:
    """Get embedded worker status."""
    running = _worker_thread is not None and _worker_thread.is_alive()
    return {
        "running": running,
        "status": _status if running else "idle",
        "jobs_completed": _jobs_completed,
        "last_error": _last_error,
    }
