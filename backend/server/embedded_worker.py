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
        # Use __builtin__ name so _recalculate_server_pools() excludes us
        _worker_daemon.worker_name = "__builtin__"

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

        logger.info("Embedded worker starting...")

        # Connect session — required for claim_jobs to work
        if not _worker_daemon._connect_session():
            _status = "error"
            _last_error = "Failed to connect worker session"
            logger.error("Embedded worker: session connect failed, aborting")
            return

        logger.info(f"Embedded worker session connected (id={_worker_daemon.session_id})")
        consecutive_empty = 0
        last_heartbeat = time.time()
        heartbeat_interval = 30

        try:
            while not _shutdown_flag:
                try:
                    # Periodic heartbeat
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        _worker_daemon._heartbeat()
                        last_heartbeat = now

                    # 1. Decide mode — check new file_tasks first, fallback to legacy
                    try:
                        from backend.db.sqlite_client import SQLiteDB
                        _qdb = SQLiteDB()
                        _cursor = _qdb.conn.cursor()

                        # New system: find phase with most pending tasks
                        _cursor.execute("""
                            SELECT
                                SUM(CASE WHEN parse_status='done' AND mc_status='pending' THEN 1 ELSE 0 END) AS mc,
                                SUM(CASE WHEN parse_status='done' AND vv_status='pending' THEN 1 ELSE 0 END) AS vv,
                                SUM(CASE WHEN mc_status='done' AND mv_status='pending' THEN 1 ELSE 0 END) AS mv
                            FROM file_tasks ft
                            JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                            WHERE aj.status = 'active'
                        """)
                        row = _cursor.fetchone()
                        mc_p, vv_p, mv_p = (row[0] or 0), (row[1] or 0), (row[2] or 0)

                        if mc_p + vv_p + mv_p > 0:
                            # Pick phase with most pending (= bottleneck)
                            candidates = {"mc": mc_p, "vv": vv_p, "mv": mv_p}
                            mode = max(candidates, key=candidates.get)
                            chunk = 7 if mode == "mc" else 20
                        else:
                            # No pending tasks in new system
                            mode = "idle"
                            chunk = 5
                    except Exception as _mode_err:
                        logger.error(f"Mode decision failed: {_mode_err}")
                        mode = "mc"
                        chunk = 5

                    _worker_daemon.processing_mode = mode

                    if mode == "idle":
                        time.sleep(10)
                        continue

                    # 2. Cap by server config max
                    try:
                        from backend.utils.config import get_config
                        max_chunk = get_config().get("server.auto_processing.batch_size", 20)
                        chunk = min(chunk, max_chunk)
                    except Exception:
                        chunk = min(chunk, 20)
                    _worker_daemon.batch_capacity = chunk

                    # 3. Claim jobs
                    logger.info(f"[EW] Claiming: mode={mode}, chunk={chunk}")
                    jobs = _worker_daemon.claim_jobs_count(chunk)
                    logger.info(f"[EW] Claimed: {len(jobs)} jobs")

                    if not jobs:
                        consecutive_empty += 1
                        wait = min(5 * consecutive_empty, 60)
                        for _ in range(wait):
                            if _shutdown_flag:
                                break
                            time.sleep(1)
                        continue

                    consecutive_empty = 0

                    # Batch processing with dynamically chosen mode
                    try:
                        results = _worker_daemon.process_batch_phased(jobs)
                        # results = [(job_id, success_bool), ...]
                        for item in results:
                            if isinstance(item, tuple) and len(item) >= 2 and item[1]:
                                _jobs_completed += 1
                    except Exception as e:
                        logger.error(f"Embedded worker batch failed: {e}", exc_info=True)

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

                    # Rest after MC batch only (GPU thermal cooldown).
                    # VV/MV are fast — no rest needed.
                    if mode == "mc":
                        try:
                            rest_s = get_config().get("server.auto_processing.rest_after_batch_s", 0)
                            if rest_s > 0:
                                _worker_daemon._current_phase = "resting"
                                _worker_daemon._current_file = f"{rest_s}s"
                                for _ in range(int(rest_s)):
                                    if _shutdown_flag:
                                        break
                                    time.sleep(1)
                                _worker_daemon._current_phase = None
                                _worker_daemon._current_file = None
                        except Exception:
                            pass

                except Exception as e:
                    logger.error(f"Embedded worker loop error: {e}", exc_info=True)
                    _last_error = str(e)
                    time.sleep(5)
        finally:
            # Disconnect session on exit
            try:
                _worker_daemon._disconnect_session()
                logger.info("Embedded worker session disconnected")
            except Exception as e:
                logger.warning(f"Embedded worker disconnect failed: {e}")

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
    """Get embedded worker status with live phase/file tracking."""
    running = _worker_thread is not None and _worker_thread.is_alive()
    result = {
        "running": running,
        "status": _status if running else "idle",
        "jobs_completed": _jobs_completed,
        "last_error": _last_error,
        "current_phase": None,
        "current_file": None,
        "phase_counts": None,
        "batch_throughput": 0.0,
        "batch_capacity": 0,
    }
    if running and _worker_daemon is not None:
        result["current_phase"] = getattr(_worker_daemon, '_current_phase', None)
        result["current_file"] = getattr(_worker_daemon, '_current_file', None)
        result["batch_throughput"] = getattr(_worker_daemon, '_batch_throughput', 0.0)
        result["batch_capacity"] = getattr(_worker_daemon, 'batch_capacity', 0)
        pc = getattr(_worker_daemon, '_phase_counts', None)
        if pc:
            result["phase_counts"] = dict(pc)
    return result
