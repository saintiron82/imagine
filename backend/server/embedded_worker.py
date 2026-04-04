"""
Embedded worker — runs WorkerDaemon in a background thread inside the FastAPI server.

Uses LocalTransport for direct DB access (no HTTP loopback).
Same WorkerDaemon class as external workers — only transport differs.

No legacy dependency (job_queue, manager.py, __builtin__ hacks).
"""

import gc
import logging
import socket
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────

_worker_thread: threading.Thread = None
_worker_daemon = None
_transport = None
_shutdown_flag = False
_status = "idle"  # idle | running | stopping | error
_last_error = None
_jobs_completed = 0


def start_worker(server_url: str, access_token: str, refresh_token: str = "") -> dict:
    """Start the embedded worker in a background thread.

    Uses LocalTransport (direct DB calls) instead of HTTP loopback.
    The server_url/tokens are kept for compatibility but not used for claim/report.
    """
    global _worker_thread, _worker_daemon, _transport, _shutdown_flag, _status, _last_error, _jobs_completed

    if _worker_thread and _worker_thread.is_alive():
        return {"success": False, "error": "Worker already running"}

    _shutdown_flag = False
    _status = "running"
    _last_error = None
    _jobs_completed = 0

    try:
        from backend.server.deps import get_db
        from backend.server.queue.analysis_manager import AnalysisJobManager
        from backend.server.queue.scheduler import WorkerScheduler
        from backend.worker.transport import LocalTransport
        from backend.worker.worker_daemon import WorkerDaemon

        db = get_db()  # shared singleton — no duplicate migrations
        manager = AnalysisJobManager(db)
        scheduler = WorkerScheduler(db)

        _transport = LocalTransport(db, scheduler, manager)
        _worker_daemon = WorkerDaemon(transport=_transport)

    except Exception as e:
        _status = "error"
        _last_error = str(e)
        logger.error(f"Failed to initialize embedded worker: {e}")
        return {"success": False, "error": str(e)}

    def _run_loop():
        global _shutdown_flag, _status, _last_error, _jobs_completed

        print("[EW] _run_loop started", flush=True)
        logger.info("Embedded worker starting...")

        # Collect hardware capability spec
        try:
            from backend.worker.capability import collect_capability
            resources = collect_capability()
            print(f"[EW] capability collected: gpu={resources.get('gpu_name','?')}", flush=True)
        except Exception as e:
            resources = {}
            print(f"[EW] capability FAILED: {e}", flush=True)

        try:
            session = _transport.connect(
                worker_name="embedded",
                hostname=socket.gethostname(),
                batch_capacity=20,
                resources=resources,
            )
            session_id = session.get("session_id")
            print(f"[EW] session created: id={session_id}", flush=True)
        except Exception as e:
            _status = "error"
            _last_error = str(e)
            print(f"[EW] session create EXCEPTION: {e}", flush=True)
            logger.error(f"Embedded worker: session create exception: {e}")
            return

        if not session_id:
            _status = "error"
            _last_error = "Failed to create worker session"
            print("[EW] session create failed — no session_id", flush=True)
            logger.error("Embedded worker: session create failed")
            return

        print(f"[EW] session OK, entering main loop", flush=True)
        logger.info(f"Embedded worker session created (id={session_id})")
        consecutive_empty = 0
        last_heartbeat = time.time()
        heartbeat_interval = 30

        try:
            while not _shutdown_flag:
                try:
                    # Periodic heartbeat (updates DB metrics)
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        _transport.heartbeat({
                            "jobs_completed": _jobs_completed,
                            "current_phase": getattr(_worker_daemon, '_current_phase', None),
                            "current_file": getattr(_worker_daemon, '_current_file', None),
                        })
                        last_heartbeat = now

                    # Claim tasks — scheduler decides phase + count
                    try:
                        result = _transport.claim()
                    except Exception as ce:
                        print(f"[EW] claim EXCEPTION: {ce}", flush=True)
                        time.sleep(5)
                        continue
                    phase = result.get("phase")
                    tasks = result.get("tasks", [])
                    if phase and tasks:
                        print(f"[EW] claimed: phase={phase} count={len(tasks)}", flush=True)

                    if not phase or not tasks:
                        # No work available
                        # Unload model when idle
                        prev = getattr(_worker_daemon, '_prev_mode', None)
                        if prev:
                            logger.info(f"[EW] Queue empty — unloading {prev} model")
                            if prev == "mc":
                                _worker_daemon._unload_vlm()
                            elif prev == "vv":
                                _worker_daemon._unload_vv()
                            elif prev == "mv":
                                _worker_daemon._unload_mv()
                            _worker_daemon._prev_mode = None

                        consecutive_empty += 1
                        wait = min(5 * consecutive_empty, 60)
                        for _ in range(wait):
                            if _shutdown_flag:
                                break
                            time.sleep(1)
                        continue

                    consecutive_empty = 0
                    logger.info(f"[EW] Assigned: phase={phase}, count={len(tasks)}")

                    # Convert tasks to job format for process_batch_phased
                    jobs = []
                    for t in tasks:
                        jobs.append({
                            "job_id": t["task_id"],
                            "file_id": t["file_id"],
                            "file_path": t["file_path"],
                            "task_id": t["task_id"],
                            "analysis_job_id": t.get("job_id"),
                        })

                    # Unload previous model if mode changed
                    prev_mode = getattr(_worker_daemon, '_prev_mode', None)
                    if prev_mode and prev_mode != phase:
                        logger.info(f"[EW] Mode switch: {prev_mode} → {phase}")
                        if prev_mode == "mc":
                            _worker_daemon._unload_vlm()
                        elif prev_mode == "vv":
                            _worker_daemon._unload_vv()
                        elif prev_mode == "mv":
                            _worker_daemon._unload_mv()

                    _worker_daemon.processing_mode = phase
                    _worker_daemon._prev_mode = phase

                    # Process batch
                    try:
                        results = _worker_daemon.process_batch_phased(jobs)
                        for item in results:
                            if isinstance(item, tuple) and len(item) >= 2 and item[1]:
                                _jobs_completed += 1
                    except Exception as e:
                        logger.error(f"Embedded worker batch failed: {e}", exc_info=True)

                except Exception as e:
                    logger.error(f"Embedded worker loop error: {e}", exc_info=True)
                    _last_error = str(e)
                    time.sleep(5)
        finally:
            try:
                _transport.disconnect(session_id)
                logger.info("Embedded worker session disconnected")
            except Exception as e:
                logger.warning(f"Embedded worker disconnect failed: {e}")

        _status = "idle"
        logger.info(f"Embedded worker stopped (completed {_jobs_completed} jobs)")

    _worker_thread = threading.Thread(target=_run_loop, daemon=True, name="embedded-worker")
    _worker_thread.start()
    logger.info("Embedded worker thread started (LocalTransport)")
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
