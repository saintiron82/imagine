"""
Embedded worker — IO thread + Analysis thread inside the FastAPI server.

Architecture:
  IO Thread:    heartbeat, result upload, next batch prefetch (DB operations)
  Analysis Thread: VLM/SigLIP2/Embedding inference (GPU operations)

Communication via thread-safe queues:
  result_queue:  analysis → IO (completed results for DB upload)
  prefetch_queue: IO → analysis (next batch of tasks)
"""

import gc
import logging
import socket
import threading
import time
from pathlib import Path
from queue import Queue, Empty

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────

_worker_thread: threading.Thread = None  # analysis thread
_io_thread: threading.Thread = None
_worker_daemon = None
_transport = None
_shutdown_flag = False
_status = "idle"
_last_error = None
_jobs_completed = 0
_start_lock = threading.Lock()

# Inter-thread queues
_result_queue: Queue = None      # analysis → IO
_prefetch_queue: Queue = None    # IO → analysis


def start_worker(server_url: str, access_token: str, refresh_token: str = "") -> dict:
    """Start the embedded worker with IO + Analysis threads."""
    global _worker_thread, _io_thread, _worker_daemon, _transport
    global _shutdown_flag, _status, _last_error, _jobs_completed
    global _result_queue, _prefetch_queue

    with _start_lock:
        if _worker_thread and _worker_thread.is_alive():
            return {"success": False, "error": "Worker already running"}
        if _status == "running":
            return {"success": False, "error": "Worker already starting"}

    _shutdown_flag = False
    _status = "running"
    _last_error = None
    _jobs_completed = 0
    _result_queue = Queue()
    _prefetch_queue = Queue(maxsize=1)

    try:
        from backend.server.deps import get_db
        from backend.server.queue.analysis_manager import AnalysisJobManager
        from backend.server.queue.scheduler import WorkerScheduler
        from backend.worker.transport import LocalTransport
        from backend.worker.worker_daemon import WorkerDaemon

        db = get_db()
        manager = AnalysisJobManager(db)
        scheduler = WorkerScheduler(db)

        _transport = LocalTransport(db, scheduler, manager)
        _worker_daemon = WorkerDaemon(transport=_transport)
        _worker_daemon._result_queue = _result_queue  # connect to IO thread

    except Exception as e:
        _status = "error"
        _last_error = str(e)
        logger.error(f"Failed to initialize embedded worker: {e}")
        return {"success": False, "error": str(e)}

    # ── Connect session ──
    try:
        from backend.worker.capability import collect_capability
        resources = collect_capability()
    except Exception:
        resources = {}

    try:
        session = _transport.connect(
            worker_name="embedded",
            hostname=socket.gethostname(),
            batch_capacity=20,
            resources=resources,
        )
        session_id = session.get("session_id")
    except Exception as e:
        _status = "error"
        _last_error = str(e)
        logger.error(f"Embedded worker session failed: {e}")
        return {"success": False, "error": str(e)}

    if not session_id:
        _status = "error"
        _last_error = "No session_id"
        return {"success": False, "error": "Session create failed"}

    logger.info(f"Embedded worker session created (id={session_id})")

    # ── IO Thread ──
    def _io_loop():
        """DB operations: heartbeat, result upload, prefetch."""
        global _jobs_completed
        heartbeat_interval = 5  # faster than before (was 30s)

        while not _shutdown_flag:
            try:
                # 1. Heartbeat — update DB with live metrics
                _transport.heartbeat({
                    "jobs_completed": _jobs_completed,
                    "current_phase": getattr(_worker_daemon, '_current_phase', None),
                    "current_file": getattr(_worker_daemon, '_current_file', None),
                    "phase_throughput": dict(getattr(_worker_daemon, '_phase_throughput', {})),
                    "batch_throughput": getattr(_worker_daemon, '_batch_throughput', 0),
                    "batch_capacity": getattr(_worker_daemon, 'batch_capacity', 0),
                })

                # 2. Upload completed results from queue
                uploaded = 0
                while not _result_queue.empty():
                    try:
                        item = _result_queue.get_nowait()
                        _save_result(item)
                        uploaded += 1
                    except Empty:
                        break
                    except Exception as e:
                        logger.warning(f"[IO] result upload failed: {e}")

                # 3. Prefetch next batch if analysis thread needs it
                if _prefetch_queue.empty():
                    try:
                        result = _transport.claim()
                        if result.get("phase") and result.get("tasks"):
                            _prefetch_queue.put(result)
                    except Exception as e:
                        logger.debug(f"[IO] prefetch claim failed: {e}")

            except Exception as e:
                logger.warning(f"[IO] loop error: {e}")

            # Sleep interval — fast enough for responsive UI
            time.sleep(heartbeat_interval)

    def _save_result(item):
        """Upload a single result to DB via transport."""
        global _jobs_completed
        rtype = item.get("type")
        file_id = item.get("file_id")
        task_id = item.get("task_id")

        # Failure report (no data to save)
        if item.get("success") is False or item.get("fields") is None and rtype == "mc":
            try:
                _transport.report_complete(
                    task_id, rtype, False, item.get("error"),
                )
            except Exception:
                pass
            return

        try:
            success = False
            if rtype == "mc":
                success = _transport.save_vision(file_id, item["fields"])
            elif rtype == "vv":
                success = _transport.save_vv(file_id, item["vector"])
            elif rtype == "mv":
                success = _transport.save_mv(file_id, item["vector"])

            _transport.report_complete(
                task_id, rtype, success,
                elapsed_s=item.get("elapsed_s"),
            )
            if success:
                _jobs_completed += 1
        except Exception as e:
            logger.warning(f"[IO] save {rtype} failed for file {file_id}: {e}")

    # ── Analysis Thread ──
    def _analysis_loop():
        """GPU operations: VLM, SigLIP2, Embedding inference."""
        global _shutdown_flag, _status, _last_error

        logger.info("[Analysis] Thread started")
        consecutive_empty = 0

        try:
            while not _shutdown_flag:
                # Get batch from IO thread's prefetch queue only
                # (IO thread handles all DB access including claim)
                batch = None
                try:
                    batch = _prefetch_queue.get(timeout=10)
                except Empty:
                    pass

                if not batch or not batch.get("tasks"):
                    consecutive_empty += 1
                    # Unload model when idle
                    prev = getattr(_worker_daemon, '_prev_mode', None)
                    if prev:
                        logger.info(f"[Analysis] Queue empty — unloading {prev}")
                        if prev == "mc":
                            _worker_daemon._unload_vlm()
                        elif prev == "vv":
                            _worker_daemon._unload_vv()
                        elif prev == "mv":
                            _worker_daemon._unload_mv()
                        _worker_daemon._prev_mode = None

                    wait = min(5 * consecutive_empty, 60)
                    for _ in range(wait):
                        if _shutdown_flag:
                            break
                        time.sleep(1)
                    continue

                consecutive_empty = 0
                phase = batch["phase"]
                tasks = batch["tasks"]
                _worker_daemon.batch_capacity = len(tasks)
                logger.info(f"[Analysis] Assigned: phase={phase}, count={len(tasks)}")

                # Convert tasks to job format
                jobs = [{
                    "job_id": t["task_id"],
                    "file_id": t["file_id"],
                    "file_path": t["file_path"],
                    "task_id": t["task_id"],
                    "analysis_job_id": t.get("job_id"),
                } for t in tasks]

                # Unload previous model if mode changed
                prev_mode = getattr(_worker_daemon, '_prev_mode', None)
                if prev_mode and prev_mode != phase:
                    logger.info(f"[Analysis] Mode switch: {prev_mode} → {phase}")
                    if prev_mode == "mc":
                        _worker_daemon._unload_vlm()
                    elif prev_mode == "vv":
                        _worker_daemon._unload_vv()
                    elif prev_mode == "mv":
                        _worker_daemon._unload_mv()

                _worker_daemon.processing_mode = phase
                _worker_daemon._prev_mode = phase

                # Process batch (GPU work)
                try:
                    _worker_daemon.process_batch_phased(jobs)
                except Exception as e:
                    logger.error(f"[Analysis] batch failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[Analysis] thread error: {e}", exc_info=True)
            _last_error = str(e)
        finally:
            try:
                _transport.disconnect(session_id)
            except Exception:
                pass
            _status = "idle"
            logger.info(f"[Analysis] Thread stopped (completed {_jobs_completed})")

    # Start both threads
    _io_thread = threading.Thread(target=_io_loop, daemon=True, name="ew-io")
    _worker_thread = threading.Thread(target=_analysis_loop, daemon=True, name="ew-analysis")
    _io_thread.start()
    _worker_thread.start()
    logger.info("Embedded worker started (IO + Analysis threads)")
    return {"success": True}


def stop_worker() -> dict:
    """Stop both threads gracefully."""
    global _shutdown_flag, _worker_thread, _io_thread, _status

    if not _worker_thread or not _worker_thread.is_alive():
        _status = "idle"
        return {"success": True, "message": "Worker was not running"}

    _status = "stopping"
    _shutdown_flag = True
    logger.info("Stopping embedded worker...")

    _worker_thread.join(timeout=60)
    if _io_thread:
        _io_thread.join(timeout=10)

    _worker_thread = None
    _io_thread = None
    _status = "idle"
    return {"success": True, "jobs_completed": _jobs_completed}


def get_status() -> dict:
    """Get live worker status from memory (thread-safe read)."""
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
        "phase_throughput": {},
    }
    if running and _worker_daemon is not None:
        result["current_phase"] = getattr(_worker_daemon, '_current_phase', None)
        result["current_file"] = getattr(_worker_daemon, '_current_file', None)
        result["batch_throughput"] = getattr(_worker_daemon, '_batch_throughput', 0.0)
        result["batch_capacity"] = getattr(_worker_daemon, 'batch_capacity', 0)
        result["phase_throughput"] = dict(getattr(_worker_daemon, '_phase_throughput', {}))
        pc = getattr(_worker_daemon, '_phase_counts', None)
        if pc:
            result["phase_counts"] = dict(pc)
    return result
