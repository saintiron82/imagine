"""
Embedded worker — thin wrapper around WorkerDaemon with LocalTransport.

Same WorkerDaemon code as external workers. Only transport differs:
  Embedded: LocalTransport (direct DB)
  External: HttpTransport (HTTP API)
"""

import logging
import socket
import threading

logger = logging.getLogger(__name__)

_daemon = None
_start_lock = threading.Lock()


def start_worker(server_url: str = "", access_token: str = "", refresh_token: str = "") -> dict:
    """Start embedded worker using LocalTransport."""
    global _daemon

    if not _start_lock.acquire(blocking=False):
        return {"success": False, "error": "Start in progress"}

    try:
        if _daemon and _daemon.is_running():
            _start_lock.release()
            return {"success": False, "error": "Worker already running"}
    except:
        pass

    try:
        from backend.server.deps import get_db
        from backend.server.queue.analysis_manager import AnalysisJobManager
        from backend.server.queue.scheduler import WorkerScheduler
        from backend.worker.transport import LocalTransport
        from backend.worker.worker_daemon import WorkerDaemon

        db = get_db()
        manager = AnalysisJobManager(db)
        scheduler = WorkerScheduler(db)
        transport = LocalTransport(db, scheduler, manager)

        _daemon = WorkerDaemon(
            transport=transport,
            origin="server-local",
            launcher="server",
        )

        # Connect session
        from backend.worker.capability import collect_capability
        resources = collect_capability()

        session = transport.connect(
            worker_name="embedded",
            hostname=socket.gethostname(),
            batch_capacity=20,
            resources=resources,
            origin="server-local",
            launcher="server",
        )
        if not session.get("session_id"):
            return {"success": False, "error": "Session create failed"}

        _daemon.session_id = session["session_id"]
        _daemon.start()
        logger.info(f"Embedded worker started (session={session['session_id']})")
        return {"success": True}

    except Exception as e:
        logger.error(f"Embedded worker start failed: {e}")
        return {"success": False, "error": str(e)}
    finally:
        _start_lock.release()


def stop_worker() -> dict:
    """Stop embedded worker."""
    global _daemon
    if not _daemon or not _daemon.is_running():
        return {"success": True, "message": "Not running"}

    _daemon.stop()
    result = {"success": True, "jobs_completed": _daemon._total_completed}
    _daemon = None
    return result


def get_status() -> dict:
    """Get live worker status from daemon memory."""
    if not _daemon:
        return {"running": False, "status": "idle", "jobs_completed": 0,
                "last_error": None, "current_phase": None, "current_file": None,
                "phase_counts": None, "batch_throughput": 0.0, "batch_capacity": 0,
                "phase_throughput": {}}

    running = _daemon.is_running()
    result = {
        "running": running,
        "status": "running" if running else "idle",
        "jobs_completed": _daemon._total_completed,
        "last_error": None,
        "current_phase": _daemon._current_phase,
        "current_file": _daemon._current_file,
        "batch_throughput": _daemon._batch_throughput,
        "batch_capacity": _daemon.batch_capacity,
        "phase_throughput": dict(_daemon._phase_throughput),
    }
    pc = _daemon._phase_counts
    if pc:
        result["phase_counts"] = dict(pc)
    return result
