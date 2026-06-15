"""
Imagine Server — FastAPI application entry point.

Usage:
    uvicorn backend.server.app:app --host 0.0.0.0 --port 8000 --reload

Or via CLI:
    python -m backend.server.app
"""

import logging
import os
import sys
import threading
from pathlib import Path

# Suppress HuggingFace/transformers progress bars (floods stderr)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import traceback as _traceback

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.server.config import get_cors_origins, get_server_config
from backend.server.deps import close_db, get_current_user

# ── Logging ──────────────────────────────────────────────────
# JSON format when piped (Electron), plain text when terminal (dev)
import os as _os
if not _os.isatty(2):
    from backend.utils.json_log_formatter import setup_json_logging
    setup_json_logging()
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)

# Capture recent log records in a ring buffer for the admin live-log view (IMGV2-26).
from backend.server import log_buffer  # noqa: E402
log_buffer.install()

# ── App ──────────────────────────────────────────────────────

app = FastAPI(
    title="Imagine Server",
    description="Image analysis & search server with distributed processing",
    version="4.0.0",
)

# CORS
_cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=("*" not in _cors_origins),  # wildcard + credentials is CORS spec violation
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler: log full traceback for 500s ────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log full traceback for any unhandled exception (500 errors)."""
    tb = _traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}:\n"
        f"{''.join(tb)}"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Lifecycle ────────────────────────────────────────────────

# Server readiness flag — False until all initialization is complete
app.state.ready = False

@app.on_event("startup")
async def startup():
    logger.info("Imagine Server starting up...")

    # Start parent watchdog — auto-exit when Electron (parent) dies unexpectedly.
    # Without this, the server process would remain orphaned and hold the port.
    try:
        from backend.utils.parent_watchdog import start_parent_watchdog
        start_parent_watchdog()
    except Exception as e:
        logger.warning(f"Parent watchdog failed to start: {e}")

    # Nothing touches DB at startup — ALL DB work happens in /server/activate after login
    logger.info("Server waiting for login (no DB access until activate)")


def _activate_server(app_instance):
    """Initialize all server subsystems. Called once after first login.

    Order matters — DB operations complete BEFORE background threads start:
    1. DB setup (admin, sessions, migrations)
    2. AnalysisJobManager init (reclaim, sync — heavy DB writes)
    3. Background pools (download, parse — start AFTER DB is stable)
    4. Scheduler + watchdog
    """
    if getattr(app_instance.state, 'ready', False):
        return  # already activated

    import time as _time
    logger.info("Activating server subsystems...")
    from backend.server.deps import get_db

    def _log(msg):
        logger.info(msg)
        print(f"[activate] {msg}", flush=True)

    t0 = _time.perf_counter()
    db = get_db()
    _log(f"get_db: {_time.perf_counter()-t0:.2f}s")

    # Phase 1: DB setup (sequential, no threads)
    t1 = _time.perf_counter()
    _create_default_admin()
    _log(f"create_default_admin: {_time.perf_counter()-t1:.2f}s")

    t1 = _time.perf_counter()
    _cleanup_stale_sessions()
    _log(f"cleanup_stale_sessions: {_time.perf_counter()-t1:.2f}s")

    # Phase 2: AnalysisJobManager init — heavy DB writes (reclaim + sync)
    try:
        from backend.server.queue.analysis_manager import AnalysisJobManager
        t1 = _time.perf_counter()
        _mgr = AnalysisJobManager(db)
        _log(f"AnalysisJobManager init: {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"AnalysisJobManager init FAILED: {e}")

    # Phase 2b: Startup health check — detect and auto-repair broken states
    t1 = _time.perf_counter()
    try:
        _startup_health_check(db, _log)
        _log(f"Phase 2b: Startup health check {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 2b: Health check failed (non-fatal): {e}")

    # Phase 3: Background pools
    t1 = _time.perf_counter()
    try:
        from backend.server.queue.download_ahead import (
            DownloadAheadPool, register_webdav_source, load_persisted_sources,
        )
        app_instance.state.download_ahead = DownloadAheadPool(db)
        app_instance.state.download_ahead.start()
        webdav_env = os.environ.get("IMAGINE_WEBDAV_SOURCES")
        n_sources = 0
        if webdav_env:
            import json as _json
            try:
                for src in _json.loads(webdav_env):
                    register_webdav_source(src)
                    n_sources += 1
            except Exception:
                pass
        # Runtime-registered sources (added via POST /api/v1/sources) survive restart.
        n_sources += load_persisted_sources()
        _log(f"Phase 3a: DownloadAheadPool started ({n_sources} WebDAV sources) {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 3a FAILED: DownloadAheadPool: {e}")

    t1 = _time.perf_counter()
    try:
        from backend.server.queue.file_task_parse_pool import FileTaskParsePool
        dl_pool = getattr(app_instance.state, 'download_ahead', None)
        app_instance.state.file_task_parse = FileTaskParsePool(db, download_pool=dl_pool)
        app_instance.state.file_task_parse.start()
        _log(f"Phase 3b: FileTaskParsePool started {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 3b FAILED: FileTaskParsePool: {e}")

    # Phase 4: Scheduler + watchdog
    t1 = _time.perf_counter()
    try:
        from backend.server.queue.scheduler import WorkerScheduler
        app_instance.state.scheduler = WorkerScheduler(db)
        _log(f"Phase 4a: WorkerScheduler started {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 4a FAILED: WorkerScheduler: {e}")

    t1 = _time.perf_counter()
    try:
        app_instance.state.heartbeat_watchdog = _start_heartbeat_watchdog()
        _log(f"Phase 4b: Heartbeat watchdog started {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 4b FAILED: Heartbeat watchdog: {e}")

    # Phase 5: License
    t1 = _time.perf_counter()
    try:
        from backend.server.licensing.license_manager import LicenseManager
        from backend.server.licensing.enforcement import set_license_manager
        lm = LicenseManager(db)
        set_license_manager(lm)
        app_instance.state.license_manager = lm
        _log(f"Phase 5: License manager loaded {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 5: License failed: {e}")

    # Phase 6: Embedded worker (if active work exists)
    t1 = _time.perf_counter()
    try:
        from backend.server.queue.analysis_manager import AnalysisJobManager
        mgr = AnalysisJobManager(db)
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM file_tasks ft
            JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
            WHERE aj.status = 'active'
              AND (mc_status = 'pending' OR vv_status = 'pending' OR mv_status = 'pending')
        """)
        pending = cursor.fetchone()[0]
        if pending > 0:
            from backend.server.local_worker import start_worker
            result = start_worker()
            _log(f"Phase 6: Local worker started ({pending} pending tasks) {_time.perf_counter()-t1:.2f}s")
        else:
            _log(f"Phase 6: No pending tasks — worker standby {_time.perf_counter()-t1:.2f}s")
    except Exception as e:
        _log(f"Phase 6 FAILED: Embedded worker: {e}")

    total = _time.perf_counter() - t0
    app_instance.state.ready = True
    _log(f"=== Server activated ({total:.1f}s total) ===")

    # Firebase re-registration on restart (best-effort, non-blocking)
    try:
        from backend.server.deps import get_db
        db = get_db()
        cur = db.conn.execute("SELECT value FROM system_meta WHERE key='group_name'")
        row = cur.fetchone()
        if row:
            from backend.server.firebase_registry import register_group
            cfg = get_server_config()
            port = cfg.get("port", 8000)
            threading.Thread(target=register_group, args=(row[0], port), daemon=True).start()
            logger.info(f"Firebase re-registration queued for '{row[0]}'")
    except Exception as e:
        logger.warning(f"Firebase re-registration skipped: {e}")

    # Phase 5: relay connector (only when explicitly configured)
    try:
        from backend.server import relay_client as _relay
        _client = _relay.maybe_start(app=app_instance, db=db)
        if _client is not None:
            logger.info("Relay connector started (endpoint=%s)", _client.endpoint)
        else:
            logger.debug("Relay connector not configured — skipping")
    except Exception as e:
        logger.warning(f"Relay connector failed to start: {e}")

    # Sprint 2 γ4: auto-tag files repeatedly flagged 'irrelevant'.
    try:
        from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags
        n_tagged = apply_feedback_to_user_tags(db, threshold=3)
        logger.info(f"auto_user_tags applied: {n_tagged} file(s) tagged")
    except Exception as e:
        logger.warning(f"auto_user_tags failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Imagine Server shutting down...")
    # Legacy pools removed (Analysis Job System v1)
    # Local worker shutdown
    try:
        from backend.server.local_worker import get_status as _lw_status, stop_worker as _lw_stop
        if _lw_status()["running"]:
            _lw_stop()
            logger.info("Local worker stopped")
    except Exception as e:
        logger.warning(f"Local worker shutdown failed: {e}")
    if hasattr(app.state, "heartbeat_watchdog") and app.state.heartbeat_watchdog:
        if hasattr(app.state.heartbeat_watchdog, "_stop_event"):
            app.state.heartbeat_watchdog._stop_event.set()
        logger.info("Heartbeat watchdog stopped")
    if hasattr(app.state, "license_check_stop") and app.state.license_check_stop:
        app.state.license_check_stop.set()
        logger.info("License check stopped")
    if hasattr(app.state, "mdns") and app.state.mdns:
        app.state.mdns.stop()
    try:
        from backend.server import relay_client as _relay
        _relay.stop_singleton()
    except Exception as e:
        logger.warning(f"Relay stop failed (non-critical): {e}")
    close_db()


# ── Routes ───────────────────────────────────────────────────

from backend.server.auth.router import router as auth_router
from backend.server.routers.admin import router as admin_router
from backend.server.routers.stats import router as stats_router
from backend.server.routers.files import router as files_router
from backend.server.routers.search import router as search_router
# Note: pipeline_router removed (legacy /api/v1/jobs/* endpoints)
from backend.server.routers.upload import router as upload_router
from backend.server.routers.workers import router as workers_router
from backend.server.routers.app_download import router as app_download_router
from backend.server.routers.sync import router as sync_router
from backend.server.routers.classification import router as classification_router
from backend.server.routers.database import router as database_router
from backend.server.routers.server_init import router as server_init_router
from backend.server.routers.license import router as license_router
from backend.server.routers.archive import router as archive_router
from backend.server.routers.analysis import router as analysis_router
from backend.server.routers.tools import router as tools_router
from backend.server.routers.connection_info import router as connection_info_router
from backend.server.routers.search_feedback import router as search_feedback_router
from backend.server.routers.feedback_dashboard import router as feedback_dashboard_router
from backend.server.routers.browse import router as browse_router
from backend.server.routers.members import router as members_router
from backend.server.routers.logs import router as logs_router

app.include_router(auth_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(files_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(workers_router, prefix="/api/v1")
app.include_router(app_download_router, prefix="/api/v1")
app.include_router(sync_router, prefix="/api/v1")
app.include_router(classification_router, prefix="/api/v1")
app.include_router(database_router, prefix="/api/v1")
app.include_router(server_init_router, prefix="/api/v1")
app.include_router(license_router, prefix="/api/v1")
app.include_router(analysis_router)  # Already has /api/v1 prefix in routes
app.include_router(archive_router, prefix="/api/v1")
app.include_router(tools_router, prefix="/api/v1")
app.include_router(connection_info_router, prefix="/api/v1")
app.include_router(search_feedback_router, prefix="/api/v1")
app.include_router(feedback_dashboard_router, prefix="/api/v1")
app.include_router(browse_router, prefix="/api/v1")
app.include_router(members_router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")


@app.post("/api/v1/server/activate")
def activate_server(
    request: Request,
    _user: dict = Depends(get_current_user),
):
    """Activate server subsystems after first login.

    Called once — initializes DB, pools, scheduler, watchdog.
    Subsequent calls are no-ops (already activated).
    """
    _activate_server(request.app)
    return {
        "success": True,
        "ready": getattr(request.app.state, 'ready', False),
    }


@app.get("/api/v1/health")
def health():
    """Health check endpoint with Firebase/CORS status for debugging."""
    import socket
    from backend.server.firebase_auth import is_firebase_available
    return {
        "status": "ok",
        "ready": getattr(app.state, 'ready', False),
        "version": "4.0.0",
        "server_name": socket.gethostname(),
        "firebase_auth": is_firebase_available(),
        "cors": "allow_all" if "*" in _cors_origins else "restricted",
    }


# ── Default admin account ────────────────────────────────────

def _create_default_admin():
    """No-op: admin is now created via POST /api/v1/server/init."""
    pass


def _cleanup_stale_sessions():
    """Reset stale worker sessions on startup (no legacy dependency)."""
    try:
        from backend.server.deps import get_db
        db = get_db()
        cursor = db.conn.cursor()
        cursor.execute(
            """UPDATE worker_sessions
               SET status = 'offline',
                   current_phase = NULL,
                   current_file = NULL,
                   current_job_id = NULL,
                   jobs_completed = 0,
                   jobs_failed = 0,
                   resources_json = NULL
               WHERE status = 'online'"""
        )
        if cursor.rowcount > 0:
            logger.info(f"Startup cleanup: marked {cursor.rowcount} stale worker sessions offline")
        db.conn.commit()

        # AnalysisJobManager handles file_tasks reclaim on __init__
    except Exception as e:
        logger.warning(f"Startup session cleanup failed: {e}")


def _startup_health_check(db, _log):
    """Detect and auto-repair broken task states on server start.

    Runs synchronously before background pools start.
    Reports all findings to server log for operator visibility.
    """
    cursor = db.conn.cursor()
    repairs = []

    # 1. Thumbnail-missing files: parse done but no thumbnail → reset to re-parse
    cursor.execute("""
        SELECT ft.id, ft.file_id, f.file_name
        FROM file_tasks ft
        JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
        JOIN files f ON ft.file_id = f.id
        WHERE aj.status = 'active'
          AND ft.parse_status = 'done'
          AND (f.thumbnail_url IS NULL OR f.thumbnail_url = '')
    """)
    thumb_missing = cursor.fetchall()
    if thumb_missing:
        task_ids = [r[0] for r in thumb_missing]
        placeholders = ",".join("?" * len(task_ids))
        cursor.execute(f"""
            UPDATE file_tasks SET
                parse_status = 'pending',
                parse_assigned_to = NULL,
                parse_started_at = NULL,
                parse_completed_at = NULL,
                mc_status = 'pending',
                mc_assigned_to = NULL,
                mc_started_at = NULL,
                mc_completed_at = NULL,
                vv_status = 'pending',
                vv_assigned_to = NULL,
                vv_started_at = NULL,
                vv_completed_at = NULL,
                mv_status = 'pending',
                mv_assigned_to = NULL,
                mv_started_at = NULL,
                mv_completed_at = NULL,
                error_message = NULL,
                updated_at = datetime('now')
            WHERE id IN ({placeholders})
        """, task_ids)
        repairs.append(f"thumbnail missing → re-parse: {len(thumb_missing)}")

    # 2. WebDAV files marked download=done but parse failed with "not yet downloaded"
    #    → temp file was lost on restart, need to re-download
    cursor.execute("""
        UPDATE file_tasks SET
            download_status = 'pending',
            parse_status = 'pending',
            parse_assigned_to = NULL,
            parse_started_at = NULL,
            parse_completed_at = NULL,
            retry_count = 0,
            max_retries = 3,
            error_message = NULL,
            updated_at = datetime('now')
        WHERE parse_status = 'failed'
          AND download_status = 'done'
          AND file_path LIKE 'webdav://%'
          AND error_message LIKE '%not yet downloaded%'
          AND analysis_job_id IN (SELECT id FROM analysis_jobs WHERE status = 'active')
    """)
    if cursor.rowcount > 0:
        repairs.append(f"WebDAV temp lost → re-download: {cursor.rowcount}")

    # 3. Stale assigned tasks: assigned but worker is offline → reset to pending
    for phase in ("mc", "vv", "mv"):
        status_col = f"{phase}_status"
        assigned_col = f"{phase}_assigned_to"
        cursor.execute(f"""
            UPDATE file_tasks SET
                {status_col} = 'pending',
                {assigned_col} = NULL,
                error_message = NULL,
                updated_at = datetime('now')
            WHERE {status_col} = 'assigned'
              AND analysis_job_id IN (SELECT id FROM analysis_jobs WHERE status = 'active')
        """)
        if cursor.rowcount > 0:
            repairs.append(f"{phase} stale assigned → pending: {cursor.rowcount}")

    # 3. Failed tasks eligible for auto-retry
    from backend.server.queue.analysis_manager import AnalysisJobManager
    mgr = AnalysisJobManager(db)
    retried = mgr.retry_failed_auto(cooldown_minutes=0)  # no cooldown on startup
    if retried > 0:
        repairs.append(f"failed → auto-retry: {retried}")

    db.conn.commit()

    # 4. Auto-complete active jobs that are actually finished
    cursor.execute("""
        SELECT aj.id, aj.name FROM analysis_jobs aj
        WHERE aj.status = 'active'
          AND NOT EXISTS (
              SELECT 1 FROM file_tasks ft
              WHERE ft.analysis_job_id = aj.id
                AND ft.dismissed_at IS NULL
                AND NOT (ft.mc_status = 'done' AND ft.vv_status = 'done' AND ft.mv_status = 'done')
                AND ft.mc_status != 'failed' AND ft.vv_status != 'failed' AND ft.mv_status != 'failed'
          )
    """)
    completable = cursor.fetchall()
    if completable:
        for job_id, job_name in completable:
            cursor.execute(
                "UPDATE analysis_jobs SET status = 'completed', completed_at = datetime('now') WHERE id = ?",
                (job_id,),
            )
        db.conn.commit()
        repairs.append(f"active → completed: {len(completable)} jobs ({', '.join(n for _, n in completable)})")

    # 5. Report summary
    if repairs:
        for r in repairs:
            _log(f"Health check: {r}")
    else:
        _log("Health check: all clean")

    # 5. Stats for operator
    cursor.execute("""
        SELECT
            SUM(CASE WHEN mc_status='pending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN vv_status='pending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN mv_status='pending' THEN 1 ELSE 0 END),
            SUM(CASE WHEN mc_status='failed' THEN 1 ELSE 0 END),
            SUM(CASE WHEN vv_status='failed' THEN 1 ELSE 0 END),
            SUM(CASE WHEN mv_status='failed' THEN 1 ELSE 0 END),
            COUNT(*)
        FROM file_tasks ft
        JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
        WHERE aj.status = 'active'
    """)
    row = cursor.fetchone()
    if row and row[6] > 0:
        _log(
            f"Task queue: MC {row[0]}p/{row[3]}f, "
            f"VV {row[1]}p/{row[4]}f, "
            f"MV {row[2]}p/{row[5]}f "
            f"(total {row[6]} tasks)"
        )


def _start_heartbeat_watchdog():
    """Background thread: detect dead workers via heartbeat timeout and reclaim jobs.

    Checks every 60s for online workers whose last_heartbeat is older than 3 minutes.
    (Heartbeat interval is 30s, so 3 minutes = 6 missed heartbeats → likely dead.)
    """
    INTERVAL = 60   # check interval (seconds)
    TIMEOUT = 15    # heartbeat timeout (minutes) — MC batch can take 10min

    _stop_event = threading.Event()

    def _check():
        while not _stop_event.is_set():
            _stop_event.wait(INTERVAL)
            if _stop_event.is_set():
                break
            try:
                from datetime import datetime as _dt
                from backend.server.deps import get_db
                from backend.server.queue.analysis_manager import AnalysisJobManager

                db = get_db()
                cursor = db.conn.cursor()
                now = _dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")

                cursor.execute(
                    """SELECT id, worker_name FROM worker_sessions
                       WHERE status = 'online'
                         AND last_heartbeat IS NOT NULL
                         AND datetime(last_heartbeat, '+' || ? || ' minutes') < datetime('now')""",
                    (TIMEOUT,)
                )
                stale_sessions = cursor.fetchall()

                if not stale_sessions:
                    continue

                mgr = AnalysisJobManager(db)
                total_reclaimed = 0
                for session_id, worker_name in stale_sessions:
                    reclaimed = mgr.reclaim_worker_tasks(session_id)
                    total_reclaimed += reclaimed
                    cursor.execute(
                        "UPDATE worker_sessions SET status = 'offline', disconnected_at = ? WHERE id = ?",
                        (now, session_id)
                    )
                    logger.warning(
                        f"Heartbeat timeout: worker '{worker_name}' (session={session_id}) "
                        f"marked offline, reclaimed {reclaimed} tasks"
                    )

                db.conn.commit()

                # Auto-retry failed AI-phase tasks (5 min cooldown, max 3 retries)
                try:
                    retried = mgr.retry_failed_auto(cooldown_minutes=5)
                    if retried:
                        logger.info(f"Watchdog: auto-retried {retried} failed tasks")
                except Exception as e2:
                    logger.warning(f"Watchdog auto-retry error: {e2}")

            except Exception as e:
                logger.error(f"Heartbeat watchdog error: {e}")
                try:
                    db.conn.rollback()
                except Exception:
                    pass

    t = threading.Thread(target=_check, daemon=True, name="heartbeat-watchdog")
    t.start()
    t._stop_event = _stop_event  # attach for clean shutdown
    return t


# ── SPA Static Serving (React frontend) ─────────────────────

DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

if DIST_DIR.exists():
    # Serve static assets (JS, CSS, images, fonts)
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="static-assets")

    # SPA fallback: non-API routes → index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve React SPA — static files or index.html fallback."""
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(DIST_DIR / "index.html"))


# ── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    cfg = get_server_config()
    host = cfg.get("host", "0.0.0.0")
    port = cfg.get("port", 8000)
    workers = cfg.get("workers", 4)

    logger.info(f"Starting Imagine Server on {host}:{port}")
    uvicorn.run(
        "backend.server.app:app",
        host=host,
        port=port,
        workers=1,  # SQLite requires single worker (single writer)
        reload=True,
    )
