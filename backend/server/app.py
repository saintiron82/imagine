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
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
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

    # Minimal startup — pools/workers start after login via POST /api/v1/server/activate
    logger.info("Server waiting for login")

    # License manager (runs before login — needed for plan enforcement)
    try:
        from backend.server.deps import get_db
        from backend.server.licensing.license_manager import LicenseManager
        from backend.server.licensing.enforcement import set_license_manager
        db = get_db()
        lm = LicenseManager(db)
        set_license_manager(lm)
        app.state.license_manager = lm
        info = await lm.verify()
        logger.info(f"License: {info.plan_id} / {info.status} (users={info.current_users}/{info.max_users})")

        app.state.license_check_stop = threading.Event()
        def _license_check():
            import asyncio
            stop = app.state.license_check_stop
            while not stop.is_set():
                stop.wait(3600)
                if stop.is_set():
                    break
                try:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(lm.verify())
                    loop.close()
                except Exception as e:
                    logger.warning(f"Periodic license check failed: {e}")
        threading.Thread(target=_license_check, daemon=True, name="license-check").start()
    except Exception as e:
        logger.warning(f"License manager failed: {e}")

    # mDNS (optional)
    try:
        from backend.server.mdns import ImagineServiceAnnouncer
        cfg = get_server_config()
        port = cfg.get("port", 8000)
        app.state.mdns = ImagineServiceAnnouncer(port)
        app.state.mdns.start()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"mDNS failed: {e}")


def _activate_server(app_instance):
    """Initialize all server subsystems. Called once after first login."""
    if getattr(app_instance.state, 'ready', False):
        return  # already activated

    logger.info("Activating server subsystems...")

    _create_default_admin()
    _cleanup_stale_sessions()

    # DownloadAheadPool
    try:
        from backend.server.queue.download_ahead import (
            DownloadAheadPool, register_webdav_source,
        )
        from backend.server.deps import get_db
        db = get_db()
        app_instance.state.download_ahead = DownloadAheadPool(db)
        app_instance.state.download_ahead.start()
        logger.info("DownloadAheadPool started")
        webdav_env = os.environ.get("IMAGINE_WEBDAV_SOURCES")
        if webdav_env:
            import json as _json
            try:
                for src in _json.loads(webdav_env):
                    register_webdav_source(src)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Download pool failed: {e}")

    # FileTaskParsePool
    try:
        from backend.server.queue.file_task_parse_pool import FileTaskParsePool
        dl_pool = getattr(app_instance.state, 'download_ahead', None)
        app_instance.state.file_task_parse = FileTaskParsePool(db, download_pool=dl_pool)
        app_instance.state.file_task_parse.start()
        logger.info("FileTaskParsePool started")
    except Exception as e:
        logger.warning(f"FileTaskParsePool failed: {e}")

    # WorkerScheduler
    try:
        from backend.server.queue.scheduler import WorkerScheduler
        app_instance.state.scheduler = WorkerScheduler(db)
        logger.info("WorkerScheduler started")
    except Exception as e:
        logger.warning(f"WorkerScheduler failed: {e}")

    # Heartbeat watchdog
    try:
        app_instance.state.heartbeat_watchdog = _start_heartbeat_watchdog()
        logger.info("Heartbeat watchdog started")
    except Exception as e:
        logger.warning(f"Heartbeat watchdog failed: {e}")

    app_instance.state.ready = True
    logger.info("Server activated — all subsystems running")

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


@app.on_event("shutdown")
async def shutdown():
    logger.info("Imagine Server shutting down...")
    # Legacy pools removed (Analysis Job System v1)
    # Embedded worker shutdown
    try:
        from backend.server.embedded_worker import get_status as _ew_status, stop_worker as _ew_stop
        if _ew_status()["running"]:
            _ew_stop()
            logger.info("Embedded worker stopped")
    except Exception as e:
        logger.warning(f"Embedded worker shutdown failed: {e}")
    if hasattr(app.state, "heartbeat_watchdog") and app.state.heartbeat_watchdog:
        if hasattr(app.state.heartbeat_watchdog, "_stop_event"):
            app.state.heartbeat_watchdog._stop_event.set()
        logger.info("Heartbeat watchdog stopped")
    if hasattr(app.state, "license_check_stop") and app.state.license_check_stop:
        app.state.license_check_stop.set()
        logger.info("License check stopped")
    if hasattr(app.state, "mdns") and app.state.mdns:
        app.state.mdns.stop()
    close_db()


# ── Routes ───────────────────────────────────────────────────

from backend.server.auth.router import router as auth_router
from backend.server.routers.admin import router as admin_router
from backend.server.routers.stats import router as stats_router
from backend.server.routers.files import router as files_router
from backend.server.routers.search import router as search_router
# Note: pipeline_router removed (legacy /api/v1/jobs/* endpoints)
from backend.server.routers.upload import router as upload_router
from backend.server.routers.worker_setup import router as worker_setup_router
from backend.server.routers.workers import router as workers_router
from backend.server.routers.app_download import router as app_download_router
from backend.server.routers.sync import router as sync_router
from backend.server.routers.classification import router as classification_router
from backend.server.routers.database import router as database_router
from backend.server.routers.server_init import router as server_init_router
from backend.server.routers.license import router as license_router
from backend.server.routers.archive import router as archive_router
from backend.server.routers.analysis import router as analysis_router

app.include_router(auth_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(files_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(worker_setup_router, prefix="/api/v1")
app.include_router(workers_router, prefix="/api/v1")
app.include_router(app_download_router, prefix="/api/v1")
app.include_router(sync_router, prefix="/api/v1")
app.include_router(classification_router, prefix="/api/v1")
app.include_router(database_router, prefix="/api/v1")
app.include_router(server_init_router, prefix="/api/v1")
app.include_router(license_router, prefix="/api/v1")
app.include_router(analysis_router)  # Already has /api/v1 prefix in routes
app.include_router(archive_router, prefix="/api/v1")


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


# ── Tunnel URL registration (Electron → Firestore) ──────────

@app.put("/api/v1/server/tunnel-url")
def set_tunnel_url(body: dict):
    """Store Cloudflare Tunnel URL and update Firestore group record.

    Called by Electron main process when tunnel starts.
    Only accessible from localhost (Electron → embedded server).
    """
    from backend.server.deps import get_db
    from backend.server.firebase_registry import register_group
    import threading

    tunnel_url = body.get("tunnel_url", "")
    db = get_db()

    # Save to system_meta
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO system_meta (key, value) VALUES ('tunnel_url', ?)",
        (tunnel_url,)
    )
    db.conn.commit()

    # Update Firestore (if group_name is configured)
    cursor.execute("SELECT value FROM system_meta WHERE key = 'group_name'")
    row = cursor.fetchone()
    if row and row[0]:
        cfg = get_server_config()
        port = cfg.get("port", 8000)
        threading.Thread(
            target=register_group,
            args=(row[0], port),
            kwargs={"tunnel_url": tunnel_url},
            daemon=True,
        ).start()
        logger.info(f"Tunnel URL registered: {tunnel_url}")

    return {"success": True, "tunnel_url": tunnel_url}


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


def _start_heartbeat_watchdog():
    """Background thread: detect dead workers via heartbeat timeout and reclaim jobs.

    Checks every 60s for online workers whose last_heartbeat is older than 3 minutes.
    (Heartbeat interval is 30s, so 3 minutes = 6 missed heartbeats → likely dead.)
    """
    INTERVAL = 60   # check interval (seconds)
    TIMEOUT = 3     # heartbeat timeout (minutes)

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
