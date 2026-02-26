"""
Imagine Server — FastAPI application entry point.

Usage:
    uvicorn backend.server.app:app --host 0.0.0.0 --port 8000 --reload

Or via CLI:
    python -m backend.server.app
"""

import logging
import sys
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.server.config import get_cors_origins, get_server_config
from backend.server.deps import close_db

# ── Logging ──────────────────────────────────────────────────
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


# ── Lifecycle ────────────────────────────────────────────────

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

    # DB will be lazily initialized on first request via get_db()
    _create_default_admin()
    _cleanup_stale_jobs()

    # Parse-ahead pool: pre-parse pending jobs in background (server-side Phase P)
    # In mc_only mode, ParseAhead also handles Phase VV (SigLIP2)
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        pa_enabled = cfg.get("server.parse_ahead.enabled", True)
        if pa_enabled:
            from backend.server.queue.parse_ahead import ParseAheadPool
            from backend.server.deps import get_db
            db = get_db()
            app.state.parse_ahead = ParseAheadPool(db)
            app.state.parse_ahead.start()
            logger.info("Parse-ahead pool started")
        else:
            logger.info("Parse-ahead pool disabled via config")
    except Exception as e:
        logger.warning(f"Parse-ahead pool failed to start: {e}")

    # Embed-ahead pool: mc_only mode — server-side Phase MV after workers upload MC
    try:
        from backend.server.queue.manager import get_processing_mode
        processing_mode = get_processing_mode()
        logger.info(f"Processing mode: {processing_mode}")
        if processing_mode == "mc_only":
            from backend.server.queue.embed_ahead import EmbedAheadPool
            from backend.server.deps import get_db
            db = get_db()
            app.state.embed_ahead = EmbedAheadPool(db)
            app.state.embed_ahead.start()
            logger.info("Embed-ahead pool started (mc_only mode)")
        else:
            logger.info("Embed-ahead pool skipped (processing_mode='full')")
    except Exception as e:
        logger.warning(f"Embed-ahead pool failed to start: {e}")

    # Determine initial processing mode (auto if no workers online + auto_processing enabled)
    try:
        from backend.server.routers.workers import _recalculate_server_pools
        from backend.server.deps import get_db
        db = get_db()
        _recalculate_server_pools(app, db)
        logger.info(f"Initial processing mode: {getattr(app.state.parse_ahead, '_processing_mode', 'unknown') if hasattr(app.state, 'parse_ahead') and app.state.parse_ahead else 'N/A'}")
    except Exception as e:
        logger.warning(f"Initial pool recalculation failed: {e}")

    # Heartbeat watchdog: periodically detect dead workers and reclaim their jobs
    try:
        app.state.heartbeat_watchdog = _start_heartbeat_watchdog()
        logger.info("Heartbeat watchdog started (60s interval, 3min timeout)")
    except Exception as e:
        logger.warning(f"Heartbeat watchdog failed to start: {e}")

    # mDNS service registration (optional — requires zeroconf)
    try:
        from backend.server.mdns import ImagineServiceAnnouncer
        cfg = get_server_config()
        port = cfg.get("port", 8000)
        app.state.mdns = ImagineServiceAnnouncer(port)
        app.state.mdns.start()
    except ImportError:
        logger.info("zeroconf not installed, mDNS discovery disabled")
    except Exception as e:
        logger.warning(f"mDNS registration failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Imagine Server shutting down...")
    if hasattr(app.state, "parse_ahead") and app.state.parse_ahead:
        app.state.parse_ahead.stop()
        logger.info("Parse-ahead pool stopped")
    if hasattr(app.state, "embed_ahead") and app.state.embed_ahead:
        app.state.embed_ahead.stop()
        logger.info("Embed-ahead pool stopped")
    if hasattr(app.state, "heartbeat_watchdog") and app.state.heartbeat_watchdog:
        if hasattr(app.state.heartbeat_watchdog, "_stop_event"):
            app.state.heartbeat_watchdog._stop_event.set()
        logger.info("Heartbeat watchdog stopped")
    if hasattr(app.state, "mdns") and app.state.mdns:
        app.state.mdns.stop()
    close_db()


# ── Routes ───────────────────────────────────────────────────

from backend.server.auth.router import router as auth_router
from backend.server.routers.admin import router as admin_router
from backend.server.routers.stats import router as stats_router
from backend.server.routers.files import router as files_router
from backend.server.routers.search import router as search_router
from backend.server.routers.pipeline import router as pipeline_router
from backend.server.routers.upload import router as upload_router
from backend.server.routers.worker_setup import router as worker_setup_router
from backend.server.routers.workers import router as workers_router
from backend.server.routers.app_download import router as app_download_router
from backend.server.routers.sync import router as sync_router
from backend.server.routers.classification import router as classification_router
from backend.server.routers.database import router as database_router

app.include_router(auth_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(stats_router, prefix="/api/v1")
app.include_router(files_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(pipeline_router)  # Already has /api/v1 prefix in routes
app.include_router(upload_router, prefix="/api/v1")
app.include_router(worker_setup_router, prefix="/api/v1")
app.include_router(workers_router, prefix="/api/v1")
app.include_router(app_download_router, prefix="/api/v1")
app.include_router(sync_router, prefix="/api/v1")
app.include_router(classification_router, prefix="/api/v1")
app.include_router(database_router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    """Health check endpoint."""
    import socket
    return {
        "status": "ok",
        "version": "4.0.0",
        "server_name": socket.gethostname(),
    }


# ── Default admin account ────────────────────────────────────

def _create_default_admin():
    """Create default admin account if no users exist (first startup)."""
    try:
        from backend.server.deps import get_db
        db = get_db()
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        if count == 0:
            import bcrypt
            password_hash = bcrypt.hashpw("admin".encode(), bcrypt.gensalt()).decode()
            cursor.execute(
                """INSERT INTO users (username, email, password_hash, role, is_active)
                   VALUES (?, ?, ?, 'admin', 1)""",
                ("admin", "admin@localhost", password_hash)
            )
            db.conn.commit()
            logger.info("Created default admin account (admin / admin)")
    except Exception as e:
        logger.warning(f"Could not create default admin: {e}")


def _cleanup_stale_jobs():
    """Reset stale assigned/processing jobs and offline worker sessions on startup."""
    try:
        from backend.server.deps import get_db
        from backend.server.queue.manager import JobQueueManager
        db = get_db()
        queue = JobQueueManager(db)
        cfg = get_server_config()
        timeout = cfg.get("queue", {}).get("assignment_timeout_minutes", 30)
        count = queue.reassign_stale_jobs(timeout)
        if count > 0:
            logger.info(f"Startup cleanup: reset {count} stale jobs to pending")

        # Mark all online worker sessions as offline (stale from previous run)
        cursor = db.conn.cursor()
        cursor.execute(
            """UPDATE worker_sessions SET status = 'offline'
               WHERE status = 'online'"""
        )
        if cursor.rowcount > 0:
            logger.info(f"Startup cleanup: marked {cursor.rowcount} stale worker sessions offline")

        # Reset stuck 'parsing' parse-ahead jobs from previous server run
        try:
            cursor.execute(
                "UPDATE job_queue SET parse_status = NULL WHERE parse_status = 'parsing'"
            )
            if cursor.rowcount > 0:
                logger.info(f"Startup cleanup: reset {cursor.rowcount} stuck parsing jobs")
        except Exception:
            pass  # Column may not exist yet (pre-migration)

        db.conn.commit()
    except Exception as e:
        logger.warning(f"Startup job cleanup failed: {e}")


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
                from backend.server.deps import get_db
                from backend.server.queue.manager import JobQueueManager, _utcnow_sql
                from backend.server.routers.workers import _recalculate_server_pools

                db = get_db()
                cursor = db.conn.cursor()
                now = _utcnow_sql()

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

                queue = JobQueueManager(db)
                total_reclaimed = 0
                for session_id, worker_name in stale_sessions:
                    reclaimed = queue.reclaim_worker_jobs(session_id)
                    total_reclaimed += reclaimed
                    cursor.execute(
                        "UPDATE worker_sessions SET status = 'offline', disconnected_at = ? WHERE id = ?",
                        (now, session_id)
                    )
                    logger.warning(
                        f"Heartbeat timeout: worker '{worker_name}' (session={session_id}) "
                        f"marked offline, reclaimed {reclaimed} jobs"
                    )

                db.conn.commit()

                if total_reclaimed > 0:
                    _recalculate_server_pools(app, db)

            except Exception as e:
                logger.error(f"Heartbeat watchdog error: {e}")

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
