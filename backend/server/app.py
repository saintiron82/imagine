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
    _startup_integrity_check()

    # Parse-ahead pool: pre-parse pending jobs in background (server-side Phase P)
    # ParseAhead handles Phase P (parse + thumbnail)
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

    # Download-ahead pool: pre-download WebDAV files to temp folder for Phase P
    try:
        if pa_enabled:
            from backend.server.queue.download_ahead import (
                DownloadAheadPool, register_webdav_source,
            )
            from backend.server.deps import get_db
            db = get_db()
            app.state.download_ahead = DownloadAheadPool(db)
            app.state.download_ahead.start()
            logger.info("Download-ahead pool started")
            # Wire references for ParseAhead ↔ DownloadAhead communication
            if hasattr(app.state, 'parse_ahead') and app.state.parse_ahead:
                app.state.parse_ahead._download_pool = app.state.download_ahead
            # Wire reference for JobQueueManager cleanup
            from backend.server.queue.manager import set_download_pool
            set_download_pool(app.state.download_ahead)
            # Pre-register WebDAV sources from environment (Electron startup)
            webdav_env = os.environ.get("IMAGINE_WEBDAV_SOURCES")
            if webdav_env:
                import json as _json
                try:
                    sources = _json.loads(webdav_env)
                    for src in sources:
                        register_webdav_source(src)
                    logger.info(
                        f"Registered {len(sources)} WebDAV source(s) from environment"
                    )
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse IMAGINE_WEBDAV_SOURCES: {e}")
    except Exception as e:
        logger.error(f"Download-ahead pool failed to start: {e}", exc_info=True)

    # EmbedAheadPool removed — tollgate architecture: server does Phase P only,
    # workers (embedded or external) handle V→VV→MV.
    logger.info("Processing mode: parse (tollgate architecture)")

    # Embedded worker: NEVER auto-start on server boot.
    # User must log in and explicitly enable via Admin UI.
    # Config "auto_processing.enabled" is remembered but only applied after login.
    logger.info("Embedded worker: standby (waiting for user login)")

    # ParseAheadPool is always parse-only — no mode recalculation needed
    from backend.server.queue.manager import set_server_pool_mode
    set_server_pool_mode("parse")

    # Heartbeat watchdog: periodically detect dead workers and reclaim their jobs
    try:
        app.state.heartbeat_watchdog = _start_heartbeat_watchdog()
        logger.info("Heartbeat watchdog started (60s interval, 3min timeout)")
    except Exception as e:
        logger.warning(f"Heartbeat watchdog failed to start: {e}")

    # License manager: hybrid verification (Firebase RTDB → cache → offline JWT → free-tier)
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

        # Periodic re-verification (every 1 hour)
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
                    logger.info("Periodic license re-verification complete")
                except Exception as e:
                    logger.warning(f"Periodic license check failed: {e}")

        t = threading.Thread(target=_license_check, daemon=True, name="license-check")
        t.start()
        logger.info("License manager started (1h periodic check)")
    except Exception as e:
        logger.warning(f"License manager failed to start: {e}")

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
    if hasattr(app.state, "download_ahead") and app.state.download_ahead:
        app.state.download_ahead.stop()
        logger.info("Download-ahead pool stopped")
    if hasattr(app.state, "parse_ahead") and app.state.parse_ahead:
        app.state.parse_ahead.stop()
        logger.info("Parse-ahead pool stopped")
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
from backend.server.routers.pipeline import router as pipeline_router
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
app.include_router(server_init_router, prefix="/api/v1")
app.include_router(license_router, prefix="/api/v1")
app.include_router(archive_router, prefix="/api/v1")


@app.get("/api/v1/health")
def health():
    """Health check endpoint with Firebase/CORS status for debugging."""
    import socket
    from backend.server.firebase_auth import is_firebase_available
    return {
        "status": "ok",
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


def _cleanup_stale_jobs():
    """Reset stale assigned/processing jobs and offline worker sessions on startup."""
    try:
        from backend.server.deps import get_db
        from backend.server.queue.manager import JobQueueManager
        db = get_db()
        queue = JobQueueManager(db)
        # Mark all online worker sessions as offline (stale from previous run)
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

        # Reclaim ALL assigned/processing jobs — server restart means
        # no worker is actively processing, so all in-flight jobs are invalid
        cursor.execute(
            """UPDATE job_queue
               SET status = 'pending',
                   assigned_to = NULL, assigned_at = NULL,
                   worker_session_id = NULL
               WHERE status IN ('assigned', 'processing')"""
        )
        if cursor.rowcount > 0:
            logger.info(f"Startup cleanup: reclaimed {cursor.rowcount} in-flight jobs to pending")

        # Reset stuck 'parsing' parse-ahead jobs from previous server run
        try:
            cursor.execute(
                "UPDATE job_queue SET parse_status = NULL WHERE parse_status = 'parsing'"
            )
            if cursor.rowcount > 0:
                logger.info(f"Startup cleanup: reset {cursor.rowcount} stuck parsing jobs")
        except Exception:
            pass  # Column may not exist yet (pre-migration)

        # Self-repair: parsed jobs must have file_ready=1 (invariant)
        try:
            cursor.execute(
                "UPDATE job_queue SET file_ready = 1 WHERE parse_status = 'parsed' AND file_ready = 0"
            )
            if cursor.rowcount > 0:
                logger.info(f"Startup fix: {cursor.rowcount} parsed jobs had file_ready=0 → fixed")
        except Exception:
            pass

        # Self-repair: WR counter mismatch (no jobs but completed_count < total)
        try:
            cursor.execute("""
                SELECT wr.id, wr.name, wr.total_files, wr.completed_count
                FROM work_requests wr
                WHERE wr.status IN ('queued', 'processing')
                  AND wr.completed_count < wr.total_files
                  AND NOT EXISTS (
                    SELECT 1 FROM job_queue jq WHERE jq.work_request_id = wr.id
                  )
            """)
            for wr_id, wr_name, wr_total, wr_done in cursor.fetchall():
                cursor.execute(
                    "UPDATE work_requests SET completed_count = total_files, status = 'completed', completed_at = datetime('now') WHERE id = ?",
                    (wr_id,)
                )
                cursor.execute(
                    "UPDATE work_subtasks SET completed_count = total_files WHERE work_request_id = ?",
                    (wr_id,)
                )
                logger.info(f"Startup fix: WR '{wr_name}' {wr_done}/{wr_total} → completed (no pending jobs)")
        except Exception:
            pass

        # Auto-recover: retry failed files that can be reprocessed
        # (files with processing_status='failed' and no job in queue)
        try:
            recovered = queue.retry_failed_jobs()
            if recovered > 0:
                logger.info(f"Startup auto-recovery: {recovered} failed files returned to queue")
        except Exception as e:
            logger.warning(f"Startup auto-recovery failed: {e}")

        db.conn.commit()
    except Exception as e:
        logger.warning(f"Startup job cleanup failed: {e}")
        try:
            db.conn.rollback()
        except Exception:
            pass


def _startup_integrity_check():
    """Run full integrity check on startup: fix metadata, audit files, cleanup queue."""
    try:
        from backend.server.deps import get_db
        from backend.server.queue.manager import JobQueueManager
        db = get_db()

        # 1. Auto-fix: relative_path etc. metadata correction (no reprocessing)
        fixed = db.fix_missing_relative_paths()
        if fixed > 0:
            logger.info(f"Startup auto-fix: filled relative_path for {fixed} files")

        # 2. Audit: full repair — re-queue incomplete files, attach to existing WRs
        queue = JobQueueManager(db)
        result = queue.audit_completed_jobs(repair=True)
        incomplete = result['incomplete_files']
        repaired = result['repaired_files']
        if repaired > 0:
            logger.warning(
                f"Startup audit: {result['total_files']} files, "
                f"{incomplete} incomplete, {repaired} re-queued"
            )
        else:
            logger.info(
                f"Startup audit: {result['total_files']} files, "
                f"{result['complete_files']} complete"
            )
        # Store for embedded worker decision
        app.state.startup_pending_jobs = incomplete
    except Exception as e:
        logger.warning(f"Startup integrity check failed: {e}")


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
                         AND worker_name != '__builtin__'
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
