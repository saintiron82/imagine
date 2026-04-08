"""
Analysis Job API — replaces legacy work_requests + job_queue endpoints.

All progress counts are query-based (no event counters).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db_safe, get_current_user, require_admin
from backend.server.queue.analysis_manager import AnalysisJobManager

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_manager(db: SQLiteDB) -> AnalysisJobManager:
    return AnalysisJobManager(db)


def _auto_start_worker():
    """Start embedded worker if not already running (called on new job creation)."""
    try:
        from backend.server.embedded_worker import _daemon
        # Already running — skip
        if _daemon and _daemon.is_running():
            return
        from backend.server.embedded_worker import start_worker
        start_worker(server_url="", access_token="")
        logger.info("Embedded worker auto-started (new analysis job created)")
    except Exception as e:
        logger.warning(f"Embedded worker auto-start failed: {e}")


# ── Models ───────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    name: str
    source_path: str
    file_paths: list[str] = []  # optional — if empty, server scans folder


class ScanJobRequest(BaseModel):
    folder_path: str
    name: Optional[str] = None
    priority: int = 0


class ClaimRequest(BaseModel):
    worker_id: int   # Server decides phase + count via scheduler


class CompletePhaseRequest(BaseModel):
    task_id: int
    phase: str       # mc, vv, mv
    success: bool
    error_message: Optional[str] = None
    elapsed_s: Optional[float] = None  # actual processing time (worker-measured)


class RetryRequest(BaseModel):
    phase: Optional[str] = None  # None = retry all phases


# ── Analysis Jobs ────────────────────────────────────────────

@router.get("/api/v1/analysis-jobs")
def list_analysis_jobs(
    include_completed: bool = False,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List analysis jobs with progress."""
    mgr = _get_manager(db)
    jobs = mgr.list_jobs(include_completed=include_completed)
    return {"success": True, "jobs": jobs}


@router.post("/api/v1/analysis-jobs")
def create_analysis_job(
    req: CreateJobRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Create a new analysis job."""
    mgr = _get_manager(db)
    result = mgr.create_job(
        name=req.name,
        source_path=req.source_path,
        file_paths=req.file_paths,
        created_by=user.get("id"),
    )
    _auto_start_worker()
    return {"success": True, **result}


@router.post("/api/v1/discover/scan")
def scan_and_create_job(
    req: ScanJobRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Scan a folder path and create an analysis job with discovered files.

    Supports local paths and webdav:// paths.
    Auto-generates job name from folder name if not provided.
    """
    from pathlib import Path, PurePosixPath

    folder_path = req.folder_path
    name = req.name or PurePosixPath(folder_path).name or folder_path

    # Discover files in folder
    file_paths = []
    if folder_path.startswith("webdav://"):
        # WebDAV: list files via WebDAV client
        try:
            from backend.server.queue.download_ahead import parse_webdav_path, get_webdav_source
            from backend.remote.webdav_client import WebDAVClient

            source_id, remote_path = parse_webdav_path(folder_path)
            source_config = get_webdav_source(source_id)
            logger.info(f"discover/scan: source_id={source_id}, remote_path={remote_path}, registered={bool(source_config)}")
            if not source_config:
                # Try auto-register from IMAGINE_WEBDAV_SOURCES env (set by Electron)
                import os as _os
                webdav_env = _os.environ.get("IMAGINE_WEBDAV_SOURCES")
                logger.info(f"discover/scan: env IMAGINE_WEBDAV_SOURCES={'set' if webdav_env else 'NOT SET'}")
                if webdav_env:
                    try:
                        import json as _j
                        from backend.server.queue.download_ahead import register_webdav_source as _reg
                        for src in _j.loads(webdav_env):
                            if src.get("id") == source_id:
                                _reg(src)
                                source_config = get_webdav_source(source_id)
                                break
                    except Exception:
                        pass
            if not source_config:
                raise HTTPException(400, f"WebDAV source '{source_id}' not registered. Restart server to reload sources.")

            client = WebDAVClient(
                base_url=source_config["url"],
                username=source_config["username"],
                password=source_config["password"],
                remote_path=remote_path,
                verify_ssl=source_config.get("verify_ssl", True),
            )
            files = client.list_files_recursive()
            client.close()
            logger.info(f"discover/scan: WebDAV found {len(files)} files in {remote_path}")

            for f in files:
                file_paths.append(f"webdav://{source_id}{f.remote_path}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"WebDAV scan failed: {e}")
    else:
        # Local folder
        local = Path(folder_path)
        if not local.exists():
            raise HTTPException(400, f"Folder not found: {folder_path}")
        supported = {".psd", ".png", ".jpg", ".jpeg"}
        for f in local.rglob("*"):
            if f.is_file() and f.suffix.lower() in supported:
                file_paths.append(str(f))

    if not file_paths:
        return {
            "success": False,
            "error": f"No supported files (.psd/.png/.jpg) found in '{PurePosixPath(folder_path).name}'",
            "jobs_created": 0,
            "scanned_path": folder_path,
        }

    # Create analysis job
    mgr = _get_manager(db)
    result = mgr.create_job(
        name=name,
        source_path=folder_path,
        file_paths=file_paths,
        created_by=user.get("id"),
    )
    _auto_start_worker()
    return {"success": True, "jobs_created": 1, **result}


@router.get("/api/v1/analysis-jobs/{job_id}")
def get_analysis_job(
    job_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get analysis job detail with progress."""
    mgr = _get_manager(db)
    job = mgr.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True, **job}


@router.get("/api/v1/analysis-jobs/{job_id}/progress")
def get_analysis_progress(
    job_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get phase-level progress (sum = total guaranteed)."""
    mgr = _get_manager(db)
    progress = mgr.get_progress(job_id)
    return {"success": True, **progress}


@router.get("/api/v1/analysis-jobs/{job_id}/metrics")
def get_analysis_metrics(
    job_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get speed metrics: throughput, phase times, bottleneck."""
    mgr = _get_manager(db)
    metrics = mgr.get_metrics(job_id)
    return {"success": True, **metrics}


# ── Job Control ──────────────────────────────────────────────

@router.post("/api/v1/analysis-jobs/{job_id}/pause")
def pause_analysis_job(
    job_id: int,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    mgr = _get_manager(db)
    if not mgr.pause_job(job_id):
        raise HTTPException(status_code=400, detail="Cannot pause (not active)")
    return {"success": True}


@router.post("/api/v1/analysis-jobs/{job_id}/resume")
def resume_analysis_job(
    job_id: int,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    mgr = _get_manager(db)
    if not mgr.resume_job(job_id):
        raise HTTPException(status_code=400, detail="Cannot resume (not paused)")
    return {"success": True}


@router.post("/api/v1/analysis-jobs/{job_id}/archive")
def archive_analysis_job(
    job_id: int,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Archive a job — hidden from dashboard, visible in history."""
    mgr = _get_manager(db)
    if not mgr.archive_job(job_id):
        raise HTTPException(status_code=400, detail="Cannot archive")
    return {"success": True}


@router.delete("/api/v1/analysis-jobs/{job_id}")
def delete_analysis_job(
    job_id: int,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Permanently delete a job + file_tasks. Search data (files/vec) preserved."""
    mgr = _get_manager(db)
    if not mgr.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"success": True}


@router.post("/api/v1/analysis-jobs/{job_id}/cancel")
def cancel_analysis_job(
    job_id: int,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    mgr = _get_manager(db)
    if not mgr.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Cannot cancel")
    return {"success": True}


@router.post("/api/v1/analysis-jobs/{job_id}/retry")
def retry_failed_tasks(
    job_id: int,
    req: RetryRequest = RetryRequest(),
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Retry failed tasks in an analysis job."""
    mgr = _get_manager(db)
    count = mgr.retry_failed(job_id, phase=req.phase)
    return {"success": True, "retried": count}


@router.get("/api/v1/analysis-jobs/{job_id}/errors")
def get_job_errors(
    job_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List failed tasks (non-dismissed) with error details."""
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT ft.file_id, f.file_name, ft.parse_status,
               ft.mc_status, ft.vv_status, ft.mv_status,
               ft.error_message, ft.retry_count, ft.max_retries,
               ft.id
        FROM file_tasks ft
        JOIN files f ON ft.file_id = f.id
        WHERE ft.analysis_job_id = ?
          AND ft.dismissed_at IS NULL
          AND (ft.parse_status = 'failed'
               OR ft.mc_status = 'failed'
               OR ft.vv_status = 'failed'
               OR ft.mv_status = 'failed')
        ORDER BY ft.file_id
    """, (job_id,))
    rows = cursor.fetchall()

    errors = []
    for r in rows:
        failed_phases = []
        if r[2] == 'failed': failed_phases.append('parse')
        if r[3] == 'failed': failed_phases.append('mc')
        if r[4] == 'failed': failed_phases.append('vv')
        if r[5] == 'failed': failed_phases.append('mv')
        permanent = r[7] >= r[8] if r[8] is not None else False
        errors.append({
            "task_id": r[9],
            "file_id": r[0],
            "file_name": r[1],
            "failed_phases": failed_phases,
            "error": r[6] or "",
            "retry_count": r[7],
            "max_retries": r[8],
            "permanent": permanent,
        })

    return {"errors": errors, "count": len(errors)}


@router.post("/api/v1/analysis-jobs/{job_id}/dismiss")
def dismiss_failed_tasks(
    job_id: int,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Dismiss permanent failures — removes them from active error counts.

    Only dismisses tasks where retry_count >= max_retries (permanent).
    """
    cursor = db.conn.cursor()
    cursor.execute("""
        UPDATE file_tasks
        SET dismissed_at = datetime('now'),
            updated_at = datetime('now')
        WHERE analysis_job_id = ?
          AND dismissed_at IS NULL
          AND retry_count >= max_retries
          AND (parse_status = 'failed'
               OR mc_status = 'failed'
               OR vv_status = 'failed'
               OR mv_status = 'failed')
    """, (job_id,))
    count = cursor.rowcount
    db.conn.commit()

    # Check if job can now be marked completed
    if count > 0:
        mgr = _get_manager(db)
        # Use any dismissed task_id to trigger completion check
        cursor.execute(
            "SELECT id FROM file_tasks WHERE analysis_job_id = ? LIMIT 1",
            (job_id,),
        )
        row = cursor.fetchone()
        if row:
            mgr._check_job_completion(row[0])

    return {"success": True, "dismissed": count}


# ── Worker Task Claim ────────────────────────────────────────

@router.post("/api/v1/tasks/claim")
def claim_tasks(
    req: ClaimRequest,
    request: Request,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Worker requests tasks. Server scheduler decides phase + count.

    Worker only sends worker_id. Scheduler determines optimal phase
    based on queue state and worker capability profile.
    """
    from backend.server.queue.scheduler import WorkerScheduler

    scheduler = getattr(request.app.state, 'scheduler', None)
    if not scheduler:
        scheduler = WorkerScheduler(db)

    # Scheduler decides phase + count
    decision = scheduler.assign(req.worker_id)
    phase = decision.get("phase")
    count = decision.get("count", 0)

    if not phase or count == 0:
        return {"success": True, "phase": None, "tasks": [], "count": 0}

    # Claim via manager
    mgr = _get_manager(db)
    tasks = mgr.claim_tasks(phase=phase, worker_id=req.worker_id, count=count)
    return {
        "success": True,
        "phase": phase,
        "tasks": tasks,
        "count": len(tasks),
    }


@router.get("/api/v1/files/{file_id}/mc")
def get_file_mc(
    file_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get MC (vision) fields for a file — used by MV workers."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT mc_caption, ai_tags, image_type, scene_type, art_style
           FROM files WHERE id = ?""",
        (file_id,),
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found")
    import json as _json
    ai_tags = row[1]
    if isinstance(ai_tags, str):
        try:
            ai_tags = _json.loads(ai_tags)
        except Exception:
            ai_tags = []
    return {
        "mc_caption": row[0] or "",
        "ai_tags": ai_tags or [],
        "image_type": row[2] or "",
        "scene_type": row[3] or "",
        "art_style": row[4] or "",
    }


@router.patch("/api/v1/files/{file_id}/vv")
async def save_vv_vector(
    file_id: int,
    request: Request,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Save VV vector directly to vec_files table."""
    import struct
    body = await request.json()
    vec = body.get("vector", [])
    if not vec:
        raise HTTPException(status_code=400, detail="No vector provided")

    # Encode as binary float32
    blob = struct.pack(f"<{len(vec)}f", *vec)
    cursor = db.conn.cursor()
    # Upsert into vec_files
    cursor.execute("DELETE FROM vec_files WHERE file_id = ?", (file_id,))
    cursor.execute("INSERT INTO vec_files (file_id, embedding) VALUES (?, ?)", (file_id, blob))
    db.conn.commit()
    return {"success": True}


@router.patch("/api/v1/files/{file_id}/mv")
async def save_mv_vector(
    file_id: int,
    request: Request,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Save MV vector directly to vec_text table."""
    import struct
    body = await request.json()
    vec = body.get("vector", [])
    if not vec:
        raise HTTPException(status_code=400, detail="No vector provided")
    blob = struct.pack(f"<{len(vec)}f", *vec)
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (file_id,))
    cursor.execute("INSERT INTO vec_text (file_id, embedding) VALUES (?, ?)", (file_id, blob))
    db.conn.commit()
    return {"success": True}


@router.patch("/api/v1/files/{file_id}/vision")
async def save_vision_fields(
    file_id: int,
    request: Request,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Save MC vision fields directly to files table."""
    body = await request.json()
    cursor = db.conn.cursor()
    # Update vision fields
    fields = ["mc_caption", "ai_tags", "image_type", "art_style", "color_palette",
              "scene_type", "time_of_day", "weather", "character_type", "item_type",
              "ui_type", "structured_meta"]
    updates = []
    values = []
    for f in fields:
        if f in body:
            import json as _json
            val = body[f]
            if isinstance(val, (list, dict)):
                val = _json.dumps(val, ensure_ascii=False)
            updates.append(f"{f} = ?")
            values.append(val)

    if updates:
        values.append(file_id)
        cursor.execute(
            f"UPDATE files SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        db.conn.commit()

    return {"success": True}


# Note: /api/v1/files/{file_id}/parse endpoint removed.
# Parse is now server-only via FileTaskParsePool. Workers never parse.


class StartPhaseRequest(BaseModel):
    task_id: int
    phase: str


@router.post("/api/v1/tasks/start")
def start_task_phase(
    req: StartPhaseRequest,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Worker reports actual processing start (not claim time)."""
    mgr = _get_manager(db)
    mgr.start_task_phase(req.task_id, req.phase)
    return {"success": True}


@router.post("/api/v1/tasks/complete")
def complete_task_phase(
    req: CompletePhaseRequest,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Worker reports phase completion for a task."""
    mgr = _get_manager(db)
    mgr.complete_task_phase(
        task_id=req.task_id,
        phase=req.phase,
        success=req.success,
        error_message=req.error_message,
        elapsed_s=req.elapsed_s,
    )
    return {"success": True}


# ── Phase Pause Control ─────────────────────────────────────

_DEFAULT_PAUSED = {"dl": False, "parse": False, "mc": False, "vv": False, "mv": False}


def get_paused_phases(db: SQLiteDB) -> dict:
    """Read paused phases from system_meta."""
    import json as _json
    raw = db._get_system_meta("paused_phases")
    if raw:
        try:
            return {**_DEFAULT_PAUSED, **_json.loads(raw)}
        except Exception:
            pass
    return dict(_DEFAULT_PAUSED)


def set_paused_phases(db: SQLiteDB, updates: dict) -> dict:
    """Update paused phases in system_meta."""
    import json as _json
    current = get_paused_phases(db)
    for k in ("dl", "parse", "mc", "vv", "mv"):
        if k in updates:
            current[k] = bool(updates[k])
    db._set_system_meta("paused_phases", _json.dumps(current))
    return current


@router.get("/api/v1/server/paused-phases")
def api_get_paused_phases(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get per-phase pause state."""
    return get_paused_phases(db)


@router.post("/api/v1/server/paused-phases")
async def api_set_paused_phases(
    request: Request,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Toggle individual phase pause. Body: {"mc": true, "vv": false, ...}"""
    body = await request.json()
    result = set_paused_phases(db, body)
    logger.info(f"Paused phases updated: {result}")
    return {"paused_phases": result}
