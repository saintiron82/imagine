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


# ── Models ───────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    name: str
    source_path: str
    file_paths: list[str]


class ClaimRequest(BaseModel):
    phase: str       # download, parse, mc, vv, mv
    worker_id: int
    count: int = 5


class CompletePhaseRequest(BaseModel):
    task_id: int
    phase: str       # download, parse, mc, vv, mv
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
    return {"success": True, **result}


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


# ── Worker Task Claim ────────────────────────────────────────

@router.post("/api/v1/tasks/claim")
def claim_tasks(
    req: ClaimRequest,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Worker claims tasks for a specific phase."""
    mgr = _get_manager(db)
    tasks = mgr.claim_tasks(
        phase=req.phase,
        worker_id=req.worker_id,
        count=req.count,
    )
    return {"success": True, "tasks": tasks, "count": len(tasks)}


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
