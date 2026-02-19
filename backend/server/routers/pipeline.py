"""
Pipeline router — job queue management, work distribution, result upload.
"""

import base64
import json
import logging
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user, require_admin
from backend.server.queue.manager import JobQueueManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pipeline"])


def _get_queue(db: SQLiteDB) -> JobQueueManager:
    return JobQueueManager(db)


# ── Schemas ──────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    count: int = 10


class ProgressUpdate(BaseModel):
    phase: str  # "parse" | "vision" | "embed"


class JobCompleteRequest(BaseModel):
    metadata: Dict[str, Any]       # Same format as upsert_metadata/update_vision_fields
    vectors: Optional[Dict[str, str]] = None  # {"vv": base64, "mv": base64, "structure": base64}


class FailReport(BaseModel):
    error_message: str


class RegisterPathsRequest(BaseModel):
    file_paths: List[str]
    priority: int = 0


# ── Job Queue Endpoints ──────────────────────────────────────

@router.post("/api/v1/jobs/claim")
def claim_jobs(
    req: ClaimRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Claim pending jobs for processing."""
    queue = _get_queue(db)
    jobs = queue.claim_jobs(user["id"], min(req.count, 20))
    return {"success": True, "jobs": jobs, "count": len(jobs)}


@router.get("/api/v1/jobs")
def list_my_jobs(
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """List jobs assigned to current user."""
    queue = _get_queue(db)
    jobs = queue.get_user_jobs(user["id"])
    return {"success": True, "jobs": jobs}


@router.get("/api/v1/jobs/stats")
def get_job_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Get job queue statistics."""
    queue = _get_queue(db)
    stats = queue.get_stats()
    return {"success": True, **stats}


@router.patch("/api/v1/jobs/{job_id}/progress")
def update_job_progress(
    job_id: int,
    req: ProgressUpdate,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Report phase completion for a job."""
    queue = _get_queue(db)
    success = queue.update_progress(job_id, user["id"], req.phase)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")
    return {"success": True}


@router.patch("/api/v1/jobs/{job_id}/complete")
def complete_job(
    job_id: int,
    req: JobCompleteRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Complete a job with analysis results (metadata + vectors)."""
    # Get job info
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path = row

    # Store metadata
    meta = req.metadata
    if "file_path" not in meta:
        meta["file_path"] = file_path

    # Phase 1: metadata
    stored_file_id = db.upsert_metadata(file_path, meta)

    # Phase 2: vision fields
    vision_keys = [
        "mc_caption", "ai_tags", "ocr_text", "dominant_color", "ai_style",
        "image_type", "art_style", "color_palette", "scene_type",
        "time_of_day", "weather", "character_type", "item_type", "ui_type",
        "structured_meta", "perceptual_hash", "dup_group_id", "caption_model",
    ]
    vision_fields = {k: v for k, v in meta.items() if k in vision_keys and v is not None}
    if vision_fields:
        db.update_vision_fields(file_path, vision_fields)

    # Phase 3: vectors
    if req.vectors:
        vv_vec = _decode_vector(req.vectors.get("vv"))
        mv_vec = _decode_vector(req.vectors.get("mv"))
        structure_vec = _decode_vector(req.vectors.get("structure"))
        db.upsert_vectors(stored_file_id, vv_vec=vv_vec, mv_vec=mv_vec, structure_vec=structure_vec)

    # Mark job complete
    queue = _get_queue(db)
    queue.complete_job(job_id, user["id"])

    logger.info(f"Job {job_id} completed by user {user['username']}: {file_path}")
    return {"success": True, "file_id": stored_file_id}


@router.patch("/api/v1/jobs/{job_id}/fail")
def fail_job(
    job_id: int,
    req: FailReport,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Report job failure."""
    queue = _get_queue(db)
    success = queue.fail_job(job_id, user["id"], req.error_message)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")
    return {"success": True}


# ── File Registration (shared_fs mode) ───────────────────────

@router.post("/api/v1/upload/register-paths")
def register_paths(
    req: RegisterPathsRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Register file paths (shared filesystem mode) and create processing jobs."""
    cursor = db.conn.cursor()
    file_ids = []
    file_paths = []

    for fpath in req.file_paths:
        # Minimal metadata insert
        meta = {
            "file_path": fpath,
            "file_name": fpath.rsplit("/", 1)[-1] if "/" in fpath else fpath.rsplit("\\", 1)[-1] if "\\" in fpath else fpath,
        }
        try:
            fid = db.upsert_metadata(fpath, meta)
            # Set uploaded_by
            cursor.execute(
                "UPDATE files SET uploaded_by = ? WHERE id = ? AND uploaded_by IS NULL",
                (user["id"], fid)
            )
            file_ids.append(fid)
            file_paths.append(fpath)
        except Exception as e:
            logger.warning(f"Failed to register {fpath}: {e}")

    db.conn.commit()

    # Create jobs
    queue = _get_queue(db)
    jobs_created = queue.create_jobs(file_ids, file_paths, req.priority)

    return {
        "success": True,
        "registered": len(file_ids),
        "jobs_created": jobs_created,
    }


# ── Admin: Stale job cleanup ─────────────────────────────────

@router.post("/api/v1/admin/jobs/cleanup")
def cleanup_stale_jobs(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
):
    """Reassign stale jobs back to pending (admin only)."""
    from backend.server.config import get_queue_config
    timeout = get_queue_config().get("assignment_timeout_minutes", 30)
    queue = _get_queue(db)
    count = queue.reassign_stale_jobs(timeout)
    return {"success": True, "reassigned": count}


# ── Helpers ──────────────────────────────────────────────────

def _decode_vector(encoded: Optional[str]) -> Optional[np.ndarray]:
    """Decode base64-encoded float32 vector."""
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded)
        return np.frombuffer(raw, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to decode vector: {e}")
        return None
