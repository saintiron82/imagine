"""
Pipeline router — job queue management, work distribution, result upload.
"""

import base64
import json
import logging
import unicodedata
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, get_current_user, require_admin
from backend.server.queue.manager import JobQueueManager, _utcnow_sql

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pipeline"])


def _get_queue(db: SQLiteDB) -> JobQueueManager:
    return JobQueueManager(db)


# ── Schemas ──────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    count: int = 10
    session_id: Optional[int] = None  # worker_session_id for per-worker tracking


class ProgressUpdate(BaseModel):
    phase: str  # "parse" | "vision" | "embed"


class JobCompleteRequest(BaseModel):
    metadata: Dict[str, Any]       # Same format as upsert_metadata/update_vision_fields
    vectors: Optional[Dict[str, str]] = None  # {"vv": base64, "mv": base64, "structure": base64}


class FailReport(BaseModel):
    error_message: str
    error_code: Optional[str] = None  # structured error code (THUMB_MISSING, VLM_FAILED, etc.)


class MCCompleteRequest(BaseModel):
    vision_fields: Dict[str, Any]  # mc_caption, ai_tags, image_type, etc.


class EmbedCompleteRequest(BaseModel):
    vectors: Dict[str, str]  # {"vv": base64, "mv": base64}


class RegisterPathsRequest(BaseModel):
    file_paths: List[str]
    priority: int = 0


class ScanFolderRequest(BaseModel):
    folder_path: str
    priority: int = 0


class PhaseStatusRequest(BaseModel):
    file_paths: List[str]


# ── Job Queue Endpoints ──────────────────────────────────────

@router.post("/api/v1/jobs/claim")
def claim_jobs(
    req: ClaimRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Claim pending jobs for processing."""
    queue = _get_queue(db)
    result = queue.claim_jobs(user["id"], min(req.count, 50), worker_session_id=req.session_id)
    # claim_jobs may return a dict with diag info (empty claim) or a list of jobs
    if isinstance(result, dict):
        return {"success": True, **result}
    return {"success": True, "jobs": result, "count": len(result)}


@router.get("/api/v1/jobs")
def list_my_jobs(
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List jobs assigned to current user."""
    queue = _get_queue(db)
    jobs = queue.get_user_jobs(user["id"])
    return {"success": True, "jobs": jobs}


@router.get("/api/v1/jobs/stats")
def get_job_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
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
    db: SQLiteDB = Depends(get_db_safe),
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
    db: SQLiteDB = Depends(get_db_safe),
):
    """Complete a job with analysis results (metadata + vectors)."""
    import sqlite3 as _sqlite3
    import time as _time
    for _attempt in range(3):
        try:
            return _complete_job_inner(job_id, req, user, db)
        except _sqlite3.OperationalError as e:
            if "locked" not in str(e) or _attempt >= 2:
                raise
            logger.warning(f"complete_job {job_id}: DB locked, retry {_attempt + 1}/3")
            _time.sleep(1.0 * (_attempt + 1))


def _complete_job_inner(job_id, req, user, db):
    """Internal complete_job implementation."""
    # Get job info (include parse_status to avoid duplicate upsert)
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path, parse_status FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path, parse_status = row

    # Normalize to NFC — job_queue may store NFD (macOS filesystem),
    # but files table stores NFC (via upsert_metadata).
    file_path = unicodedata.normalize('NFC', file_path)

    # Store metadata
    meta = req.metadata
    if "file_path" not in meta:
        meta["file_path"] = file_path

    # Phase 1: metadata — skip if ParseAheadPool already saved it
    if parse_status == 'parsed':
        # Pre-parsed: metadata already in DB, just look up file_id
        cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
        existing = cursor.fetchone()
        stored_file_id = existing[0] if existing else db.upsert_metadata(file_path, meta)
    else:
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

    # Verify data integrity: completed = ALL data must exist (mc + vv + mv)
    integrity = db.verify_data_integrity(
        stored_file_id, expect_mc=True, expect_vv=True, expect_mv=True
    )

    queue = _get_queue(db)
    if integrity["valid"]:
        queue.complete_job(job_id, user["id"])
    else:
        logger.warning(
            f"Job {job_id} integrity mismatch: missing={integrity['missing']}. "
            f"Saving with actual phases: {integrity['actual_phases']}"
        )
        queue.complete_job_with_phases(job_id, user["id"], integrity["actual_phases"])

    logger.info(f"Job {job_id} completed by user {user['username']}: {file_path}")
    return {"success": True, "file_id": stored_file_id, "integrity": integrity["valid"]}


@router.patch("/api/v1/jobs/{job_id}/fail")
def fail_job(
    job_id: int,
    req: FailReport,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Report job failure."""
    queue = _get_queue(db)
    success = queue.fail_job(job_id, user["id"], req.error_message, req.error_code)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")
    return {"success": True}


@router.patch("/api/v1/jobs/{job_id}/complete_mc")
def complete_mc(
    job_id: int,
    req: MCCompleteRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """MC-only worker: store vision fields, mark MC done (not fully complete).

    In mc_only mode, workers only generate MC (VLM caption/tags).
    Server handles VV (ParseAhead) and MV (EmbedAhead) separately.
    Job stays in 'processing' status until EmbedAhead completes MV.
    """
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path = row

    # Normalize to NFC — job_queue may store NFD (macOS filesystem),
    # but files table stores NFC (via upsert_metadata).
    file_path = unicodedata.normalize('NFC', file_path)

    # Look up stored file_id (pre-parsed by ParseAhead)
    cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
    existing = cursor.fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail="File not found in DB — ParseAhead may not have processed this job")
    stored_file_id = existing[0]

    # Store vision fields (mc_caption, ai_tags, image_type, etc.)
    vision_keys = [
        "mc_caption", "ai_tags", "ocr_text", "dominant_color", "ai_style",
        "image_type", "art_style", "color_palette", "scene_type",
        "time_of_day", "weather", "character_type", "item_type", "ui_type",
        "structured_meta", "perceptual_hash", "dup_group_id", "caption_model",
    ]
    vision_fields = {k: v for k, v in req.vision_fields.items() if k in vision_keys and v is not None}

    # mc_caption is required — EmbedAhead needs it to build MV.
    # If VLM failed and worker sent empty fields, reject rather than
    # marking vision=true with no MC data (which causes EmbedAhead failure).
    if not vision_fields or "mc_caption" not in vision_fields:
        logger.warning(f"MC-only job {job_id}: empty or missing mc_caption in vision_fields")
        raise HTTPException(
            status_code=422,
            detail="mc_caption is required in vision_fields — VLM may have failed"
        )

    db.update_vision_fields(file_path, vision_fields)

    # Post-save verification: confirm mc_caption was actually stored
    integrity = db.verify_data_integrity(stored_file_id, expect_mc=True)
    if not integrity["has_mc"]:
        logger.error(
            f"MC-only job {job_id}: mc_caption save verification failed "
            f"(data not found in DB after update_vision_fields)"
        )
        raise HTTPException(
            status_code=500,
            detail="mc_caption save failed — data not found in DB after write"
        )

    # Update job: mark vision done, release back to pending for VV/MV workers
    queue = _get_queue(db)
    queue.update_phase(job_id, "mc", True)
    mc_now = _utcnow_sql()
    cursor.execute(
        """UPDATE job_queue
           SET status = 'pending', assigned_to = NULL, worker_session_id = NULL,
               mc_completed_at = ?
           WHERE id = ?""",
        (mc_now, job_id)
    )
    db.conn.commit()

    logger.info(f"MC-only job {job_id} vision done by {user['username']}: {file_path}")
    return {"success": True, "file_id": stored_file_id}


@router.patch("/api/v1/jobs/{job_id}/complete_embed")
def complete_embed(
    job_id: int,
    req: EmbedCompleteRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Embed-only worker: store VV+MV vectors and mark job complete.

    In embed_only mode, workers only generate VV (SigLIP2) and MV (Qwen3-Embedding).
    Server or full workers handle Parse + Vision (MC). Job must already have vision=true.
    """
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path = row
    file_path = unicodedata.normalize('NFC', file_path)

    # Look up stored file_id (vision must have already been completed)
    cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
    existing = cursor.fetchone()
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail="File not found in DB — vision phase may not have completed"
        )
    stored_file_id = existing[0]

    # Store VV + MV vectors
    vv_vec = _decode_vector(req.vectors.get("vv"))
    mv_vec = _decode_vector(req.vectors.get("mv"))
    db.upsert_vectors(stored_file_id, vv_vec=vv_vec, mv_vec=mv_vec)

    # Verify all data exists: completed = mc + vv + mv all present
    integrity = db.verify_data_integrity(
        stored_file_id, expect_mc=True, expect_vv=True, expect_mv=True
    )

    queue = _get_queue(db)
    if integrity["valid"]:
        queue.complete_job(job_id, user["id"])
    else:
        logger.warning(
            f"Embed-only job {job_id} integrity mismatch: missing={integrity['missing']}. "
            f"Saving with actual phases: {integrity['actual_phases']}"
        )
        queue.complete_job_with_phases(job_id, user["id"], integrity["actual_phases"])

    logger.info(f"Embed-only job {job_id} completed by {user['username']}: {file_path}")
    return {"success": True, "file_id": stored_file_id, "integrity": integrity["valid"]}


@router.patch("/api/v1/jobs/{job_id}/complete_vv")
def complete_vv(
    job_id: int,
    req: EmbedCompleteRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """VV-only worker: store VV vector. Job stays processing until MV is also done."""
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path = row
    file_path = unicodedata.normalize('NFC', file_path)

    cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
    existing = cursor.fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail="File not found in DB")
    stored_file_id = existing[0]

    vv_vec = _decode_vector(req.vectors.get("vv"))
    db.upsert_vectors(stored_file_id, vv_vec=vv_vec)

    # Update phase_completed: mark vv=true
    queue = _get_queue(db)
    queue.update_phase(job_id, "vv", True)

    # Check if all phases complete
    integrity = db.verify_data_integrity(
        stored_file_id, expect_mc=True, expect_vv=True, expect_mv=True
    )
    if integrity["valid"]:
        queue.complete_job(job_id, user["id"])
    else:
        # Release back to pending for remaining phases (MV)
        cursor.execute(
            "UPDATE job_queue SET status = 'pending', assigned_to = NULL, worker_session_id = NULL WHERE id = ?",
            (job_id,)
        )
        db.conn.commit()

    return {"success": True, "file_id": stored_file_id}


@router.patch("/api/v1/jobs/{job_id}/complete_mv")
def complete_mv(
    job_id: int,
    req: EmbedCompleteRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """MV-only worker: store MV vector. Job stays processing until VV is also done."""
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path = row
    file_path = unicodedata.normalize('NFC', file_path)

    cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
    existing = cursor.fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail="File not found in DB")
    stored_file_id = existing[0]

    mv_vec = _decode_vector(req.vectors.get("mv"))
    db.upsert_vectors(stored_file_id, mv_vec=mv_vec)

    # Update phase_completed: mark mv=true
    queue = _get_queue(db)
    queue.update_phase(job_id, "mv", True)

    # Check if all phases complete
    integrity = db.verify_data_integrity(
        stored_file_id, expect_mc=True, expect_vv=True, expect_mv=True
    )
    if integrity["valid"]:
        queue.complete_job(job_id, user["id"])
    else:
        # Release back to pending for remaining phases (VV)
        cursor.execute(
            "UPDATE job_queue SET status = 'pending', assigned_to = NULL, worker_session_id = NULL WHERE id = ?",
            (job_id,)
        )
        db.conn.commit()

    return {"success": True, "file_id": stored_file_id}


class BatchVectorRequest(BaseModel):
    items: List[dict]  # [{job_id, vv/mv: base64_vector}, ...]


@router.patch("/api/v1/admin/jobs/batch-complete-vv")
def batch_complete_vv(
    req: BatchVectorRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Batch VV upload: process multiple VV vectors in one request."""
    queue = _get_queue(db)
    cursor = db.conn.cursor()
    results = []
    errors = []

    for item in req.items:
        job_id = item.get("job_id")
        try:
            cursor.execute(
                "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
                (job_id, user["id"])
            )
            row = cursor.fetchone()
            if not row:
                errors.append({
                    "job_id": job_id,
                    "error": "Job not found or not assigned to this worker user",
                })
                results.append(False)
                continue

            file_id, file_path = row
            file_path = unicodedata.normalize('NFC', file_path)
            cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
            existing = cursor.fetchone()
            if not existing:
                errors.append({
                    "job_id": job_id,
                    "error": f"File not found in DB for path: {file_path}",
                })
                results.append(False)
                continue
            stored_file_id = existing[0]

            vv_vec = _decode_vector(item.get("vv"))
            db.upsert_vectors(stored_file_id, vv_vec=vv_vec)
            queue.update_phase(job_id, "vv", True)

            integrity = db.verify_data_integrity(stored_file_id, expect_mc=True, expect_vv=True, expect_mv=True)
            if integrity["valid"]:
                queue.complete_job(job_id, user["id"])
            else:
                cursor.execute(
                    "UPDATE job_queue SET status = 'pending', assigned_to = NULL, worker_session_id = NULL WHERE id = ?",
                    (job_id,)
                )
            results.append(True)
        except Exception as e:
            logger.error(f"Batch VV job {job_id} failed: {e}")
            errors.append({"job_id": job_id, "error": str(e)})
            results.append(False)

    db.conn.commit()
    return {"success": True, "results": results, "errors": errors}


@router.patch("/api/v1/admin/jobs/batch-complete-mv")
def batch_complete_mv(
    req: BatchVectorRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Batch MV upload: process multiple MV vectors in one request."""
    queue = _get_queue(db)
    cursor = db.conn.cursor()
    results = []

    for item in req.items:
        job_id = item.get("job_id")
        try:
            cursor.execute(
                "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
                (job_id, user["id"])
            )
            row = cursor.fetchone()
            if not row:
                results.append(False)
                continue

            file_id, file_path = row
            file_path = unicodedata.normalize('NFC', file_path)
            cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
            existing = cursor.fetchone()
            if not existing:
                results.append(False)
                continue
            stored_file_id = existing[0]

            mv_vec = _decode_vector(item.get("mv"))
            db.upsert_vectors(stored_file_id, mv_vec=mv_vec)
            queue.update_phase(job_id, "mv", True)

            integrity = db.verify_data_integrity(stored_file_id, expect_mc=True, expect_vv=True, expect_mv=True)
            if integrity["valid"]:
                queue.complete_job(job_id, user["id"])
            else:
                cursor.execute(
                    "UPDATE job_queue SET status = 'pending', assigned_to = NULL, worker_session_id = NULL WHERE id = ?",
                    (job_id,)
                )
            results.append(True)
        except Exception as e:
            logger.error(f"Batch MV job {job_id} failed: {e}")
            results.append(False)

    db.conn.commit()
    return {"success": True, "results": results}


@router.patch("/api/v1/jobs/{job_id}/complete_parse")
def complete_parse(
    job_id: int,
    req: JobCompleteRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """parse_thumb worker: upload parse results (metadata + thumbnail) to server.

    Worker has downloaded the file, parsed it, generated a thumbnail,
    and now uploads the results. Server stores metadata in files table
    and marks parse phase complete in job_queue.
    """
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT file_id, file_path FROM job_queue WHERE id = ? AND assigned_to = ?",
        (job_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found or not assigned to you")

    file_id, file_path = row
    file_path = unicodedata.normalize('NFC', file_path)

    meta = req.metadata or {}
    meta["file_path"] = file_path

    # Upsert metadata to files table
    stored_file_id = db.upsert_metadata(file_path, meta)

    # Build mc_raw context for downstream MC workers
    from backend.pipeline.ingest_engine import _build_mc_raw, _set_tier_metadata
    from backend.parser.schema import AssetMeta

    asset_meta = AssetMeta()
    for k, v in meta.items():
        if hasattr(asset_meta, k):
            setattr(asset_meta, k, v)
    _set_tier_metadata(asset_meta)
    mc_raw = _build_mc_raw(asset_meta)

    # Store parsed_metadata in job_queue
    parsed_metadata = {
        "metadata": meta,
        "thumb_path": meta.get("thumbnail_url"),
        "mc_raw": mc_raw,
    }
    cursor.execute(
        """UPDATE job_queue
           SET parse_status = 'parsed', file_ready = 1,
               parsed_metadata = ?,
               phase_completed = json_set(COALESCE(phase_completed, '{}'), '$.parse', json('true')),
               parsed_at = datetime('now')
           WHERE id = ?""",
        (json.dumps(parsed_metadata, ensure_ascii=False, default=str), job_id)
    )
    db.conn.commit()

    logger.info(f"Parse job {job_id} completed by worker {user['username']}: {file_path}")
    return {"success": True, "file_id": stored_file_id}


# ── Discover: Browse & Scan server filesystem ────────────────

SUPPORTED_EXTENSIONS = {'.psd', '.png', '.jpg', '.jpeg'}


@router.get("/api/v1/discover/browse")
def browse_folders(
    path: str = "/",
    user: dict = Depends(get_current_user),
):
    """Browse server filesystem directories (authenticated users).

    Returns subdirectories and count of supported image files.
    """
    from pathlib import Path as P

    target = P(path).resolve()
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    dirs = []
    files_count = 0

    try:
        for entry in sorted(target.iterdir(), key=lambda e: e.name.lower()):
            if entry.name.startswith('.'):
                continue
            if entry.is_dir():
                # Count supported files in subdir (non-recursive, for quick preview)
                sub_count = sum(
                    1 for f in entry.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
                )
                dirs.append({"name": entry.name, "path": str(entry), "files_count": sub_count})
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                files_count += 1
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return {
        "current": str(target),
        "parent": str(target.parent) if target != target.parent else None,
        "dirs": dirs,
        "files_count": files_count,
    }


@router.post("/api/v1/discover/scan")
def scan_folder(
    req: ScanFolderRequest,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """DFS scan a folder and create a work request with sub-tasks (admin only).

    Reuses the existing discover_files() logic from the pipeline engine.
    Groups files by immediate subfolder for sub-task tracking.
    """
    from pathlib import Path as P
    from backend.pipeline.ingest_engine import discover_files

    folder = P(req.folder_path).resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    # DFS scan
    discovered = discover_files(folder)
    if not discovered:
        return {"success": True, "discovered": 0, "jobs_created": 0, "skipped": 0}

    # Register files and group by immediate subfolder
    cursor = db.conn.cursor()
    file_groups = {}  # folder_path → [(file_id, file_path), ...]
    skipped = 0

    for file_path, folder_str, depth, folder_tags in discovered:
        # Normalize to NFC — macOS filesystem returns NFD for Korean paths
        fpath_str = unicodedata.normalize('NFC', str(file_path))

        # Check if job already exists for this file (avoid duplicates)
        cursor.execute(
            "SELECT id FROM job_queue WHERE file_path = ?",
            (fpath_str,)
        )
        if cursor.fetchone():
            skipped += 1
            continue

        meta = {
            "file_path": fpath_str,
            "file_name": file_path.name,
            "folder_path": folder_str,
            "folder_depth": depth,
            "folder_tags": folder_tags,
        }
        try:
            fid = db.upsert_metadata(fpath_str, meta)
            cursor.execute(
                "UPDATE files SET uploaded_by = ? WHERE id = ? AND uploaded_by IS NULL",
                (admin["id"], fid)
            )
            # Promote preview-only files to searchable
            cursor.execute(
                "UPDATE files SET preview_only = 0 WHERE id = ? AND preview_only = 1",
                (fid,)
            )

            # Group by immediate subfolder (1st level relative to scan root)
            try:
                rel = file_path.relative_to(folder)
                if len(rel.parts) > 1:
                    group_key = str(folder / rel.parts[0])
                else:
                    group_key = str(folder)
            except ValueError:
                group_key = str(folder)

            file_groups.setdefault(group_key, []).append((fid, fpath_str))
        except Exception as e:
            logger.warning(f"Failed to register {fpath_str}: {e}")

    db.conn.commit()

    # Create work request with sub-tasks
    queue = _get_queue(db)
    if file_groups:
        result = queue.create_work_request(
            name=folder.name,
            source_path=str(folder),
            file_groups=file_groups,
            priority=req.priority,
            created_by=admin["id"],
        )
        jobs_created = result.get("jobs_created", 0)
        work_request_id = result.get("work_request_id", 0)
    else:
        jobs_created = 0
        work_request_id = 0

    logger.info(
        f"Discover scan: {req.folder_path} → {len(discovered)} found, "
        f"{jobs_created} jobs created, {skipped} skipped, wr_id={work_request_id}"
    )
    return {
        "success": True,
        "discovered": len(discovered),
        "jobs_created": jobs_created,
        "skipped": skipped,
        "work_request_id": work_request_id,
    }


# ── File Registration (shared_fs mode) ───────────────────────

@router.post("/api/v1/upload/register-paths")
def register_paths(
    req: RegisterPathsRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
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
    db: SQLiteDB = Depends(get_db_safe),
):
    """Reassign stale jobs back to pending (admin only)."""
    from backend.server.config import get_queue_config
    timeout = get_queue_config().get("assignment_timeout_minutes", 30)
    queue = _get_queue(db)
    count = queue.reassign_stale_jobs(timeout)
    return {"success": True, "reassigned": count}


@router.get("/api/v1/jobs/list")
def list_all_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    archived: bool = False,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List all jobs with optional status filter and pagination."""
    queue = _get_queue(db)
    result = queue.list_jobs(status=status, limit=min(limit, 100), offset=offset, archived=archived)
    return {"success": True, **result}


@router.patch("/api/v1/jobs/{job_id}/cancel")
def cancel_job(
    job_id: int,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Cancel a pending/assigned/failed job."""
    queue = _get_queue(db)
    success = queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled (may be processing or already completed)")
    return {"success": True}


@router.post("/api/v1/admin/jobs/retry-failed")
def retry_failed_jobs(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Retry all failed jobs by resetting them to pending (admin only)."""
    queue = _get_queue(db)
    count = queue.retry_failed_jobs()
    return {"success": True, "retried": count}


@router.post("/api/v1/admin/jobs/force-retry-failed")
def force_retry_failed_jobs(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Force retry ALL failed/stuck jobs from scratch (admin only).

    Unlike retry-failed, this resets everything including non-retryable errors
    (FILE_NOT_FOUND, PARSE_FAILED) and stuck pending jobs with high retry_count.
    All phase data is cleared — jobs will be re-processed from Phase P.
    """
    queue = _get_queue(db)
    count = queue.force_retry_failed_jobs()
    return {"success": True, "retried": count}


@router.post("/api/v1/admin/jobs/audit-integrity")
def audit_integrity(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Scan completed jobs for data integrity issues and repair (admin only).

    Checks all completed jobs against actual DB data (mc_caption, vec_files,
    vec_text). Jobs with missing data are reset to pending for re-processing.
    """
    queue = _get_queue(db)
    result = queue.audit_completed_jobs()
    return {"success": True, **result}


@router.delete("/api/v1/admin/jobs/clear-completed")
def clear_completed_jobs(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Archive all completed jobs (admin only)."""
    queue = _get_queue(db)
    count = queue.clear_completed_jobs()
    return {"success": True, "archived": count}


@router.post("/api/v1/admin/jobs/archive")
def archive_jobs(
    body: dict,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Archive specific jobs or all jobs with a given status (admin only).

    Body: {"job_ids": [1,2,3]} or {"status": "completed"}
    """
    queue = _get_queue(db)
    if "job_ids" in body:
        count = queue.archive_jobs(body["job_ids"])
    elif "status" in body and body["status"] == "completed":
        count = queue.archive_completed_jobs()
    else:
        raise HTTPException(status_code=400, detail="Provide job_ids or status")
    return {"success": True, "archived": count}


@router.get("/api/v1/admin/history/sessions")
def list_history_sessions(
    limit: int = 50,
    offset: int = 0,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List work request sessions for history view."""
    queue = _get_queue(db)
    result = queue.list_history_sessions(limit=min(limit, 100), offset=offset)
    return {"success": True, **result}


@router.get("/api/v1/admin/history/sessions/{wr_id}/jobs")
def list_history_jobs(
    wr_id: int,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List job history for a specific work request session."""
    queue = _get_queue(db)
    result = queue.list_history_jobs(
        work_request_id=wr_id, status_filter=status,
        limit=min(limit, 100), offset=offset
    )
    return {"success": True, **result}


@router.delete("/api/v1/admin/jobs/permanently-failed")
def dismiss_permanently_failed(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Delete permanently failed jobs and clean up incomplete file records."""
    queue = _get_queue(db)
    result = queue.dismiss_permanently_failed_jobs()
    return {"success": True, **result}


@router.post("/api/v1/admin/jobs/cleanup")
def cleanup_queue(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Comprehensive job queue cleanup — remove completed, duplicate, and dangling jobs."""
    queue = _get_queue(db)
    result = queue.cleanup_queue()
    return {"success": True, **result}


# ── File Phase Status ────────────────────────────────────────

@router.post("/api/v1/files/phase-status")
def get_files_phase_status(
    req: PhaseStatusRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Return per-file MC/VV/MV presence status."""
    paths = req.file_paths[:500]  # limit
    status = db.get_files_phase_status(paths)
    return {"success": True, "status": status}


# ── Data Repair ──────────────────────────────────────────────

@router.post("/api/v1/admin/fix-relative-paths")
def fix_relative_paths(
    _admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Auto-fill missing relative_path from file_path (admin only)."""
    fixed = db.fix_missing_relative_paths()
    return {"success": True, "fixed": fixed}


# ── Work Request API ──────────────────────────────────────────

class ReorderRequest(BaseModel):
    ordered_ids: List[int]


@router.get("/api/v1/admin/work-requests")
def list_work_requests(
    include_completed: bool = False,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List work requests with counters."""
    queue = _get_queue(db)
    return {"success": True, "work_requests": queue.get_work_requests(include_completed)}


@router.get("/api/v1/admin/work-requests/{wr_id}")
def get_work_request(
    wr_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get work request detail with sub-tasks."""
    queue = _get_queue(db)
    detail = queue.get_work_request_detail(wr_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Work request not found")
    return {"success": True, **detail}


@router.put("/api/v1/admin/work-requests/order")
def reorder_work_requests(
    req: ReorderRequest,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Update work request processing order."""
    queue = _get_queue(db)
    queue.reorder_work_requests(req.ordered_ids)
    return {"success": True}


@router.post("/api/v1/admin/work-requests/{wr_id}/pause")
def pause_work_request(
    wr_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Pause a work request."""
    queue = _get_queue(db)
    if not queue.pause_work_request(wr_id):
        raise HTTPException(status_code=400, detail="Cannot pause this work request")
    return {"success": True}


@router.post("/api/v1/admin/work-requests/{wr_id}/resume")
def resume_work_request(
    wr_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Resume a paused work request."""
    queue = _get_queue(db)
    if not queue.resume_work_request(wr_id):
        raise HTTPException(status_code=400, detail="Cannot resume this work request")
    return {"success": True}


@router.post("/api/v1/admin/work-requests/{wr_id}/cancel")
def cancel_work_request(
    wr_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Cancel a work request and remove pending jobs."""
    queue = _get_queue(db)
    result = queue.cancel_work_request(wr_id)
    return {"success": True, **result}


# ── Helpers ──────────────────────────────────────────────────

def _decode_vector(encoded: Optional[str]) -> Optional[np.ndarray]:
    """Decode base64-encoded float32 vector, rejecting NaN/inf."""
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded)
        vec = np.frombuffer(raw, dtype=np.float32).copy()
        if not np.all(np.isfinite(vec)):
            logger.warning(f"Vector contains NaN/inf ({np.sum(~np.isfinite(vec))} bad values), rejected")
            return None
        return vec
    except Exception as e:
        logger.warning(f"Failed to decode vector: {e}")
        return None
