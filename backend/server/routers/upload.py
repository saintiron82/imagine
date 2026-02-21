"""
Upload router — image upload and download for distributed processing.
Thumbnails are stored on both server and client (dual storage).
"""

import logging
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user
from backend.server.config import get_storage_config
from backend.server.queue.manager import JobQueueManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])


def _get_upload_dir() -> Path:
    cfg = get_storage_config()
    upload_dir = Path(cfg.get("upload_dir", "./uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _get_thumbnail_dir() -> Path:
    cfg = get_storage_config()
    thumb_dir = Path(cfg.get("thumbnail_dir", "./thumbnails"))
    thumb_dir.mkdir(parents=True, exist_ok=True)
    return thumb_dir


def _get_max_file_size() -> int:
    cfg = get_storage_config()
    return cfg.get("max_file_size_mb", 500) * 1024 * 1024


# ── Image Upload ─────────────────────────────────────────────

@router.post("/images")
async def upload_images(
    files: List[UploadFile] = File(...),
    priority: int = Form(0),
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Upload image files to server. Creates processing jobs automatically."""
    upload_dir = _get_upload_dir()
    max_size = _get_max_file_size()

    results = []
    file_ids = []
    file_paths = []

    for upload_file in files:
        # Validate extension
        ext = Path(upload_file.filename or "").suffix.lower()
        if ext not in (".psd", ".png", ".jpg", ".jpeg"):
            results.append({
                "filename": upload_file.filename,
                "success": False,
                "error": f"Unsupported format: {ext}",
            })
            continue

        # Save to upload directory
        dest = upload_dir / upload_file.filename
        # Avoid overwriting: append numeric suffix
        counter = 1
        while dest.exists():
            stem = Path(upload_file.filename).stem
            dest = upload_dir / f"{stem}_{counter}{ext}"
            counter += 1

        try:
            content = await upload_file.read()
            if len(content) > max_size:
                results.append({
                    "filename": upload_file.filename,
                    "success": False,
                    "error": f"File too large (max {max_size // (1024*1024)}MB)",
                })
                continue

            dest.write_bytes(content)

            # Register in DB
            meta = {
                "file_path": str(dest),
                "file_name": dest.name,
                "file_size": len(content),
                "format": ext.lstrip(".").upper(),
            }
            fid = db.upsert_metadata(str(dest), meta)

            # Set uploaded_by
            db.conn.execute(
                "UPDATE files SET uploaded_by = ? WHERE id = ? AND uploaded_by IS NULL",
                (user["id"], fid)
            )
            db.conn.commit()

            file_ids.append(fid)
            file_paths.append(str(dest))

            results.append({
                "filename": upload_file.filename,
                "success": True,
                "file_id": fid,
                "stored_path": str(dest),
            })

        except Exception as e:
            logger.error(f"Upload failed for {upload_file.filename}: {e}")
            results.append({
                "filename": upload_file.filename,
                "success": False,
                "error": str(e),
            })

    # Create processing jobs
    queue = JobQueueManager(db)
    jobs_created = queue.create_jobs(file_ids, file_paths, priority) if file_ids else 0

    return {
        "success": True,
        "uploaded": len(file_ids),
        "jobs_created": jobs_created,
        "results": results,
    }


# ── Image Download (for workers) ─────────────────────────────

@router.get("/download/{file_id}")
def download_file(
    file_id: int,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Download original image for processing (worker must have an assigned job)."""
    cursor = db.conn.cursor()

    # Check if user has an assigned job for this file
    cursor.execute(
        """SELECT jq.id FROM job_queue jq
           WHERE jq.file_id = ? AND jq.assigned_to = ? AND jq.status IN ('assigned', 'processing')""",
        (file_id, user["id"])
    )
    if cursor.fetchone() is None:
        raise HTTPException(
            status_code=403,
            detail="No active job assignment for this file",
        )

    cursor.execute("SELECT file_path FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = Path(row[0])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(
        str(file_path),
        media_type="application/octet-stream",
        filename=file_path.name,
    )


# ── Thumbnail Upload (dual storage: server + client) ─────────

@router.post("/thumbnails/{file_id}")
async def upload_thumbnail(
    file_id: int,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Upload thumbnail for a file (server-side storage).

    Client also keeps a local copy. Server stores canonical version.
    """
    thumb_dir = _get_thumbnail_dir()
    cursor = db.conn.cursor()

    cursor.execute("SELECT file_name FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    # Save thumbnail
    file_name = row[0]
    stem = Path(file_name).stem
    thumb_path = thumb_dir / f"{stem}_thumb.png"

    content = await file.read()
    thumb_path.write_bytes(content)

    # Update DB
    cursor.execute(
        "UPDATE files SET thumbnail_url = ? WHERE id = ?",
        (str(thumb_path), file_id)
    )
    db.conn.commit()

    return {"success": True, "thumbnail_path": str(thumb_path)}
