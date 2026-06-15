"""Database admin operations — reset / export / import with password verification."""

import logging
import os
import tempfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import require_admin, get_db_safe
from backend.server.security import audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/database", tags=["database"])

_MAX_IMPORT_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB guard


class ResetRequest(BaseModel):
    password: str


def _verify_admin_password(db: SQLiteDB, admin: dict, password: str) -> None:
    """Re-verify the admin's password for destructive ops; raise 403 if wrong."""
    from backend.server.auth.router import _verify_password
    cursor = db.conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE id = ?", (admin["id"],))
    row = cursor.fetchone()
    if not row or not _verify_password(password, row[0]):
        raise HTTPException(status_code=403, detail="Invalid password")


@router.post("/reset")
def reset_database(
    req: ResetRequest,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Reset all file data. Requires admin password re-verification."""
    _verify_admin_password(db, admin, req.password)

    result = db.reset_file_data()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Reset failed"))

    audit_log.record(
        db, "database_reset",
        actor_user_id=admin.get("id"), actor_username=admin.get("username"),
        target_kind="database", detail=f"files={result.get('files')} vectors={result.get('vectors')}",
    )
    return result


@router.get("/export")
def export_database(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Stream a consistent full snapshot of the database (IMGV2-48). Admin only."""
    fd, tmp = tempfile.mkstemp(suffix=".db", prefix="imagine-export-")
    os.close(fd)
    os.unlink(tmp)  # VACUUM INTO requires the target file to NOT exist

    result = db.export_snapshot(tmp)
    if not result["success"]:
        try:
            os.path.exists(tmp) and os.unlink(tmp)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=result.get("error", "Export failed"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"imagine-backup-{stamp}.db"
    logger.info("Admin %s exported DB snapshot (%s bytes)", admin.get("username"), result.get("bytes"))
    audit_log.record(
        db, "database_exported",
        actor_user_id=admin.get("id"), actor_username=admin.get("username"),
        target_kind="database", detail=f"bytes={result.get('bytes')} files={result.get('file_count')}",
    )
    return FileResponse(
        tmp,
        media_type="application/x-sqlite3",
        filename=filename,
        background=BackgroundTask(lambda: os.path.exists(tmp) and os.unlink(tmp)),
    )


@router.post("/import")
async def import_database(
    password: str = Form(...),
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Restore the database from an uploaded snapshot (IMGV2-48). DESTRUCTIVE.

    Re-verifies the admin password (same gate as reset) and validates that the
    upload is intact and carries our schema before overwriting the live database.

    Caveats (surfaced to the operator): the restore replaces ALL data including
    users/auth — if the snapshot lacks your account you must re-login with the
    restored credentials afterward. The online backup needs an exclusive write,
    so run it with processing paused. A pre-import copy is kept at
    `<db>.preimport.bak` for manual recovery.
    """
    _verify_admin_password(db, admin, password)

    fd, tmp = tempfile.mkstemp(suffix=".db", prefix="imagine-import-")
    try:
        total = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_IMPORT_BYTES:
                    raise HTTPException(status_code=413, detail="Upload too large")
                out.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty upload")

        result = db.import_snapshot(tmp)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Import failed"))
        logger.info("Admin %s imported DB snapshot (%s files)", admin.get("username"), result.get("file_count"))
        audit_log.record(
            db, "database_imported",
            actor_user_id=admin.get("id"), actor_username=admin.get("username"),
            target_kind="database", detail=f"files={result.get('file_count')} bytes={total}",
        )
        return result
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
