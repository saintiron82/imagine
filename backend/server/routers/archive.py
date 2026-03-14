"""
Archive browse router — folder listing, file listing with phase status.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db_safe, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/archive", tags=["archive"])


@router.get("/folders")
def list_archive_folders(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List distinct folder_path values with file counts and phase stats."""
    folders = db.get_archive_folders()
    return {"success": True, "folders": folders}


@router.get("/files")
def list_archive_files(
    folder_path: Optional[str] = Query(None),
    image_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List files for archive browsing with phase status badges."""
    result = db.get_archive_files(
        folder_path=folder_path,
        image_type=image_type,
        limit=limit,
        offset=offset,
    )
    return {"success": True, **result}


@router.get("/image-types")
def list_image_types(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get image_type distribution across all files."""
    types = db.get_image_type_stats()
    return {"success": True, "types": types}
