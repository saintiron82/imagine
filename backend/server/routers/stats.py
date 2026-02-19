"""
Stats router — database statistics endpoints.
Wraps existing SQLiteDB.get_stats() etc.
"""

from fastapi import APIRouter, Depends

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/db")
def get_db_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Get database statistics (total files, MC/VV/MV counts, format distribution)."""
    stats = db.get_stats()
    return {"success": True, **stats}


@router.get("/incomplete")
def get_incomplete_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Get incomplete file stats grouped by storage_root."""
    stats = db.get_incomplete_stats()
    return {"success": True, **stats}


@router.get("/folders/{root_path:path}")
def get_folder_phase_stats(
    root_path: str,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Get per-folder phase completion stats under root_path prefix."""
    stats = db.get_folder_phase_stats(root_path)
    return {"success": True, "folders": stats}
