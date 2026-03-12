"""
Stats router — database statistics endpoints.
Wraps existing SQLiteDB.get_stats() etc.
"""

from fastapi import APIRouter, Depends

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, get_current_user, require_admin

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/db")
def get_db_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get database statistics (total files, MC/VV/MV counts, format distribution)."""
    stats = db.get_stats()
    return {"success": True, **stats}


@router.get("/incomplete")
def get_incomplete_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get incomplete file stats grouped by storage_root."""
    stats = db.get_incomplete_stats()
    return {"success": True, **stats}


@router.get("/folders/{root_path:path}")
def get_folder_phase_stats(
    root_path: str,
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get per-folder phase completion stats under root_path prefix."""
    stats = db.get_folder_phase_stats(root_path)
    return {"success": True, "folders": stats}


@router.get("/thumbnails")
def get_thumbnail_stats(
    _user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Get thumbnail availability stats grouped by storage_root."""
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT
            COALESCE(storage_root, 'local') AS source,
            COUNT(*) AS total,
            SUM(CASE WHEN thumbnail_url IS NOT NULL AND thumbnail_url != '' THEN 1 ELSE 0 END) AS has_thumb,
            SUM(CASE WHEN thumbnail_url IS NULL OR thumbnail_url = '' THEN 1 ELSE 0 END) AS missing
        FROM files
        GROUP BY COALESCE(storage_root, 'local')
        ORDER BY total DESC
    """)
    rows = cursor.fetchall()
    sources = []
    grand_total = 0
    grand_has = 0
    grand_missing = 0
    for row in rows:
        src = {
            "source": row[0],
            "total": row[1],
            "has_thumb": row[2],
            "missing": row[3],
        }
        sources.append(src)
        grand_total += row[1]
        grand_has += row[2]
        grand_missing += row[3]
    return {
        "success": True,
        "sources": sources,
        "grand_total": grand_total,
        "grand_has_thumb": grand_has,
        "grand_missing": grand_missing,
    }
