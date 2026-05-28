"""Sprint 2 γ2: admin-visible aggregation of search_feedback."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db_safe, require_admin

router = APIRouter(tags=["search-feedback-admin"])


@router.get("/admin/search-feedback/summary")
def search_feedback_summary(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    cur = db.conn.cursor()

    total_30d = cur.execute(
        """SELECT COUNT(*) FROM search_feedback
           WHERE datetime(created_at) >= datetime('now', '-30 days')"""
    ).fetchone()[0]

    top_files_rows = cur.execute(
        """SELECT file_id, COUNT(*) AS n FROM search_feedback
           WHERE datetime(created_at) >= datetime('now', '-30 days')
           GROUP BY file_id
           ORDER BY n DESC, file_id ASC
           LIMIT 20"""
    ).fetchall()
    top_files = [{"file_id": r[0], "count": r[1]} for r in top_files_rows]

    top_queries_rows = cur.execute(
        """SELECT query, COUNT(*) AS n FROM search_feedback
           WHERE datetime(created_at) >= datetime('now', '-30 days')
           GROUP BY query
           ORDER BY n DESC, query ASC
           LIMIT 10"""
    ).fetchall()
    top_queries = [{"query": r[0], "count": r[1]} for r in top_queries_rows]

    return {
        "total_30d": int(total_30d or 0),
        "top_files": top_files,
        "top_queries": top_queries,
    }
