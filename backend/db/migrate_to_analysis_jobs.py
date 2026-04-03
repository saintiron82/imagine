#!/usr/bin/env python3
"""
Migrate legacy queue data to Analysis Job System v1.

Converts:
  work_requests (non-Recovery) → analysis_jobs
  job_queue → file_tasks (with phase status derived from files DB)

Safe to run multiple times (uses INSERT OR IGNORE).
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def migrate(db):
    """Run migration from legacy tables to analysis_jobs + file_tasks."""
    cursor = db.conn.cursor()

    # Ensure new tables exist
    schema_path = Path(__file__).parent / "schema_analysis.sql"
    if schema_path.exists():
        db.conn.executescript(schema_path.read_text())
        logger.info("Analysis tables created/verified")

    # ── Step 1: Migrate work_requests → analysis_jobs ──
    cursor.execute("""
        SELECT id, name, source_path, status, total_files, created_by, created_at, completed_at
        FROM work_requests
        WHERE name NOT LIKE '[Recovery]%'
        ORDER BY id
    """)
    wrs = cursor.fetchall()
    migrated_jobs = 0

    for wr in wrs:
        wr_id, name, source_path, status, total, created_by, created_at, completed_at = wr

        # Map status
        new_status = {
            "queued": "active", "processing": "active",
            "paused": "paused", "completed": "completed",
            "cancelled": "cancelled",
        }.get(status, "active")

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO analysis_jobs
                (id, name, source_path, status, total_files, created_by, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (wr_id, name, source_path, new_status, total, created_by, created_at, completed_at))
            if cursor.rowcount > 0:
                migrated_jobs += 1
        except Exception as e:
            logger.warning(f"Skip WR-{wr_id}: {e}")

    logger.info(f"Migrated {migrated_jobs} work_requests → analysis_jobs")

    # ── Step 2: Migrate files → file_tasks ──
    # For each analysis_job, find files under its source_path
    cursor.execute("SELECT id, source_path FROM analysis_jobs")
    jobs = cursor.fetchall()
    migrated_tasks = 0

    for job_id, source_path in jobs:
        if not source_path:
            continue

        # Find files under this path (don't mangle existing ://)
        like_path = source_path.rstrip("/") + "%"
        cursor.execute(
            "SELECT id, file_path FROM files WHERE file_path LIKE ?",
            (like_path,),
        )
        files = cursor.fetchall()

        is_webdav = source_path.startswith("webdav://")

        for file_id, file_path in files:
            # Determine phase status from actual data in files DB
            cursor.execute("""
                SELECT
                    mc_caption IS NOT NULL AND mc_caption != '' AS has_mc,
                    EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) AS has_vv,
                    EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) AS has_mv,
                    thumbnail_url IS NOT NULL AS has_thumb
                FROM files f WHERE f.id = ?
            """, (file_id,))
            row = cursor.fetchone()
            has_mc, has_vv, has_mv, has_thumb = row if row else (0, 0, 0, 0)

            dl_status = "done" if (not is_webdav or has_thumb) else "n/a" if not is_webdav else "pending"
            parse_status = "done" if has_thumb else "pending"
            mc_status = "done" if has_mc else "pending"
            vv_status = "done" if has_vv else "pending"
            mv_status = "done" if has_mv else ("pending" if has_mc else "pending")

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO file_tasks
                    (analysis_job_id, file_id, file_path,
                     download_status, parse_status, mc_status, vv_status, mv_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (job_id, file_id, file_path,
                      dl_status, parse_status, mc_status, vv_status, mv_status))
                if cursor.rowcount > 0:
                    migrated_tasks += 1
            except Exception as e:
                logger.warning(f"Skip file {file_id}: {e}")

        # Update total_files to actual count
        cursor.execute(
            "UPDATE analysis_jobs SET total_files = (SELECT COUNT(*) FROM file_tasks WHERE analysis_job_id = ?) WHERE id = ?",
            (job_id, job_id),
        )

    db.conn.commit()
    logger.info(f"Migrated {migrated_tasks} files → file_tasks")

    # ── Step 3: Re-evaluate job completion ──
    cursor.execute("SELECT id FROM analysis_jobs WHERE status = 'completed'")
    for (job_id,) in cursor.fetchall():
        # Check if truly complete
        cursor.execute("""
            SELECT COUNT(*) FROM file_tasks
            WHERE analysis_job_id = ?
              AND NOT (mc_status = 'done' AND vv_status = 'done' AND mv_status = 'done')
        """, (job_id,))
        incomplete = cursor.fetchone()[0]
        if incomplete > 0:
            cursor.execute(
                "UPDATE analysis_jobs SET status = 'active', completed_at = NULL WHERE id = ?",
                (job_id,),
            )
            logger.info(f"Job {job_id}: reopened ({incomplete} incomplete files)")

    db.conn.commit()
    logger.info("Migration complete")


if __name__ == "__main__":
    from backend.db.sqlite_client import SQLiteDB
    db = SQLiteDB()
    migrate(db)
