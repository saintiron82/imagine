"""
Job queue manager — work distribution for distributed processing.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)


class JobQueueManager:
    """Manages the job queue for distributed file processing."""

    def __init__(self, db: SQLiteDB):
        self.db = db

    def create_jobs(self, file_ids: List[int], file_paths: List[str], priority: int = 0) -> int:
        """Create pending jobs for files. Returns count of jobs created."""
        cursor = self.db.conn.cursor()
        created = 0
        for fid, fpath in zip(file_ids, file_paths):
            try:
                cursor.execute(
                    """INSERT INTO job_queue (file_id, file_path, status, priority)
                       VALUES (?, ?, 'pending', ?)
                       ON CONFLICT DO NOTHING""",
                    (fid, fpath, priority)
                )
                if cursor.rowcount > 0:
                    created += 1
            except Exception as e:
                logger.warning(f"Failed to create job for file_id={fid}: {e}")
        self.db.conn.commit()
        return created

    def claim_jobs(self, user_id: int, count: int = 10) -> List[Dict[str, Any]]:
        """Claim up to N pending jobs for a worker.

        Uses serialized access (SQLite single-writer) to avoid race conditions.
        """
        cursor = self.db.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Select pending jobs ordered by priority (desc) then age (asc)
        cursor.execute(
            """SELECT id, file_id, file_path, priority
               FROM job_queue
               WHERE status = 'pending'
               ORDER BY priority DESC, created_at ASC
               LIMIT ?""",
            (count,)
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        claimed = []
        for row in rows:
            job_id, file_id, file_path, priority = row
            cursor.execute(
                """UPDATE job_queue
                   SET status = 'assigned', assigned_to = ?, assigned_at = ?
                   WHERE id = ? AND status = 'pending'""",
                (user_id, now, job_id)
            )
            if cursor.rowcount > 0:
                claimed.append({
                    "job_id": job_id,
                    "file_id": file_id,
                    "file_path": file_path,
                    "priority": priority,
                })

        self.db.conn.commit()
        logger.info(f"User {user_id} claimed {len(claimed)} jobs")
        return claimed

    def update_progress(self, job_id: int, user_id: int, phase: str) -> bool:
        """Update phase completion for a job."""
        cursor = self.db.conn.cursor()

        # Verify ownership
        cursor.execute(
            "SELECT phase_completed FROM job_queue WHERE id = ? AND assigned_to = ?",
            (job_id, user_id)
        )
        row = cursor.fetchone()
        if row is None:
            return False

        phases = json.loads(row[0])
        if phase in phases:
            phases[phase] = True

        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """UPDATE job_queue
               SET phase_completed = ?, status = 'processing', started_at = COALESCE(started_at, ?)
               WHERE id = ?""",
            (json.dumps(phases), now, job_id)
        )
        self.db.conn.commit()
        return True

    def complete_job(self, job_id: int, user_id: int) -> bool:
        """Mark a job as completed."""
        cursor = self.db.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            """UPDATE job_queue
               SET status = 'completed', completed_at = ?,
                   phase_completed = '{"parse":true,"vision":true,"embed":true}'
               WHERE id = ? AND assigned_to = ?""",
            (now, job_id, user_id)
        )
        success = cursor.rowcount > 0
        self.db.conn.commit()
        return success

    def fail_job(self, job_id: int, user_id: int, error_message: str) -> bool:
        """Mark a job as failed. May be retried."""
        cursor = self.db.conn.cursor()

        # Check retry count
        cursor.execute(
            "SELECT retry_count, max_retries FROM job_queue WHERE id = ? AND assigned_to = ?",
            (job_id, user_id)
        )
        row = cursor.fetchone()
        if row is None:
            return False

        retry_count, max_retries = row
        new_status = "pending" if retry_count < max_retries else "failed"

        cursor.execute(
            """UPDATE job_queue
               SET status = ?, error_message = ?,
                   retry_count = retry_count + 1,
                   assigned_to = NULL, assigned_at = NULL
               WHERE id = ?""",
            (new_status, error_message, job_id)
        )
        self.db.conn.commit()
        if new_status == "pending":
            logger.info(f"Job {job_id} will be retried (attempt {retry_count + 1}/{max_retries})")
        else:
            logger.warning(f"Job {job_id} permanently failed: {error_message}")
        return True

    def get_stale_jobs(self, timeout_minutes: int = 30) -> List[int]:
        """Find jobs that have been assigned but not progressed within timeout."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id FROM job_queue
               WHERE status IN ('assigned', 'processing')
                 AND assigned_at IS NOT NULL
                 AND datetime(assigned_at, '+' || ? || ' minutes') < datetime('now')""",
            (timeout_minutes,)
        )
        return [row[0] for row in cursor.fetchall()]

    def reassign_stale_jobs(self, timeout_minutes: int = 30) -> int:
        """Reset stale jobs back to pending for reassignment."""
        stale_ids = self.get_stale_jobs(timeout_minutes)
        if not stale_ids:
            return 0

        cursor = self.db.conn.cursor()
        placeholders = ",".join("?" * len(stale_ids))
        cursor.execute(
            f"""UPDATE job_queue
                SET status = 'pending', assigned_to = NULL, assigned_at = NULL
                WHERE id IN ({placeholders})""",
            stale_ids
        )
        self.db.conn.commit()
        logger.info(f"Reassigned {cursor.rowcount} stale jobs")
        return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Get job queue statistics."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT
                status,
                COUNT(*) as count
            FROM job_queue
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM job_queue")
        total = cursor.fetchone()[0]

        return {
            "total": total,
            "pending": status_counts.get("pending", 0),
            "assigned": status_counts.get("assigned", 0),
            "processing": status_counts.get("processing", 0),
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
        }

    def get_user_jobs(self, user_id: int) -> List[Dict[str, Any]]:
        """Get jobs assigned to or completed by a user."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, file_id, file_path, status, phase_completed,
                      assigned_at, started_at, completed_at, error_message
               FROM job_queue
               WHERE assigned_to = ?
               ORDER BY created_at DESC
               LIMIT 100""",
            (user_id,)
        )
        return [
            {
                "job_id": row[0], "file_id": row[1], "file_path": row[2],
                "status": row[3], "phase_completed": json.loads(row[4] or "{}"),
                "assigned_at": row[5], "started_at": row[6],
                "completed_at": row[7], "error_message": row[8],
            }
            for row in cursor.fetchall()
        ]
