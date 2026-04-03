"""
AnalysisJobManager — Analysis Job System v1.

Replaces the legacy JobQueueManager (manager.py) with a clean design:
- analysis_jobs: user-created folder analysis requests
- file_tasks: per-file phase tracking with independent MC/VV status

Key differences from legacy:
- No event counters (completed_count/failed_count) — all counts from queries
- No subtasks — flat file list per job
- No Recovery WRs — retry in same job
- MC and VV are independent (parallel possible)
- MV depends on MC completion
- Per-phase timestamps for speed measurement
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("analysis_manager")


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


class AnalysisJobManager:
    """Manages analysis jobs and file task lifecycle."""

    def __init__(self, db):
        self.db = db
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        schema_path = Path(__file__).parent.parent / "db" / "schema_analysis.sql"
        if schema_path.exists():
            self.db.conn.executescript(schema_path.read_text())

    # ── Analysis Job CRUD ────────────────────────────────────

    def create_job(self, name: str, source_path: str, file_paths: List[str],
                   created_by: int = None) -> Dict[str, Any]:
        """Create a new analysis job with file tasks.

        Args:
            name: Display name for the job
            source_path: Root folder path (webdav:// or local)
            file_paths: List of file paths to analyze
            created_by: User ID who created this job
        """
        cursor = self.db.conn.cursor()
        now = _now()

        # Create job
        cursor.execute(
            """INSERT INTO analysis_jobs (name, source_path, total_files, created_by, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (name, source_path, len(file_paths), created_by, now),
        )
        job_id = cursor.lastrowid

        # Create file tasks
        is_webdav = source_path.startswith("webdav://")
        created = 0
        for fp in file_paths:
            # Get or create file record
            file_id = self._ensure_file(cursor, fp)
            if file_id is None:
                continue

            dl_status = "pending" if is_webdav else "n/a"
            try:
                cursor.execute(
                    """INSERT OR IGNORE INTO file_tasks
                       (analysis_job_id, file_id, file_path, download_status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (job_id, file_id, fp, dl_status, now, now),
                )
                if cursor.rowcount > 0:
                    created += 1
            except Exception as e:
                logger.warning(f"Failed to create task for {fp}: {e}")

        # Update total to actual created count (minus duplicates)
        cursor.execute(
            "UPDATE analysis_jobs SET total_files = ? WHERE id = ?",
            (created, job_id),
        )
        self.db.conn.commit()

        logger.info(f"Analysis job '{name}' created: id={job_id}, files={created}")
        return {"job_id": job_id, "total_files": created}

    def _ensure_file(self, cursor, file_path: str) -> Optional[int]:
        """Get file_id from files table, creating if needed."""
        cursor.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        if row:
            return row[0]
        # Create minimal file record
        try:
            cursor.execute(
                "INSERT INTO files (file_path, file_name, created_at) VALUES (?, ?, ?)",
                (file_path, PurePosixPath(file_path).name, _now()),
            )
            return cursor.lastrowid
        except Exception:
            return None

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get a single analysis job with progress."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        job = dict(zip(cols, row))
        job["progress"] = self.get_progress(job_id)
        return job

    def list_jobs(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        """List analysis jobs with progress summaries."""
        cursor = self.db.conn.cursor()
        if include_completed:
            cursor.execute(
                "SELECT * FROM analysis_jobs ORDER BY created_at DESC"
            )
        else:
            cursor.execute(
                "SELECT * FROM analysis_jobs WHERE status IN ('active', 'paused') "
                "ORDER BY created_at DESC"
            )
        cols = [d[0] for d in cursor.description]
        jobs = []
        for row in cursor.fetchall():
            job = dict(zip(cols, row))
            job["progress"] = self.get_progress(job["id"])
            jobs.append(job)
        return jobs

    def pause_job(self, job_id: int) -> bool:
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE analysis_jobs SET status = 'paused' WHERE id = ? AND status = 'active'",
            (job_id,),
        )
        self.db.conn.commit()
        return cursor.rowcount > 0

    def resume_job(self, job_id: int) -> bool:
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE analysis_jobs SET status = 'active' WHERE id = ? AND status = 'paused'",
            (job_id,),
        )
        self.db.conn.commit()
        return cursor.rowcount > 0

    def cancel_job(self, job_id: int) -> bool:
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE analysis_jobs SET status = 'cancelled' WHERE id = ?",
            (job_id,),
        )
        self.db.conn.commit()
        return cursor.rowcount > 0

    # ── Progress Queries ─────────────────────────────────────

    def get_progress(self, job_id: int) -> Dict[str, Any]:
        """Get phase-level progress for an analysis job.

        All counts from single query — sum always equals total.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN download_status IN ('done', 'n/a') THEN 1 ELSE 0 END) AS downloaded,
                SUM(CASE WHEN parse_status = 'done' THEN 1 ELSE 0 END) AS parsed,
                SUM(CASE WHEN mc_status = 'done' THEN 1 ELSE 0 END) AS mc_done,
                SUM(CASE WHEN vv_status = 'done' THEN 1 ELSE 0 END) AS vv_done,
                SUM(CASE WHEN mv_status = 'done' THEN 1 ELSE 0 END) AS mv_done,
                SUM(CASE WHEN mc_status = 'done' AND vv_status = 'done' AND mv_status = 'done'
                    THEN 1 ELSE 0 END) AS complete,

                -- Phase breakdown: each file in exactly one phase (most behind)
                SUM(CASE WHEN download_status NOT IN ('done', 'n/a') THEN 1 ELSE 0 END) AS phase_download,
                SUM(CASE WHEN download_status IN ('done', 'n/a') AND parse_status != 'done'
                    THEN 1 ELSE 0 END) AS phase_parse,
                SUM(CASE WHEN parse_status = 'done'
                         AND (mc_status != 'done' OR vv_status != 'done')
                         AND NOT (mc_status = 'done' AND vv_status = 'done')
                    THEN 1 ELSE 0 END) AS phase_ai,
                SUM(CASE WHEN mc_status = 'done' AND vv_status = 'done' AND mv_status != 'done'
                    THEN 1 ELSE 0 END) AS phase_mv,

                -- Failures
                SUM(CASE WHEN download_status = 'failed' THEN 1 ELSE 0 END) AS download_failed,
                SUM(CASE WHEN parse_status = 'failed' THEN 1 ELSE 0 END) AS parse_failed,
                SUM(CASE WHEN mc_status = 'failed' THEN 1 ELSE 0 END) AS mc_failed,
                SUM(CASE WHEN vv_status = 'failed' THEN 1 ELSE 0 END) AS vv_failed,
                SUM(CASE WHEN mv_status = 'failed' THEN 1 ELSE 0 END) AS mv_failed
            FROM file_tasks
            WHERE analysis_job_id = ?
        """, (job_id,))
        r = cursor.fetchone()
        if not r or r[0] == 0:
            return {"total": 0}

        total = r[0]
        complete = r[6]
        pct = round(complete / total * 100, 1) if total > 0 else 0

        return {
            "total": total,
            "complete": complete,
            "pct": pct,
            "downloaded": r[1],
            "parsed": r[2],
            "mc_done": r[3],
            "vv_done": r[4],
            "mv_done": r[5],
            "phases": {
                "download": r[7],
                "parse": r[8],
                "ai": r[9],      # MC and/or VV pending
                "mv": r[10],
                "done": complete,
            },
            "failed": {
                "download": r[11],
                "parse": r[12],
                "mc": r[13],
                "vv": r[14],
                "mv": r[15],
            },
        }

    # ── Metrics ──────────────────────────────────────────────

    def get_metrics(self, job_id: int) -> Dict[str, Any]:
        """Get speed metrics for an analysis job."""
        cursor = self.db.conn.cursor()
        progress = self.get_progress(job_id)

        # Phase-level average times
        cursor.execute("""
            SELECT
                AVG(julianday(mc_completed_at) - julianday(mc_started_at)) * 86400,
                AVG(julianday(vv_completed_at) - julianday(vv_started_at)) * 86400,
                AVG(julianday(mv_completed_at) - julianday(mv_started_at)) * 86400,
                MAX(julianday(mc_completed_at) - julianday(mc_started_at)) * 86400,
                MAX(julianday(vv_completed_at) - julianday(vv_started_at)) * 86400,
                MAX(julianday(mv_completed_at) - julianday(mv_started_at)) * 86400
            FROM file_tasks
            WHERE analysis_job_id = ?
              AND mc_status = 'done' AND vv_status = 'done' AND mv_status = 'done'
        """, (job_id,))
        times = cursor.fetchone()

        # Recent throughput (5-min window — files fully completed)
        cursor.execute("""
            SELECT COUNT(*) FROM file_tasks
            WHERE analysis_job_id = ?
              AND mc_status = 'done' AND vv_status = 'done' AND mv_status = 'done'
              AND mv_completed_at > datetime('now', '-5 minutes')
        """, (job_id,))
        recent_5m = cursor.fetchone()[0]
        throughput = round(recent_5m / 5.0, 1) if recent_5m > 0 else 0

        remaining = progress["total"] - progress["complete"]
        eta_seconds = round(remaining / throughput * 60) if throughput > 0 else None

        # Bottleneck: phase with most pending items
        phases = progress.get("phases", {})
        bottleneck = max(
            [(k, v) for k, v in phases.items() if k != "done" and v > 0],
            key=lambda x: x[1], default=("none", 0)
        )[0]

        return {
            "progress": progress,
            "throughput": {
                "files_per_min": throughput,
                "eta_seconds": eta_seconds,
            },
            "phase_metrics": {
                "mc": {"avg_s": round(times[0] or 0, 2), "max_s": round(times[3] or 0, 2)},
                "vv": {"avg_s": round(times[1] or 0, 2), "max_s": round(times[4] or 0, 2)},
                "mv": {"avg_s": round(times[2] or 0, 2), "max_s": round(times[5] or 0, 2)},
            },
            "bottleneck": bottleneck,
        }

    # ── Task Claim (for workers) ─────────────────────────────

    def claim_tasks(self, phase: str, worker_id: int, count: int = 5
                    ) -> List[Dict[str, Any]]:
        """Claim pending tasks for a specific phase.

        Args:
            phase: 'download', 'parse', 'mc', 'vv', 'mv'
            worker_id: Worker session ID
            count: Max tasks to claim
        """
        cursor = self.db.conn.cursor()
        now = _now()

        # Phase-specific WHERE clause and columns
        claim_sql = {
            "download": (
                "download_status = 'pending'",
                "download_status", "download_assigned_to", "download_started_at"
            ),
            "parse": (
                "download_status IN ('done', 'n/a') AND parse_status = 'pending'",
                "parse_status", "parse_assigned_to", "parse_started_at"
            ),
            "mc": (
                "parse_status = 'done' AND mc_status = 'pending'",
                "mc_status", "mc_assigned_to", "mc_started_at"
            ),
            "vv": (
                "parse_status = 'done' AND vv_status = 'pending'",
                "vv_status", "vv_assigned_to", "vv_started_at"
            ),
            "mv": (
                "mc_status = 'done' AND mv_status = 'pending'",
                "mv_status", "mv_assigned_to", "mv_started_at"
            ),
        }

        if phase not in claim_sql:
            return []

        where, status_col, assigned_col, started_col = claim_sql[phase]

        # Find claimable tasks (only from active jobs)
        cursor.execute(f"""
            SELECT ft.id, ft.file_id, ft.file_path, ft.analysis_job_id
            FROM file_tasks ft
            JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
            WHERE aj.status = 'active'
              AND {where}
            ORDER BY ft.priority DESC, ft.created_at ASC
            LIMIT ?
        """, (count,))
        rows = cursor.fetchall()

        if not rows:
            return []

        # Assign to worker
        task_ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(task_ids))
        cursor.execute(f"""
            UPDATE file_tasks
            SET {status_col} = 'assigned',
                {assigned_col} = ?,
                {started_col} = ?,
                updated_at = ?
            WHERE id IN ({placeholders})
        """, [worker_id, now, now] + task_ids)
        self.db.conn.commit()

        return [
            {"task_id": r[0], "file_id": r[1], "file_path": r[2], "job_id": r[3]}
            for r in rows
        ]

    def complete_task_phase(self, task_id: int, phase: str,
                           success: bool, error_message: str = None):
        """Mark a phase as done or failed for a task.

        On success, triggers verification and next-phase notification.
        """
        cursor = self.db.conn.cursor()
        now = _now()

        status_col = f"{phase}_status"
        completed_col = f"{phase}_completed_at"

        new_status = "done" if success else "failed"
        cursor.execute(f"""
            UPDATE file_tasks
            SET {status_col} = ?,
                {completed_col} = ?,
                error_message = CASE WHEN ? IS NOT NULL THEN ? ELSE error_message END,
                updated_at = ?
            WHERE id = ?
        """, (new_status, now, error_message, error_message, now, task_id))
        self.db.conn.commit()

        # Verify result if successful
        if success:
            self._verify_phase_result(task_id, phase)

        # Check if entire job is complete
        self._check_job_completion(task_id)

    def _verify_phase_result(self, task_id: int, phase: str):
        """Verify actual data exists after phase completion."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT file_id FROM file_tasks WHERE id = ?", (task_id,)
        )
        row = cursor.fetchone()
        if not row:
            return
        file_id = row[0]

        checks = {
            "mc": "SELECT mc_caption FROM files WHERE id = ? AND mc_caption IS NOT NULL AND mc_caption != ''",
            "vv": "SELECT 1 FROM vec_files WHERE file_id = ?",
            "mv": "SELECT 1 FROM vec_text WHERE file_id = ?",
        }

        if phase in checks:
            cursor.execute(checks[phase], (file_id,))
            if not cursor.fetchone():
                # Data missing despite status=done → rollback
                status_col = f"{phase}_status"
                logger.warning(
                    f"Verify failed: task {task_id} phase {phase} "
                    f"done but data missing for file {file_id}"
                )
                cursor.execute(f"""
                    UPDATE file_tasks
                    SET {status_col} = 'failed',
                        error_message = 'Verification failed: data missing after completion',
                        updated_at = ?
                    WHERE id = ?
                """, (_now(), task_id))
                self.db.conn.commit()

    def _check_job_completion(self, task_id: int):
        """Check if the analysis job is fully complete."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT analysis_job_id FROM file_tasks WHERE id = ?",
            (task_id,),
        )
        row = cursor.fetchone()
        if not row:
            return
        job_id = row[0]

        # All tasks done?
        cursor.execute("""
            SELECT COUNT(*) FROM file_tasks
            WHERE analysis_job_id = ?
              AND NOT (mc_status = 'done' AND vv_status = 'done' AND mv_status = 'done')
              AND mc_status != 'failed' AND vv_status != 'failed' AND mv_status != 'failed'
        """, (job_id,))
        pending = cursor.fetchone()[0]

        if pending == 0:
            # All tasks either done or permanently failed
            cursor.execute(
                "UPDATE analysis_jobs SET status = 'completed', completed_at = ? WHERE id = ? AND status = 'active'",
                (_now(), job_id),
            )
            if cursor.rowcount > 0:
                logger.info(f"Analysis job {job_id} completed")
                self.db.conn.commit()

    # ── Retry Failed ─────────────────────────────────────────

    def retry_failed(self, job_id: int, phase: str = None) -> int:
        """Reset failed tasks to pending for retry.

        Args:
            job_id: Analysis job ID
            phase: Optional — retry only specific phase failures
        """
        cursor = self.db.conn.cursor()
        now = _now()

        if phase:
            status_col = f"{phase}_status"
            cursor.execute(f"""
                UPDATE file_tasks
                SET {status_col} = 'pending', error_message = NULL, updated_at = ?
                WHERE analysis_job_id = ? AND {status_col} = 'failed'
                  AND retry_count < max_retries
            """, (now, job_id))
        else:
            # Retry all phases
            for p in ("download", "parse", "mc", "vv", "mv"):
                col = f"{p}_status"
                cursor.execute(f"""
                    UPDATE file_tasks
                    SET {col} = 'pending', error_message = NULL, updated_at = ?
                    WHERE analysis_job_id = ? AND {col} = 'failed'
                      AND retry_count < max_retries
                """, (now, job_id))

        retried = cursor.rowcount
        self.db.conn.commit()

        # Re-activate job if it was completed
        if retried > 0:
            cursor.execute(
                "UPDATE analysis_jobs SET status = 'active', completed_at = NULL "
                "WHERE id = ? AND status = 'completed'",
                (job_id,),
            )
            self.db.conn.commit()

        return retried
