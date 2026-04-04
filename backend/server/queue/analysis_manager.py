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
        self._ensure_elapsed_columns()
        self._ensure_snapshot_columns()
        self._fix_check_constraint()
        self._reclaim_stale_assigned()
        self._sync_with_files_db()

    def _reclaim_stale_assigned(self):
        """Reset assigned tasks back to pending on startup.

        Prevents tasks from being permanently stuck in 'assigned' state
        after worker crashes or server restarts.
        """
        cursor = self.db.conn.cursor()
        total = 0
        for phase in ("download", "parse", "mc", "vv", "mv"):
            col = f"{phase}_status"
            assigned_col = f"{phase}_assigned_to"
            cursor.execute(f"""
                UPDATE file_tasks
                SET {col} = 'pending', {assigned_col} = NULL
                WHERE {col} = 'assigned'
            """)
            total += cursor.rowcount
        if total > 0:
            self.db.conn.commit()
            logger.info(f"Reclaimed {total} stale assigned tasks → pending")

    def reclaim_worker_tasks(self, worker_id: int) -> int:
        """Reclaim tasks assigned to a specific worker back to pending.

        Called when a worker disconnects or is blocked.
        """
        cursor = self.db.conn.cursor()
        total = 0
        for phase in ("mc", "vv", "mv"):
            col = f"{phase}_status"
            assigned_col = f"{phase}_assigned_to"
            cursor.execute(f"""
                UPDATE file_tasks
                SET {col} = 'pending', {assigned_col} = NULL
                WHERE {col} = 'assigned' AND {assigned_col} = ?
            """, (worker_id,))
            total += cursor.rowcount
        if total > 0:
            self.db.conn.commit()
            logger.info(f"Reclaimed {total} tasks from worker {worker_id} → pending")
        else:
            self.db.conn.commit()
        return total

    def _sync_with_files_db(self):
        """Sync file_tasks status with actual files DB state.

        Handles cases where legacy pipeline processed files but
        file_tasks wasn't updated (e.g. legacy system ran first).
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
            UPDATE file_tasks SET
                download_status = CASE
                    WHEN download_status IN ('pending', 'assigned') AND
                         (SELECT thumbnail_url FROM files WHERE id = file_tasks.file_id) IS NOT NULL
                    THEN 'done' ELSE download_status END,
                parse_status = CASE
                    WHEN parse_status IN ('pending', 'assigned') AND
                         (SELECT thumbnail_url FROM files WHERE id = file_tasks.file_id) IS NOT NULL
                    THEN 'done' ELSE parse_status END,
                mc_status = CASE
                    WHEN mc_status IN ('pending', 'assigned') AND
                         (SELECT mc_caption FROM files WHERE id = file_tasks.file_id) IS NOT NULL AND
                         (SELECT mc_caption FROM files WHERE id = file_tasks.file_id) != ''
                    THEN 'done' ELSE mc_status END,
                vv_status = CASE
                    WHEN vv_status IN ('pending', 'assigned') AND
                         EXISTS(SELECT 1 FROM vec_files WHERE file_id = file_tasks.file_id)
                    THEN 'done' ELSE vv_status END,
                mv_status = CASE
                    WHEN mv_status IN ('pending', 'assigned') AND
                         EXISTS(SELECT 1 FROM vec_text WHERE file_id = file_tasks.file_id)
                    THEN 'done' ELSE mv_status END
        """)
        synced = cursor.rowcount
        if synced > 0:
            self.db.conn.commit()
            logger.info(f"Synced {synced} file_tasks with files DB")

    def _ensure_snapshot_columns(self):
        """Add completion snapshot columns to analysis_jobs if missing."""
        cursor = self.db.conn.cursor()
        columns = {
            "completed_files": "INTEGER",
            "avg_mc_speed": "REAL",
            "avg_vv_speed": "REAL",
            "avg_mv_speed": "REAL",
            "total_elapsed_s": "REAL",
            "worker_count": "INTEGER",
            "started_at": "TEXT",
            "worker_stats_json": "TEXT",
        }
        for col, typedef in columns.items():
            try:
                cursor.execute(f"SELECT {col} FROM analysis_jobs LIMIT 1")
            except Exception:
                try:
                    cursor.execute(
                        f"ALTER TABLE analysis_jobs ADD COLUMN {col} {typedef}"
                    )
                    self.db.conn.commit()
                except Exception:
                    pass

    def _ensure_elapsed_columns(self):
        """Add elapsed_s columns if missing."""
        cursor = self.db.conn.cursor()
        for col in ("parse_elapsed_s", "mc_elapsed_s", "vv_elapsed_s", "mv_elapsed_s"):
            try:
                cursor.execute(f"SELECT {col} FROM file_tasks LIMIT 1")
            except Exception:
                cursor.execute(f"ALTER TABLE file_tasks ADD COLUMN {col} REAL")
                self.db.conn.commit()

    def _fix_check_constraint(self):
        """Ensure analysis_jobs CHECK includes 'archived'."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE name='analysis_jobs'")
            row = cursor.fetchone()
            if row and "'archived'" not in (row[0] or ""):
                cursor.execute("PRAGMA writable_schema = ON")
                cursor.execute("""
                    UPDATE sqlite_master SET sql = REPLACE(sql,
                        "status IN ('active', 'paused', 'completed', 'cancelled')",
                        "status IN ('active', 'paused', 'completed', 'cancelled', 'archived')")
                    WHERE name = 'analysis_jobs'
                """)
                cursor.execute("PRAGMA writable_schema = OFF")
                self.db.conn.commit()
        except Exception:
            pass

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
                "INSERT INTO files (file_path, file_name, relative_path, created_at) VALUES (?, ?, ?, ?)",
                (file_path, PurePosixPath(file_path).name, file_path, _now()),
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
                "SELECT * FROM analysis_jobs WHERE status != 'archived' ORDER BY created_at DESC"
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

    def archive_job(self, job_id: int) -> bool:
        """Archive a job — hidden from worker dashboard, visible in history."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            "UPDATE analysis_jobs SET status = 'archived' WHERE id = ?",
            (job_id,),
        )
        self.db.conn.commit()
        return cursor.rowcount > 0

    def delete_job(self, job_id: int) -> bool:
        """Permanently delete a job and its file_tasks (CASCADE).

        files/vec_files/vec_text are NOT affected — search data persists.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM analysis_jobs WHERE id = ?", (job_id,))
        deleted = cursor.rowcount > 0
        self.db.conn.commit()
        if deleted:
            logger.info(f"Analysis job {job_id} permanently deleted (file_tasks cascaded)")
        return deleted

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

        # Phase-level times from worker-measured elapsed_s (precise, not datetime)
        cursor.execute("""
            SELECT AVG(mc_elapsed_s), MAX(mc_elapsed_s)
            FROM file_tasks WHERE analysis_job_id = ? AND mc_status = 'done' AND mc_elapsed_s IS NOT NULL
        """, (job_id,))
        mc_times = cursor.fetchone() or (0, 0)

        cursor.execute("""
            SELECT AVG(vv_elapsed_s), MAX(vv_elapsed_s)
            FROM file_tasks WHERE analysis_job_id = ? AND vv_status = 'done' AND vv_elapsed_s IS NOT NULL
        """, (job_id,))
        vv_times = cursor.fetchone() or (0, 0)

        cursor.execute("""
            SELECT AVG(mv_elapsed_s), MAX(mv_elapsed_s)
            FROM file_tasks WHERE analysis_job_id = ? AND mv_status = 'done' AND mv_elapsed_s IS NOT NULL
        """, (job_id,))
        mv_times = cursor.fetchone() or (0, 0)

        # Per-phase recent throughput (5-min window)
        phase_tput = {}
        for phase, col in [("mc", "mc_completed_at"), ("vv", "vv_completed_at"), ("mv", "mv_completed_at")]:
            cursor.execute(f"""
                SELECT COUNT(*) FROM file_tasks
                WHERE analysis_job_id = ? AND {phase}_status = 'done'
                  AND {col} > datetime('now', '-5 minutes')
            """, (job_id,))
            cnt = cursor.fetchone()[0]
            phase_tput[phase] = round(cnt / 5.0, 1) if cnt > 0 else 0

        # Overall throughput = bottleneck phase speed (slowest active)
        active_speeds = {k: v for k, v in phase_tput.items() if v > 0}
        # Phase-level ETA: remaining per phase / phase speed
        total = progress.get("total", 0)
        mc_remaining = total - progress.get("mc_done", 0)
        vv_remaining = total - progress.get("vv_done", 0)
        mv_remaining = total - progress.get("mv_done", 0)

        phase_etas = {}  # minutes
        if phase_tput.get("mc", 0) > 0 and mc_remaining > 0:
            phase_etas["mc"] = mc_remaining / phase_tput["mc"]
        if phase_tput.get("vv", 0) > 0 and vv_remaining > 0:
            phase_etas["vv"] = vv_remaining / phase_tput["vv"]
        if phase_tput.get("mv", 0) > 0 and mv_remaining > 0:
            phase_etas["mv"] = mv_remaining / phase_tput["mv"]

        if phase_etas:
            bottleneck = max(phase_etas, key=phase_etas.get)
            eta_seconds = round(phase_etas[bottleneck] * 60)
            throughput = phase_tput.get(bottleneck, 0)
        else:
            bottleneck = "none"
            eta_seconds = None
            throughput = min(active_speeds.values()) if active_speeds else 0

        return {
            "progress": progress,
            "throughput": {
                "files_per_min": throughput,
                "eta_seconds": eta_seconds,
            },
            "phase_metrics": {
                "mc": {"avg_s": round(mc_times[0] or 0, 2), "max_s": round(mc_times[1] or 0, 2), "fpm": phase_tput.get("mc", 0)},
                "vv": {"avg_s": round(vv_times[0] or 0, 2), "max_s": round(vv_times[1] or 0, 2), "fpm": phase_tput.get("vv", 0)},
                "mv": {"avg_s": round(mv_times[0] or 0, 2), "max_s": round(mv_times[1] or 0, 2), "fpm": phase_tput.get("mv", 0)},
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
        # Workers only claim AI phases (mc/vv/mv).
        # Download is handled by DownloadAheadPool, parse by FileTaskParsePool.
        claim_sql = {
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
                updated_at = ?
            WHERE id IN ({placeholders})
        """, [worker_id, now] + task_ids)

        # Mark job started_at (first claim ever)
        job_ids = set(r[3] for r in rows)
        for jid in job_ids:
            cursor.execute(
                "UPDATE analysis_jobs SET started_at = ? WHERE id = ? AND started_at IS NULL",
                (now, jid),
            )

        self.db.conn.commit()

        return [
            {"task_id": r[0], "file_id": r[1], "file_path": r[2], "job_id": r[3]}
            for r in rows
        ]

    def start_task_phase(self, task_id: int, phase: str):
        """Record actual processing start time (not claim time)."""
        cursor = self.db.conn.cursor()
        started_col = f"{phase}_started_at"
        cursor.execute(f"""
            UPDATE file_tasks SET {started_col} = ? WHERE id = ? AND {started_col} IS NULL
        """, (_now(), task_id))
        self.db.conn.commit()

    def complete_task_phase(self, task_id: int, phase: str,
                           success: bool, error_message: str = None,
                           elapsed_s: float = None):
        """Mark a phase as done or failed for a task.

        Args:
            elapsed_s: Actual processing time measured by worker (seconds).
                       If provided, started_at is back-calculated from now - elapsed_s.
        """
        cursor = self.db.conn.cursor()
        now = _now()
        completed_col = f"{phase}_completed_at"
        status_col = f"{phase}_status"

        new_status = "done" if success else "failed"
        elapsed_col = f"{phase}_elapsed_s"
        cursor.execute(f"""
            UPDATE file_tasks
            SET {status_col} = ?,
                {completed_col} = ?,
                {elapsed_col} = ?,
                error_message = CASE WHEN ? IS NOT NULL THEN ? ELSE error_message END,
                updated_at = ?
            WHERE id = ?
        """, (new_status, now, elapsed_s, error_message, error_message, now, task_id))
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
            "parse": "SELECT thumbnail_url FROM files WHERE id = ? AND thumbnail_url IS NOT NULL AND thumbnail_url != ''",
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
        """Check if the analysis job is fully complete. Write snapshot on completion."""
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
            now = _now()
            # Write completion snapshot
            cursor.execute("""
                UPDATE analysis_jobs SET
                    status = 'completed',
                    completed_at = ?,
                    completed_files = (
                        SELECT COUNT(*) FROM file_tasks
                        WHERE analysis_job_id = ?
                          AND mc_status = 'done' AND vv_status = 'done' AND mv_status = 'done'
                    ),
                    avg_mc_speed = (
                        SELECT CASE WHEN AVG(mc_elapsed_s) > 0
                            THEN ROUND(60.0 / AVG(mc_elapsed_s), 1) END
                        FROM file_tasks
                        WHERE analysis_job_id = ? AND mc_elapsed_s > 0
                    ),
                    avg_vv_speed = (
                        SELECT CASE WHEN AVG(vv_elapsed_s) > 0
                            THEN ROUND(60.0 / AVG(vv_elapsed_s), 1) END
                        FROM file_tasks
                        WHERE analysis_job_id = ? AND vv_elapsed_s > 0
                    ),
                    avg_mv_speed = (
                        SELECT CASE WHEN AVG(mv_elapsed_s) > 0
                            THEN ROUND(60.0 / AVG(mv_elapsed_s), 1) END
                        FROM file_tasks
                        WHERE analysis_job_id = ? AND mv_elapsed_s > 0
                    ),
                    total_elapsed_s = (
                        SELECT ROUND(
                            (julianday(?) - julianday(started_at)) * 86400, 1
                        ) FROM analysis_jobs WHERE id = ?
                    ),
                    worker_count = (
                        SELECT COUNT(DISTINCT w) FROM (
                            SELECT mc_assigned_to AS w FROM file_tasks
                                WHERE analysis_job_id = ? AND mc_assigned_to IS NOT NULL
                            UNION
                            SELECT vv_assigned_to FROM file_tasks
                                WHERE analysis_job_id = ? AND vv_assigned_to IS NOT NULL
                            UNION
                            SELECT mv_assigned_to FROM file_tasks
                                WHERE analysis_job_id = ? AND mv_assigned_to IS NOT NULL
                        )
                    )
                WHERE id = ? AND status = 'active'
            """, (now, job_id, job_id, job_id, job_id, now, job_id, job_id, job_id, job_id, job_id))

            if cursor.rowcount > 0:
                # Build per-worker breakdown JSON
                self._write_worker_stats(cursor, job_id)
                self.db.conn.commit()
                logger.info(f"Analysis job {job_id} completed (snapshot written)")

    def _write_worker_stats(self, cursor, job_id: int):
        """Build per-worker breakdown and store as JSON in analysis_jobs."""
        import json as _json

        # Collect all unique workers that touched this job
        cursor.execute("""
            SELECT DISTINCT w, phase FROM (
                SELECT mc_assigned_to AS w, 'mc' AS phase FROM file_tasks
                    WHERE analysis_job_id = ? AND mc_assigned_to IS NOT NULL
                UNION ALL
                SELECT vv_assigned_to, 'vv' FROM file_tasks
                    WHERE analysis_job_id = ? AND vv_assigned_to IS NOT NULL
                UNION ALL
                SELECT mv_assigned_to, 'mv' FROM file_tasks
                    WHERE analysis_job_id = ? AND mv_assigned_to IS NOT NULL
            )
        """, (job_id, job_id, job_id))

        worker_phases = {}  # {worker_id: set of phases}
        for wid, phase in cursor.fetchall():
            worker_phases.setdefault(wid, set()).add(phase)

        stats = []
        for wid in worker_phases:
            entry = {"worker_id": wid}

            # Worker name from worker_sessions
            cursor.execute(
                "SELECT worker_name FROM worker_sessions WHERE id = ?", (wid,)
            )
            name_row = cursor.fetchone()
            entry["name"] = name_row[0] if name_row else f"worker-{wid}"

            # Per-phase: count + avg speed
            for phase in ("mc", "vv", "mv"):
                assigned_col = f"{phase}_assigned_to"
                elapsed_col = f"{phase}_elapsed_s"
                cursor.execute(f"""
                    SELECT COUNT(*),
                           CASE WHEN AVG({elapsed_col}) > 0
                               THEN ROUND(60.0 / AVG({elapsed_col}), 1) END
                    FROM file_tasks
                    WHERE analysis_job_id = ? AND {assigned_col} = ? AND {phase}_status = 'done'
                """, (job_id, wid))
                row = cursor.fetchone()
                entry[f"{phase}_count"] = row[0] or 0
                entry[f"{phase}_speed"] = row[1]  # files/min, None if no data

            stats.append(entry)

        cursor.execute(
            "UPDATE analysis_jobs SET worker_stats_json = ? WHERE id = ?",
            (_json.dumps(stats, ensure_ascii=False), job_id),
        )

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
