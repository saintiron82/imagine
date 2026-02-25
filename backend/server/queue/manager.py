"""
Job queue manager — work distribution for distributed processing.
"""

import json
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)


def get_processing_mode() -> str:
    """Get effective processing mode from config.

    Checks server.processing_mode first (set by Admin API at runtime),
    then falls back to worker.processing_mode (set by WorkerPage UI,
    stored in user-settings.yaml). Defaults to "full".

    This resolves the config path mismatch where WorkerPage saves to
    'worker.processing_mode' but the server reads 'server.processing_mode'.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        return cfg.get("server.processing_mode") or cfg.get("worker.processing_mode") or "full"
    except Exception:
        return "full"


def _utcnow_sql() -> str:
    """Return current UTC time in SQLite-native format: YYYY-MM-DD HH:MM:SS.

    Using this format (no 'T' separator, no timezone suffix, no microseconds)
    ensures correct lexicographic comparison with SQLite datetime() results.
    Python's isoformat() produces '2026-02-23T04:22:25.505000+00:00' which
    compares incorrectly as raw string ('T' > ' ').
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class JobQueueManager:
    """Manages the job queue for distributed file processing."""

    def __init__(self, db: SQLiteDB):
        self.db = db

    def _get_processing_mode(self) -> str:
        """Get server processing mode from config (always fresh).

        No caching — config.get() reads from an in-memory dict, so it's cheap.
        This ensures runtime mode changes via Admin API propagate immediately.
        """
        return get_processing_mode()

    def create_jobs(self, file_ids: List[int], file_paths: List[str], priority: int = 0) -> int:
        """Create pending jobs for files. Returns count of jobs created."""
        cursor = self.db.conn.cursor()
        created = 0
        for fid, fpath in zip(file_ids, file_paths):
            # Normalize to NFC — macOS filesystem returns NFD (decomposed Korean),
            # but files table stores NFC (via upsert_metadata). Must match.
            fpath = unicodedata.normalize('NFC', fpath)
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

    def claim_jobs(self, user_id: int, count: int = 10, worker_session_id: int = None) -> List[Dict[str, Any]]:
        """Claim up to N pending jobs for a worker.

        Job selection is routed based on the worker's effective processing_mode:
        - "full":       Prefer pre-parsed jobs, fall back to unparsed. Worker does P→V→VV→MV.
        - "mc_only":    Pre-parsed jobs only. Worker does Phase V (VLM/MC). Server runs ParseAhead+EmbedAhead.
        - "embed_only": Vision-complete jobs only (vision=true, embed=false). Worker does Phase VV+MV.

        Per-worker processing_mode_override takes precedence over global config.
        Resource-aware: throttle_level from worker session limits claim count.
        """
        cursor = self.db.conn.cursor()
        now = _utcnow_sql()

        # Determine effective processing mode for this specific worker.
        # Per-worker override (auto-detected or admin-set) > global config.
        processing_mode = self._get_processing_mode()
        if worker_session_id is not None:
            cursor.execute(
                "SELECT resources_json, processing_mode_override FROM worker_sessions WHERE id = ?",
                (worker_session_id,)
            )
            session_row = cursor.fetchone()
            if session_row:
                mode_override = session_row[1]
                if mode_override:
                    processing_mode = mode_override

                # Resource-aware throttling
                if session_row[0]:
                    try:
                        resources = json.loads(session_row[0])
                        throttle = resources.get("throttle_level", "normal")
                        if throttle == "critical":
                            logger.info(
                                f"Claim denied for session {worker_session_id}: "
                                f"throttle_level=critical (mode={processing_mode})"
                            )
                            return []
                        elif throttle == "danger":
                            count = min(count, 1)
                        elif throttle == "warning":
                            count = max(1, int(count * 0.5))
                    except (json.JSONDecodeError, TypeError):
                        pass

        if processing_mode == "embed_only":
            # embed_only workers: claim jobs where vision (MC) is done but embed (VV/MV) is not.
            # These arise when a full worker completes Phase V then fails before embed,
            # or when Phase-splitting is explicitly used in mixed-worker scenarios.
            cursor.execute(
                """SELECT id, file_id, file_path, priority, parsed_metadata
                   FROM job_queue
                   WHERE status = 'pending'
                     AND json_extract(phase_completed, '$.vision') = 1
                     AND (json_extract(phase_completed, '$.embed') IS NULL
                          OR json_extract(phase_completed, '$.embed') = 0)
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (count,)
            )
            rows = list(cursor.fetchall())

        elif processing_mode == "mc_only":
            # mc_only workers: only claim pre-parsed jobs (Phase P done by ParseAhead).
            # complete_mc() requires file metadata already upserted by ParseAhead.
            cursor.execute(
                """SELECT id, file_id, file_path, priority, parsed_metadata
                   FROM job_queue
                   WHERE status = 'pending' AND parse_status = 'parsed'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (count,)
            )
            rows = list(cursor.fetchall())

        else:
            # full workers: prefer pre-parsed jobs (skip Phase P), fallback to unparsed.
            # 1) Prefer pre-parsed jobs (server already ran Phase P)
            cursor.execute(
                """SELECT id, file_id, file_path, priority, parsed_metadata
                   FROM job_queue
                   WHERE status = 'pending' AND parse_status = 'parsed'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (count,)
            )
            rows = list(cursor.fetchall())

            # 2) Fill remainder with unparsed jobs
            if len(rows) < count:
                remainder = count - len(rows)
                claimed_ids = [r[0] for r in rows]
                if claimed_ids:
                    placeholders = ",".join("?" * len(claimed_ids))
                    cursor.execute(
                        f"""SELECT id, file_id, file_path, priority, parsed_metadata
                            FROM job_queue
                            WHERE status = 'pending'
                              AND (parse_status IS NULL OR parse_status = 'pending' OR parse_status = 'failed')
                              AND id NOT IN ({placeholders})
                            ORDER BY priority DESC, created_at ASC
                            LIMIT ?""",
                        (*claimed_ids, remainder)
                    )
                else:
                    cursor.execute(
                        """SELECT id, file_id, file_path, priority, parsed_metadata
                           FROM job_queue
                           WHERE status = 'pending'
                             AND (parse_status IS NULL OR parse_status = 'pending' OR parse_status = 'failed')
                           ORDER BY priority DESC, created_at ASC
                           LIMIT ?""",
                        (remainder,)
                    )
                rows.extend(cursor.fetchall())

        # Signal demand to ParseAheadPool BEFORE early return.
        # Uses requested count (not actual claimed count) — represents
        # "workers want N jobs" regardless of what's available.
        # This prevents the chicken-and-egg deadlock in mc_only mode where
        # 0 pre-parsed jobs → no record_claim → no demand → no pre-parsing.
        if worker_session_id is not None:
            try:
                from backend.server.queue.base_ahead_pool import BaseAheadPool
                BaseAheadPool.record_claim(
                    session_id=worker_session_id, count=count
                )
            except ImportError:
                pass

        if not rows:
            return []

        # For embed_only mode: pre-fetch vision fields from files table.
        # mc_caption, ai_tags etc. are stored in files by Phase V but NOT in
        # parsed_metadata (which only contains Phase P output). Embed workers
        # need these fields to build the MV (text embedding) input.
        embed_vision_map = {}
        if processing_mode == "embed_only":
            file_paths_nfc = [unicodedata.normalize('NFC', r[2]) for r in rows]
            placeholders = ",".join("?" * len(file_paths_nfc))
            cursor.execute(
                f"""SELECT file_path, mc_caption, ai_tags, image_type, scene_type, art_style
                    FROM files WHERE file_path IN ({placeholders})""",
                file_paths_nfc
            )
            for frow in cursor.fetchall():
                try:
                    ai_tags = json.loads(frow[2]) if frow[2] else []
                except (json.JSONDecodeError, TypeError):
                    ai_tags = []
                embed_vision_map[frow[0]] = {
                    "mc_caption": frow[1],
                    "ai_tags": ai_tags,
                    "image_type": frow[3],
                    "scene_type": frow[4],
                    "art_style": frow[5],
                }

        claimed = []
        for row in rows:
            job_id, file_id, file_path, priority, parsed_metadata = row
            cursor.execute(
                """UPDATE job_queue
                   SET status = 'assigned', assigned_to = ?, assigned_at = ?,
                       worker_session_id = ?
                   WHERE id = ? AND status = 'pending'""",
                (user_id, now, worker_session_id, job_id)
            )
            if cursor.rowcount > 0:
                job_data = {
                    "job_id": job_id,
                    "file_id": file_id,
                    "file_path": file_path,
                    "priority": priority,
                    "pre_parsed": False,
                }
                # Attach pre-parsed metadata if available
                if parsed_metadata:
                    try:
                        pm = json.loads(parsed_metadata)
                        job_data["pre_parsed"] = True
                        job_data["metadata"] = pm.get("metadata", {})
                        job_data["mc_raw"] = pm.get("mc_raw", {})
                        job_data["thumb_path"] = pm.get("thumb_path")
                    except (json.JSONDecodeError, TypeError):
                        job_data["pre_parsed"] = False

                # Attach vision data for embed_only mode (mc_caption needed for MV)
                if processing_mode == "embed_only":
                    nfc_path = unicodedata.normalize('NFC', file_path)
                    vision = embed_vision_map.get(nfc_path)
                    if vision:
                        job_data["vision_data"] = vision

                claimed.append(job_data)

        self.db.conn.commit()
        pre_parsed_count = sum(1 for j in claimed if j.get("pre_parsed"))
        logger.info(
            f"User {user_id} claimed {len(claimed)} jobs "
            f"({pre_parsed_count} pre-parsed, {len(claimed) - pre_parsed_count} unparsed)"
        )

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

        now = _utcnow_sql()
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
        now = _utcnow_sql()
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
        """Get job queue statistics including throughput."""
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

        # Determine processing mode for throughput calculation
        processing_mode = get_processing_mode()

        # Throughput: sliding windows
        # mc_only: use mc_completed_at (worker MC speed, not EmbedAhead MV speed)
        # full:    use completed_at (full pipeline completion)
        if processing_mode == "mc_only":
            cursor.execute("""
                SELECT COUNT(*) FROM job_queue
                WHERE mc_completed_at IS NOT NULL
                  AND datetime(mc_completed_at) > datetime('now', '-5 minutes')
            """)
            recent_5min = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM job_queue
                WHERE mc_completed_at IS NOT NULL
                  AND datetime(mc_completed_at) > datetime('now', '-1 minute')
            """)
            recent_1min = cursor.fetchone()[0]
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM job_queue
                WHERE status = 'completed'
                  AND completed_at IS NOT NULL
                  AND datetime(completed_at) > datetime('now', '-5 minutes')
            """)
            recent_5min = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM job_queue
                WHERE status = 'completed'
                  AND completed_at IS NOT NULL
                  AND datetime(completed_at) > datetime('now', '-1 minute')
            """)
            recent_1min = cursor.fetchone()[0]

        # Use 1-min window if active, otherwise 5-min average
        if recent_1min > 0:
            throughput = float(recent_1min)
        elif recent_5min > 0:
            throughput = round(recent_5min / 5.0, 1)
        else:
            throughput = 0.0

        # Phase-level progress counts
        phase_stats = {}
        try:
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN json_extract(phase_completed, '$.parse') = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN json_extract(phase_completed, '$.vision') = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN json_extract(phase_completed, '$.embed') = 1 THEN 1 ELSE 0 END)
                FROM job_queue
                WHERE status IN ('pending', 'assigned', 'processing', 'completed')
            """)
            phase_row = cursor.fetchone()
            phase_stats = {
                "phase_parse_done": phase_row[0] or 0,
                "phase_vision_done": phase_row[1] or 0,
                "phase_embed_done": phase_row[2] or 0,
            }
        except Exception:
            pass

        # Parse-ahead stats
        parse_ahead_stats = {}
        try:
            cursor.execute("""
                SELECT parse_status, COUNT(*) FROM job_queue
                WHERE status = 'pending' AND parse_status IS NOT NULL
                GROUP BY parse_status
            """)
            pa_counts = dict(cursor.fetchall())
            parse_ahead_stats = {
                "parse_ahead_parsed": pa_counts.get("parsed", 0),
                "parse_ahead_parsing": pa_counts.get("parsing", 0),
                "parse_ahead_failed": pa_counts.get("failed", 0),
            }
        except Exception:
            pass

        # ETA: estimated seconds to complete remaining jobs
        pending = status_counts.get("pending", 0)
        assigned = status_counts.get("assigned", 0)
        processing = status_counts.get("processing", 0)
        remaining = pending + assigned + processing
        if throughput > 0 and remaining > 0:
            eta_seconds = round((remaining / throughput) * 60)
        else:
            eta_seconds = None

        return {
            "total": total,
            "pending": pending,
            "assigned": assigned,
            "processing": processing,
            "completed": status_counts.get("completed", 0),
            "failed": status_counts.get("failed", 0),
            "throughput": throughput,
            "recent_1min": recent_1min,
            "recent_5min": recent_5min,
            "eta_seconds": eta_seconds,
            **phase_stats,
            **parse_ahead_stats,
        }

    def list_jobs(self, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all jobs with optional status filter and pagination."""
        cursor = self.db.conn.cursor()

        # Total count
        if status:
            cursor.execute("SELECT COUNT(*) FROM job_queue WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM job_queue")
        total = cursor.fetchone()[0]

        # Fetch page
        if status:
            cursor.execute(
                """SELECT id, file_path, status, phase_completed, priority,
                          error_message, retry_count, created_at, started_at, completed_at
                   FROM job_queue
                   WHERE status = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (status, limit, offset)
            )
        else:
            cursor.execute(
                """SELECT id, file_path, status, phase_completed, priority,
                          error_message, retry_count, created_at, started_at, completed_at
                   FROM job_queue
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset)
            )
        jobs = [
            {
                "job_id": row[0],
                "file_path": row[1],
                "status": row[2],
                "phase_completed": json.loads(row[3] or "{}"),
                "priority": row[4],
                "error_message": row[5],
                "retry_count": row[6],
                "created_at": row[7],
                "started_at": row[8],
                "completed_at": row[9],
            }
            for row in cursor.fetchall()
        ]
        return {"jobs": jobs, "total": total}

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a job (only pending/assigned/failed)."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """UPDATE job_queue SET status = 'cancelled', assigned_to = NULL
               WHERE id = ? AND status IN ('pending', 'assigned', 'failed')""",
            (job_id,)
        )
        self.db.conn.commit()
        success = cursor.rowcount > 0
        if success:
            logger.info(f"Job {job_id} cancelled")
        return success

    def retry_failed_jobs(self) -> int:
        """Retry all failed jobs by resetting them to pending.

        Also resets parse_status='failed' back to NULL so ParseAhead
        can re-attempt pre-parsing (prevents permanent parse deadlock).
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            """UPDATE job_queue
               SET status = 'pending', retry_count = 0,
                   error_message = NULL, assigned_to = NULL, assigned_at = NULL,
                   parse_status = CASE
                       WHEN parse_status = 'failed' THEN NULL
                       ELSE parse_status
                   END
               WHERE status = 'failed'"""
        )
        self.db.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Retried {count} failed jobs")
        return count

    def clear_completed_jobs(self) -> int:
        """Delete all completed jobs."""
        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM job_queue WHERE status = 'completed'")
        self.db.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleared {count} completed jobs")
        return count

    def queue_backfill(self) -> Dict[str, int]:
        """Detect files with incomplete vector data and auto-create backfill jobs.

        Scans for files that have VV (vec_files) but are missing Structure
        (vec_structure) vectors. Creates jobs with parse_status='backfill'
        so ParseAheadPool can process them (DINOv2 only, skip full parse).

        Skips files that already have an active backfill job in the queue.

        Returns:
            Dict with counts of created jobs by type, e.g. {"structure": 5}.
        """
        cursor = self.db.conn.cursor()
        created = {"structure": 0}

        # Files with VV but no Structure vector, no active backfill job
        try:
            cursor.execute("""
                SELECT f.id, f.file_path
                FROM files f
                WHERE EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                  AND NOT EXISTS(SELECT 1 FROM vec_structure WHERE file_id = f.id)
                  AND NOT EXISTS(
                      SELECT 1 FROM job_queue jq
                      WHERE jq.file_id = f.id
                        AND jq.parse_status = 'backfill'
                        AND jq.status IN ('pending', 'assigned', 'processing')
                  )
            """)
            rows = cursor.fetchall()
        except Exception as e:
            logger.warning(f"Backfill scan failed: {e}")
            return created

        if not rows:
            return created

        now = _utcnow_sql()
        for file_id, file_path in rows:
            try:
                cursor.execute(
                    """INSERT INTO job_queue
                       (file_id, file_path, status, parse_status,
                        phase_completed, parsed_metadata, created_at)
                       VALUES (?, ?, 'pending', 'backfill',
                               '{"parse":true,"vision":true,"embed":false}',
                               '{"backfill":"structure"}', ?)""",
                    (file_id, file_path, now),
                )
                if cursor.rowcount > 0:
                    created["structure"] += 1
            except Exception as e:
                logger.warning(f"Backfill job creation failed for file_id={file_id}: {e}")

        self.db.conn.commit()
        total = sum(created.values())
        if total > 0:
            logger.info(f"Backfill: queued {total} jobs (structure={created['structure']})")
        return created

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
