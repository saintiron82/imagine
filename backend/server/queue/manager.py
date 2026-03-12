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

# Structured error codes for job failure classification.
# Non-retryable errors are permanent — retrying will never succeed.
NON_RETRYABLE_ERRORS = frozenset({
    "THUMB_MISSING",   # No thumbnail available (V/VV phases impossible)
    "FILE_NOT_FOUND",  # Source file inaccessible (deleted, WebDAV 404)
    "PARSE_FAILED",    # File parsing failed (corrupt file, unsupported format)
})


def _infer_error_code(error_message: str) -> str | None:
    """Infer structured error_code from legacy free-text error_message.

    Used by audit to backfill error_code on jobs that failed before
    the error code system was introduced.
    """
    if not error_message:
        return None
    msg = error_message.lower()
    if "file unavailable" in msg or "file not found" in msg or "cannot access" in msg:
        return "FILE_NOT_FOUND"
    if "no thumbnail" in msg or ("thumbnail" in msg and "requires" in msg):
        return "THUMB_MISSING"
    if "parse failed" in msg:
        return "PARSE_FAILED"
    return None


def get_processing_mode() -> str:
    """Get effective processing mode from config.

    Returns "mc_only", "parse_only", "auto", or "builtin_worker" (default: auto).
    - mc_only: Server P+VV+MV, workers do V(MC) only.
    - parse_only: Server P only (zero GPU), workers do V+VV+MV (full mode).
    - auto: Server P + gap-fill, workers distribute V/VV/MV by capability.
    - builtin_worker: Server processes full P→V→VV→MV always, regardless of workers.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        mode = cfg.get("server.processing_mode") or "auto"
        # Normalize legacy values
        if mode not in ("mc_only", "parse_only", "auto", "builtin_worker"):
            mode = "auto"
        return mode
    except Exception:
        return "auto"


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

    def _batch_check_existing_data(self, file_ids: List[int]) -> Dict[int, dict]:
        """Check actual DB data existence for a batch of files.

        Returns a mapping of file_id → {"has_mc": bool, "has_vv": bool, "has_mv": bool}
        Uses a single query with subqueries for efficiency.
        """
        if not file_ids:
            return {}

        cursor = self.db.conn.cursor()
        placeholders = ",".join("?" * len(file_ids))
        cursor.execute(f"""
            SELECT f.id,
                   (f.mc_caption IS NOT NULL AND f.mc_caption != '') AS has_mc,
                   EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) AS has_vv,
                   EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) AS has_mv
            FROM files f
            WHERE f.id IN ({placeholders})
        """, file_ids)

        result = {}
        for row in cursor.fetchall():
            result[row[0]] = {
                "has_mc": bool(row[1]),
                "has_vv": bool(row[2]),
                "has_mv": bool(row[3]),
            }
        return result

    def create_jobs(self, file_ids: List[int], file_paths: List[str], priority: int = 0) -> int:
        """Create pending jobs for files with accurate phase_completed.

        Checks actual DB data to set initial phase_completed correctly,
        preventing re-processing of already-complete phases.

        Returns count of jobs created.
        """
        cursor = self.db.conn.cursor()

        # Batch check existing data for all files
        existing_data = self._batch_check_existing_data(file_ids)

        created = 0
        for fid, fpath in zip(file_ids, file_paths):
            # Normalize to NFC — macOS filesystem returns NFD (decomposed Korean),
            # but files table stores NFC (via upsert_metadata). Must match.
            fpath = unicodedata.normalize('NFC', fpath)

            # Determine accurate phase_completed from actual data
            data = existing_data.get(fid, {})
            has_mc = data.get("has_mc", False)
            has_vv = data.get("has_vv", False)
            has_mv = data.get("has_mv", False)
            phase_completed = json.dumps({
                "parse": True,  # Files being registered already have metadata
                "vision": has_mc,
                "embed": has_vv and has_mv,
            })

            try:
                cursor.execute(
                    """INSERT INTO job_queue (file_id, file_path, status, priority, phase_completed)
                       VALUES (?, ?, 'pending', ?, ?)
                       ON CONFLICT DO NOTHING""",
                    (fid, fpath, priority, phase_completed)
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

        # In builtin_worker mode, only the built-in worker may claim jobs
        if processing_mode == "builtin_worker" and worker_session_id is not None:
            cursor.execute(
                "SELECT worker_name FROM worker_sessions WHERE id = ?",
                (worker_session_id,)
            )
            ws_row = cursor.fetchone()
            if ws_row and ws_row[0] != "__builtin__":
                logger.info(
                    f"Claim denied for session {worker_session_id}: "
                    f"builtin_worker mode active, external workers blocked"
                )
                return []

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
            # embed_only (lightweight) workers: claim pre-parsed + vision-done jobs.
            # Server gap-fills V(MC) for these jobs. Worker does VV+MV only.
            cursor.execute(
                """SELECT id, file_id, file_path, priority, parsed_metadata
                   FROM job_queue
                   WHERE status = 'pending'
                     AND parse_status = 'parsed'
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
            # full workers: only claim pre-parsed jobs.
            # Server (ParseAheadPool) always handles Phase P — workers never parse.
            # Priority: vision-done jobs first (VV+MV only), then regular (V+VV+MV).
            # 1) Vision-done jobs (server gap-filled MC — just needs VV+MV)
            cursor.execute(
                """SELECT id, file_id, file_path, priority, parsed_metadata
                   FROM job_queue
                   WHERE status = 'pending' AND parse_status = 'parsed'
                     AND json_extract(phase_completed, '$.vision') = 1
                     AND (json_extract(phase_completed, '$.embed') IS NULL
                          OR json_extract(phase_completed, '$.embed') = 0)
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (count,)
            )
            rows = list(cursor.fetchall())
            vision_done_ids = {r[0] for r in rows}  # Track which jobs have vision done

            # 2) Regular pre-parsed jobs (needs V+VV+MV)
            if len(rows) < count:
                remainder = count - len(rows)
                claimed_ids = [r[0] for r in rows]
                if claimed_ids:
                    placeholders = ",".join("?" * len(claimed_ids))
                    cursor.execute(
                        f"""SELECT id, file_id, file_path, priority, parsed_metadata
                            FROM job_queue
                            WHERE status = 'pending' AND parse_status = 'parsed'
                              AND (json_extract(phase_completed, '$.vision') IS NULL
                                   OR json_extract(phase_completed, '$.vision') = 0)
                              AND id NOT IN ({placeholders})
                            ORDER BY priority DESC, created_at ASC
                            LIMIT ?""",
                        (*claimed_ids, remainder)
                    )
                else:
                    cursor.execute(
                        """SELECT id, file_id, file_path, priority, parsed_metadata
                           FROM job_queue
                           WHERE status = 'pending' AND parse_status = 'parsed'
                             AND (json_extract(phase_completed, '$.vision') IS NULL
                                  OR json_extract(phase_completed, '$.vision') = 0)
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
            # Diagnostic: log queue state when no jobs available for worker
            if worker_session_id is not None:
                try:
                    diag = cursor.execute(
                        """SELECT
                            COUNT(*) FILTER (WHERE status = 'pending') as pending,
                            COUNT(*) FILTER (WHERE status = 'pending' AND parse_status IS NULL) as unparsed,
                            COUNT(*) FILTER (WHERE status = 'pending' AND parse_status = 'parsing') as parsing,
                            COUNT(*) FILTER (WHERE status = 'pending' AND parse_status = 'parsed') as parsed,
                            COUNT(*) FILTER (WHERE status = 'pending' AND parse_status = 'failed') as parse_failed,
                            COUNT(*) FILTER (WHERE status = 'assigned') as assigned,
                            COUNT(*) FILTER (WHERE status = 'completed') as completed,
                            COUNT(*) FROM job_queue"""
                    ).fetchone()
                    logger.info(
                        f"[CLAIM-DIAG] session={worker_session_id} mode={processing_mode} | "
                        f"pending={diag[0]} (unparsed={diag[1]} parsing={diag[2]} "
                        f"parsed={diag[3]} failed={diag[4]}) assigned={diag[5]} "
                        f"completed={diag[6]} total={diag[7]}"
                    )
                except Exception:
                    pass
            return []

        # Pre-fetch vision fields from files table for workers that need them.
        # embed_only workers always need vision_data (VV+MV only, MC from server).
        # full workers need vision_data for vision-done jobs (server gap-filled MC).
        # mc_caption, ai_tags etc. are stored in files by Phase V but NOT in
        # parsed_metadata (which only contains Phase P output).
        embed_vision_map = {}
        if processing_mode in ("embed_only", "full"):
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
                        job_data["mc_raw"] = pm.get("mc_raw") or None
                        job_data["thumb_path"] = pm.get("thumb_path")
                    except (json.JSONDecodeError, TypeError):
                        job_data["pre_parsed"] = False

                # Fallback: look up files.thumbnail_url when
                # parsed_metadata doesn't have thumb_path
                # (e.g. old WebDAV files browsed before job creation fix)
                if not job_data.get("thumb_path"):
                    try:
                        thumb_row = cursor.execute(
                            "SELECT thumbnail_url FROM files WHERE id = ?",
                            (file_id,)
                        ).fetchone()
                        if thumb_row and thumb_row[0]:
                            job_data["thumb_path"] = thumb_row[0]
                    except Exception:
                        pass

                # Attach vision data for workers that need MC from server.
                # embed_only: always (VV+MV only, all jobs have vision done).
                # full: only for jobs where phase_completed.vision = 1
                #       (NOT based on files table — old mc_caption from previous
                #        sessions should not suppress worker's own Vision phase).
                if processing_mode == "embed_only":
                    nfc_path = unicodedata.normalize('NFC', file_path)
                    vision = embed_vision_map.get(nfc_path)
                    if vision:
                        job_data["vision_data"] = vision
                elif processing_mode == "full" and job_id in vision_done_ids:
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

    def complete_job_with_phases(self, job_id: int, user_id: int, phases: dict) -> bool:
        """Complete a job with explicit phase status based on actual data.

        If all phases are done → status='completed'.
        If some phases are missing → status='pending' (re-claimable for retry).

        Args:
            job_id: Job ID
            user_id: Assigned worker's user ID
            phases: {"parse": bool, "vision": bool, "embed": bool}

        Returns:
            True if updated successfully.
        """
        cursor = self.db.conn.cursor()
        now = _utcnow_sql()

        all_done = all(phases.values())
        if all_done:
            # Fully complete
            cursor.execute(
                """UPDATE job_queue
                   SET status = 'completed', completed_at = ?,
                       phase_completed = ?
                   WHERE id = ? AND assigned_to = ?""",
                (now, json.dumps(phases), job_id, user_id)
            )
        else:
            # Partial completion → check retry count before releasing
            missing = [k for k, v in phases.items() if not v]
            cursor.execute(
                "SELECT retry_count, max_retries FROM job_queue WHERE id = ?",
                (job_id,)
            )
            retry_row = cursor.fetchone()
            retry_count = retry_row[0] if retry_row else 0
            max_retries = retry_row[1] if retry_row else 3

            if retry_count >= max_retries:
                # Too many retries — mark as permanently failed
                logger.warning(
                    f"Job {job_id} permanently failed after {retry_count} retries "
                    f"(missing: {missing})."
                )
                cursor.execute(
                    """UPDATE job_queue
                       SET status = 'failed', phase_completed = ?,
                           error_message = ?,
                           assigned_to = NULL, assigned_at = NULL,
                           worker_session_id = NULL
                       WHERE id = ? AND assigned_to = ?""",
                    (json.dumps(phases),
                     f"Partial after {retry_count} retries, missing: {missing}",
                     job_id, user_id)
                )
            else:
                logger.warning(
                    f"Job {job_id} partially complete (missing: {missing}). "
                    f"Retry {retry_count + 1}/{max_retries}."
                )
                cursor.execute(
                    """UPDATE job_queue
                       SET status = 'pending', phase_completed = ?,
                           retry_count = retry_count + 1,
                           assigned_to = NULL, assigned_at = NULL,
                           worker_session_id = NULL
                       WHERE id = ? AND assigned_to = ?""",
                    (json.dumps(phases), job_id, user_id)
                )

        success = cursor.rowcount > 0
        self.db.conn.commit()
        return success

    def fail_job(self, job_id: int, user_id: int, error_message: str,
                 error_code: str = None) -> bool:
        """Mark a job as failed with optional structured error code.

        Non-retryable error codes (THUMB_MISSING, FILE_NOT_FOUND, PARSE_FAILED)
        skip retries and fail immediately. Retryable errors follow the existing
        retry_count/max_retries logic.
        """
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

        # Non-retryable errors → immediate failure (no retry)
        if error_code and error_code in NON_RETRYABLE_ERRORS:
            new_status = "failed"
        else:
            new_status = "pending" if retry_count < max_retries else "failed"

        cursor.execute(
            """UPDATE job_queue
               SET status = ?, error_message = ?, error_code = ?,
                   retry_count = retry_count + 1,
                   assigned_to = NULL, assigned_at = NULL,
                   worker_session_id = NULL
               WHERE id = ?""",
            (new_status, error_message, error_code, job_id)
        )
        self.db.conn.commit()
        if new_status == "pending":
            logger.info(f"Job {job_id} will be retried (attempt {retry_count + 1}/{max_retries})")
        else:
            code_info = f" [{error_code}]" if error_code else ""
            logger.warning(f"Job {job_id} permanently failed{code_info}: {error_message}")
        return True

    def reclaim_worker_jobs(self, worker_session_id: int) -> int:
        """Reclaim all jobs assigned to a worker session back to pending.

        Called when a worker disconnects, is blocked, or times out.
        Preserves phase_completed so Smart Skip avoids re-processing
        already-completed phases.
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            """UPDATE job_queue
               SET status = 'pending',
                   assigned_to = NULL,
                   assigned_at = NULL,
                   worker_session_id = NULL
               WHERE worker_session_id = ?
                 AND status IN ('assigned', 'processing')""",
            (worker_session_id,)
        )
        self.db.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Reclaimed {count} jobs from worker session {worker_session_id}")
        return count

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
        # Preserve phase_completed (avoid re-processing done phases),
        # clear worker_session_id to remove stale references.
        cursor.execute(
            f"""UPDATE job_queue
                SET status = 'pending',
                    assigned_to = NULL,
                    assigned_at = NULL,
                    worker_session_id = NULL
                WHERE id IN ({placeholders})""",
            stale_ids
        )
        self.db.conn.commit()
        logger.info(f"Reassigned {cursor.rowcount} stale jobs (phase_completed preserved)")
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

        Skips non-retryable errors (THUMB_MISSING, FILE_NOT_FOUND, PARSE_FAILED).
        Also resets parse_status='failed' back to NULL so ParseAhead
        can re-attempt pre-parsing (prevents permanent parse deadlock).
        """
        cursor = self.db.conn.cursor()
        non_retryable_list = ",".join(f"'{c}'" for c in NON_RETRYABLE_ERRORS)
        cursor.execute(
            f"""UPDATE job_queue
               SET status = 'pending', retry_count = 0,
                   error_message = NULL, error_code = NULL,
                   assigned_to = NULL, assigned_at = NULL,
                   parse_status = CASE
                       WHEN parse_status = 'failed' THEN NULL
                       ELSE parse_status
                   END
               WHERE status = 'failed'
               AND (error_code IS NULL OR error_code NOT IN ({non_retryable_list}))"""
        )
        self.db.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Retried {count} failed jobs")
        return count

    def audit_completed_jobs(self) -> Dict[str, Any]:
        """Full data integrity audit across ALL files in the database.

        File-centric: each file is either complete (mc+vv+mv) or not.
        For incomplete files, ensures a pending job exists for re-processing.

        Returns file-centric results:
            {
                "total_files": int,
                "complete_files": int,
                "incomplete_files": int,
                "repaired_files": int,   # files that were fixed
                "details": [{"file_id", "file_path", "missing"}]
            }
        """
        cursor = self.db.conn.cursor()
        details = []
        repaired_files = 0
        skipped_non_retryable = 0

        # ── Pass 1: Scan ALL files for data completeness ──
        cursor.execute("""
            SELECT f.id, f.file_path,
                   (f.mc_caption IS NOT NULL AND f.mc_caption != '') AS has_mc,
                   EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) AS has_vv,
                   EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) AS has_mv,
                   f.thumbnail_url
            FROM files f
        """)
        all_files = cursor.fetchall()
        total_files = len(all_files)
        complete_files = 0
        incomplete_file_ids = set()

        for file_id, file_path, has_mc, has_vv, has_mv, thumbnail_url in all_files:
            if has_mc and has_vv and has_mv:
                complete_files += 1
                continue

            # This file is incomplete
            incomplete_file_ids.add(file_id)
            missing = []
            if not has_mc:
                missing.append("mc")
            if not has_vv:
                missing.append("vv")
            if not has_mv:
                missing.append("mv")

            actual_phases = json.dumps({
                "parse": True,
                "vision": bool(has_mc),
                "embed": bool(has_vv and has_mv),
            })

            # Build parsed_metadata so workers recognize this as pre-parsed
            parsed_metadata = json.dumps({
                "metadata": {},
                "thumb_path": thumbnail_url,
                "mc_raw": None,
            }, ensure_ascii=False)

            # Check what jobs exist for this file
            cursor.execute("""
                SELECT id, status, phase_completed, parsed_metadata,
                       error_code, retry_count, error_message
                FROM job_queue WHERE file_id = ?
                ORDER BY
                    CASE status
                        WHEN 'processing' THEN 1
                        WHEN 'pending' THEN 2
                        WHEN 'assigned' THEN 3
                        WHEN 'completed' THEN 4
                        WHEN 'failed' THEN 5
                    END
                LIMIT 1
            """, (file_id,))
            job_row = cursor.fetchone()

            # Skip files without thumbnails — can't process V/VV phases
            if not thumbnail_url:
                if job_row is None:
                    # No job and no thumbnail — skip entirely
                    skipped_non_retryable += 1
                    continue
                elif job_row[1] != 'failed':
                    # Has job but no thumbnail — mark as THUMB_MISSING
                    cursor.execute(
                        """UPDATE job_queue SET status = 'failed',
                           error_code = 'THUMB_MISSING',
                           error_message = 'No thumbnail available for processing'
                           WHERE id = ?""",
                        (job_row[0],)
                    )
                skipped_non_retryable += 1
                continue

            # Ensure this file has a pending job for re-processing
            if job_row is None:
                # No job at all — create one with parsed_metadata
                try:
                    cursor.execute(
                        """INSERT INTO job_queue
                           (file_id, file_path, status, priority,
                            phase_completed, parse_status, parsed_metadata)
                           VALUES (?, ?, 'pending', 0, ?, 'parsed', ?)""",
                        (file_id, file_path, actual_phases, parsed_metadata)
                    )
                except Exception as e:
                    logger.warning(f"Audit: failed to create job for file_id={file_id}: {e}")

            elif job_row[1] == 'completed':
                # False-completed — reset to pending, fill parsed_metadata if missing
                pm_update = parsed_metadata if not job_row[3] else job_row[3]
                cursor.execute(
                    """UPDATE job_queue
                       SET status = 'pending', phase_completed = ?,
                           parsed_metadata = COALESCE(parsed_metadata, ?),
                           assigned_to = NULL, assigned_at = NULL,
                           worker_session_id = NULL
                       WHERE id = ?""",
                    (actual_phases, pm_update, job_row[0])
                )

            elif job_row[1] == 'failed':
                # job_row: (id, status, phase_completed, parsed_metadata,
                #           error_code, retry_count, error_message)
                existing_error_code = job_row[4]
                retry_count = job_row[5] or 0
                err_msg_text = job_row[6] or ""

                # Backfill: infer error_code from legacy error_message
                if not existing_error_code:
                    existing_error_code = _infer_error_code(err_msg_text)
                    if existing_error_code:
                        cursor.execute(
                            "UPDATE job_queue SET error_code = ? WHERE id = ?",
                            (existing_error_code, job_row[0])
                        )

                if existing_error_code and existing_error_code in NON_RETRYABLE_ERRORS:
                    # Non-retryable — leave it failed, don't count as repaired
                    skipped_non_retryable += 1
                    continue
                elif retry_count >= 3:
                    # Exhausted retries — don't reset again
                    skipped_non_retryable += 1
                    continue
                else:
                    # Retryable failure with retries remaining — reset to pending
                    cursor.execute(
                        """UPDATE job_queue
                           SET status = 'pending', phase_completed = ?,
                               parsed_metadata = COALESCE(parsed_metadata, ?),
                               retry_count = 0, error_message = NULL,
                               error_code = NULL,
                               assigned_to = NULL, assigned_at = NULL,
                               worker_session_id = NULL
                           WHERE id = ?""",
                        (actual_phases, parsed_metadata, job_row[0])
                    )

            elif job_row[1] in ('pending', 'assigned', 'processing'):
                # Already in pipeline — fix phase_completed, fill parsed_metadata if missing
                cursor.execute(
                    """UPDATE job_queue
                       SET phase_completed = ?,
                           parsed_metadata = COALESCE(parsed_metadata, ?)
                       WHERE id = ?""",
                    (actual_phases, parsed_metadata, job_row[0])
                )

            repaired_files += 1
            details.append({
                "file_id": file_id,
                "file_path": file_path,
                "missing": missing,
            })

        # ── Pass 2: Completed jobs referencing deleted files ──
        cursor.execute("""
            SELECT jq.id, jq.file_id FROM job_queue jq
            WHERE jq.status = 'completed'
            AND NOT EXISTS(SELECT 1 FROM files WHERE id = jq.file_id)
        """)
        dangling_rows = cursor.fetchall()
        for job_id, file_id in dangling_rows:
            cursor.execute(
                "UPDATE job_queue SET status = 'failed', error_message = 'file missing from DB', error_code = 'FILE_NOT_FOUND' WHERE id = ?",
                (job_id,)
            )

        self.db.conn.commit()

        incomplete_files = total_files - complete_files

        if repaired_files > 0 or skipped_non_retryable > 0:
            parts = [f"Audit: {total_files} files, {incomplete_files} incomplete"]
            if repaired_files > 0:
                parts.append(f"{repaired_files} repaired")
            if skipped_non_retryable > 0:
                parts.append(f"{skipped_non_retryable} permanently failed (skipped)")
            logger.warning(", ".join(parts))
        else:
            logger.info(f"Audit: {total_files} files scanned, all complete")

        return {
            "total_files": total_files,
            "complete_files": complete_files,
            "incomplete_files": incomplete_files,
            "repaired_files": repaired_files,
            "skipped_non_retryable": skipped_non_retryable,
            "details": details,
        }

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
