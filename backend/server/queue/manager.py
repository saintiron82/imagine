"""
Job queue manager — work distribution for distributed processing.
"""

import json
import logging
import os
import unicodedata
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)

# Module-level state for CLAIM-DIAG deduplication
_last_claim_diag: dict = {}

# Module-level reference to DownloadAheadPool (set by app.py on startup).
_download_pool_ref = None


def set_download_pool(pool):
    """Register the DownloadAheadPool instance for temp file cleanup."""
    global _download_pool_ref
    _download_pool_ref = pool


def _get_download_pool():
    """Get the registered DownloadAheadPool, if any."""
    return _download_pool_ref


# Structured error codes for job failure classification.
# Non-retryable errors are permanent — retrying will never succeed.
NON_RETRYABLE_ERRORS = frozenset({
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

            # WebDAV files need download before processing (file_ready=0)
            file_ready = 0 if fpath.startswith("webdav://") else 1

            try:
                cursor.execute(
                    """INSERT INTO job_queue (file_id, file_path, status, priority, phase_completed, file_ready)
                       VALUES (?, ?, 'pending', ?, ?, ?)
                       ON CONFLICT DO NOTHING""",
                    (fid, fpath, priority, phase_completed, file_ready)
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
        import sqlite3 as _sqlite3
        try:
            return self._claim_jobs_inner(user_id, count, worker_session_id)
        except _sqlite3.OperationalError as e:
            if "locked" in str(e):
                logger.warning(f"claim_jobs: DB busy, returning empty (will retry next poll)")
                return []
            raise

    def _claim_jobs_inner(self, user_id: int, count: int, worker_session_id: int = None) -> List[Dict[str, Any]]:
        """Internal claim implementation."""
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
                            throttle_key = ("throttle", worker_session_id)
                            if throttle_key != _last_claim_diag.get("throttle_key"):
                                _last_claim_diag["throttle_key"] = throttle_key
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
                   WHERE status = 'pending' AND file_ready = 1
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
                   WHERE status = 'pending' AND file_ready = 1
                     AND parse_status = 'parsed'
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
                   WHERE status = 'pending' AND file_ready = 1
                     AND parse_status = 'parsed'
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
                            WHERE status = 'pending' AND file_ready = 1
                              AND parse_status = 'parsed'
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
                           WHERE status = 'pending' AND file_ready = 1
                             AND parse_status = 'parsed'
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
            # Diagnostic: log queue state once when transitioning to idle
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
                            COUNT(*) FROM job_queue,
                            COUNT(*) FILTER (WHERE status = 'pending' AND file_ready = 0) as not_ready"""
                    ).fetchone()
                    diag_key = (worker_session_id, diag[0], diag[6], diag[7])
                    if diag_key != _last_claim_diag.get("key"):
                        _last_claim_diag["key"] = diag_key
                        not_ready = diag[8] if len(diag) > 8 else 0
                        logger.info(
                            f"[CLAIM-DIAG] session={worker_session_id} mode={processing_mode} | "
                            f"pending={diag[0]} (unparsed={diag[1]} parsing={diag[2]} "
                            f"parsed={diag[3]} failed={diag[4]} not_ready={not_ready}) "
                            f"assigned={diag[5]} completed={diag[6]} total={diag[7]}"
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
        """Complete a job: cleanup temp files, log completion, DELETE from queue."""
        cursor = self.db.conn.cursor()

        # Verify ownership and get file_id before deletion
        cursor.execute(
            "SELECT file_id FROM job_queue WHERE id = ? AND assigned_to = ?",
            (job_id, user_id)
        )
        row = cursor.fetchone()
        if not row:
            self.db.conn.rollback()
            return False

        file_id = row[0]

        # Cleanup temp files BEFORE deleting the job row
        self._cleanup_temp_file(job_id)

        # Clear any processing_status on the file (successful completion)
        cursor.execute(
            "UPDATE files SET processing_status = NULL, processing_error = NULL WHERE id = ?",
            (file_id,)
        )

        # Log completion for throughput tracking
        cursor.execute(
            "INSERT INTO job_completions (file_id, worker_session_id) VALUES (?, ?)",
            (file_id, self._get_worker_session_id(cursor, user_id))
        )

        # DELETE the completed job
        cursor.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
        self.db.conn.commit()
        return True

    def _get_worker_session_id(self, cursor, user_id: int) -> Optional[int]:
        """Get current active worker session ID for a user."""
        try:
            cursor.execute(
                "SELECT id FROM worker_sessions WHERE user_id = ? AND status = 'online' "
                "ORDER BY last_heartbeat DESC LIMIT 1",
                (user_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def _cleanup_temp_file(self, job_id: int):
        """Delete temp file for a WebDAV job and release buffer slot.

        Called after job completion to free disk space in the bounded
        download-ahead buffer.
        """
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT file_id, file_path, parsed_metadata FROM job_queue WHERE id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
            if not row:
                return
            file_id, file_path, pm_str = row

            if not file_path or not file_path.startswith("webdav://"):
                return

            # Release slot via the module-level download pool reference
            pool = _get_download_pool()
            if pool:
                pool.release_slot(file_id)
                return

            # Fallback: clean up temp file directly from parsed_metadata
            if pm_str:
                from pathlib import Path
                try:
                    pm = json.loads(pm_str)
                    temp_path = pm.get("temp_local_path")
                    if temp_path:
                        p = Path(temp_path)
                        if p.exists():
                            p.unlink()
                            logger.debug(
                                f"Cleaned up temp file for job {job_id}: {p.name}"
                            )
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception as e:
            logger.warning(f"_cleanup_temp_file error for job {job_id}: {e}")

    def complete_job_with_phases(self, job_id: int, user_id: int, phases: dict) -> bool:
        """Complete a job with explicit phase status based on actual data.

        If all phases are done → DELETE from queue + log completion.
        If some phases are missing → retry or permanently fail.

        Args:
            job_id: Job ID
            user_id: Assigned worker's user ID
            phases: {"parse": bool, "vision": bool, "embed": bool}

        Returns:
            True if updated successfully.
        """
        cursor = self.db.conn.cursor()

        all_done = all(phases.values())
        if all_done:
            # Fully complete — delegate to complete_job (DELETE + log)
            return self.complete_job(job_id, user_id)
        else:
            # Partial completion → check retry count before releasing
            missing = [k for k, v in phases.items() if not v]
            cursor.execute(
                "SELECT file_id, retry_count, max_retries FROM job_queue WHERE id = ? AND assigned_to = ?",
                (job_id, user_id)
            )
            retry_row = cursor.fetchone()
            if not retry_row:
                self.db.conn.rollback()
                return False

            file_id = retry_row[0]
            retry_count = retry_row[1] or 0
            max_retries = retry_row[2] or 3

            if retry_count >= max_retries:
                # Too many retries — permanently fail
                error_msg = f"Partial after {retry_count} retries, missing: {missing}"
                logger.warning(
                    f"Job {job_id} permanently failed after {retry_count} retries "
                    f"(missing: {missing})."
                )
                # Cleanup temp file before deletion
                self._cleanup_temp_file(job_id)
                # Mark file as failed
                cursor.execute(
                    "UPDATE files SET processing_status = 'failed', processing_error = ? WHERE id = ?",
                    (error_msg, file_id)
                )
                # DELETE the job
                cursor.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
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
        """Handle a job failure with optional structured error code.

        Non-retryable error codes (FILE_NOT_FOUND, PARSE_FAILED)
        skip retries and permanently fail immediately.
        Retryable errors follow retry_count/max_retries logic.

        Permanent failure: DELETE from job_queue + mark files.processing_status='failed'.
        Retryable failure: reset to pending with incremented retry_count.
        """
        cursor = self.db.conn.cursor()

        cursor.execute(
            "SELECT file_id, retry_count, max_retries FROM job_queue WHERE id = ? AND assigned_to = ?",
            (job_id, user_id)
        )
        row = cursor.fetchone()
        if row is None:
            self.db.conn.rollback()
            return False

        file_id, retry_count, max_retries = row

        # Non-retryable errors → immediate permanent failure
        non_retryable = error_code and error_code in NON_RETRYABLE_ERRORS
        permanent_fail = non_retryable or retry_count >= max_retries

        if permanent_fail:
            code_info = f" [{error_code}]" if error_code else ""
            logger.warning(f"Job {job_id} permanently failed{code_info}: {error_message}")

            # Cleanup temp file before deletion
            self._cleanup_temp_file(job_id)

            # Mark file as failed
            cursor.execute(
                "UPDATE files SET processing_status = 'failed', processing_error = ? WHERE id = ?",
                (error_message, file_id)
            )

            # DELETE the job from queue
            cursor.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
        else:
            cursor.execute(
                """UPDATE job_queue
                   SET status = 'pending', error_message = ?, error_code = ?,
                       retry_count = retry_count + 1,
                       assigned_to = NULL, assigned_at = NULL,
                       worker_session_id = NULL
                   WHERE id = ?""",
                (error_message, error_code, job_id)
            )
            logger.info(f"Job {job_id} will be retried (attempt {retry_count + 1}/{max_retries})")

        self.db.conn.commit()
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

        # Prune old completion records (> 1 hour) for housekeeping
        cursor.execute(
            "DELETE FROM job_completions WHERE datetime(completed_at) < datetime('now', '-1 hour')"
        )

        # Throughput from job_completions table (sliding windows)
        processing_mode = get_processing_mode()

        if processing_mode == "mc_only":
            # mc_only mode: use mc_completed_at from job_queue (pending MC jobs)
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
            # full mode: use job_completions table
            cursor.execute("""
                SELECT COUNT(*) FROM job_completions
                WHERE datetime(completed_at) > datetime('now', '-5 minutes')
            """)
            recent_5min = cursor.fetchone()[0]
            cursor.execute("""
                SELECT COUNT(*) FROM job_completions
                WHERE datetime(completed_at) > datetime('now', '-1 minute')
            """)
            recent_1min = cursor.fetchone()[0]

        # Use 1-min window if active, otherwise 5-min average
        if recent_1min > 0:
            throughput = float(recent_1min)
        elif recent_5min > 0:
            throughput = round(recent_5min / 5.0, 1)
        else:
            throughput = 0.0

        # Phase-level progress counts — deferred to file-centric block below
        phase_stats = {}

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

        # file_ready stats (2-stage pipeline: download waiting vs processing ready)
        file_ready_stats = {}
        try:
            cursor.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE file_ready = 0 AND status = 'pending'),
                    COUNT(*) FILTER (WHERE file_ready = 1 AND status = 'pending')
                FROM job_queue
            """)
            fr_row = cursor.fetchone()
            file_ready_stats = {
                "download_waiting": fr_row[0] or 0,
                "ready_pending": fr_row[1] or 0,
            }
        except Exception:
            pass

        # Download buffer stats (from DownloadAheadPool if available)
        dl_pool = _get_download_pool()
        download_buffer = dl_pool.get_stats() if dl_pool else None

        # ETA: estimated seconds to complete remaining jobs
        pending = status_counts.get("pending", 0)
        assigned = status_counts.get("assigned", 0)
        processing = status_counts.get("processing", 0)
        # Exclude download_waiting from remaining (can't be processed yet)
        download_waiting = file_ready_stats.get("download_waiting", 0)
        remaining = pending - download_waiting + assigned + processing
        if throughput > 0 and remaining > 0:
            eta_seconds = round((remaining / throughput) * 60)
        else:
            eta_seconds = None

        # ── File-centric counts ──
        cursor.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM files f
            WHERE (f.mc_caption IS NOT NULL AND f.mc_caption != '')
              AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
              AND EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
        """)
        complete_files = cursor.fetchone()[0]

        # Phase progress: only incomplete files (exclude fully done files)
        # Shows remaining work, not total inventory
        incomplete = total_files - complete_files
        if incomplete > 0:
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN f.mc_caption IS NOT NULL AND f.mc_caption != ''
                        THEN 1 ELSE 0 END),
                    SUM(CASE WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                        THEN 1 ELSE 0 END),
                    SUM(CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
                        THEN 1 ELSE 0 END)
                FROM files f
                WHERE NOT (
                    (f.mc_caption IS NOT NULL AND f.mc_caption != '')
                    AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
                    AND EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
                )
            """)
            inc_row = cursor.fetchone()
            mc_done = inc_row[0] or 0
            vv_done = inc_row[1] or 0
            mv_done = inc_row[2] or 0
        else:
            mc_done = vv_done = mv_done = 0

        phase_stats = {
            "phase_total": incomplete,  # denominator: incomplete files only
            "phase_parse_done": max(0, incomplete - download_waiting),  # exclude files awaiting download
            "phase_vision_done": mc_done,
            "phase_embed_done": min(vv_done, mv_done),
        }

        # Failed files count (from files table, not job_queue)
        cursor.execute(
            "SELECT COUNT(*) FROM files WHERE processing_status = 'failed'"
        )
        failed_files = cursor.fetchone()[0]

        self.db.conn.commit()  # commit the pruning DELETE above

        return {
            "total": total_files,
            "total_files": total_files,
            "complete_files": complete_files,
            "pending": pending,
            "assigned": assigned,
            "processing": processing,
            "completed": complete_files,  # files-based: fully processed
            "failed": failed_files,       # files-based: permanently failed
            "throughput": throughput,
            "recent_1min": recent_1min,
            "recent_5min": recent_5min,
            "eta_seconds": eta_seconds,
            **phase_stats,
            **parse_ahead_stats,
            **file_ready_stats,
            "download_buffer": download_buffer,
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
        """Retry retryable failed files by creating new pending jobs.

        Skips non-retryable errors (FILE_NOT_FOUND, PARSE_FAILED).
        Resets processing_status and creates fresh pending jobs.
        """
        cursor = self.db.conn.cursor()

        # Find retryable failed files (exclude FILE_NOT_FOUND and PARSE_FAILED)
        non_retryable_patterns = [f"%{code}%" for code in NON_RETRYABLE_ERRORS]
        cursor.execute("""
            SELECT id, file_path, processing_error FROM files
            WHERE processing_status = 'failed'
        """)
        failed_files = cursor.fetchall()

        count = 0
        for file_id, file_path, error in failed_files:
            # Skip non-retryable errors
            if error:
                skip = False
                for code in NON_RETRYABLE_ERRORS:
                    if code.lower() in error.lower():
                        skip = True
                        break
                if skip:
                    continue

            # Reset processing status
            cursor.execute(
                "UPDATE files SET processing_status = NULL, processing_error = NULL WHERE id = ?",
                (file_id,)
            )
            # Create new pending job (if not already in queue)
            cursor.execute(
                "SELECT COUNT(*) FROM job_queue WHERE file_id = ?", (file_id,)
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    """INSERT INTO job_queue (file_id, file_path, status, priority, max_retries)
                       VALUES (?, ?, 'pending', 0, 3)""",
                    (file_id, file_path)
                )
                count += 1

        self.db.conn.commit()
        if count > 0:
            logger.info(f"Retried {count} failed files")
        return count

    def force_retry_failed_jobs(self) -> int:
        """Force retry ALL permanently failed files from scratch.

        Finds files with processing_status='failed', clears their AI data,
        resets processing_status, and creates new pending jobs.
        Also handles any remaining failed/cancelled jobs in queue (legacy).
        """
        cursor = self.db.conn.cursor()

        # 1. Find files marked as permanently failed
        cursor.execute(
            "SELECT id, file_path FROM files WHERE processing_status = 'failed'"
        )
        failed_files = cursor.fetchall()
        file_ids = [r[0] for r in failed_files]

        if not file_ids:
            # Also check legacy failed/cancelled jobs still in queue
            cursor.execute(
                "SELECT file_id FROM job_queue WHERE status IN ('failed', 'cancelled')"
            )
            legacy_ids = [r[0] for r in cursor.fetchall() if r[0]]
            if legacy_ids:
                # Clean up legacy: delete those jobs
                cursor.execute(
                    f"DELETE FROM job_queue WHERE status IN ('failed', 'cancelled')"
                )
                self.db.conn.commit()
                logger.info(f"Cleaned up {cursor.rowcount} legacy failed/cancelled jobs")
            return 0

        placeholders = ",".join("?" * len(file_ids))

        # 2. Clear partial AI data from files table
        cursor.execute(
            f"""UPDATE files
                SET mc_caption = NULL, ai_tags = NULL,
                    image_type = NULL, scene_type = NULL, art_style = NULL,
                    processing_status = NULL, processing_error = NULL
                WHERE id IN ({placeholders})""",
            file_ids,
        )

        # 3. Delete vector data (VV + MV)
        cursor.execute(
            f"DELETE FROM vec_files WHERE file_id IN ({placeholders})",
            file_ids,
        )
        cursor.execute(
            f"DELETE FROM vec_text WHERE file_id IN ({placeholders})",
            file_ids,
        )
        logger.info(f"Cleared AI data for {len(file_ids)} files (mc/vv/mv)")

        # 4. Delete any existing jobs for these files (cleanup)
        cursor.execute(
            f"DELETE FROM job_queue WHERE file_id IN ({placeholders})",
            file_ids,
        )

        # 5. Create new pending jobs
        count = 0
        for file_id, file_path in failed_files:
            cursor.execute(
                """INSERT INTO job_queue (file_id, file_path, status, priority, max_retries)
                   VALUES (?, ?, 'pending', 0, 3)""",
                (file_id, file_path)
            )
            count += 1

        self.db.conn.commit()
        if count > 0:
            logger.info(f"Force-retried {count} permanently failed files from scratch")
        return count

    def audit_completed_jobs(self) -> Dict[str, Any]:
        """Full data integrity audit across ALL files in the database.

        2-pass design (simplified for TODO-only queue):

        Pass 1: File-centric scan
          - Complete (mc+vv+mv) → OK, delete any residual jobs
          - Permanently failed (processing_status='failed') → SKIP (count only)
          - Incomplete + no pending job → CREATE pending job
          - Incomplete + pending job exists → OK (already queued)

        Pass 2: Unmatched job cleanup
          - Jobs with file_id not in files → DELETE

        Returns file-centric results.
        """
        cursor = self.db.conn.cursor()
        details = []
        perm_failed_details = []
        repaired_files = 0
        skipped_non_retryable = 0

        # ── Pass 1: Scan ALL files for data completeness ──
        cursor.execute("""
            SELECT f.id, f.file_path,
                   (f.mc_caption IS NOT NULL AND f.mc_caption != '') AS has_mc,
                   EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) AS has_vv,
                   EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) AS has_mv,
                   f.thumbnail_url,
                   f.processing_status, f.processing_error
            FROM files f
        """)
        all_files = cursor.fetchall()
        total_files = len(all_files)
        complete_files = 0

        for (file_id, file_path, has_mc, has_vv, has_mv,
             thumbnail_url, proc_status_col, proc_error) in all_files:

            if has_mc and has_vv and has_mv:
                complete_files += 1
                # Delete any residual jobs for this file
                cursor.execute(
                    "DELETE FROM job_queue WHERE file_id = ?", (file_id,)
                )
                # Clear processing_status if somehow set on a complete file
                if proc_status_col:
                    cursor.execute(
                        "UPDATE files SET processing_status = NULL, processing_error = NULL WHERE id = ?",
                        (file_id,)
                    )
                continue

            # Check if permanently failed
            if proc_status_col == 'failed':
                skipped_non_retryable += 1
                perm_failed_details.append({
                    "file_id": file_id,
                    "file_path": file_path,
                    "error": proc_error or "unknown",
                })
                # Ensure no jobs exist for permanently failed files
                cursor.execute(
                    "DELETE FROM job_queue WHERE file_id = ?", (file_id,)
                )
                continue

            # Incomplete file — ensure a pending job exists
            missing = []
            if not has_mc:
                missing.append("mc")
            if not has_vv:
                missing.append("vv")
            if not has_mv:
                missing.append("mv")

            # Check if a pending/assigned/processing job already exists
            cursor.execute(
                """SELECT id FROM job_queue
                   WHERE file_id = ? AND status IN ('pending', 'assigned', 'processing')
                   LIMIT 1""",
                (file_id,)
            )
            existing_job = cursor.fetchone()

            if existing_job:
                # Already in pipeline — update phase_completed to reflect actual state
                actual_phases = json.dumps({
                    "parse": True,
                    "vision": bool(has_mc),
                    "embed": bool(has_vv and has_mv),
                })
                parsed_metadata = json.dumps({
                    "metadata": {},
                    "thumb_path": thumbnail_url,
                    "mc_raw": None,
                }, ensure_ascii=False)
                cursor.execute(
                    """UPDATE job_queue
                       SET phase_completed = ?,
                           parsed_metadata = COALESCE(parsed_metadata, ?)
                       WHERE id = ?""",
                    (actual_phases, parsed_metadata, existing_job[0])
                )
            else:
                # No job — create one
                actual_phases = json.dumps({
                    "parse": True,
                    "vision": bool(has_mc),
                    "embed": bool(has_vv and has_mv),
                })
                parsed_metadata = json.dumps({
                    "metadata": {},
                    "thumb_path": thumbnail_url,
                    "mc_raw": None,
                }, ensure_ascii=False)
                is_webdav = file_path.startswith("webdav://") if file_path else False
                audit_file_ready = 0 if is_webdav else 1
                try:
                    cursor.execute(
                        """INSERT INTO job_queue
                           (file_id, file_path, status, priority,
                            phase_completed, parse_status, parsed_metadata, file_ready)
                           VALUES (?, ?, 'pending', 0, ?, 'parsed', ?, ?)""",
                        (file_id, file_path, actual_phases, parsed_metadata, audit_file_ready)
                    )
                    repaired_files += 1
                    details.append({
                        "file_id": file_id,
                        "file_path": file_path,
                        "missing": missing,
                        "status": "download_waiting" if is_webdav else "no_job",
                    })
                except Exception as e:
                    logger.warning(f"Audit: failed to create job for file_id={file_id}: {e}")

        # ── Pass 2: Unmatched job cleanup ──
        # Delete jobs whose file_id no longer exists in files table
        cursor.execute("""
            DELETE FROM job_queue
            WHERE file_id IS NOT NULL
              AND NOT EXISTS(SELECT 1 FROM files WHERE id = job_queue.file_id)
        """)
        dangling_removed = cursor.rowcount

        # Also clean up any legacy completed/failed jobs still in queue
        cursor.execute(
            "DELETE FROM job_queue WHERE status IN ('completed', 'failed', 'cancelled')"
        )
        legacy_removed = cursor.rowcount

        # Deduplicate: if multiple pending jobs exist for same file, keep only latest
        cursor.execute("""
            SELECT jq.id FROM job_queue jq
            WHERE jq.status = 'pending'
              AND EXISTS(
                  SELECT 1 FROM job_queue jq2
                  WHERE jq2.file_id = jq.file_id
                    AND jq2.id > jq.id
                    AND jq2.status = 'pending'
              )
        """)
        dup_ids = [r[0] for r in cursor.fetchall()]
        if dup_ids:
            placeholders = ",".join("?" * len(dup_ids))
            cursor.execute(
                f"DELETE FROM job_queue WHERE id IN ({placeholders})", dup_ids
            )

        self.db.conn.commit()

        incomplete_files = total_files - complete_files

        if repaired_files > 0 or skipped_non_retryable > 0:
            parts = [f"Audit: {total_files} files, {incomplete_files} incomplete"]
            if repaired_files > 0:
                parts.append(f"{repaired_files} repaired")
            if skipped_non_retryable > 0:
                parts.append(f"{skipped_non_retryable} permanently failed (skipped)")
            if dangling_removed > 0:
                parts.append(f"{dangling_removed} unmatched jobs removed")
            if repaired_files > 0:
                logger.warning(", ".join(parts))
            else:
                logger.info(", ".join(parts))
        else:
            logger.info(f"Audit: {total_files} files scanned, all complete")

        return {
            "total_files": total_files,
            "complete_files": complete_files,
            "incomplete_files": incomplete_files,
            "repaired_files": repaired_files,
            "skipped_non_retryable": skipped_non_retryable,
            "failed_stuck_jobs": skipped_non_retryable,  # files-based
            "details": details,
            "permanently_failed_details": perm_failed_details,
        }

    def clear_completed_jobs(self) -> int:
        """Delete any remaining completed jobs (legacy cleanup).

        In the new design, completed jobs are deleted immediately.
        This method handles any legacy completed jobs still in queue.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM job_queue WHERE status = 'completed'")
        self.db.conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleared {count} legacy completed jobs")
        return count

    def cleanup_queue(self) -> Dict[str, int]:
        """Comprehensive job queue cleanup.

        Removes:
        1. Legacy completed/failed jobs (should not exist in new design)
        2. Duplicate pending jobs (same file_id, keep only latest)
        3. Jobs referencing non-existent files (file_id not in files table)
        4. Jobs for already-complete files (mc+vv+mv all present)

        Returns counts of each type removed.
        """
        cursor = self.db.conn.cursor()
        removed_legacy = 0
        removed_duplicates = 0
        removed_dangling = 0
        removed_complete = 0

        # 1. Legacy completed/failed jobs (should not exist anymore)
        cursor.execute(
            "DELETE FROM job_queue WHERE status IN ('completed', 'failed', 'cancelled')"
        )
        removed_legacy = cursor.rowcount

        # 2. Duplicate pending jobs (keep only latest per file_id)
        cursor.execute("""
            SELECT jq.id FROM job_queue jq
            WHERE EXISTS(
                SELECT 1 FROM job_queue jq2
                WHERE jq2.file_id = jq.file_id
                  AND jq2.id > jq.id
                  AND jq2.status = 'pending'
            )
            AND jq.status = 'pending'
        """)
        ids = [r[0] for r in cursor.fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"DELETE FROM job_queue WHERE id IN ({placeholders})", ids
            )
            removed_duplicates = len(ids)

        # 3. Jobs referencing non-existent files
        cursor.execute("""
            SELECT jq.id FROM job_queue jq
            WHERE jq.file_id IS NOT NULL
              AND NOT EXISTS(SELECT 1 FROM files WHERE id = jq.file_id)
        """)
        ids = [r[0] for r in cursor.fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"DELETE FROM job_queue WHERE id IN ({placeholders})", ids
            )
            removed_dangling = len(ids)

        # 4. Jobs for already-complete files (mc+vv+mv present)
        cursor.execute("""
            SELECT jq.id FROM job_queue jq
            WHERE jq.file_id IS NOT NULL
              AND jq.status = 'pending'
              AND EXISTS(
                  SELECT 1 FROM files f
                  WHERE f.id = jq.file_id
                    AND f.mc_caption IS NOT NULL AND f.mc_caption != ''
              )
              AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = jq.file_id)
              AND EXISTS(SELECT 1 FROM vec_text WHERE file_id = jq.file_id)
        """)
        ids = [r[0] for r in cursor.fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"DELETE FROM job_queue WHERE id IN ({placeholders})", ids
            )
            removed_complete = len(ids)

        self.db.conn.commit()

        total = removed_legacy + removed_duplicates + removed_dangling + removed_complete
        if total > 0:
            logger.info(
                f"Queue cleanup: {removed_legacy} legacy, "
                f"{removed_duplicates} duplicates, "
                f"{removed_dangling} unmatched, "
                f"{removed_complete} already-complete — {total} total removed"
            )

        return {
            "removed_completed": removed_legacy + removed_complete,
            "removed_duplicates": removed_duplicates,
            "removed_dangling": removed_dangling,
            "total_removed": total,
        }

    def dismiss_permanently_failed_jobs(self) -> Dict[str, Any]:
        """Delete permanently failed files and their incomplete data.

        Finds files with processing_status='failed', removes their incomplete
        data (mc, vv, mv, fts), and deletes the file records.

        Returns:
            {"dismissed_jobs": 0, "cleaned_files": int}
        """
        cursor = self.db.conn.cursor()

        # Find files marked as permanently failed
        cursor.execute(
            "SELECT id FROM files WHERE processing_status = 'failed'"
        )
        file_ids = [r[0] for r in cursor.fetchall()]

        if not file_ids:
            # Also cleanup any legacy failed jobs still in queue
            cursor.execute(
                "DELETE FROM job_queue WHERE status IN ('failed', 'cancelled')"
            )
            legacy_cleaned = cursor.rowcount
            self.db.conn.commit()
            if legacy_cleaned > 0:
                logger.info(f"Cleaned {legacy_cleaned} legacy failed/cancelled jobs from queue")
            return {"dismissed_jobs": legacy_cleaned, "cleaned_files": 0}

        # Clean up file records and their data
        cleaned_files = 0
        for fid in file_ids:
            # Check if file data is incomplete (missing mc/vv/mv)
            cursor.execute("""
                SELECT (mc_caption IS NOT NULL AND mc_caption != '') AS has_mc,
                       EXISTS(SELECT 1 FROM vec_files WHERE file_id = ?) AS has_vv,
                       EXISTS(SELECT 1 FROM vec_text WHERE file_id = ?) AS has_mv
                FROM files WHERE id = ?
            """, (fid, fid, fid))
            row = cursor.fetchone()
            if not row:
                continue
            has_mc, has_vv, has_mv = row
            if has_mc and has_vv and has_mv:
                # File is actually complete — just clear the failed status
                cursor.execute(
                    "UPDATE files SET processing_status = NULL, processing_error = NULL WHERE id = ?",
                    (fid,)
                )
                continue

            # Incomplete file — clean up everything
            cursor.execute("DELETE FROM vec_files WHERE file_id = ?", (fid,))
            cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (fid,))
            cursor.execute("DELETE FROM files_fts WHERE rowid = ?", (fid,))
            cursor.execute("DELETE FROM job_queue WHERE file_id = ?", (fid,))
            cursor.execute("DELETE FROM files WHERE id = ?", (fid,))
            cleaned_files += 1

        self.db.conn.commit()
        logger.info(f"Dismissed {cleaned_files} permanently failed file records")
        return {"dismissed_jobs": 0, "cleaned_files": cleaned_files}

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
