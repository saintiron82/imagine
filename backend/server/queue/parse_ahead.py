"""
Parse-ahead pool — server-side Phase P pre-parser.

PSD 파싱 + 썸네일 생성만 담당. AI 모델 없음.
워커는 pre-parsed 결과(썸네일 + 메타데이터)를 받아 AI 처리(MC/VV/MV)만 수행.
"""

import json
import logging
import shutil
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Optional

from backend.server.queue.base_ahead_pool import BaseAheadPool
from backend.server.queue.manager import _utcnow_sql
from backend.utils.meta_helpers import meta_to_dict

logger = logging.getLogger(__name__)


class ParseAheadPool(BaseAheadPool):
    """Server-side pre-parser: PSD 파싱 + 썸네일 생성만 담당.

    AI 모델 없음. 워커가 AI 처리(MC/VV/MV) 전담.
    """

    def __init__(self, db):
        super().__init__(db)
        self._processing_mode = "parse_only"  # Fixed: parse + thumbnail only
        # Backfill (DINOv2) removed — AI models are worker-only
        self._last_retry_reset = 0.0  # Timestamp of last parse_status='failed' reset
        logger.info("ParseAheadPool initialized (parse + thumbnail only)")

    def _unload_models(self):
        """No AI models to unload — parse + thumbnail only."""
        pass

    # ── Buffer management ────────────────────────────────────────

    def _calculate_buffer_target(self) -> int:
        """Calculate how many pre-parsed jobs to maintain.

        ParseAheadPool runs independently — always parses downloaded files.
        Each stage (download → parse → worker) runs at its own pace with
        buffers in between, not synchronized to each other.
        """
        # Always parse downloaded files (file_ready=1, not yet parsed)
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) FROM job_queue
                   WHERE status = 'pending' AND file_ready = 1
                   AND (parse_status IS NULL OR parse_status = 'pending')"""
            )
            unparsed_ready = cursor.fetchone()[0]
            if unparsed_ready > 0:
                return unparsed_ready  # Parse everything that's downloaded
        except Exception:
            pass

        # Worker demand: maintain buffer for workers to claim
        if self.has_recent_demand():
            return max(self.get_total_demand(), 10)

        # Workers online but not claiming yet: keep minimum buffer
        try:
            from backend.server.embedded_worker import get_status
            if get_status().get("running"):
                return 10
        except Exception:
            pass
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM worker_sessions WHERE status = 'online'"
            )
            if cursor.fetchone()[0] > 0:
                return 10
        except Exception:
            pass

        return 0

    def _self_repair(self):
        """Fix invariant violations periodically. Silent if nothing to fix."""
        try:
            cursor = self.db.conn.cursor()
            fixed = 0

            # 0. FTS entries with empty content → rebuild from file data
            cursor.execute(
                "SELECT rowid FROM files_fts WHERE meta_strong = '' AND meta_weak = '' LIMIT 100"
            )
            empty_fts = [r[0] for r in cursor.fetchall()]
            if empty_fts:
                for fid in empty_fts:
                    try:
                        self.db._refresh_fts_row(cursor, fid)
                    except Exception:
                        pass
                fixed += len(empty_fts)
                logger.warning(f"Self-repair: rebuilt {len(empty_fts)} empty FTS entries")

            # 1. Invariant: parsed → file_ready=1
            cursor.execute(
                "UPDATE job_queue SET file_ready = 1 WHERE parse_status = 'parsed' AND file_ready = 0"
            )
            if cursor.rowcount > 0:
                fixed += cursor.rowcount
                logger.warning(f"Self-repair: {cursor.rowcount} parsed jobs had file_ready=0 → fixed")

            # 2. WR counter mismatch: no jobs left but completed_count < total
            #    Files may be fully done but WR counter wasn't updated
            cursor.execute("""
                SELECT wr.id, wr.name, wr.total_files, wr.completed_count
                FROM work_requests wr
                WHERE wr.status IN ('queued', 'processing')
                  AND wr.completed_count < wr.total_files
                  AND NOT EXISTS (
                    SELECT 1 FROM job_queue jq WHERE jq.work_request_id = wr.id
                  )
            """)
            stale_wrs = cursor.fetchall()
            for wr_id, wr_name, wr_total, wr_done in stale_wrs:
                # Check if all files are actually complete
                cursor.execute("""
                    SELECT COUNT(*) FROM work_subtasks ws
                    JOIN job_queue jq ON jq.work_subtask_id = ws.id
                    WHERE ws.work_request_id = ?
                """, (wr_id,))
                pending_jobs = cursor.fetchone()[0]

                if pending_jobs == 0:
                    # No jobs → mark WR as completed (files are done)
                    cursor.execute(
                        "UPDATE work_requests SET completed_count = total_files, status = 'completed', completed_at = datetime('now') WHERE id = ?",
                        (wr_id,)
                    )
                    cursor.execute(
                        "UPDATE work_subtasks SET completed_count = total_files WHERE work_request_id = ?",
                        (wr_id,)
                    )
                    fixed += 1
                    logger.warning(
                        f"Self-repair: WR '{wr_name}' counter {wr_done}/{wr_total} → "
                        f"marked completed (no pending jobs, files are done)"
                    )

            if fixed > 0:
                self.db.conn.commit()
            else:
                self.db.conn.rollback()
        except Exception:
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _run_pre_parse_buffer(self) -> bool:
        """Run one cycle of pre-parse buffer filling.

        Calculates target based on worker demand, finds unparsed jobs,
        and pre-parses them (Phase P only, with VV in mc_only mode).

        Returns True if at least one job was parsed.
        """
        target = self._calculate_buffer_target()
        if target <= 0:
            return False

        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM job_queue "
            "WHERE parse_status = 'parsed' AND status = 'pending' AND file_ready = 1"
        )
        current_parsed = cursor.fetchone()[0]
        deficit = target - current_parsed

        if deficit <= 0:
            return False

        # Select jobs to pre-parse: must be file_ready=1
        # (local files are always ready, WebDAV files need download first)
        cursor.execute(
            """SELECT id, file_id, file_path FROM job_queue
               WHERE status = 'pending'
                 AND file_ready = 1
                 AND (parse_status IS NULL OR parse_status = 'pending')
               ORDER BY priority DESC, created_at ASC
               LIMIT ?""",
            (deficit,),
        )
        jobs_to_parse = cursor.fetchall()

        if not jobs_to_parse:
            # No unparsed jobs — retry parse-failed jobs that are actually downloadable.
            # Only retry if file_ready=1 (temp file exists) and retry_count < 3.
            now = time.time()
            if now - self._last_retry_reset > 60:
                cursor.execute(
                    """UPDATE job_queue SET parse_status = NULL, retry_count = COALESCE(retry_count, 0) + 1
                       WHERE status = 'pending' AND parse_status = 'failed'
                       AND file_ready = 1
                       AND COALESCE(retry_count, 0) < 3"""
                )
                if cursor.rowcount > 0:
                    logger.info(
                        f"ParseAhead: reset {cursor.rowcount} parse-failed "
                        f"jobs for retry (max 3 attempts)"
                    )
                self.db.conn.commit()
                self._last_retry_reset = now
            return False

        parsed_count = 0
        for row in jobs_to_parse:
            if not self._running:
                break

            job_id, file_id, file_path = row

            # Atomically claim the job for parsing (prevent race condition)
            cursor.execute(
                """UPDATE job_queue
                   SET parse_status = 'parsing'
                   WHERE id = ?
                     AND (parse_status IS NULL OR parse_status = 'pending')
                     AND status = 'pending'""",
                (job_id,),
            )
            self.db.conn.commit()

            if cursor.rowcount == 0:
                # Another thread/process claimed it already
                continue

            success = False
            try:
                success = self._parse_single_job(job_id, file_id, file_path)
            except Exception as e:
                logger.error(
                    f"ParseAhead job {job_id} exception: {e}\n"
                    f"{traceback.format_exc()}"
                )

            now = _utcnow_sql()
            if success:
                cursor.execute(
                    "UPDATE job_queue SET parse_status = 'parsed', file_ready = 1, parsed_at = ? WHERE id = ?",
                    (now, job_id),
                )
                parsed_count += 1

                # Release DL cache immediately after successful parse (tollgate C1).
                # Thumbnail is now generated — original file no longer needed.
                if file_path and file_path.startswith("webdav://"):
                    try:
                        from backend.server.queue.manager import _get_download_pool
                        pool = _get_download_pool()
                        if pool:
                            pool.release_slot(file_id)
                            logger.debug(
                                f"DL cache released after parse: file_id={file_id}"
                            )
                    except Exception as e:
                        logger.warning(f"DL cache release failed for file_id={file_id}: {e}")
            else:
                # Parse failed — keep DL cache for retry (tollgate C2)
                cursor.execute(
                    "UPDATE job_queue SET parse_status = 'failed' WHERE id = ?",
                    (job_id,),
                )
            self.db.conn.commit()

        return parsed_count > 0

    def _loop(self):
        """Main loop: Phase P only (PSD 파싱 + 썸네일 생성)."""
        logger.info("ParseAheadPool loop started (parse + thumbnail only)")
        poll_interval_s = self._get_config_value("server.parse_ahead.poll_interval_s", 2)

        try:
            while self._running:
                try:
                    if not hasattr(self, '_diag_counter'):
                        self._diag_counter = 0
                    self._diag_counter += 1

                    # Periodic diagnostics (only when active)
                    if self._diag_counter % 15 == 1 and self.has_recent_demand():
                        target = self._calculate_buffer_target()
                        logger.info(f"[PA-DIAG] target={target}")

                    # Periodic self-repair: fix invariant violations
                    repair_interval = self._get_config_value("server.parse_ahead.self_repair_interval_s", 600)
                    repair_ticks = max(1, int(repair_interval / poll_interval_s))
                    if self._diag_counter % repair_ticks == 0:
                        self._self_repair()

                    self._run_pre_parse_buffer()

                    time.sleep(poll_interval_s)

                except Exception as e:
                    logger.error(
                        f"ParseAheadPool iteration error: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    time.sleep(5)
                finally:
                    # Safety: release any WAL write lock from uncommitted DML.
                    try:
                        self.db.conn.rollback()
                    except Exception:
                        pass

        except Exception as e:
            logger.critical(
                f"ParseAheadPool loop crashed: {e}\n"
                f"{traceback.format_exc()}"
            )

        logger.info("ParseAheadPool loop exited")

    def _get_temp_file(self, job_id: int, file_path: str) -> Optional[Path]:
        """Look up a temp local copy of a WebDAV file from DownloadAheadPool.

        Checks both the in-memory registry and parsed_metadata in job_queue.
        Returns Path if found, None otherwise.
        """
        try:
            from backend.server.queue.download_ahead import DownloadAheadPool
            # Try DownloadAheadPool's active files registry
            pool = getattr(self, '_download_pool', None)
            if pool:
                # Get file_id from job
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "SELECT file_id, parsed_metadata FROM job_queue WHERE id = ?",
                    (job_id,),
                )
                row = cursor.fetchone()
                if row:
                    file_id = row[0]
                    temp_path = pool.get_temp_path(file_id)
                    if temp_path and Path(temp_path).exists():
                        return Path(temp_path)
                    # Also check parsed_metadata
                    if row[1]:
                        try:
                            import json
                            pm = json.loads(row[1])
                            tlp = pm.get("temp_local_path")
                            if tlp and Path(tlp).exists():
                                return Path(tlp)
                        except (json.JSONDecodeError, TypeError):
                            pass
            else:
                # No pool reference — check parsed_metadata directly
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "SELECT parsed_metadata FROM job_queue WHERE id = ?",
                    (job_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    try:
                        import json
                        pm = json.loads(row[0])
                        tlp = pm.get("temp_local_path")
                        if tlp and Path(tlp).exists():
                            return Path(tlp)
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception as e:
            logger.debug(f"_get_temp_file failed for job {job_id}: {e}")
        return None

    def _parse_single_job(self, job_id: int, file_id: int, file_path: str) -> bool:
        """Execute Phase P for a single job.

        Steps:
            1. Parse file with ParserFactory
            2. Compute content hash
            3. Set tier metadata
            4. Copy thumbnail to server thumbnails/ directory
            5. Upsert metadata to files table
            6. Build mc_raw context
            7. Store parsed_metadata JSON in job_queue

        Returns:
            True on success, False on failure.
        """
        from backend.pipeline.ingest_engine import (
            ParserFactory,
            _set_tier_metadata,
            _build_mc_raw,
        )
        from backend.utils.content_hash import compute_content_hash

        file_p = Path(file_path)
        if not file_p.exists():
            # WebDAV files: check if DownloadAhead has a temp copy
            if file_path.startswith("webdav://"):
                temp_path = self._get_temp_file(job_id, file_path)
                if temp_path:
                    file_p = temp_path
                else:
                    logger.debug(
                        f"ParseAhead: WebDAV file not yet downloaded: {file_path}"
                    )
                    return False
            else:
                logger.warning(f"ParseAhead: file not found: {file_path}")
                return False

        # 1. Parse
        parser = ParserFactory.get_parser(file_p)
        if not parser:
            logger.warning(f"ParseAhead: no parser for {file_path}")
            return False

        result = parser.parse(file_p)
        if not result.success:
            logger.warning(f"ParseAhead: parse failed for {file_path}: {result.errors}")
            return False

        meta = result.asset_meta

        # 2. Content hash
        try:
            meta.content_hash = compute_content_hash(file_p)
        except Exception as e:
            logger.warning(f"ParseAhead: content_hash failed: {e}")

        # 3. Folder metadata from path
        parent = file_p.parent
        if parent.name and parent.name not in (".", ""):
            meta.folder_path = parent.name
            meta.folder_depth = 0
            meta.folder_tags = [parent.name]

        # 4. Tier metadata
        _set_tier_metadata(meta)

        # 5. Copy thumbnail to server thumbnails/ directory
        server_thumb_path = None
        if meta.thumbnail_url:
            src_thumb = Path(meta.thumbnail_url)
            if src_thumb.exists():
                try:
                    thumb_dir = self._get_thumbnail_dir()
                    dest_name = f"{file_p.stem}_thumb.png"
                    server_thumb_path = thumb_dir / dest_name
                    shutil.copy2(str(src_thumb), str(server_thumb_path))
                    logger.debug(f"ParseAhead: thumbnail copied to {server_thumb_path}")
                except Exception as e:
                    logger.warning(f"ParseAhead: thumbnail copy failed: {e}")
                    server_thumb_path = None

        # 6. Upsert metadata to files table
        meta_dict = meta_to_dict(meta)
        # Use canonical path for DB storage (webdav:// for remote files)
        # file_p may be a temp local copy, but DB stores the original path
        canonical = file_path if file_path.startswith("webdav://") else str(file_p)
        nfc_path = unicodedata.normalize('NFC', canonical)
        meta_dict["file_path"] = nfc_path

        try:
            stored_file_id = self.db.upsert_metadata(nfc_path, meta_dict)
            logger.debug(f"ParseAhead: metadata upserted, file_id={stored_file_id}")
        except Exception as e:
            logger.error(f"ParseAhead: metadata upsert failed: {e}")
            return False

        # Update thumbnail_url in files table if we have a server copy
        if server_thumb_path:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "UPDATE files SET thumbnail_url = ? WHERE id = ?",
                    (str(server_thumb_path), stored_file_id),
                )
                # UPDATE trigger resets FTS to '' — must refresh
                self.db._refresh_fts_row(cursor, stored_file_id)
                self.db.conn.commit()
            except Exception as e:
                logger.warning(f"ParseAhead: thumbnail_url update failed: {e}")

        # 7. Build mc_raw context
        mc_raw = _build_mc_raw(meta)

        # 8. Construct parsed_metadata JSON
        parsed_metadata = {
            "metadata": meta_dict,
            "thumb_path": str(server_thumb_path) if server_thumb_path else None,
            "mc_raw": mc_raw,
        }

        # 9. Store parsed_metadata in job_queue
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "UPDATE job_queue SET parsed_metadata = ? WHERE id = ?",
                (json.dumps(parsed_metadata, ensure_ascii=False, default=str), job_id),
            )
            self.db.conn.commit()
        except Exception as e:
            logger.error(f"ParseAhead: parsed_metadata storage failed: {e}")
            return False

        logger.info(f"ParseAhead: job {job_id} pre-parsed OK ({file_p.name})")
        return True

    # _process_backfill_batch removed — DINOv2 is an AI model, belongs to worker

    def _get_thumbnail_dir(self) -> Path:
        """Get server thumbnail directory (same logic as upload.py)."""
        from backend.server.config import get_storage_config

        cfg = get_storage_config()
        thumb_dir = Path(cfg.get("thumbnail_dir", "./thumbnails"))
        thumb_dir.mkdir(parents=True, exist_ok=True)
        return thumb_dir

    def get_stats(self) -> dict:
        """Get current parse-ahead pool statistics.

        Returns:
            Dict with parsed_count, parsing_count, failed_count, buffer_target.
        """
        try:
            cursor = self.db.conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM job_queue "
                "WHERE parse_status = 'parsed' AND status = 'pending'"
            )
            parsed_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM job_queue WHERE parse_status = 'parsing'"
            )
            parsing_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM job_queue WHERE parse_status = 'failed'"
            )
            failed_count = cursor.fetchone()[0]

            buffer_target = self._calculate_buffer_target()

            return {
                "parsed_count": parsed_count,
                "parsing_count": parsing_count,
                "failed_count": failed_count,
                "buffer_target": buffer_target,
            }

        except Exception as e:
            logger.warning(f"Failed to get parse-ahead stats: {e}")
            return {
                "parsed_count": 0,
                "parsing_count": 0,
                "failed_count": 0,
                "buffer_target": 0,
            }
