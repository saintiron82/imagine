"""
Parse-ahead pool — server-side Phase P pre-parser (tollgate architecture).

Monitors connected workers' total capacity and pre-parses pending jobs
so that workers receive thumbnails (~200KB) instead of raw files (~500MB).

The pool runs as a background daemon thread, continuously maintaining
a buffer of pre-parsed jobs proportional to worker demand.

Mode: parse_only — Server does Phase P only (zero GPU models loaded).
All GPU work (V+VV+MV) is delegated to workers running in full mode.
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
    """Server-side pre-parser that runs Phase P ahead of worker demand.

    Tollgate architecture: parse_only mode fixed.
    Server does Phase P only (zero GPU), workers do V+VV+MV.
    """

    def __init__(self, db):
        super().__init__(db)
        self._processing_mode = "parse_only"  # Fixed: tollgate architecture
        self._vv_encoder = None  # Used by _process_backfill_batch (DINOv2)
        self._structure_encoder = None  # Used by _process_backfill_batch (DINOv2)
        self._last_retry_reset = 0.0  # Timestamp of last parse_status='failed' reset
        logger.info("ParseAheadPool initialized (parse_only mode — tollgate architecture)")

        # Auto-audit on startup: repair completed jobs with missing data
        self._startup_integrity_audit()

    def _startup_integrity_audit(self):
        """Run integrity audit on server startup to repair incomplete files."""
        try:
            from backend.server.queue.manager import JobQueueManager
            mgr = JobQueueManager(self.db)
            result = mgr.audit_completed_jobs()
            if result["repaired_files"] > 0:
                logger.warning(
                    f"Startup audit: {result['total_files']} files, "
                    f"{result['repaired_files']} incomplete → repaired"
                )
            else:
                logger.info(f"Startup audit: {result['total_files']} files, all complete")
        except Exception as e:
            logger.warning(f"Startup integrity audit failed (non-fatal): {e}")

    def _unload_models(self):
        """Unload VV and Structure encoders if loaded (backfill mode)."""
        if self._vv_encoder is not None:
            try:
                self._vv_encoder.unload()
                logger.info("ParseAheadPool: VV encoder unloaded")
            except Exception as e:
                logger.warning(f"ParseAheadPool: VV encoder unload error: {e}")
            self._vv_encoder = None
        if self._structure_encoder is not None:
            try:
                self._structure_encoder.unload()
                logger.info("ParseAheadPool: DINOv2 Structure encoder unloaded")
            except Exception as e:
                logger.warning(f"ParseAheadPool: DINOv2 Structure encoder unload error: {e}")
            self._structure_encoder = None

    # ── Buffer management ────────────────────────────────────────

    def _calculate_buffer_target(self) -> int:
        """Calculate how many pre-parsed jobs to maintain.

        Demand-driven: uses actual worker claim counts as the prediction.
        Each worker's last claim count is recorded by JobQueueManager,
        and we sum them to get the total expected demand.

        Returns:
            Sum of recent per-worker claim counts, or 0 if no demand.
        """
        if not self.has_recent_demand():
            return 0

        return self.get_total_demand()

    def _run_pre_parse_buffer(self) -> bool:
        """Run one cycle of pre-parse buffer filling.

        Calculates target based on worker demand, finds unparsed jobs,
        and pre-parses them (Phase P only, with VV in mc_only mode).

        Returns True if at least one job was parsed.
        """
        target = self._calculate_buffer_target()
        if target <= 0:
            self._process_backfill_batch()
            return False

        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM job_queue "
            "WHERE parse_status = 'parsed' AND status = 'pending'"
        )
        current_parsed = cursor.fetchone()[0]
        deficit = target - current_parsed

        if deficit <= 0:
            self._process_backfill_batch()
            return False

        # Select jobs to pre-parse (include WebDAV files with temp downloads)
        cursor.execute(
            """SELECT id, file_id, file_path FROM job_queue
               WHERE status = 'pending'
                 AND (parse_status IS NULL OR parse_status = 'pending')
                 AND (file_path NOT LIKE 'webdav://%'
                      OR parsed_metadata LIKE '%temp_local_path%')
               ORDER BY priority DESC, created_at ASC
               LIMIT ?""",
            (deficit,),
        )
        jobs_to_parse = cursor.fetchall()

        if not jobs_to_parse:
            # No unparsed jobs — check if all are parse-failed (deadlock).
            # Reset parse_status='failed' back to NULL every 60s for retry.
            now = time.time()
            if now - self._last_retry_reset > 60:
                cursor.execute(
                    """UPDATE job_queue SET parse_status = NULL
                       WHERE status = 'pending' AND parse_status = 'failed'"""
                )
                if cursor.rowcount > 0:
                    logger.info(
                        f"ParseAhead: reset {cursor.rowcount} parse-failed "
                        f"jobs for retry"
                    )
                self.db.conn.commit()  # Always commit to release WAL write lock
                self._last_retry_reset = now
            self._process_backfill_batch()
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
                    "UPDATE job_queue SET parse_status = 'parsed', parsed_at = ? WHERE id = ?",
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
        """Main loop: Phase P only (parse_only mode fixed).

        Tollgate architecture: server pre-parses pending jobs (Phase P)
        and workers handle AI processing (V→VV→MV).
        """
        logger.info("ParseAheadPool loop started (parse_only mode)")
        poll_interval_s = self._get_config_value("server.parse_ahead.poll_interval_s", 2)

        # Auto-queue backfill jobs on startup
        try:
            from backend.server.queue.manager import JobQueueManager
            mgr = JobQueueManager(self.db)
            backfill_counts = mgr.queue_backfill()
            total = sum(backfill_counts.values())
            if total > 0:
                logger.info(f"ParseAheadPool: auto-queued {total} backfill jobs: {backfill_counts}")
        except Exception as e:
            logger.warning(f"ParseAheadPool: backfill queue scan failed: {e}")

        try:
            while self._running:
                try:
                    # Periodic diagnostics
                    if not hasattr(self, '_diag_counter'):
                        self._diag_counter = 0
                    self._diag_counter += 1
                    if self._diag_counter % 15 == 1:  # Every ~30s
                        target = self._calculate_buffer_target()
                        demand = self.has_recent_demand()
                        logger.info(
                            f"[PA-DIAG] mode=parse_only "
                            f"demand={demand} target={target}"
                        )

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

    def _process_backfill_batch(self, batch_size: int = 8) -> int:
        """Process queued backfill jobs (DINOv2 structure vector only).

        Picks up jobs with parse_status='backfill', generates the missing
        structure vector, and marks them completed. Runs during idle time
        without interfering with normal parsing.

        Returns:
            Number of jobs processed.
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, file_id, file_path FROM job_queue
               WHERE status = 'pending' AND parse_status = 'backfill'
               ORDER BY created_at ASC
               LIMIT ?""",
            (batch_size,),
        )
        jobs = cursor.fetchall()
        if not jobs:
            return 0

        # Lazy load DINOv2
        if self._structure_encoder is None:
            from backend.vector.dinov2_encoder import DinoV2Encoder
            self._structure_encoder = DinoV2Encoder()
            logger.info("ParseAheadPool: DINOv2 loaded for structure backfill")

        from PIL import Image

        processed = 0
        for job_id, file_id, file_path in jobs:
            if not self._running:
                break

            # Atomically claim
            cursor.execute(
                "UPDATE job_queue SET status = 'processing' WHERE id = ? AND status = 'pending'",
                (job_id,),
            )
            self.db.conn.commit()
            if cursor.rowcount == 0:
                continue

            # Find best image source (thumbnail preferred)
            cursor.execute(
                "SELECT thumbnail_url FROM files WHERE id = ?", (file_id,)
            )
            row = cursor.fetchone()
            thumb_url = row[0] if row else None

            img_source = None
            if thumb_url:
                p = Path(thumb_url)
                if p.exists():
                    img_source = p
            if img_source is None:
                p = Path(file_path)
                if p.exists():
                    img_source = p

            now = _utcnow_sql()
            if img_source is None:
                logger.warning(f"Backfill: no image for job {job_id} (file_id={file_id}), marking failed")
                cursor.execute(
                    "UPDATE files SET processing_status = 'failed', processing_error = 'No image for backfill' WHERE id = ?",
                    (file_id,),
                )
                cursor.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
                self.db.conn.commit()
                continue

            try:
                img = Image.open(str(img_source)).convert("RGB")
                try:
                    structure_vec = self._structure_encoder.encode_image(img)
                finally:
                    img.close()
                self.db.upsert_vectors(file_id, structure_vec=structure_vec)
                # Complete — log and DELETE
                cursor.execute(
                    "INSERT INTO job_completions (file_id) VALUES (?)",
                    (file_id,),
                )
                cursor.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
                processed += 1
            except Exception as e:
                logger.warning(f"Backfill: DINOv2 failed for job {job_id}: {e}")
                cursor.execute(
                    "UPDATE files SET processing_status = 'failed', processing_error = ? WHERE id = ?",
                    (str(e), file_id),
                )
                cursor.execute("DELETE FROM job_queue WHERE id = ?", (job_id,))
            self.db.conn.commit()

        if processed > 0:
            logger.info(f"Backfill: completed {processed} structure vector jobs")
        return processed

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
