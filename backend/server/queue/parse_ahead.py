"""
Parse-ahead pool — server-side Phase P pre-parser for worker optimization.

Monitors connected workers' total capacity and pre-parses pending jobs
so that workers receive thumbnails (~200KB) instead of raw files (~500MB).

The pool runs as a background daemon thread, continuously maintaining
a buffer of pre-parsed jobs proportional to worker demand.

mc_only mode: Also runs Phase VV (SigLIP2 + DINOv2) on parsed jobs since
VV/Structure only need the image (independent of MC). Both encoders stay
loaded for the session.

auto mode: When no workers are connected and auto_processing is enabled,
the server processes all phases (P→V→VV→MV) itself. Models are loaded
per-phase and unloaded between phases to minimize memory usage.
"""

import gc
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

    Monitors connected workers' total capacity and pre-parses
    pending jobs to have thumbnails + metadata ready before claim.

    In mc_only mode, also runs Phase VV (SigLIP2 visual embedding +
    DINOv2 structure embedding) since both only need the image pixel
    data, not MC text.
    """

    def __init__(self, db):
        super().__init__(db)
        from backend.server.queue.manager import get_processing_mode
        self._processing_mode = get_processing_mode()
        self._vv_encoder = None  # Lazy-loaded SigLIP2, stays resident in mc_only mode
        self._structure_encoder = None  # Lazy-loaded DINOv2, stays resident in mc_only mode
        logger.info(f"ParseAheadPool initialized (processing_mode={self._processing_mode})")

    def _unload_models(self):
        """Unload VV and Structure encoders if loaded (mc_only mode)."""
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

    # ── Auto mode: full pipeline P→V→VV→MV ────────────────────

    def _process_auto_batch(self) -> int:
        """Auto mode: server processes full pipeline P→V→VV→MV.

        Loads and unloads models per-phase to minimize GPU memory.
        Returns number of files processed.
        """
        batch_size = self._get_config_value("server.auto_processing.batch_size", 5)

        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, file_id, file_path FROM job_queue
               WHERE status = 'pending'
                 AND (parse_status IS NULL OR parse_status = 'pending')
               ORDER BY priority DESC, created_at ASC
               LIMIT ?""",
            (batch_size,),
        )
        jobs = cursor.fetchall()
        if not jobs:
            return 0

        logger.info(f"Auto processing: starting batch of {len(jobs)} files")

        # Mark jobs as processing
        now = _utcnow_sql()
        for job_id, _, _ in jobs:
            cursor.execute(
                """UPDATE job_queue
                   SET status = 'processing', started_at = ?, parse_status = 'parsing'
                   WHERE id = ? AND status = 'pending'""",
                (now, job_id),
            )
        self.db.conn.commit()

        contexts = []  # [(job_id, file_id, file_path, thumb_path, mc_raw)]

        # ── Phase P: Parse ──
        for job_id, file_id, file_path in jobs:
            if not self._running or self._processing_mode != "auto":
                break

            success = False
            try:
                success = self._parse_single_job(job_id, file_id, file_path)
            except Exception as e:
                logger.error(f"Auto Parse job {job_id}: {e}")

            if success:
                cursor.execute(
                    "SELECT parsed_metadata FROM job_queue WHERE id = ?", (job_id,)
                )
                row = cursor.fetchone()
                pm = json.loads(row[0]) if row and row[0] else {}
                contexts.append(
                    (job_id, file_id, file_path, pm.get("thumb_path"), pm.get("mc_raw"))
                )
                cursor.execute(
                    "UPDATE job_queue SET parse_status = 'parsed', parsed_at = ? WHERE id = ?",
                    (_utcnow_sql(), job_id),
                )
            else:
                cursor.execute(
                    """UPDATE job_queue SET status = 'failed',
                       parse_status = 'failed', error_message = 'Auto parse failed'
                       WHERE id = ?""",
                    (job_id,),
                )
        self.db.conn.commit()

        if not contexts or not self._running or self._processing_mode != "auto":
            return len(contexts)

        # ── Phase V: Vision/VLM (MC generation) ──
        logger.info(f"Auto Phase V: processing {len(contexts)} files with VLM")
        self._auto_run_vision_batch(contexts)
        self._auto_unload_vlm()

        if not self._running or self._processing_mode != "auto":
            return len(contexts)

        # ── Phase VV: SigLIP2 visual embedding ──
        logger.info(f"Auto Phase VV: processing {len(contexts)} files with SigLIP2")
        for ctx in contexts:
            job_id, file_id, file_path, thumb_path, _ = ctx
            if not self._running or self._processing_mode != "auto":
                break
            if thumb_path:
                try:
                    self._run_vv_embedding(
                        file_id, Path(file_path),
                        Path(thumb_path) if thumb_path else None,
                    )
                except Exception as e:
                    logger.warning(f"Auto VV failed for job {job_id}: {e}")
        # VV encoder stays loaded (lightweight, reusable for next batch)

        if not self._running or self._processing_mode != "auto":
            return len(contexts)

        # ── Phase MV: Qwen3-Embedding text embedding ──
        logger.info(f"Auto Phase MV: processing {len(contexts)} files with text embedder")
        self._auto_run_mv_batch(contexts)
        self._auto_unload_mv()

        # ── Mark all completed ──
        now = _utcnow_sql()
        for ctx in contexts:
            job_id = ctx[0]
            cursor.execute(
                """UPDATE job_queue SET status = 'completed', completed_at = ?,
                   phase_completed = '{"parse":true,"vision":true,"embed":true}'
                   WHERE id = ? AND status = 'processing'""",
                (now, job_id),
            )
        self.db.conn.commit()

        logger.info(f"Auto processing: {len(contexts)} files completed (P→V→VV→MV)")
        return len(contexts)

    def _auto_run_vision_batch(self, contexts: list):
        """Phase V: Generate MC (caption/tags) with VLM. Pattern from worker_daemon."""
        from PIL import Image

        try:
            from backend.vision.vision_factory import get_vision_analyzer
            analyzer = get_vision_analyzer()
        except Exception as e:
            logger.error(f"Auto Vision: failed to load VLM: {e}")
            return

        for ctx in contexts:
            job_id, file_id, file_path, thumb_path, mc_raw = ctx
            if not self._running or self._processing_mode != "auto":
                break
            if not thumb_path or not Path(thumb_path).exists():
                continue

            try:
                raw_img = Image.open(thumb_path)
                if raw_img.mode == "RGBA":
                    img = Image.new("RGB", raw_img.size, (255, 255, 255))
                    img.paste(raw_img, mask=raw_img.split()[3])
                    raw_img.close()
                elif raw_img.mode != "RGB":
                    img = raw_img.convert("RGB")
                else:
                    img = raw_img

                vision_result = analyzer.analyze(img, mc_raw or {})
                img.close()

                if vision_result and isinstance(vision_result, dict):
                    fields = {}
                    if "caption" in vision_result:
                        fields["mc_caption"] = vision_result["caption"]
                    if "tags" in vision_result:
                        fields["ai_tags"] = vision_result["tags"]
                    for key in [
                        "image_type", "art_style", "scene_type", "ocr_text",
                        "dominant_color", "character_type", "item_type", "ui_type",
                    ]:
                        if vision_result.get(key) is not None:
                            fields[key] = vision_result[key]

                    if fields:
                        self.db.update_file_fields(file_id, fields)
                        cursor = self.db.conn.cursor()
                        cursor.execute(
                            """UPDATE job_queue SET phase_completed =
                               json_set(COALESCE(phase_completed, '{}'), '$.vision', 1)
                               WHERE id = ?""",
                            (job_id,),
                        )
                        self.db.conn.commit()
                        logger.debug(f"Auto Vision: job {job_id} MC generated")

            except Exception as e:
                logger.warning(f"Auto Vision failed for job {job_id}: {e}")

    def _auto_run_mv_batch(self, contexts: list):
        """Phase MV: Generate meaning vectors from MC text with Qwen3-Embedding."""
        try:
            from backend.vector.text_embedding import get_text_embedding_provider
            provider = get_text_embedding_provider()
        except Exception as e:
            logger.error(f"Auto MV: failed to load text embedder: {e}")
            return

        for ctx in contexts:
            job_id, file_id, _, _, _ = ctx
            if not self._running or self._processing_mode != "auto":
                break

            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT mc_caption, ai_tags FROM files WHERE id = ?", (file_id,)
            )
            row = cursor.fetchone()
            if not row or not (row[0] or row[1]):
                continue

            mc_caption = row[0] or ""
            ai_tags = row[1] or ""
            if isinstance(ai_tags, list):
                ai_tags = ", ".join(str(t) for t in ai_tags)

            mv_text = f"{mc_caption} {ai_tags}".strip()
            if not mv_text:
                continue

            try:
                mv_vec = provider.encode(mv_text)
                self.db.upsert_vectors(file_id, mv_vec=mv_vec)
                logger.debug(f"Auto MV: job {job_id} embedded OK")
            except Exception as e:
                logger.warning(f"Auto MV failed for job {job_id}: {e}")

    def _auto_unload_vlm(self):
        """Unload VLM after Phase V to free GPU memory."""
        try:
            from backend.vision.vision_factory import VisionAnalyzerFactory
            VisionAnalyzerFactory.reset()
            self._gc_cleanup()
            logger.info("Auto: VLM unloaded")
        except Exception as e:
            logger.warning(f"Auto: VLM unload failed: {e}")

    def _auto_unload_mv(self):
        """Unload MV text embedder after Phase MV to free GPU memory."""
        try:
            from backend.vector.text_embedding import reset_provider
            reset_provider()
            self._gc_cleanup()
            logger.info("Auto: MV text embedder unloaded")
        except Exception as e:
            logger.warning(f"Auto: MV unload failed: {e}")

    @staticmethod
    def _gc_cleanup():
        """Force garbage collection and GPU cache cleanup."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _interruptible_sleep(self, seconds: float):
        """Sleep that wakes up immediately if mode changes from auto."""
        interval = 1.0
        elapsed = 0.0
        while elapsed < seconds and self._running:
            if self._processing_mode != "auto":
                logger.info("Mode changed during rest, resuming immediately")
                return
            time.sleep(min(interval, seconds - elapsed))
            elapsed += interval

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

    def _loop(self):
        """Main loop: continuously pre-parse pending jobs to fill the buffer."""
        logger.info("ParseAheadPool loop started")
        poll_interval_s = self._get_config_value("server.parse_ahead.poll_interval_s", 2)
        last_retry_reset = 0.0  # timestamp of last parse_status='failed' reset

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
                    # Auto mode: server processes all phases (P→V→VV→MV) when no workers
                    if self._processing_mode == "auto":
                        processed = self._process_auto_batch()
                        if processed > 0:
                            rest_s = self._get_config_value(
                                "server.auto_processing.rest_after_batch_s", 30
                            )
                            logger.info(
                                f"Auto batch done ({processed} files), resting {rest_s}s"
                            )
                            self._interruptible_sleep(rest_s)
                        else:
                            self._process_backfill_batch()
                            time.sleep(poll_interval_s)
                        continue

                    # Process backfill jobs during idle (no demand or buffer full)
                    target = self._calculate_buffer_target()
                    if target <= 0:
                        self._process_backfill_batch()
                        time.sleep(5)
                        continue

                    # Count currently parsed-and-pending jobs
                    cursor = self.db.conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM job_queue "
                        "WHERE parse_status = 'parsed' AND status = 'pending'"
                    )
                    current_parsed = cursor.fetchone()[0]
                    deficit = target - current_parsed

                    if deficit <= 0:
                        self._process_backfill_batch()
                        time.sleep(poll_interval_s)
                        continue

                    # Select jobs to pre-parse
                    cursor.execute(
                        """SELECT id, file_id, file_path FROM job_queue
                           WHERE status = 'pending'
                             AND (parse_status IS NULL OR parse_status = 'pending')
                           ORDER BY priority DESC, created_at ASC
                           LIMIT ?""",
                        (deficit,),
                    )
                    jobs_to_parse = cursor.fetchall()

                    if not jobs_to_parse:
                        # No unparsed jobs — check if all are parse-failed (deadlock).
                        # Reset parse_status='failed' back to NULL every 60s
                        # so they can be retried.
                        now = time.time()
                        if now - last_retry_reset > 60:
                            cursor.execute(
                                """UPDATE job_queue SET parse_status = NULL
                                   WHERE status = 'pending' AND parse_status = 'failed'"""
                            )
                            if cursor.rowcount > 0:
                                self.db.conn.commit()
                                logger.info(
                                    f"ParseAhead: reset {cursor.rowcount} parse-failed "
                                    f"jobs for retry"
                                )
                            last_retry_reset = now
                        self._process_backfill_batch()
                        time.sleep(poll_interval_s)
                        continue

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
                        else:
                            cursor.execute(
                                "UPDATE job_queue SET parse_status = 'failed' WHERE id = ?",
                                (job_id,),
                            )
                        self.db.conn.commit()

                    time.sleep(poll_interval_s)

                except Exception as e:
                    logger.error(
                        f"ParseAheadPool iteration error: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    time.sleep(5)

        except Exception as e:
            logger.critical(
                f"ParseAheadPool loop crashed: {e}\n"
                f"{traceback.format_exc()}"
            )

        logger.info("ParseAheadPool loop exited")

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
        # Normalize to NFC — macOS Path may preserve NFD from filesystem,
        # but files table must store NFC for consistent lookups.
        nfc_path = unicodedata.normalize('NFC', str(file_p))
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

        # 10. mc_only mode: Phase VV — encode image with SigLIP2
        # Re-check processing_mode dynamically (may change via Admin API at runtime)
        from backend.server.queue.manager import get_processing_mode
        current_mode = get_processing_mode()
        if current_mode == "mc_only":
            try:
                self._run_vv_embedding(stored_file_id, file_p, server_thumb_path)
            except Exception as e:
                logger.warning(f"ParseAhead: VV embedding failed for job {job_id}: {e}")
                # VV failure is non-fatal — job still counts as parsed

        logger.info(f"ParseAhead: job {job_id} pre-parsed OK ({file_p.name})")
        return True

    def _run_vv_embedding(self, file_id: int, file_path: Path, thumb_path: Optional[Path] = None):
        """Run SigLIP2 VV + DINOv2 Structure embedding on a single image (mc_only mode).

        Loads SigLIP2 and DINOv2 once and keeps them resident for the session.
        Uses thumbnail if available, falls back to original file.
        """
        from PIL import Image

        if self._vv_encoder is None:
            from backend.vector.siglip2_encoder import SigLIP2Encoder
            self._vv_encoder = SigLIP2Encoder()
            logger.info("ParseAheadPool: SigLIP2 VV encoder loaded (mc_only mode, will stay resident)")

        if self._structure_encoder is None:
            from backend.vector.dinov2_encoder import DinoV2Encoder
            self._structure_encoder = DinoV2Encoder()
            logger.info("ParseAheadPool: DINOv2 Structure encoder loaded (mc_only mode, will stay resident)")

        # Prefer thumbnail (smaller, faster), fall back to original
        img_source = thumb_path if thumb_path and thumb_path.exists() else file_path
        try:
            img = Image.open(str(img_source)).convert("RGB")
        except Exception as e:
            logger.warning(f"ParseAhead VV: cannot open image {img_source}: {e}")
            return

        try:
            vv_vec = self._vv_encoder.encode_image(img)
            # DINOv2 Structure vector (same image)
            structure_vec = None
            try:
                structure_vec = self._structure_encoder.encode_image(img)
            except Exception as e:
                logger.warning(f"ParseAhead Structure: DINOv2 encoding failed for file_id={file_id}: {e}")
        finally:
            img.close()
        self.db.upsert_vectors(file_id, vv_vec=vv_vec, structure_vec=structure_vec)
        logger.debug(f"ParseAhead VV+Structure: file_id={file_id} embedded OK")

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
                    "UPDATE job_queue SET status = 'failed', completed_at = ? WHERE id = ?",
                    (now, job_id),
                )
                self.db.conn.commit()
                continue

            try:
                img = Image.open(str(img_source)).convert("RGB")
                try:
                    structure_vec = self._structure_encoder.encode_image(img)
                finally:
                    img.close()
                self.db.upsert_vectors(file_id, structure_vec=structure_vec)
                cursor.execute(
                    """UPDATE job_queue SET status = 'completed', completed_at = ?,
                       phase_completed = '{"parse":true,"vision":true,"embed":true}'
                       WHERE id = ?""",
                    (now, job_id),
                )
                processed += 1
            except Exception as e:
                logger.warning(f"Backfill: DINOv2 failed for job {job_id}: {e}")
                cursor.execute(
                    "UPDATE job_queue SET status = 'failed', completed_at = ?, error_message = ? WHERE id = ?",
                    (now, str(e), job_id),
                )
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
