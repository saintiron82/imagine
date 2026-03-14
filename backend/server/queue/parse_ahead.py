"""
Parse-ahead pool — server-side Phase P pre-parser for worker optimization.

Monitors connected workers' total capacity and pre-parses pending jobs
so that workers receive thumbnails (~200KB) instead of raw files (~500MB).

The pool runs as a background daemon thread, continuously maintaining
a buffer of pre-parsed jobs proportional to worker demand.

Modes:
- auto: No workers connected — server processes all phases (P→V→VV→MV).
  Models loaded per-phase and unloaded between phases.
- mc_only: Also runs Phase VV (SigLIP2 + DINOv2) on parsed jobs since
  VV/Structure only need the image (independent of MC). Workers handle
  V(MC) only; EmbedAheadPool handles MV.
- parse_only: Server does Phase P only (zero GPU models loaded). All GPU
  work (V+VV+MV) is delegated to workers running in full mode. When no
  workers are online, server keeps parsing and queues jobs (no auto fallback).
- distribute: Pre-parse + gap-fill V(MC) for lightweight workers. Full
  workers handle V+VV+MV, lightweight workers handle VV+MV. Server fills
  vision gaps so lightweight workers can claim vision-done jobs.
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

    Modes:
    - auto: Full pipeline P→V→VV→MV (no workers connected).
    - mc_only: P + VV (SigLIP2 + DINOv2); workers do V(MC) only.
    - parse_only: P only (zero GPU); workers do V+VV+MV (full mode).
    - distribute: P + gap-fill V(MC) for lightweight workers.
    """

    def __init__(self, db):
        super().__init__(db)
        self._processing_mode = "parse_only"  # Fixed: tollgate architecture
        self._vv_encoder = None  # Legacy, kept for _unload_models() compatibility
        self._structure_encoder = None  # Legacy, kept for _unload_models() compatibility
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
        Picks up ALL pending jobs, including already-parsed ones (Phase P skip).
        Returns number of files processed.
        """
        batch_size = self._get_config_value("server.auto_processing.batch_size", 5)

        cursor = self.db.conn.cursor()

        # Reclaim stuck processing jobs from builtin worker (crash/mode-switch recovery)
        cursor.execute(
            """UPDATE job_queue SET status = 'pending',
               assigned_to = NULL, assigned_at = NULL, worker_session_id = NULL
               WHERE status = 'processing'
                 AND worker_session_id = (
                     SELECT id FROM worker_sessions
                     WHERE worker_name = '__builtin__' LIMIT 1
                 )"""
        )
        if cursor.rowcount > 0:
            logger.info(
                f"Auto: reclaimed {cursor.rowcount} stuck processing jobs "
                f"from builtin worker"
            )
        self.db.conn.commit()  # Always commit to release WAL write lock

        # Pick up pending jobs that are file-ready (local or downloaded)
        # file_ready gate: only process files that are locally available
        cursor.execute(
            """SELECT id, file_id, file_path, parse_status, parsed_metadata, phase_completed
               FROM job_queue
               WHERE status = 'pending' AND file_ready = 1
               ORDER BY priority DESC, created_at ASC
               LIMIT ?""",
            (batch_size,),
        )
        jobs = cursor.fetchall()
        if not jobs:
            return 0

        logger.info(f"Auto processing: starting batch of {len(jobs)} files")
        self._update_builtin_session("parse", f"batch({len(jobs)})")

        # Parse phase_completed for each job
        job_phases = {}  # job_id → {"parse": bool, "vision": bool, "embed": bool}
        for row in jobs:
            job_id = row[0]
            phase_str = row[5] if len(row) > 5 else None
            try:
                phases = json.loads(phase_str) if phase_str else {}
            except (json.JSONDecodeError, TypeError):
                phases = {}
            job_phases[job_id] = phases

        # Mark jobs as processing
        now = _utcnow_sql()
        for row in jobs:
            job_id = row[0]
            cursor.execute(
                """UPDATE job_queue
                   SET status = 'processing', started_at = ?
                   WHERE id = ? AND status = 'pending'""",
                (now, job_id),
            )
        self.db.conn.commit()

        contexts = []  # [(job_id, file_id, file_path, thumb_path, mc_raw)]

        # ── Phase P: Parse (skip already-parsed jobs) ──
        for job_id, file_id, file_path, parse_status, parsed_metadata, _phase_json in jobs:
            if not self._running or self._processing_mode != "auto":
                break

            # Already parsed — reuse existing metadata, skip re-parse
            if parse_status == "parsed":
                pm = json.loads(parsed_metadata) if parsed_metadata else {}
                contexts.append(
                    (job_id, file_id, file_path, pm.get("thumb_path"), pm.get("mc_raw"))
                )
                logger.debug(f"Auto: job {job_id} already parsed, skipping Phase P")
                continue

            success = False
            try:
                cursor.execute(
                    "UPDATE job_queue SET parse_status = 'parsing' WHERE id = ?",
                    (job_id,),
                )
                self.db.conn.commit()
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
                # Parse failed permanently — mark file and DELETE job
                cursor.execute(
                    "UPDATE files SET processing_status = 'failed', processing_error = 'Auto parse failed' WHERE id = ?",
                    (file_id,),
                )
                cursor.execute(
                    "DELETE FROM job_queue WHERE id = ?",
                    (job_id,),
                )
        self.db.conn.commit()

        if not contexts or not self._running or self._processing_mode != "auto":
            return len(contexts)

        # ── Phase V → VV → MV via unified PhaseRunner ──
        from backend.pipeline.protocols import PhaseItem, FixedBatchStrategy
        from backend.pipeline.model_manager import ModelManager
        from backend.pipeline.phase_runner import PhaseRunner
        from backend.pipeline.storage_direct import DirectSQLStorage

        # Convert contexts to PhaseItems
        phase_items = []
        skipped_jobs = set()  # Jobs skipped (e.g. no thumbnail) — exclude from integrity check
        for ctx in contexts:
            job_id, file_id, file_path, thumb_path, mc_raw = ctx
            phases_done = job_phases.get(job_id, {})

            # Fallback: if thumb_path missing, look up files.thumbnail_url
            # (covers WebDAV files where browse generated thumb but
            #  parsed_metadata.thumb_path wasn't set)
            if not thumb_path or not Path(thumb_path).exists():
                try:
                    row = cursor.execute(
                        "SELECT thumbnail_url FROM files WHERE id = ?",
                        (file_id,)
                    ).fetchone()
                    if row and row[0] and Path(row[0]).exists():
                        thumb_path = row[0]
                except Exception:
                    pass

            # No thumbnail available — release back to pending so it can be
            # retried later when thumbnail becomes available (e.g. after re-download).
            if not thumb_path or not Path(thumb_path).exists():
                logger.info(
                    f"ParseAhead job {job_id}: no thumbnail, releasing to pending")
                cursor.execute(
                    """UPDATE job_queue SET status = 'pending',
                       assigned_to = NULL, assigned_at = NULL,
                       worker_session_id = NULL
                       WHERE id = ?""",
                    (job_id,),
                )
                skipped_jobs.add(job_id)
                continue

            # For resume: if vision is done but embed isn't, mc_raw needs
            # the VLM results (mc_caption, ai_tags) from DB for MV encoding.
            effective_mc_raw = mc_raw if phases_done.get("vision") else None
            if effective_mc_raw and phases_done.get("vision") and not phases_done.get("embed"):
                try:
                    db_row = cursor.execute(
                        "SELECT mc_caption, ai_tags, image_type, scene_type, art_style "
                        "FROM files WHERE id = ?",
                        (file_id,),
                    ).fetchone()
                    if db_row:
                        if db_row[0]:
                            effective_mc_raw["mc_caption"] = db_row[0]
                        if db_row[1]:
                            effective_mc_raw["ai_tags"] = db_row[1]
                        if db_row[2]:
                            effective_mc_raw["image_type"] = db_row[2]
                        if db_row[3]:
                            effective_mc_raw["scene_type"] = db_row[3]
                        if db_row[4]:
                            effective_mc_raw["art_style"] = db_row[4]
                except Exception as e:
                    logger.debug(f"Auto: failed to enrich mc_raw for file {file_id}: {e}")

            phase_items.append(PhaseItem(
                job_id=job_id,
                file_id=file_id,
                file_path=file_path,
                thumb_path=thumb_path,
                mc_raw=effective_mc_raw,
                skip_vision=bool(phases_done.get("vision")),
                skip_vv=bool(phases_done.get("embed")),
                skip_mv=bool(phases_done.get("embed")),
            ))

        # Commit any skipped-job status changes before running PhaseRunner
        if skipped_jobs:
            self.db.conn.commit()
            logger.info(f"Auto: {len(skipped_jobs)} jobs released to pending (no thumbnail)")

        # Create PhaseRunner with DirectSQL storage
        models = ModelManager()
        storage = DirectSQLStorage(self.db)
        batch_strategy = FixedBatchStrategy(vision=1, vv=batch_size, mv=batch_size)
        runner = PhaseRunner(
            models=models,
            storage=storage,
            batch_strategy=batch_strategy,
            stop_check=lambda: not self._running or self._processing_mode != "auto",
        )

        # Run V → VV → MV (PhaseRunner handles skip flags and model unloading)
        self._update_builtin_session("vision", f"batch({len(phase_items)})")
        phase_items = runner.run_vision(phase_items)

        if self._running and self._processing_mode == "auto":
            self._update_builtin_session("embed_vv", f"batch({len(phase_items)})")
            phase_items = runner.run_vv(phase_items)

        if self._running and self._processing_mode == "auto":
            self._update_builtin_session("embed_mv", f"batch({len(phase_items)})")
            phase_items = runner.run_mv(phase_items)

        storage.flush()

        # ── Mark completed with per-file integrity verification ──
        now = _utcnow_sql()
        completed_count = 0
        partial_count = 0
        failed_count = 0
        for ctx in contexts:
            job_id, file_id = ctx[0], ctx[1]

            # Skip jobs that were released back to pending (no thumbnail etc.)
            if job_id in skipped_jobs:
                continue

            # Skip jobs already marked as failed (e.g. THUMB_MISSING)
            status_row = cursor.execute(
                "SELECT status FROM job_queue WHERE id = ?", (job_id,)
            ).fetchone()
            if status_row and status_row[0] == 'failed':
                failed_count += 1
                continue

            verify = self.db.verify_data_integrity(
                file_id, expect_mc=True, expect_vv=True, expect_mv=True
            )
            if verify["valid"]:
                # Complete — log completion and DELETE job
                cursor.execute(
                    "INSERT INTO job_completions (file_id) VALUES (?)",
                    (file_id,),
                )
                cursor.execute(
                    "DELETE FROM job_queue WHERE id = ?",
                    (job_id,),
                )
                # Clear any processing_status
                cursor.execute(
                    "UPDATE files SET processing_status = NULL, processing_error = NULL WHERE id = ?",
                    (file_id,),
                )
                completed_count += 1
            else:
                # Partial: check retry_count before releasing
                phase_json = json.dumps(verify["actual_phases"])
                retry_row = cursor.execute(
                    "SELECT retry_count, max_retries FROM job_queue WHERE id = ?",
                    (job_id,)
                ).fetchone()
                retry_count = retry_row[0] if retry_row else 0
                max_retries = retry_row[1] if retry_row else 3

                if retry_count >= max_retries:
                    # Permanently failed — mark file and DELETE job
                    error_msg = (
                        f"Auto: partial after {retry_count} retries, "
                        f"missing={verify['missing']}"
                    )
                    cursor.execute(
                        "UPDATE files SET processing_status = 'failed', processing_error = ? WHERE id = ?",
                        (error_msg, file_id),
                    )
                    cursor.execute(
                        "DELETE FROM job_queue WHERE id = ?",
                        (job_id,),
                    )
                    failed_count += 1
                    logger.warning(
                        f"Auto job {job_id} permanently failed after "
                        f"{retry_count} retries (missing={verify['missing']})"
                    )
                else:
                    cursor.execute(
                        """UPDATE job_queue SET status = 'pending', phase_completed = ?,
                           retry_count = retry_count + 1,
                           assigned_to = NULL, assigned_at = NULL,
                           worker_session_id = NULL
                           WHERE id = ? AND status = 'processing'""",
                        (phase_json, job_id),
                    )
                    partial_count += 1
                    logger.warning(
                        f"Auto job {job_id} integrity mismatch "
                        f"(missing={verify['missing']}). "
                        f"Retry {retry_count + 1}/{max_retries}."
                    )
        self.db.conn.commit()

        # Release download buffer slots for completed/failed WebDAV jobs
        dl_pool = getattr(self, '_download_pool', None)
        if dl_pool:
            for ctx in contexts:
                job_id, file_id = ctx[0], ctx[1]
                status_row = cursor.execute(
                    "SELECT status, file_path FROM job_queue WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if status_row and status_row[1] and status_row[1].startswith("webdav://"):
                    if status_row[0] in ('completed', 'failed'):
                        dl_pool.release_slot(file_id)

        if partial_count > 0 or failed_count > 0:
            logger.warning(
                f"Auto processing: {completed_count} completed, "
                f"{partial_count} partial, {failed_count} failed"
            )
        else:
            logger.info(f"Auto processing: {completed_count} files completed (P→V→VV→MV)")
        self._update_builtin_session(None, None, jobs_done=completed_count)
        return completed_count

    def _update_builtin_session(self, phase: str = None, file_name: str = None,
                                 jobs_done: int = 0):
        """Update virtual builtin worker session for UI visibility.

        Only active when global mode is builtin_worker.
        """
        try:
            from backend.server.queue.manager import get_processing_mode, _utcnow_sql
            if get_processing_mode() != "builtin_worker":
                return

            cursor = self.db.conn.cursor()
            now = _utcnow_sql()

            if jobs_done > 0:
                cursor.execute(
                    """UPDATE worker_sessions
                       SET current_phase = ?, current_file = ?,
                           jobs_completed = COALESCE(jobs_completed, 0) + ?,
                           last_heartbeat = ?
                       WHERE worker_name = '__builtin__' AND status = 'online'""",
                    (phase, file_name, jobs_done, now),
                )
            else:
                cursor.execute(
                    """UPDATE worker_sessions
                       SET current_phase = ?, current_file = ?,
                           last_heartbeat = ?
                       WHERE worker_name = '__builtin__' AND status = 'online'""",
                    (phase, file_name, now),
                )
            self.db.conn.commit()
        except Exception as e:
            logger.debug(f"Builtin session update failed: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _auto_run_vision_batch(self, contexts: list):
        """Phase V: Generate MC (caption/tags) with VLM.

        Used by both auto mode (full pipeline) and distribute mode (gap-fill).
        Caller is responsible for mode checks between phases.
        """
        from PIL import Image

        try:
            from backend.vision.vision_factory import get_vision_analyzer
            analyzer = get_vision_analyzer()
        except Exception as e:
            logger.error(f"Auto Vision: failed to load VLM: {e}")
            return

        for ctx in contexts:
            job_id, file_id, file_path, thumb_path, mc_raw = ctx
            if not self._running:
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
                        self.db.update_vision_fields(file_path, fields)
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
                try:
                    self.db.conn.rollback()
                except Exception:
                    pass

    def _auto_unload_vlm(self):
        """Unload VLM after Phase V to free GPU memory."""
        try:
            from backend.vision.vision_factory import VisionAnalyzerFactory
            VisionAnalyzerFactory.reset()
            self._gc_cleanup()
            logger.info("Auto: VLM unloaded")
        except Exception as e:
            logger.warning(f"Auto: VLM unload failed: {e}")

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
            else:
                cursor.execute(
                    "UPDATE job_queue SET parse_status = 'failed' WHERE id = ?",
                    (job_id,),
                )
            self.db.conn.commit()

        return parsed_count > 0

    def _fill_vision_gaps(self) -> int:
        """Generate V(MC) for parsed+pending+vision-not-done jobs (gap-fill).

        In distribute mode with lightweight workers, these workers can only
        handle VV+MV (they lack GPU for VLM). Server generates MC here so
        lightweight workers can claim the vision-done jobs for VV+MV.

        Full workers that already claimed jobs won't be affected — they
        handle V+VV+MV themselves.

        Returns number of jobs processed.
        """
        batch_size = self._get_config_value("server.auto_processing.batch_size", 5)

        cursor = self.db.conn.cursor()
        cursor.execute(
            """SELECT id, file_id, file_path, parsed_metadata
               FROM job_queue
               WHERE status = 'pending'
                 AND parse_status = 'parsed'
                 AND (json_extract(phase_completed, '$.vision') IS NULL
                      OR json_extract(phase_completed, '$.vision') = 0)
               ORDER BY priority DESC, created_at ASC
               LIMIT ?""",
            (batch_size,),
        )
        jobs = cursor.fetchall()
        if not jobs:
            return 0

        # Build contexts for vision batch
        contexts = []
        for job_id, file_id, file_path, parsed_metadata in jobs:
            pm = json.loads(parsed_metadata) if parsed_metadata else {}
            contexts.append((
                job_id, file_id, file_path,
                pm.get("thumb_path"),
                pm.get("mc_raw"),
            ))

        logger.info(
            f"Gap-fill V: {len(contexts)} parsed jobs → VLM MC generation "
            f"for lightweight workers"
        )

        # Run Phase V (MC generation) — reuse auto mode's vision batch
        self._auto_run_vision_batch(contexts)
        self._auto_unload_vlm()

        return len(contexts)

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
