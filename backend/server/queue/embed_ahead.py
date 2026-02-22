"""
Embed-ahead pool — server-side Phase MV post-processor for mc_only mode.

After workers upload MC text (vision fields), this pool picks up jobs
with vision=done + embed=pending, runs Qwen3-Embedding to generate MV
vectors, and marks jobs as fully completed.

The MV model (Qwen3-Embedding) is loaded once and stays resident.
"""

import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)


class EmbedAheadPool:
    """Server-side MV embedder that runs after workers upload MC.

    In mc_only mode:
    - ParseAhead handles Phase P + Phase VV
    - Workers handle Phase V (MC generation only)
    - EmbedAhead handles Phase MV (this class)

    The Qwen3-Embedding model is loaded once and stays resident.
    """

    def __init__(self, db: SQLiteDB):
        self.db = db
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._mv_provider = None  # Lazy-loaded, stays resident

    def start(self):
        """Start the background embed-ahead daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("EmbedAheadPool already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._embed_loop,
            name="EmbedAheadPool",
            daemon=True,
        )
        self._thread.start()
        logger.info("EmbedAheadPool started")

    def stop(self):
        """Gracefully stop the embed-ahead daemon thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                logger.warning("EmbedAheadPool thread did not stop within timeout")
            else:
                logger.info("EmbedAheadPool stopped")
        self._thread = None

        # Unload MV provider
        if self._mv_provider is not None:
            try:
                if hasattr(self._mv_provider, 'unload'):
                    self._mv_provider.unload()
                logger.info("EmbedAheadPool: MV provider unloaded")
            except Exception as e:
                logger.warning(f"EmbedAheadPool: MV provider unload error: {e}")
            self._mv_provider = None

    def _embed_loop(self):
        """Main loop: find MC-completed jobs and run MV embedding."""
        logger.info("EmbedAheadPool loop started")
        poll_interval_s = self._get_config_value("server.embed_ahead.poll_interval_s", 2)
        batch_size = self._get_config_value("server.embed_ahead.batch_size", 32)

        try:
            while self._running:
                try:
                    jobs = self._get_mc_completed_jobs(limit=batch_size)
                    if not jobs:
                        time.sleep(poll_interval_s)
                        continue

                    self._process_mv_batch(jobs)

                except Exception as e:
                    logger.error(
                        f"EmbedAheadPool iteration error: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    time.sleep(5)

        except Exception as e:
            logger.critical(
                f"EmbedAheadPool loop crashed: {e}\n"
                f"{traceback.format_exc()}"
            )

        logger.info("EmbedAheadPool loop exited")

    def _get_mc_completed_jobs(self, limit: int = 32):
        """Find jobs where MC is done but MV embedding is pending.

        These are jobs in 'processing' status with vision=true, embed=false.
        """
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """SELECT jq.id, jq.file_id, jq.file_path
                   FROM job_queue jq
                   WHERE jq.status = 'processing'
                     AND json_extract(jq.phase_completed, '$.vision') = 1
                     AND json_extract(jq.phase_completed, '$.embed') = 0
                   ORDER BY jq.priority DESC, jq.created_at ASC
                   LIMIT ?""",
                (limit,),
            )
            return cursor.fetchall()
        except Exception as e:
            logger.warning(f"EmbedAhead: failed to query MC-completed jobs: {e}")
            return []

    def _process_mv_batch(self, jobs):
        """Run batch MV embedding and mark jobs as completed.

        Args:
            jobs: List of (job_id, file_id, file_path) tuples.
        """
        from backend.vector.text_embedding import build_document_text

        if self._mv_provider is None:
            from backend.vector.text_embedding import get_text_embedding_provider
            self._mv_provider = get_text_embedding_provider()
            logger.info("EmbedAheadPool: MV provider loaded (will stay resident)")

        # Collect MC text from DB for each job
        texts = []
        valid_jobs = []
        cursor = self.db.conn.cursor()

        for job_id, file_id, file_path in jobs:
            # Look up file_id from files table (file_id in job_queue may be stale)
            cursor.execute(
                "SELECT id, mc_caption, ai_tags, image_type, art_style, scene_type "
                "FROM files WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            if row is None:
                logger.warning(f"EmbedAhead: file not found for job {job_id}: {file_path}")
                continue

            stored_file_id, mc_caption, ai_tags, image_type, art_style, scene_type = row
            mc_caption = mc_caption or ""
            ai_tags = ai_tags or "[]"

            # Parse ai_tags
            try:
                tags = json.loads(ai_tags) if isinstance(ai_tags, str) else ai_tags
            except (json.JSONDecodeError, TypeError):
                tags = [ai_tags] if ai_tags else []

            # Build facts dict for richer MV embedding
            facts = {}
            if image_type:
                facts["image_type"] = image_type
            if art_style:
                facts["art_style"] = art_style
            if scene_type:
                facts["scene_type"] = scene_type

            doc_text = build_document_text(mc_caption, tags, facts=facts)
            if not doc_text.strip():
                logger.warning(f"EmbedAhead: empty document text for job {job_id}")
                continue

            texts.append(doc_text)
            valid_jobs.append((job_id, stored_file_id, file_path))

        if not texts:
            return

        # Batch MV encoding
        try:
            mv_vecs = self._mv_provider.encode_batch(texts)
        except Exception as e:
            logger.error(f"EmbedAhead: batch MV encoding failed: {e}")
            return

        # Store vectors + mark jobs completed
        now = datetime.now(timezone.utc).isoformat()
        completed_phase = json.dumps({"parse": True, "vision": True, "embed": True})

        for (job_id, stored_file_id, file_path), mv_vec in zip(valid_jobs, mv_vecs):
            try:
                self.db.upsert_vectors(stored_file_id, mv_vec=mv_vec, commit=False)
                cursor.execute(
                    """UPDATE job_queue
                       SET status = 'completed', completed_at = ?,
                           phase_completed = ?
                       WHERE id = ?""",
                    (now, completed_phase, job_id),
                )
                logger.debug(f"EmbedAhead: job {job_id} MV done, completed")
            except Exception as e:
                logger.error(f"EmbedAhead: failed to store MV for job {job_id}: {e}")

        try:
            self.db.conn.commit()
        except Exception as e:
            logger.error(f"EmbedAhead: commit failed: {e}")

        logger.info(f"EmbedAhead: processed {len(valid_jobs)} jobs (MV batch)")

    def _get_config_value(self, dotted_key: str, default):
        """Read a value from config.yaml by dotted key."""
        try:
            from backend.utils.config import get_config
            return get_config().get(dotted_key, default)
        except Exception:
            return default

    def get_stats(self) -> dict:
        """Get current embed-ahead pool statistics."""
        try:
            cursor = self.db.conn.cursor()

            # Jobs waiting for MV embedding
            cursor.execute(
                """SELECT COUNT(*) FROM job_queue
                   WHERE status = 'processing'
                     AND json_extract(phase_completed, '$.vision') = 1
                     AND json_extract(phase_completed, '$.embed') = 0"""
            )
            pending_mv = cursor.fetchone()[0]

            return {
                "pending_mv": pending_mv,
                "mv_provider_loaded": self._mv_provider is not None,
            }

        except Exception as e:
            logger.warning(f"EmbedAhead stats error: {e}")
            return {"pending_mv": 0, "mv_provider_loaded": False}
