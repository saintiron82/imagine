"""
phase_runner.py — Single Source of Truth for V / VV / MV phase execution.

Replaces duplicated phase logic in:
  - ingest_engine.py (phase2_vision_adaptive, phase3a_vv_adaptive, phase3b_mv_adaptive)
  - worker_daemon.py (_run_vision, _run_embed_vv, _run_embed_mv)
  - parse_ahead.py (_auto_run_vision_batch, _auto_run_mv_batch)

All three pipeline modes inject different StorageBackend / BatchStrategy
but share the same phase execution code here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image

from backend.pipeline.model_manager import ModelManager
from backend.pipeline.protocols import (
    BatchStrategy,
    FixedBatchStrategy,
    NullProgressReporter,
    PhaseItem,
    ProgressReporter,
    StorageBackend,
)

logger = logging.getLogger("pipeline.phase_runner")


class PhaseRunner:
    """
    Unified V → VV → MV phase executor.

    Phase P (Parse) is NOT included — each mode handles parsing differently:
      - Local: ThreadPool parallel parsing
      - Worker: server pre-parse (no local parsing)
      - Server: single-file synchronous parsing

    Phase V/VV/MV logic is identical across all modes — this class is that
    single implementation.
    """

    def __init__(
        self,
        models: ModelManager,
        storage: StorageBackend,
        progress: Optional[ProgressReporter] = None,
        batch_strategy: Optional[BatchStrategy] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self.models = models
        self.storage = storage
        self.progress = progress or NullProgressReporter()
        self.batch_strategy = batch_strategy or FixedBatchStrategy()
        self.stop_check = stop_check or (lambda: False)

    # ------------------------------------------------------------------
    # Phase V: Vision (VLM → MC generation)
    # ------------------------------------------------------------------

    def run_vision(self, items: List[PhaseItem]) -> List[PhaseItem]:
        """
        Run VLM to generate MC (caption + tags + classification).

        Items with skip_vision=True or existing VLM results are skipped.
        mc_raw may contain parse-context metadata (file_name, layers) without
        VLM results — only skip when actual MC (caption/tags) is present.
        Results are saved incrementally via StorageBackend.
        VLM is unloaded after completion.
        """
        needed = [it for it in items if not it.skip_vision and not self._has_vlm_result(it)]
        if not needed:
            logger.info("[PHASE:vision] all items skipped")
            return items

        count = len(needed)
        self.progress.phase_start("vision", count)
        t0 = time.perf_counter()

        try:
            analyzer = self.models.get_vlm()
        except Exception as e:
            logger.error(f"[PHASE:vision] VLM load failed: {e}", exc_info=True)
            for it in needed:
                it.error = f"VLM load failed: {e}"
            self.progress.phase_complete("vision", time.perf_counter() - t0)
            return items

        # Log VLM model info for diagnostics
        _vlm_model = getattr(analyzer, 'model_id', None) or getattr(analyzer, 'model_name', None) or 'unknown'
        _vlm_backend = type(analyzer).__name__
        _vlm_device = getattr(analyzer, 'device', 'unknown')
        logger.info(
            f"[PHASE:vision] VLM ready: {_vlm_backend} "
            f"(model={_vlm_model}, device={_vlm_device})"
        )

        # Get active domain for VLM
        domain = self._get_active_domain()

        batch_size = self.batch_strategy.get_batch_size("vision")
        processed = 0

        for start in range(0, count, batch_size):
            if self.stop_check():
                logger.info("[PHASE:vision] stop requested")
                break

            chunk = needed[start:start + batch_size]
            t_sub = time.perf_counter()

            for i, item in enumerate(chunk):
                try:
                    img = self._load_thumbnail(item)
                    if img is None:
                        item.error = (
                            f"thumbnail load failed "
                            f"(path={item.thumb_path or item.file_path})"
                        )
                        logger.warning(
                            f"[PHASE:vision] {item.file_name}: {item.error}"
                        )
                        self.progress.file_done(
                            "vision", processed + i, count, item.file_name, False)
                        continue

                    # Build context from item metadata
                    context = self._build_vision_context(item)

                    # Call VLM
                    t_vlm = time.perf_counter()
                    if hasattr(analyzer, "classify_and_analyze"):
                        result = analyzer.classify_and_analyze(
                            img, context=context, domain=domain)
                    elif hasattr(analyzer, "analyze"):
                        result = analyzer.analyze(img, context or {})
                    else:
                        result = {}
                    vlm_elapsed = time.perf_counter() - t_vlm

                    # Compute perceptual hash before closing image (P02)
                    # Independent of VLM success/failure — runs even if caption empty
                    if isinstance(result, dict):
                        try:
                            from backend.utils.dhash import dhash64
                            hash_val = dhash64(img)
                            # SQLite INTEGER is signed 64-bit
                            if hash_val >= 2**63:
                                hash_val -= 2**64
                            result["perceptual_hash"] = hash_val
                        except Exception as dhash_err:
                            logger.debug(
                                f"[PHASE:vision] dHash failed for "
                                f"{item.file_name}: {dhash_err}"
                            )
                        # Record actual VLM that produced this result (P13/P14)
                        result["caption_model"] = _vlm_model
                        # Snapshot full VLM JSON for SQL-queryable structured_meta (P01)
                        result["structured_meta"] = json.dumps(
                            result, ensure_ascii=False
                        )

                    img.close()

                    # Store result on item
                    item.vision_result = result
                    item.mc_raw = result

                    # Validate MC result — empty caption = VLM failure
                    # (matches _run_vision_phase behavior in worker_daemon)
                    _caption = ""
                    if isinstance(result, dict):
                        _caption = (
                            result.get("mc_caption")
                            or result.get("caption")
                            or ""
                        )
                    if not _caption:
                        result_keys = (
                            list(result.keys())
                            if isinstance(result, dict)
                            else type(result).__name__
                        )
                        item.error = (
                            f"VLM returned empty caption "
                            f"(elapsed={vlm_elapsed:.1f}s, "
                            f"result_keys={result_keys})"
                        )
                        logger.warning(
                            f"[PHASE:vision] {item.file_name}: {item.error}"
                        )
                        if isinstance(result, dict):
                            result["processing_status"] = "error"
                            result["processing_error"] = f"vision: {item.error}"
                    else:
                        logger.debug(
                            f"[PHASE:vision] {item.file_name}: OK "
                            f"({vlm_elapsed:.1f}s, "
                            f"caption_len={len(_caption)})"
                        )
                        if isinstance(result, dict):
                            result["processing_status"] = "vision_done"
                            result["processing_error"] = None

                    # Save incrementally
                    self.storage.save_vision(item, result)

                    self.progress.file_done(
                        "vision", processed + i, count,
                        item.file_name, item.error is None)

                except Exception as e:
                    logger.warning(
                        f"[PHASE:vision] {item.file_name}: "
                        f"{type(e).__name__}: {e}"
                    )
                    item.error = f"{type(e).__name__}: {e}"
                    # P05: persist error status so future runs can identify failures
                    try:
                        self.storage.save_vision(item, {
                            "processing_status": "error",
                            "processing_error": f"vision: {item.error}",
                        })
                    except Exception:
                        pass
                    self.progress.file_done(
                        "vision", processed + i, count, item.file_name, False)

            elapsed_sub = time.perf_counter() - t_sub
            self.batch_strategy.after_sub_batch("vision", elapsed_sub, len(chunk))
            processed += len(chunk)

        # Unload VLM
        self.models.unload_vlm()

        elapsed = time.perf_counter() - t0
        self.progress.phase_complete("vision", elapsed)
        ok = sum(1 for it in needed if it.error is None)
        fail = count - ok
        logger.info(
            f"[PHASE:vision] {processed}/{count} in {elapsed:.1f}s "
            f"(ok={ok}, fail={fail}, "
            f"backend={_vlm_backend}, model={_vlm_model})"
        )

        return items

    # ------------------------------------------------------------------
    # Phase VV: SigLIP2 visual vector encoding
    # ------------------------------------------------------------------

    def run_vv(self, items: List[PhaseItem]) -> List[PhaseItem]:
        """
        Encode thumbnails into visual vectors (SigLIP2).

        Items with skip_vv=True are skipped.
        Uses batch encoding for efficiency (encode_image_batch).
        SigLIP2 is unloaded after completion.
        """
        needed = [it for it in items if not it.skip_vv and it.error is None]
        if not needed:
            logger.info("[PHASE:vv] all items skipped")
            return items

        count = len(needed)
        self.progress.phase_start("vv", count)
        t0 = time.perf_counter()

        encoder = self.models.get_vv_encoder()
        batch_size = self.batch_strategy.get_batch_size("vv")
        processed = 0

        for start in range(0, count, batch_size):
            if self.stop_check():
                logger.info("[PHASE:vv] stop requested")
                break

            chunk = needed[start:start + batch_size]
            t_sub = time.perf_counter()

            # Load images
            images = []
            valid_items = []
            for item in chunk:
                try:
                    img = self._load_thumbnail(item)
                    if img is not None:
                        images.append(img)
                        valid_items.append(item)
                    else:
                        item.error = "VV thumbnail load failed"
                except Exception as e:
                    logger.warning(f"[PHASE:vv] load {item.file_name}: {e}")
                    item.error = str(e)

            # Batch encode
            if images:
                try:
                    if hasattr(encoder, "encode_image_batch") and len(images) > 1:
                        vv_vectors = encoder.encode_image_batch(images)
                    else:
                        vv_vectors = [encoder.encode_image(img) for img in images]

                    for j, vec in enumerate(vv_vectors):
                        item = valid_items[j]
                        item.vv_embedding = vec

                        # Optional structure vector
                        if hasattr(encoder, "encode_structure"):
                            try:
                                item.structure_embedding = encoder.encode_structure(
                                    images[j])
                            except Exception:
                                pass

                        # Save incrementally
                        self.storage.save_vv(
                            item, vec, item.structure_embedding)

                        self.progress.file_done(
                            "vv", processed + j, count, item.file_name, True)

                except Exception as e:
                    logger.error(f"[PHASE:vv] batch encode failed: {e}")
                    for item in valid_items:
                        if item.vv_embedding is None:
                            item.error = f"VV encode failed: {e}"

                # Close images
                for img in images:
                    try:
                        img.close()
                    except Exception:
                        pass
                del images, vv_vectors

            elapsed_sub = time.perf_counter() - t_sub
            self.batch_strategy.after_sub_batch("vv", elapsed_sub, len(chunk))
            processed += len(chunk)

        # Unload SigLIP2
        self.models.unload_vv()

        elapsed = time.perf_counter() - t0
        self.progress.phase_complete("vv", elapsed)
        logger.info(f"[PHASE:vv] {processed}/{count} in {elapsed:.1f}s")

        return items

    # ------------------------------------------------------------------
    # Phase MV: Qwen3-Embedding meaning vector encoding
    # ------------------------------------------------------------------

    def run_mv(self, items: List[PhaseItem]) -> List[PhaseItem]:
        """
        Encode MC text into meaning vectors (Qwen3-Embedding).

        Items with skip_mv=True or no MC data are skipped.
        Uses batch encoding with fallback to individual encoding.
        MV encoder is unloaded after completion.
        """
        needed = [it for it in items
                  if not it.skip_mv and it.error is None and it.mc_raw]
        if not needed:
            logger.info("[PHASE:mv] all items skipped")
            return items

        # Compose MV texts
        mv_pairs = []  # (item, text)
        for item in needed:
            text = self._compose_mv_text(item)
            if text:
                mv_pairs.append((item, text))
                item.mv_text = text

        if not mv_pairs:
            logger.info("[PHASE:mv] no texts to encode")
            return items

        count = len(mv_pairs)
        self.progress.phase_start("mv", count)
        t0 = time.perf_counter()

        provider = self.models.get_mv_encoder()
        batch_size = self.batch_strategy.get_batch_size("mv")
        processed = 0

        for start in range(0, count, batch_size):
            if self.stop_check():
                logger.info("[PHASE:mv] stop requested")
                break

            chunk = mv_pairs[start:start + batch_size]
            t_sub = time.perf_counter()

            items_chunk = [pair[0] for pair in chunk]
            texts = [pair[1] for pair in chunk]

            try:
                # Batch encode (with fallback)
                if hasattr(provider, "encode_batch") and len(texts) > 1:
                    vecs = provider.encode_batch(texts)
                else:
                    vecs = [provider.encode(t) for t in texts]

                for j, vec in enumerate(vecs):
                    item = items_chunk[j]
                    item.mv_embedding = vec

                    self.storage.save_mv(item, vec, texts[j])

                    self.progress.file_done(
                        "mv", processed + j, count, item.file_name, True)

            except Exception as e:
                logger.warning(f"[PHASE:mv] batch failed: {e}, trying individual")
                # Fallback to individual encoding
                for j, (item, text) in enumerate(chunk):
                    try:
                        vec = provider.encode(text)
                        item.mv_embedding = vec
                        self.storage.save_mv(item, vec, text)
                        self.progress.file_done(
                            "mv", processed + j, count, item.file_name, True)
                    except Exception as e2:
                        logger.warning(f"[PHASE:mv] {item.file_name}: {e2}")
                        item.error = str(e2)
                        self.progress.file_done(
                            "mv", processed + j, count, item.file_name, False)

            elapsed_sub = time.perf_counter() - t_sub
            self.batch_strategy.after_sub_batch("mv", elapsed_sub, len(chunk))
            processed += len(chunk)

        # Unload MV
        self.models.unload_mv()

        elapsed = time.perf_counter() - t0
        self.progress.phase_complete("mv", elapsed)
        logger.info(f"[PHASE:mv] {processed}/{count} in {elapsed:.1f}s")

        return items

    # ------------------------------------------------------------------
    # run_all: V → VV → MV sequential execution
    # ------------------------------------------------------------------

    def run_all(self, items: List[PhaseItem]) -> Tuple[int, int]:
        """
        Execute V → VV → MV phases sequentially.

        Phase P is NOT included — the caller handles parsing.

        Returns:
            (success_count, fail_count)
        """
        items = self.run_vision(items)
        items = self.run_vv(items)
        items = self.run_mv(items)

        # Flush any buffered storage writes
        try:
            self.storage.flush()
        except Exception as e:
            logger.error(f"Storage flush failed: {e}")

        success = sum(1 for it in items if it.succeeded)
        failed = len(items) - success
        return success, failed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_thumbnail(self, item: PhaseItem) -> Optional[Image.Image]:
        """Load thumbnail image for VLM / VV processing."""
        path = item.thumb_path or item.file_path
        if not path or not Path(path).exists():
            return None
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            logger.warning(f"Thumbnail load failed {path}: {e}")
            return None

    @staticmethod
    def _has_vlm_result(item: PhaseItem) -> bool:
        """Check if item already has actual VLM results (not just parse context).

        mc_raw from ParseAhead contains only parse-context metadata
        (file_name, layers, folder_path etc.) — NOT VLM results.
        Real VLM output always includes 'mc_caption'/'caption' or 'ai_tags'/'tags'.
        """
        if not item.mc_raw or not isinstance(item.mc_raw, dict):
            return False
        mc = item.mc_raw
        return bool(
            mc.get("mc_caption") or mc.get("caption")
            or mc.get("ai_tags") or mc.get("tags")
        )

    def _build_vision_context(self, item: PhaseItem) -> Optional[dict]:
        """Build context dict for VLM from PhaseItem metadata."""
        ctx = {}
        if item.folder_path:
            ctx["folder_path"] = item.folder_path
        if item.folder_tags:
            ctx["folder_tags"] = item.folder_tags
        if item.mc_raw and isinstance(item.mc_raw, dict):
            ctx.update(item.mc_raw)
        return ctx if ctx else None

    def _compose_mv_text(self, item: PhaseItem) -> Optional[str]:
        """
        Compose MV input text from MC data.

        Uses build_document_text() from text_embedding module for
        consistent [SEMANTIC] + [FACTS] format.
        """
        if not item.mc_raw:
            return None

        mc = item.mc_raw
        caption = mc.get("mc_caption", "") or mc.get("caption", "")
        tags = mc.get("ai_tags", "") or mc.get("tags", [])

        if not caption and not tags:
            return None

        # Build facts from available metadata (image content only, no folder path)
        # Folder path belongs to FTS (metadata axis), not MV (semantic axis)
        facts = {}
        for key in ("image_type", "scene_type", "art_style"):
            val = mc.get(key)
            if val:
                facts[key] = val

        try:
            from backend.vector.text_embedding import build_document_text
            return build_document_text(caption, tags, facts=facts)
        except ImportError:
            # Fallback: simple concatenation
            if isinstance(tags, list):
                tags = ", ".join(str(t) for t in tags)
            return f"{caption} {tags}".strip() or None

    def _get_active_domain(self):
        """Get the active classification domain as a DomainProfile object."""
        try:
            from backend.utils.config import get_config
            cfg = get_config()
            domain_id = cfg.get("classification.active_domain")
            if not domain_id:
                return None
            from backend.vision.domain_loader import load_domain
            return load_domain(domain_id)
        except Exception as e:
            logger.warning(f"Failed to load active domain: {e}")
            return None
