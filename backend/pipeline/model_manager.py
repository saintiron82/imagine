"""
model_manager.py — Unified model lifecycle management.

Single source of truth for loading/unloading VLM, SigLIP2, Qwen3-Embedding.
Shared by ingest_engine (local pipeline) and worker_daemon (server/worker).
"""

from __future__ import annotations

import gc
import logging
import sys
from typing import Any, Optional

logger = logging.getLogger("pipeline.model_manager")


def _try_empty_gpu_cache():
    """Empty GPU memory cache (CUDA / MPS)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


class ModelManager:
    """
    Unified model lifecycle singleton.

    Lazy-loads models on first access, explicitly unloads after each phase.
    All three pipeline modes (local, worker, server) use the same manager.
    """

    def __init__(self):
        self._vlm: Any = None
        self._vv_encoder: Any = None
        self._mv_provider: Any = None

    # ------------------------------------------------------------------
    # VLM (Vision-Language Model)
    # ------------------------------------------------------------------

    def get_vlm(self, timeout_s: float = 180.0):
        """
        Get or load the VLM analyzer.

        Uses VisionAnalyzerFactory (vision_factory.py) which handles
        platform-specific backend selection and fallback chain.
        """
        if self._vlm is not None:
            return self._vlm

        from backend.vision.vision_factory import get_vision_analyzer

        analyzer = get_vision_analyzer()

        # Trigger model preload if supported
        if hasattr(analyzer, "_load_model"):
            try:
                analyzer._load_model()
                logger.info(f"VLM loaded: {type(analyzer).__name__}")
            except Exception as e:
                logger.error(f"VLM load failed: {e}")
                raise

        self._vlm = analyzer
        return self._vlm

    def unload_vlm(self) -> None:
        """Unload VLM and free GPU memory."""
        try:
            from backend.vision.vision_factory import VisionAnalyzerFactory
            if self._vlm is not None:
                if hasattr(self._vlm, "unload_model"):
                    self._vlm.unload_model()
            VisionAnalyzerFactory.reset()
        except Exception as e:
            logger.debug(f"VLM unload note: {e}")

        self._vlm = None
        gc.collect()
        _try_empty_gpu_cache()
        logger.info("VLM unloaded")

    # ------------------------------------------------------------------
    # SigLIP2 (Visual Vector encoder)
    # ------------------------------------------------------------------

    def get_vv_encoder(self):
        """Get or load the SigLIP2 encoder."""
        if self._vv_encoder is not None:
            return self._vv_encoder

        from backend.vector.siglip2_encoder import SigLIP2Encoder

        self._vv_encoder = SigLIP2Encoder()
        logger.info(f"SigLIP2 loaded: {self._vv_encoder.model_name}")
        return self._vv_encoder

    def unload_vv(self) -> None:
        """Unload SigLIP2 encoder and free GPU memory."""
        if self._vv_encoder is not None:
            if hasattr(self._vv_encoder, "unload"):
                self._vv_encoder.unload()
            self._vv_encoder = None

        gc.collect()
        _try_empty_gpu_cache()
        logger.info("SigLIP2 unloaded")

    # ------------------------------------------------------------------
    # Qwen3-Embedding (Meaning Vector encoder)
    # ------------------------------------------------------------------

    def get_mv_encoder(self):
        """Get or load the text embedding provider."""
        if self._mv_provider is not None:
            return self._mv_provider

        from backend.vector.text_embedding import get_text_embedding_provider

        self._mv_provider = get_text_embedding_provider()
        logger.info(f"MV encoder loaded: {type(self._mv_provider).__name__}")
        return self._mv_provider

    def unload_mv(self) -> None:
        """Unload MV encoder and free GPU memory."""
        if self._mv_provider is not None:
            if hasattr(self._mv_provider, "unload"):
                self._mv_provider.unload()
            self._mv_provider = None

        # Also reset the module-level singleton
        try:
            from backend.vector.text_embedding import reset_provider
            reset_provider()
        except Exception:
            pass

        gc.collect()
        _try_empty_gpu_cache()
        logger.info("MV encoder unloaded")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def unload_all(self) -> None:
        """Unload all models."""
        self.unload_vlm()
        self.unload_vv()
        self.unload_mv()

    @property
    def vlm_loaded(self) -> bool:
        return self._vlm is not None

    @property
    def vv_loaded(self) -> bool:
        return self._vv_encoder is not None

    @property
    def mv_loaded(self) -> bool:
        return self._mv_provider is not None
