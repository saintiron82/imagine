"""
Encoder management for search: lazy-loading/unloading of AI encoders.

Manages SigLIP2 (VV), Qwen3-Embedding (MV), and DINOv2 (Structure)
encoder lifecycle, extracted from SqliteVectorSearch.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EncoderManager:
    """Manages lazy-loading/unloading of AI encoders (SigLIP2, Qwen3-Embedding, DINOv2)."""

    def __init__(self, db=None):
        """
        Initialize encoder manager.

        Args:
            db: SQLiteDB instance (needed for text_search_enabled check)
        """
        self._encoder = None          # SigLIP2 (VV)
        self._text_provider = None     # Qwen3-Embedding (MV)
        self._structure_encoder = None  # DINOv2 (Structure)
        self._text_enabled = None      # Cache for MV availability check
        self.db = db

    @property
    def vv_encoder(self):
        """Lazy load VV embedding encoder (SigLIP2)."""
        if self._encoder is None:
            from backend.vector.siglip2_encoder import SigLIP2Encoder
            self._encoder = SigLIP2Encoder()
            logger.info(f"Embedding encoder loaded: {self._encoder.model_name}")
        return self._encoder

    @property
    def mv_encoder(self):
        """Lazy load MV provider (Qwen3-Embedding)."""
        if self._text_provider is None:
            from backend.vector.text_embedding import get_text_embedding_provider
            self._text_provider = get_text_embedding_provider()
        return self._text_provider

    @property
    def structure_encoder(self):
        """Lazy load DINOv2 structure encoder."""
        if self._structure_encoder is None:
            from backend.vector.dinov2_encoder import DinoV2Encoder
            self._structure_encoder = DinoV2Encoder()
            logger.info(f"Structure encoder loaded: {self._structure_encoder.model_name}")
        return self._structure_encoder

    @property
    def text_search_enabled(self) -> bool:
        """Check if MV search is available (vec_text table exists with data)."""
        if self._text_enabled is None:
            if self.db is None:
                self._text_enabled = False
            else:
                try:
                    count = self.db.conn.execute("SELECT COUNT(*) FROM vec_text").fetchone()[0]
                    self._text_enabled = count > 0
                except Exception:
                    self._text_enabled = False
        return self._text_enabled

    def encode_text(self, query: str) -> np.ndarray:
        """
        Encode text query to VV embedding vector (SigLIP2).

        Args:
            query: Text query

        Returns:
            Embedding vector
        """
        return self.vv_encoder.encode_text(query)

    def encode_structure(self, image) -> np.ndarray:
        """
        Encode image to Structure embedding vector (DINOv2).

        Args:
            image: PIL Image or scalar

        Returns:
            768-dim embedding vector
        """
        return self.structure_encoder.encode_image(image)

    def unload_all(self) -> None:
        """Unload all loaded encoders to free memory."""
        if self._encoder is not None:
            del self._encoder
            self._encoder = None
            logger.info("VV encoder unloaded")
        if self._text_provider is not None:
            del self._text_provider
            self._text_provider = None
            logger.info("MV encoder unloaded")
        if self._structure_encoder is not None:
            del self._structure_encoder
            self._structure_encoder = None
            logger.info("Structure encoder unloaded")
