"""
Text Embedding Provider for S-axis semantic search.

Uses Ollama's embed API with Qwen3-Embedding-0.6B (1024-dim) by default.
Encodes mc_caption + ai_tags into a dense vector for text-to-text retrieval,
complementing the V-axis (SigLIP 2 image-to-text) search.

v3.1: Supports 3-Tier architecture with MRL (Matryoshka Representation Learning)
truncation and re-normalization for efficient embedding.

Usage:
    provider = get_text_embedding_provider()
    query_vec = provider.encode("fantasy character with sword", is_query=True)
    doc_vec   = provider.encode("A warrior holding a glowing sword in a dark forest")
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode text to a normalized embedding vector.

        Args:
            text: Input text string
            is_query: If True, apply query instruction prefix

        Returns:
            L2-normalized numpy array of shape (dimensions,)
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        ...


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Text embedding via Ollama's /api/embed endpoint.

    Default model: qwen3-embedding:0.6b (1024 dimensions)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        dimensions: Optional[int] = None,
        instruction_prefix: Optional[str] = None,
        normalize: bool = False,
    ):
        from ..utils.config import get_config
        cfg = get_config()

        self.model = model or cfg.get("embedding.text.model", "qwen3-embedding:0.6b")
        self.host = host or cfg.get("vision.ollama_host", "http://localhost:11434")
        self._dimensions = dimensions or cfg.get("embedding.text.dimensions", 1024)
        self._instruction_prefix = instruction_prefix or cfg.get(
            "embedding.text.instruction_prefix", ""
        )
        self._normalize = normalize  # v3.1: MRL re-normalization flag
        self._embed_url = f"{self.host.rstrip('/')}/api/embed"

        logger.info(
            f"OllamaEmbeddingProvider initialized "
            f"(model={self.model}, dims={self._dimensions}, normalize={normalize})"
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode text via Ollama embed API.

        Args:
            text: Input text
            is_query: If True, prepend instruction_prefix for retrieval queries

        Returns:
            L2-normalized float32 numpy array
        """
        if not text or not text.strip():
            return np.zeros(self._dimensions, dtype=np.float32)

        input_text = text.strip()
        if is_query and self._instruction_prefix:
            input_text = f"{self._instruction_prefix}{input_text}"

        payload = {
            "model": self.model,
            "input": input_text,
        }

        try:
            resp = requests.post(self._embed_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            embeddings = data.get("embeddings", [])
            if not embeddings:
                logger.error(f"Empty embeddings response from Ollama")
                return np.zeros(self._dimensions, dtype=np.float32)

            vec = np.array(embeddings[0], dtype=np.float32)

            # v3.1: MRL Truncation + Re-normalize (if enabled)
            if self._normalize and len(vec) > self._dimensions:
                vec = vec[:self._dimensions]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            else:
                # Standard L2 normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

            return vec

        except requests.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.host}. "
                f"Is Ollama running? (ollama serve)"
            )
            return np.zeros(self._dimensions, dtype=np.float32)
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return np.zeros(self._dimensions, dtype=np.float32)

    def encode_batch(self, texts: list[str], is_query: bool = False) -> list[np.ndarray]:
        """
        Encode multiple texts in a single API call.

        Args:
            texts: List of input texts
            is_query: If True, prepend instruction_prefix

        Returns:
            List of L2-normalized float32 numpy arrays
        """
        if not texts:
            return []

        inputs = []
        for t in texts:
            t = t.strip() if t else ""
            if is_query and self._instruction_prefix:
                t = f"{self._instruction_prefix}{t}"
            inputs.append(t)

        payload = {
            "model": self.model,
            "input": inputs,
        }

        try:
            resp = requests.post(self._embed_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for emb in data.get("embeddings", []):
                vec = np.array(emb, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results.append(vec)
            return results

        except Exception as e:
            logger.error(f"Batch text embedding failed: {e}")
            return [np.zeros(self._dimensions, dtype=np.float32) for _ in texts]


def build_document_text(caption: str, tags: list, facts: dict = None) -> str:
    """
    Build a document string for MV (text_vec) embedding.

    v3.1: Two-section format for better semantic separation:
    - [SEMANTIC]: MC caption + AI tags (high-level meaning)
    - [FACTS]: Structured metadata (image_type, path, fonts, etc.)

    Args:
        caption: MC caption (AI-generated with file metadata context)
        tags: AI-generated keyword tags
        facts: Optional structured metadata dict
               Keys: image_type, scene_type, art_style, fonts, path

    Returns:
        Combined text suitable for embedding

    Example:
        >>> build_document_text(
        ...     "A warrior holding a sword",
        ...     ["character", "fantasy", "weapon"],
        ...     {"image_type": "character", "fonts": "NotoSans"}
        ... )
        '[SEMANTIC]\\nA warrior holding a sword\\nKeywords: character, fantasy, weapon\\n\\n[FACTS]\\nimage_type=character; fonts=NotoSans'
    """
    parts = []

    # Section 1: SEMANTIC (caption + tags)
    semantic_parts = []
    if caption and caption.strip():
        semantic_parts.append(caption.strip())
    if tags:
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = [tags]
        if isinstance(tags, list) and tags:
            tag_str = ", ".join(str(t) for t in tags if t)
            if tag_str:
                semantic_parts.append(f"Keywords: {tag_str}")

    if semantic_parts:
        parts.append("[SEMANTIC]")
        parts.append("\n".join(semantic_parts))

    # Section 2: FACTS (structured metadata)
    if facts and isinstance(facts, dict):
        fact_pairs = []
        for key in ["image_type", "scene_type", "art_style", "fonts", "path"]:
            val = facts.get(key)
            if val and str(val).strip():
                fact_pairs.append(f"{key}={val}")

        if fact_pairs:
            parts.append("\n[FACTS]")
            parts.append("; ".join(fact_pairs))

    return "\n".join(parts) if parts else ""


# ── Singleton / Factory ───────────────────────────────────

_provider_instance: Optional[EmbeddingProvider] = None


def get_text_embedding_provider() -> EmbeddingProvider:
    """
    Return a singleton text embedding provider based on config.yaml.

    v3.1: Tier-aware configuration (Standard/Pro/Ultra).
    """
    global _provider_instance
    if _provider_instance is None:
        from ..utils.tier_config import get_active_tier
        from ..utils.config import get_config

        cfg = get_config()
        tier_name, tier_config = get_active_tier()
        text_config = tier_config.get("text_embed", {})

        # Extract tier-specific settings
        model = text_config.get("model") or cfg.get("embedding.text.model", "qwen3-embedding:0.6b")
        dimensions = text_config.get("dimensions") or cfg.get("embedding.text.dimensions", 1024)
        normalize = text_config.get("normalize", False)  # MRL re-normalize flag
        host = cfg.get("vision.ollama_host", "http://localhost:11434")
        instruction_prefix = cfg.get("embedding.text.instruction_prefix", "")

        logger.info(
            f"Initializing text embedding provider (tier: {tier_name}, "
            f"model: {model}, dims: {dimensions}, MRL normalize: {normalize})"
        )

        _provider_instance = OllamaEmbeddingProvider(
            model=model,
            host=host,
            dimensions=dimensions,
            instruction_prefix=instruction_prefix,
            normalize=normalize,
        )

    return _provider_instance
