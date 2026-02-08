"""
Text Embedding Provider for T-axis semantic search.

Uses Ollama's embed API with Qwen3-Embedding-0.6B (1024-dim) by default.
Encodes ai_caption + ai_tags into a dense vector for text-to-text retrieval,
complementing the V-axis (SigLIP 2 image-to-text) search.

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
    ):
        from ..utils.config import get_config
        cfg = get_config()

        self.model = model or cfg.get("embedding.text.model", "qwen3-embedding:0.6b")
        self.host = host or cfg.get("vision.ollama_host", "http://localhost:11434")
        self._dimensions = dimensions or cfg.get("embedding.text.dimensions", 1024)
        self._instruction_prefix = instruction_prefix or cfg.get(
            "embedding.text.instruction_prefix", ""
        )
        self._embed_url = f"{self.host.rstrip('/')}/api/embed"

        logger.info(
            f"OllamaEmbeddingProvider initialized "
            f"(model={self.model}, dims={self._dimensions})"
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

            # L2 normalize
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


def build_document_text(caption: str, tags: list) -> str:
    """
    Build a document string from AI caption and tags for T-axis embedding.

    Args:
        caption: AI-generated image description
        tags: AI-generated keyword tags

    Returns:
        Combined text suitable for embedding
    """
    parts = []
    if caption and caption.strip():
        parts.append(caption.strip())
    if tags:
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = [tags]
        if isinstance(tags, list) and tags:
            tag_str = ", ".join(str(t) for t in tags if t)
            if tag_str:
                parts.append(f"Tags: {tag_str}")
    return ". ".join(parts) if parts else ""


# ── Singleton / Factory ───────────────────────────────────

_provider_instance: Optional[EmbeddingProvider] = None


def get_text_embedding_provider() -> EmbeddingProvider:
    """
    Return a singleton text embedding provider based on config.yaml.

    Currently only OllamaEmbeddingProvider is supported.
    """
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OllamaEmbeddingProvider()
    return _provider_instance
