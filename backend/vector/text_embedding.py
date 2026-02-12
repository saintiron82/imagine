"""
MV (Meaning Vector) Provider for semantic search.

Supports two backends:
- Transformers: Direct HuggingFace model loading (default for macOS/pro tier)
- Ollama: /api/embed endpoint (fallback)

Default model: Qwen/Qwen3-Embedding-0.6B (1024-dim)

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
    MV generation via Ollama's /api/embed endpoint.

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
            import requests
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

        except Exception as e:
            if "ConnectionError" in type(e).__name__:
                logger.error(
                    f"Cannot connect to Ollama at {self.host}. "
                    f"Is Ollama running? (ollama serve)"
                )
            else:
                logger.error(f"MV generation failed: {e}")
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
            import requests
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
            logger.error(f"Batch MV generation failed: {e}")
            return [np.zeros(self._dimensions, dtype=np.float32) for _ in texts]


class TransformersEmbeddingProvider(EmbeddingProvider):
    """
    MV generation via HuggingFace transformers (no Ollama dependency).

    Uses Qwen3-Embedding-0.6B by default. Runs on MPS/CUDA/CPU.
    """

    # Map Ollama model names to HuggingFace model IDs
    _MODEL_MAP = {
        "qwen3-embedding:0.6b": "Qwen/Qwen3-Embedding-0.6B",
        "qwen3-embedding:8b": "Qwen/Qwen3-Embedding-8B",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        instruction_prefix: Optional[str] = None,
        normalize: bool = False,
        device: Optional[str] = None,
    ):
        from ..utils.config import get_config
        cfg = get_config()

        raw_model = model or cfg.get("embedding.text.model", "qwen3-embedding:0.6b")
        self._hf_model_id = self._MODEL_MAP.get(raw_model, raw_model)
        self._dimensions = dimensions or cfg.get("embedding.text.dimensions", 1024)
        self._instruction_prefix = instruction_prefix or cfg.get(
            "embedding.text.instruction_prefix", ""
        )
        self._normalize = normalize

        # Device selection
        import torch
        if device and device != "auto":
            self._device = device
        elif torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        # Lazy-load model and tokenizer
        self._tokenizer = None
        self._model = None

        logger.info(
            f"TransformersEmbeddingProvider initialized "
            f"(model={self._hf_model_id}, dims={self._dimensions}, "
            f"device={self._device}, normalize={normalize})"
        )

    def _ensure_loaded(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Loading MV model: {self._hf_model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hf_model_id, padding_side='left'
        )
        self._model = AutoModel.from_pretrained(
            self._hf_model_id, torch_dtype=torch.float16
        ).to(self._device).eval()
        logger.info(f"MV model loaded on {self._device}")

    def unload(self):
        """Unload MV model to free GPU memory."""
        import gc
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info(f"MV model unloaded ({self._hf_model_id})")

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        """Extract the last non-padding token's hidden state."""
        import torch
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, text: str, is_query: bool = False) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self._dimensions, dtype=np.float32)

        self._ensure_loaded()

        import torch

        input_text = text.strip()
        if is_query and self._instruction_prefix:
            input_text = f"{self._instruction_prefix}{input_text}"

        try:
            batch_dict = self._tokenizer(
                [input_text], padding=True, truncation=True,
                max_length=8192, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**batch_dict)

            emb = self._last_token_pool(
                outputs.last_hidden_state, batch_dict['attention_mask']
            )
            vec = emb[0].float().cpu().numpy()

            # Free GPU tensors immediately (MPS doesn't auto-reclaim)
            del batch_dict, outputs, emb
            if self._device == "mps":
                torch.mps.empty_cache()

            # MRL truncation if needed
            if self._normalize and len(vec) > self._dimensions:
                vec = vec[:self._dimensions]

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            return vec.astype(np.float32)

        except Exception as e:
            logger.error(f"MV generation failed: {e}")
            return np.zeros(self._dimensions, dtype=np.float32)

    def encode_batch(self, texts: list[str], is_query: bool = False) -> list[np.ndarray]:
        if not texts:
            return []

        self._ensure_loaded()

        import torch

        inputs = []
        for t in texts:
            t = t.strip() if t else ""
            if is_query and self._instruction_prefix:
                t = f"{self._instruction_prefix}{t}"
            inputs.append(t)

        try:
            batch_dict = self._tokenizer(
                inputs, padding=True, truncation=True,
                max_length=8192, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**batch_dict)

            embs = self._last_token_pool(
                outputs.last_hidden_state, batch_dict['attention_mask']
            )

            results = []
            for i in range(embs.shape[0]):
                vec = embs[i].float().cpu().numpy()
                if self._normalize and len(vec) > self._dimensions:
                    vec = vec[:self._dimensions]
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                results.append(vec.astype(np.float32))

            # Free GPU tensors immediately (MPS doesn't auto-reclaim)
            del batch_dict, outputs, embs
            if self._device == "mps":
                torch.mps.empty_cache()

            return results

        except Exception as e:
            logger.error(f"Batch MV generation failed: {e}")
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
    Return a singleton MV provider based on config.yaml.

    v3.1: Tier-aware configuration (Standard/Pro/Ultra).
    v3.2: Prefers TransformersEmbeddingProvider (no Ollama dependency).
          Falls back to OllamaEmbeddingProvider if transformers unavailable.
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
        instruction_prefix = cfg.get("embedding.text.instruction_prefix", "")

        logger.info(
            f"Initializing MV provider (tier: {tier_name}, "
            f"model: {model}, dims: {dimensions}, MRL normalize: {normalize})"
        )

        # Prefer Transformers backend (works without Ollama, supports MPS)
        try:
            _provider_instance = TransformersEmbeddingProvider(
                model=model,
                dimensions=dimensions,
                instruction_prefix=instruction_prefix,
                normalize=normalize,
            )
            logger.info("Using TransformersEmbeddingProvider (HuggingFace direct)")
        except Exception as e:
            logger.warning(f"TransformersEmbeddingProvider failed: {e}, falling back to Ollama")
            import requests as _req
            host = cfg.get("vision.ollama_host", "http://localhost:11434")
            _provider_instance = OllamaEmbeddingProvider(
                model=model,
                host=host,
                dimensions=dimensions,
                instruction_prefix=instruction_prefix,
                normalize=normalize,
            )

    return _provider_instance
