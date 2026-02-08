"""
SigLIP 2 Vision-Language Encoder for image/text embeddings.

Replaces CLIP ViT-L-14 (768-dim) with SigLIP 2 So400m (1152-dim).
Uses HuggingFace transformers AutoModel (not sentence-transformers).

Note: AutoProcessor has a tokenizer mapping bug in transformers 5.x,
so we load AutoImageProcessor and GemmaTokenizerFast separately.
Forward pass (model(**inputs)) returns pooled image_embeds/text_embeds.
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)


class SigLIP2Encoder:
    """
    Lazy-loaded SigLIP 2 encoder for image and text embeddings.

    Usage:
        encoder = SigLIP2Encoder()                        # config-driven
        img_vec = encoder.encode_image(pil_image)          # np.ndarray (1152,)
        txt_vec = encoder.encode_text("search query")     # np.ndarray (1152,)
    """

    def __init__(self, model_name: Optional[str] = None):
        from ..utils.config import get_config
        cfg = get_config()

        self.model_name = model_name or cfg.get(
            "embedding.visual.model", "google/siglip2-so400m-patch14-384"
        )
        self._dimensions = cfg.get("embedding.visual.dimensions", 1152)
        self._model = None
        self._image_processor = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"SigLIP2Encoder initialized (model: {self.model_name}, device: {self._device})")

    def _load(self):
        """Lazy-load model, image processor, and tokenizer on first use."""
        if self._model is not None and self._image_processor is not None and self._tokenizer is not None:
            return

        from transformers import AutoModel, AutoImageProcessor, GemmaTokenizerFast

        logger.info(f"Loading SigLIP 2 model: {self.model_name}...")
        try:
            self._model = AutoModel.from_pretrained(
                self.model_name, dtype=torch.float16, local_files_only=True
            ).to(self._device).eval()
            self._image_processor = AutoImageProcessor.from_pretrained(
                self.model_name, local_files_only=True
            )
            self._tokenizer = GemmaTokenizerFast.from_pretrained(
                self.model_name, local_files_only=True
            )
            logger.info(f"SigLIP 2 model loaded ({self._device}, fp16)")
        except OSError:
            logger.info("Local cache not found, downloading from HuggingFace...")
            self._model = AutoModel.from_pretrained(
                self.model_name, dtype=torch.float16
            ).to(self._device).eval()
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._tokenizer = GemmaTokenizerFast.from_pretrained(self.model_name)
            logger.info(f"SigLIP 2 model downloaded and loaded ({self._device}, fp16)")
        except Exception as e:
            logger.error(f"Failed to load SigLIP 2 model: {e}")
            self._model = None
            self._image_processor = None
            self._tokenizer = None
            raise

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image to a normalized embedding vector.

        Uses forward pass to get pooled image_embeds (not per-patch features).

        Args:
            image: PIL Image (RGB)

        Returns:
            L2-normalized numpy array of shape (1152,)
        """
        self._load()
        img_inputs = self._image_processor(images=[image], return_tensors="pt")
        img_inputs = {k: v.to(self._device) for k, v in img_inputs.items()}

        # Dummy text input for forward pass (required by model)
        txt_inputs = self._tokenizer([""], return_tensors="pt", padding=True)
        txt_inputs = {k: v.to(self._device) for k, v in txt_inputs.items()}

        with torch.no_grad():
            outputs = self._model(**img_inputs, **txt_inputs)

        vec = outputs.image_embeds[0].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query to a normalized embedding vector.

        Uses forward pass to get pooled text_embeds.

        Args:
            text: Search query string

        Returns:
            L2-normalized numpy array of shape (1152,)
        """
        self._load()
        txt_inputs = self._tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        )
        txt_inputs = {k: v.to(self._device) for k, v in txt_inputs.items()}

        # Dummy image input for forward pass (required by model)
        dummy_img = Image.new("RGB", (384, 384), (128, 128, 128))
        img_inputs = self._image_processor(images=[dummy_img], return_tensors="pt")
        img_inputs = {k: v.to(self._device) for k, v in img_inputs.items()}

        with torch.no_grad():
            outputs = self._model(**img_inputs, **txt_inputs)

        vec = outputs.text_embeds[0].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
