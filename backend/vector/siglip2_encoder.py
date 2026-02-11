"""
SigLIP 2 NaFlex VV (Visual Vector) Encoder.

Uses SigLIP 2 So400m NaFlex (1152-dim) with dynamic resolution.
NaFlex preserves native aspect ratio for better embedding quality.

v3.1: Supports 3-Tier architecture with automatic model selection.

API: get_image_features / get_text_features → pooler_output (768/1152/1664,)
"""

import base64
import logging
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SigLIP2Encoder:
    """
    Lazy-loaded SigLIP 2 NaFlex encoder for VV generation.

    Usage:
        encoder = SigLIP2Encoder()                        # config-driven
        img_vec = encoder.encode_image(pil_image)          # np.ndarray (1152,)
        txt_vec = encoder.encode_text("search query")     # np.ndarray (1152,)
    """

    def __init__(self, model_name: Optional[str] = None):
        from ..utils.config import get_config
        from ..utils.tier_config import get_active_tier

        cfg = get_config()

        # v3.1: Tier-aware model selection
        tier_name, tier_config = get_active_tier()
        visual_config = tier_config.get("visual", {})

        # Priority: explicit param > tier config > legacy config
        self.model_name = model_name or visual_config.get("model") or cfg.get(
            "embedding.visual.model", "google/siglip2-so400m-patch16-naflex"
        )
        self._dimensions = visual_config.get("dimensions") or cfg.get(
            "embedding.visual.dimensions", 1152
        )
        self.tier_name = tier_name
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info(
            f"SigLIP2Encoder initialized (tier: {tier_name}, model: {self.model_name}, "
            f"dimensions: {self._dimensions}, device: {self._device})"
        )

    def _load(self):
        """Lazy-load model and processor on first use."""
        if self._model is not None and self._processor is not None:
            return

        from transformers import AutoModel, AutoImageProcessor

        logger.info(f"Loading SigLIP 2 model: {self.model_name}...")
        try:
            self._model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.float16, local_files_only=True
            ).to(self._device).eval()
            self._processor = AutoImageProcessor.from_pretrained(
                self.model_name, local_files_only=True
            )
            logger.info(f"SigLIP 2 model loaded ({self._device}, fp16)")
        except OSError:
            logger.info("Local cache not found, downloading from HuggingFace...")
            self._model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self._device).eval()
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            logger.info(f"SigLIP 2 model downloaded and loaded ({self._device}, fp16)")
        except Exception as e:
            logger.error(f"Failed to load SigLIP 2 model: {e}")
            self._model = None
            self._processor = None
            raise

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image to a normalized embedding vector.

        NaFlex preserves native aspect ratio with dynamic patch count.

        Args:
            image: PIL Image (RGB)

        Returns:
            L2-normalized numpy array of shape (1152,)
        """
        self._load()
        inputs = self._processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # get_image_features returns BaseModelOutputWithPooling or tensor
        if hasattr(image_features, 'pooler_output'):
            vec = image_features.pooler_output[0]
        elif image_features.dim() == 2:
            vec = image_features[0]
        else:
            vec = image_features[1][0]  # fallback: index [1] = pooler_output

        vec = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def encode_image_from_base64(self, base64_data: str) -> np.ndarray:
        """
        Encode a base64-encoded image to a normalized embedding vector.

        Accepts raw base64 or data URL (data:image/...;base64,...).

        Args:
            base64_data: Base64-encoded image string

        Returns:
            L2-normalized numpy array of shape (dimensions,)
        """
        # Strip data URL prefix if present
        if "," in base64_data and base64_data.startswith("data:"):
            base64_data = base64_data.split(",", 1)[1]

        raw = base64.b64decode(base64_data)
        image = Image.open(BytesIO(raw)).convert("RGB")
        return self.encode_image(image)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text query to a normalized embedding vector.

        Processor handles lowercasing + padding automatically.

        Args:
            text: Search query string

        Returns:
            L2-normalized numpy array of shape (1152,)
        """
        self._load()
        inputs = self._processor(
            text=[text], return_tensors="pt",
            padding="max_length", max_length=64, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

        # get_text_features returns BaseModelOutputWithPooling or tensor
        if hasattr(text_features, 'pooler_output'):
            vec = text_features.pooler_output[0]
        elif text_features.dim() == 2:
            vec = text_features[0]
        else:
            vec = text_features[1][0]  # fallback: index [1] = pooler_output

        vec = vec.float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)
