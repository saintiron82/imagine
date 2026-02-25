"""
DINOv2 Encoder Module.

This module provides the DinoV2Encoder class for generating
structural/texture-based image embeddings using DINOv2.

Model: facebook/dinov2-base (768-dim)
"""

import logging
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Union
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


class DinoV2Encoder:
    """
    DINOv2 encoder for structural image embeddings.

    Uses 'facebook/dinov2-base' model.
    Output dimension: 768
    """

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        self.model_name = model_name
        self.device = self._get_device()
        self.processor = None
        self.model = None
        self.dimensions = 768  # DINOv2 Base
        
    def _get_device(self) -> str:
        """Select best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self):
        """Lazy load the model and processor (auto-download on first use)."""
        if self.model is not None:
            return

        logger.info(f"Loading DINOv2 model: {self.model_name} on {self.device}...")
        try:
            # Stage 1: try local cache first (fast, no network)
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(
                self.model_name, local_files_only=True).to(self.device)
            self.model.eval()
            logger.info("DINOv2 model loaded from cache.")
        except OSError:
            # Stage 2: auto-download from HuggingFace
            logger.info("DINOv2 local cache not found, downloading from HuggingFace...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("DINOv2 model downloaded and loaded.")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise

    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode a single image into a 768-dim vector.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Normalized numpy float32 array (768,)
        """
        self.load_model()
        
        # Ensure RGB
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # DINOv2's pooler_output or last_hidden_state class token
            # AutoModel for dinov2-base usually outputs last_hidden_state.
            # We use the [CLS] token (index 0) which aggregates global structural info.
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states[0, 0, :]  # (1, seq_len, hidden_size) -> (hidden_size,)
            
            # Normalize
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            logger.error(f"DINOv2 encoding failed: {e}")
            # Return zero vector on failure to avoid crashing pipeline
            return np.zeros(self.dimensions, dtype=np.float32)

    def unload(self):
        """Unload model to free VRAM."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            logger.info("DINOv2 model unloaded.")
