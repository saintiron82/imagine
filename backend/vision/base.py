"""
Base Vision Analyzer - Abstract base class for all vision adapters.

Extracted from vision_factory.py to avoid circular imports when adapters
need to inherit from this base class.

All adapters (Transformers, MLX, Ollama, vLLM) must inherit from
BaseVisionAnalyzer and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from PIL import Image


class BaseVisionAnalyzer(ABC):
    """
    Abstract base for all vision adapters.

    Subclasses must implement:
      - analyze()
      - classify()
      - analyze_structured()
      - classify_and_analyze()
      - unload_model()

    Default implementations provided for:
      - analyze_file()  (opens image, delegates to analyze())
      - classify_and_analyze_sequence()  (sequential loop over items)
    """

    @abstractmethod
    def analyze(self, image: Image.Image, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Legacy single-pass analysis.

        Returns:
            Dictionary with caption, tags, ocr, color, style
        """
        ...

    def analyze_file(self, image_path, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze an image file (legacy single-pass)."""
        raise NotImplementedError

    # -- v3 P0: 2-Stage Pipeline ------------------------------------

    @abstractmethod
    def classify(self, image: Image.Image, keep_alive: str = None) -> Dict[str, Any]:
        """Stage 1: Classify image type. Returns {image_type, confidence}."""
        ...

    @abstractmethod
    def analyze_structured(self, image: Image.Image, image_type: str, keep_alive: str = None) -> Dict[str, Any]:
        """Stage 2: Type-specific structured analysis."""
        ...

    @abstractmethod
    def classify_and_analyze(self, image: Image.Image, keep_alive: str = None) -> Dict[str, Any]:
        """Full 2-Stage pipeline: classify -> analyze_structured."""
        ...

    def classify_and_analyze_sequence(
        self,
        items: list,
        progress_callback=None,
    ) -> list:
        """
        Process multiple images sequentially through 2-stage pipeline.
        Model stays loaded across calls to avoid repeated lazy-load overhead.

        Args:
            items: List of (PIL Image, context dict) tuples
            progress_callback: Optional fn(index, total, result)

        Returns:
            List of vision result dicts (same order as input)
        """
        results = []
        for idx, (image, context) in enumerate(items):
            try:
                result = self.classify_and_analyze(image, context=context)
            except Exception:
                result = {"caption": "", "tags": [], "image_type": "other"}
            results.append(result)
            if progress_callback:
                progress_callback(idx, len(items), result)
        return results

    @abstractmethod
    def unload_model(self):
        """Explicitly unload model from VRAM. Subclasses should override."""
        ...
