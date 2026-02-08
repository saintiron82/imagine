"""
Vision Analyzer Factory - Environment-based adapter selection.

Automatically chooses between:
- Transformers (development): High accuracy, requires GPU/CPU resources
- Ollama (deployment): Memory-efficient, automatic model management
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

# Load environment variables
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded environment from {env_path}")
    else:
        logging.debug(f".env file not found at {env_path}, using system environment")
except ImportError:
    logging.warning("python-dotenv not installed, using system environment only")

logger = logging.getLogger(__name__)


class VisionAnalyzerFactory:
    """
    Factory for creating vision analyzers based on environment configuration.

    Environment Variables:
        VISION_BACKEND: 'transformers' or 'ollama' (default: 'transformers')
        VISION_MODEL: Model to use (backend-specific)
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    """

    _cached_analyzer = None

    @classmethod
    def create(cls) -> "BaseVisionAnalyzer":
        """
        Create vision analyzer based on environment configuration.

        Returns:
            VisionAnalyzer (Transformers) or OllamaVisionAdapter
        """
        # Use cached instance for efficiency
        if cls._cached_analyzer is not None:
            return cls._cached_analyzer

        backend = os.getenv('VISION_BACKEND', 'transformers').lower()

        if backend == 'ollama':
            logger.info("Using Ollama vision backend (deployment mode)")
            from .ollama_adapter import OllamaVisionAdapter

            # OllamaVisionAdapter reads from config.yaml internally
            cls._cached_analyzer = OllamaVisionAdapter()

        else:
            logger.info("Using Transformers vision backend (development mode)")
            from .analyzer import VisionAnalyzer

            model = os.getenv('VISION_MODEL', 'Qwen/Qwen2-VL-2B-Instruct')
            device = os.getenv('VISION_DEVICE', None)  # auto-detect

            cls._cached_analyzer = VisionAnalyzer(device=device, model_id=model)

        return cls._cached_analyzer

    @classmethod
    def reset(cls):
        """Reset cached analyzer (for testing or switching backends)."""
        cls._cached_analyzer = None


class BaseVisionAnalyzer:
    """
    Base interface for vision analyzers.

    All adapters (Transformers, Ollama) must implement this interface.
    """

    def analyze(self, image: Image.Image, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Legacy single-pass analysis.

        Returns:
            Dictionary with caption, tags, ocr, color, style
        """
        raise NotImplementedError

    def analyze_file(self, image_path, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze an image file (legacy single-pass)."""
        raise NotImplementedError

    # ── v3 P0: 2-Stage Pipeline ────────────────────────────

    def classify(self, image: Image.Image, keep_alive: str = None) -> Dict[str, Any]:
        """Stage 1: Classify image type. Returns {image_type, confidence}."""
        raise NotImplementedError

    def analyze_structured(self, image: Image.Image, image_type: str, keep_alive: str = None) -> Dict[str, Any]:
        """Stage 2: Type-specific structured analysis."""
        raise NotImplementedError

    def classify_and_analyze(self, image: Image.Image, keep_alive: str = None) -> Dict[str, Any]:
        """Full 2-Stage pipeline: classify → analyze_structured."""
        raise NotImplementedError

    def unload_model(self):
        """Explicitly unload model from VRAM."""
        pass


def get_vision_analyzer() -> BaseVisionAnalyzer:
    """
    Convenience function to get vision analyzer.

    Usage:
        analyzer = get_vision_analyzer()
        result = analyzer.analyze(image)
    """
    return VisionAnalyzerFactory.create()
