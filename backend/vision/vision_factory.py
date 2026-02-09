"""
Vision Analyzer Factory - Environment-based adapter selection.

Automatically chooses between:
- Transformers (development): High accuracy, requires GPU/CPU resources
- Ollama (deployment): Memory-efficient, automatic model management

v3.1: Supports 3-Tier architecture (Standard/Pro/Ultra) with automatic VRAM detection.
"""

import os
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image

# v3.1: Tier-aware configuration
from backend.utils.tier_config import get_active_tier

# v3.1.1: Platform-specific optimization
from backend.utils.platform_detector import (
    get_optimal_backend,
    get_optimal_batch_size,
    get_platform_info
)

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

        v3.1: Uses tier-aware configuration (Standard/Pro/Ultra).
        v3.1.1: Platform-specific optimization (auto backend selection).

        Returns:
            VisionAnalyzer (Transformers), OllamaVisionAdapter, or VLLMAdapter
        """
        # Use cached instance for efficiency
        if cls._cached_analyzer is not None:
            return cls._cached_analyzer

        # v3.1: Load tier configuration
        tier_name, tier_config = get_active_tier()
        vlm_config = tier_config.get("vlm", {})

        # Backend selection: tier config > env var > default
        # Fix: Tier config should take priority over .env (GUI settings must be respected)
        backend = vlm_config.get("backend") or os.getenv('VISION_BACKEND') or "transformers"
        backend = backend.lower()

        # v3.1.1: AUTO mode - platform-specific optimization
        model = None  # Will be set by platform config or fallback
        if backend == 'auto':
            current_platform = platform.system().lower()  # 'windows', 'darwin', 'linux'

            logger.info(f"AUTO mode enabled (platform: {current_platform}, tier: {tier_name})")

            # Try platform-specific config first
            platform_configs = vlm_config.get("backends", {})
            if current_platform in platform_configs:
                platform_config = platform_configs[current_platform]
                backend = platform_config.get("backend", "ollama")
                model = platform_config.get("model")

                logger.info(f"✓ Platform-specific config found: {backend} ({model})")
            else:
                # Fallback to platform detector
                backend = get_optimal_backend(tier_name)
                logger.info(f"✓ Platform detector selected: {backend}")

        # Backend instantiation
        if backend == 'vllm':
            logger.info(f"Using vLLM vision backend (tier: {tier_name})")

            # Check platform compatibility
            if platform.system() == 'Windows':
                logger.error("vLLM is not supported on Windows. Falling back to Ollama.")
                backend = 'ollama'
            else:
                from .vllm_adapter import VLLMAdapter

                # Model selection
                model = model or vlm_config.get("model") or os.getenv('VISION_MODEL') or "Qwen/Qwen3-VL-8B-Instruct"

                cls._cached_analyzer = VLLMAdapter(
                    model=model,
                    tier_name=tier_name
                )
                return cls._cached_analyzer

        if backend == 'ollama':
            logger.info(f"Using Ollama vision backend (tier: {tier_name})")
            from .ollama_adapter import OllamaVisionAdapter

            # Fix: Tier-specific model takes priority over env override
            model = model or vlm_config.get("model") or os.getenv('VISION_MODEL') or "qwen3-vl:4b"
            cls._cached_analyzer = OllamaVisionAdapter(model=model)

        else:  # transformers (default)
            logger.info(f"Using Transformers vision backend (tier: {tier_name})")
            from .analyzer import VisionAnalyzer

            # Fix: Tier-specific model takes priority over env override
            model = model or vlm_config.get("model") or os.getenv('VISION_MODEL') or "Qwen/Qwen2-VL-2B-Instruct"
            device = vlm_config.get("device") or os.getenv('VISION_DEVICE') or "auto"
            dtype = vlm_config.get("dtype", "float16")

            cls._cached_analyzer = VisionAnalyzer(
                device=device,
                model_id=model,
                dtype=dtype,
                tier_name=tier_name  # v3.1: Pass tier for metadata tracking
            )

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
