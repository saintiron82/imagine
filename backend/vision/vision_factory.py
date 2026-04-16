"""
Vision Analyzer Factory - Fallback-chain based adapter selection.

Automatically chooses between:
- MLX (Apple Silicon): Native acceleration, best performance on Mac
- vLLM (Linux/Mac): High-throughput batch processing
- Ollama (all platforms): Memory-efficient, automatic model management
- Transformers (all platforms): Universal fallback, always available

Backend selection uses an explicit fallback chain:
  1. Read primary backend from config.yaml (tier + platform)
  2. Read `fallback` field for next option
  3. Always append `transformers` as final safety net

v3.1: Supports 3-Tier architecture (Standard/Pro/Ultra).
v6.4: MLX backend for native Apple Silicon acceleration.
v8.0: Explicit fallback chain — every platform+tier combination
      degrades gracefully to transformers.
"""

import os
import logging
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image

# v3.1: Tier-aware configuration
from backend.utils.tier_config import get_active_tier

# v3.1.1: Platform-specific optimization
# v6.4: MLX backend detection
# v8.0: All availability checks for pre-flight validation
from backend.utils.platform_detector import (
    get_optimal_backend,
    get_optimal_batch_size,
    get_platform_info,
    is_mlx_vlm_available,
    is_vllm_available,
    is_ollama_available,
)

# ABC base class for all vision adapters (extracted to avoid circular imports)
from backend.vision.base import BaseVisionAnalyzer

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
    Factory for creating vision analyzers with explicit fallback chains.

    Uses config.yaml `fallback` field to build an ordered list of backends
    to try. Always ends with `transformers` as the universal safety net.

    Environment Variables (override only, tier config takes priority):
        VISION_BACKEND: Force specific backend
        VISION_MODEL: Force specific model
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
    """

    _cached_analyzer = None

    # ── Platform-aware model resolution (v3 P14/P15/P17) ─────

    @staticmethod
    def resolve_platform_model(vlm_config: dict, plat_key: str, target_backend: str):
        """Resolve VLM model for (platform, backend) pair.

        Priority:
        1. backends.{plat}.{target_backend}.model        (v3 P14 schema)
        2. mlx_model when target_backend == 'mlx'        (legacy darwin shortcut)
        3. backends.{plat}.model (when backends.{plat}.backend == target)  (legacy)
        4. vlm_config.model                              (top-level fallback)
        """
        plat_cfg = (vlm_config.get("backends", {}) or {}).get(plat_key, {}) or {}
        per_backend = plat_cfg.get(target_backend)
        if isinstance(per_backend, dict) and per_backend.get("model"):
            return per_backend["model"]
        if target_backend == "mlx" and vlm_config.get("mlx_model"):
            return vlm_config["mlx_model"]
        if plat_cfg.get("backend") == target_backend and plat_cfg.get("model"):
            return plat_cfg["model"]
        return vlm_config.get("model")

    @staticmethod
    def resolve_batch_size(vlm_config: dict, plat_key: str, target_backend: str):
        plat_cfg = (vlm_config.get("backends", {}) or {}).get(plat_key, {}) or {}
        per_backend = plat_cfg.get(target_backend)
        if isinstance(per_backend, dict) and per_backend.get("batch_size"):
            return per_backend["batch_size"]
        return plat_cfg.get("batch_size") or vlm_config.get("batch_size")

    # ── Fallback Chain Resolution ────────────────────────────

    @classmethod
    def _resolve_backend_chain(
        cls, vlm_config: dict, tier_name: str
    ) -> List[Dict[str, Any]]:
        """
        Build ordered fallback chain from config.yaml.

        Reads `backend` + `fallback` from platform-specific config.
        Always appends `transformers` as the final safety net.

        Returns:
            List of dicts: [{"backend": str, "model": str|None, "batch_size": int|None}, ...]
        """
        # Backend resolution priority:
        # 1. VISION_BACKEND env var (explicit override)
        # 2. user-settings.yaml ai_mode.vlm_backend (per-system)
        # 3. config.yaml tier vlm.backend
        # 4. auto-detect via platform detector
        from backend.utils.config import get_config
        app_cfg = get_config()
        backend = (
            os.getenv('VISION_BACKEND')
            or app_cfg.get("ai_mode.vlm_backend")
            or vlm_config.get("backend")
            or "auto"
        ).lower()

        chain = []

        # Resolve platform-specific config.
        # v3 P14/P15: prefer backends.{plat}.{backend}.model (explicit per-backend),
        # fall back to legacy backends.{plat}.model (whole-platform default).
        import platform as _plat
        plat_key = {"Darwin": "darwin", "Windows": "windows", "Linux": "linux"}.get(_plat.system(), "linux")

        if backend == 'auto':
            # Auto-detect: use platform detector to find optimal backend
            detected = get_optimal_backend(tier_name)
            model = VisionAnalyzerFactory.resolve_platform_model(vlm_config, plat_key, detected)
            logger.info(f"Auto-detected backend: {detected} (tier: {tier_name}, model: {model})")
            chain.append({
                "backend": detected,
                "model": model,
                "batch_size": VisionAnalyzerFactory.resolve_batch_size(vlm_config, plat_key, detected),
            })
        else:
            # Explicit backend — use per-backend model if defined
            model = VisionAnalyzerFactory.resolve_platform_model(vlm_config, plat_key, backend)
            logger.info(f"Explicit backend: {backend} (tier: {tier_name}, model: {model})")
            chain.append({
                "backend": backend,
                "model": model,
                "batch_size": VisionAnalyzerFactory.resolve_batch_size(vlm_config, plat_key, backend),
            })

        # Always ensure transformers is the final fallback
        seen = {entry["backend"] for entry in chain}
        if "transformers" not in seen:
            chain.append({
                "backend": "transformers",
                "model": None,
                "batch_size": None,
            })

        return chain

    # ── Pre-flight Availability Check ────────────────────────

    @staticmethod
    def _check_backend_available(backend_name: str) -> bool:
        """
        Quick pre-flight check before attempting instantiation.

        Returns True if the backend is likely available on this system.
        """
        if backend_name == 'mlx':
            return is_mlx_vlm_available()
        elif backend_name == 'vllm':
            return is_vllm_available()
        elif backend_name == 'ollama':
            return is_ollama_available()
        elif backend_name == 'transformers':
            return True  # Always available (torch is a required dependency)
        else:
            return False

    # ── Backend Instantiation ────────────────────────────────

    @classmethod
    def _instantiate_backend(
        cls,
        backend_name: str,
        entry: Dict[str, Any],
        vlm_config: dict,
        tier_name: str,
    ) -> "BaseVisionAnalyzer":
        """
        Create a single backend instance. Raises on failure.

        Args:
            backend_name: 'mlx', 'vllm', 'ollama', or 'transformers'
            entry: Chain entry with model/batch_size overrides
            vlm_config: Full VLM config from tier
            tier_name: Active tier name
        """
        model = entry.get("model") or vlm_config.get("model")

        if backend_name == 'mlx':
            if platform.system() != 'Darwin':
                raise RuntimeError("MLX is only supported on macOS")
            from .mlx_adapter import MLXVisionAdapter

            # Prefer mlx_model (quantized) over generic model (HF transformers ID)
            mlx_model = vlm_config.get("mlx_model")
            if mlx_model:
                model = mlx_model
            model = model or "mlx-community/Qwen3.5-9B-MLX-4bit"
            return MLXVisionAdapter(model=model, tier_name=tier_name)

        elif backend_name == 'vllm':
            if platform.system() == 'Windows':
                raise RuntimeError("vLLM is not supported on Windows")
            from .vllm_adapter import VLLMAdapter

            model = (
                model
                or os.getenv('VISION_MODEL')
                or "Qwen/Qwen3.5-9B"
            )
            return VLLMAdapter(model=model, tier_name=tier_name)

        elif backend_name == 'ollama':
            from .ollama_adapter import OllamaVisionAdapter

            model = (
                model
                or os.getenv('VISION_MODEL')
                or "qwen3.5:4b"
            )
            return OllamaVisionAdapter(model=model)

        elif backend_name == 'transformers':
            from .analyzer import VisionAnalyzer

            model = (
                model
                or os.getenv('VISION_MODEL')
                or "Qwen/Qwen2-VL-2B-Instruct"
            )
            device = vlm_config.get("device") or os.getenv('VISION_DEVICE') or "auto"
            dtype = vlm_config.get("dtype", "float16")
            return VisionAnalyzer(
                device=device,
                model_id=model,
                dtype=dtype,
                tier_name=tier_name,
            )

        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    # ── Main Factory Method ──────────────────────────────────

    @classmethod
    def create(cls) -> "BaseVisionAnalyzer":
        """
        Create vision analyzer using fallback chain.

        Builds an ordered list of backends to try, checks availability,
        and instantiates the first one that succeeds.

        v8.0: Explicit fallback chain — always ends at transformers.
        """
        if cls._cached_analyzer is not None:
            return cls._cached_analyzer

        tier_name, tier_config = get_active_tier()
        vlm_config = tier_config.get("vlm", {})

        # Build fallback chain
        chain = cls._resolve_backend_chain(vlm_config, tier_name)
        chain_names = [e["backend"] for e in chain]
        logger.info(
            f"VLM backend chain (tier: {tier_name}): "
            f"{' → '.join(chain_names)}"
        )

        # Try each backend in order
        last_error = None
        for entry in chain:
            backend_name = entry["backend"]

            # Pre-flight availability check
            if not cls._check_backend_available(backend_name):
                logger.info(
                    f"[SKIP] {backend_name}: not available, trying next"
                )
                continue

            # Attempt instantiation
            try:
                analyzer = cls._instantiate_backend(
                    backend_name, entry, vlm_config, tier_name
                )
                logger.info(
                    f"[OK] VLM backend: {backend_name} (tier: {tier_name})"
                )
                cls._cached_analyzer = analyzer
                return analyzer
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[FAIL] {backend_name}: {e}, trying next"
                )
                continue

        # All backends exhausted
        raise RuntimeError(
            f"All VLM backends failed for tier '{tier_name}'. "
            f"Chain: {chain_names}. Last error: {last_error}"
        )

    @classmethod
    def reset(cls):
        """Reset cached analyzer — unload model first to free GPU memory."""
        if cls._cached_analyzer is not None:
            try:
                if hasattr(cls._cached_analyzer, 'unload_model'):
                    cls._cached_analyzer.unload_model()
            except Exception as e:
                logger.warning(f"Error unloading cached analyzer: {e}")
        cls._cached_analyzer = None


def get_vision_analyzer() -> BaseVisionAnalyzer:
    """
    Convenience function to get vision analyzer.

    Usage:
        analyzer = get_vision_analyzer()
        result = analyzer.analyze(image)
    """
    return VisionAnalyzerFactory.create()
