"""
Singleton configuration loader for ImageParser v3.

Loads config.yaml with .env overrides for backward compatibility.
"""

import os
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# .env key → config.yaml dotted path
_ENV_OVERRIDES = {
    "VISION_MODEL": "vision.model",
    "OLLAMA_HOST": "vision.ollama_host",
}

_instance: Optional["AppConfig"] = None


class AppConfig:
    """Read-only configuration backed by config.yaml + .env overrides."""

    def __init__(self, path: Path = _CONFIG_PATH):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._data: dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded config from {path}")
        else:
            self._data = {}
            logger.warning(f"config.yaml not found at {path}, using defaults")

        self._apply_env_overrides()

    # ── public API ──────────────────────────────────────

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Retrieve a value by dotted path.

        Example:
            cfg.get("vision.model")          -> "qwen3-vl:8b"
            cfg.get("search.rrf.k", 60)      -> 60
            cfg.get("embedding.visual.model") -> "clip-ViT-L-14"
        """
        parts = dotted_key.split(".")
        node = self._data
        for p in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(p)
            if node is None:
                return default
        return node

    def section(self, key: str) -> dict:
        """Return a top-level section as a dict (shallow copy)."""
        val = self._data.get(key, {})
        return dict(val) if isinstance(val, dict) else {}

    # ── internals ───────────────────────────────────────

    def _apply_env_overrides(self):
        """Apply .env / environment variable overrides."""
        for env_key, dotted_path in _ENV_OVERRIDES.items():
            val = os.environ.get(env_key)
            if val is not None:
                self._set_dotted(dotted_path, val)
                logger.debug(f"env override: {env_key} -> {dotted_path}")

    def _set_dotted(self, dotted_key: str, value: Any):
        parts = dotted_key.split(".")
        node = self._data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value


def get_config() -> AppConfig:
    """Return the singleton AppConfig instance."""
    global _instance
    if _instance is None:
        _instance = AppConfig()
    return _instance
