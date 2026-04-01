"""
Singleton configuration loader for ImageParser v3.

Layered config: system config.yaml + per-user user-settings.yaml.
User settings take precedence over system defaults.
"""

import os
import sys
import logging
import platform
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# PyInstaller bundle: config.yaml is extracted to _MEIPASS root
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    _PROJECT_ROOT = Path(sys._MEIPASS)
else:
    _PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

# .env key → config.yaml dotted path
_ENV_OVERRIDES = {
    "VISION_MODEL": "vision.model",
    "OLLAMA_HOST": "vision.ollama_host",
}

_instance: Optional["AppConfig"] = None


def _resolve_user_settings_path() -> Optional[Path]:
    """Resolve user-settings.yaml path from env var or platform default."""
    # 1. Explicit env var (set by Electron IPC)
    env_path = os.environ.get("IMAGINE_USER_SETTINGS_PATH")
    if env_path:
        return Path(env_path)

    # 2. Platform-specific app data directory
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "Imagine"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / "Imagine"
    else:  # Linux and others
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        base = Path(xdg) / "Imagine"

    return base / "user-settings.yaml"


class AppConfig:
    """Layered configuration: user-settings.yaml overrides config.yaml."""

    def __init__(self, path: Path = _CONFIG_PATH,
                 user_settings_path: Optional[Path] = None):
        # System config (config.yaml)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._data: dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded system config from {path}")
        else:
            self._data = {}
            logger.warning(f"config.yaml not found at {path}, using defaults")

        # User settings (user-settings.yaml)
        self._user_data: dict = {}
        self._user_settings_path = user_settings_path or _resolve_user_settings_path()
        self._load_user_settings()

        self._apply_env_overrides()

    # ── public API ──────────────────────────────────────

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Retrieve a value by dotted path.
        User settings take precedence over system config.

        Example:
            cfg.get("vision.model")          -> "qwen3.5:9b"
            cfg.get("search.rrf.k", 60)      -> 60
            cfg.get("ai_mode.override")      -> "pro"  (from user-settings)
        """
        # 1. User settings first
        val = self._get_from_dict(self._user_data, dotted_key)
        if val is not None:
            return val
        # 2. System config fallback
        val = self._get_from_dict(self._data, dotted_key)
        if val is not None:
            return val
        return default

    def section(self, key: str) -> dict:
        """Return a top-level section as a dict, with user overrides merged."""
        system_val = self._data.get(key, {})
        user_val = self._user_data.get(key, {})

        if isinstance(system_val, dict) and isinstance(user_val, dict):
            merged = dict(system_val)
            merged.update(user_val)
            return merged
        elif isinstance(user_val, dict) and user_val:
            return dict(user_val)
        elif isinstance(system_val, dict):
            return dict(system_val)
        return {}

    def set_user_settings_path(self, path: Path):
        """Set user settings path at runtime and reload."""
        self._user_settings_path = path
        self._load_user_settings()

    @property
    def user_settings_path(self) -> Optional[Path]:
        return self._user_settings_path

    # ── internals ───────────────────────────────────────

    def _load_user_settings(self):
        """Load user-settings.yaml if it exists."""
        if self._user_settings_path and self._user_settings_path.exists():
            try:
                with open(self._user_settings_path, "r", encoding="utf-8") as f:
                    self._user_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded user settings from {self._user_settings_path}")
            except Exception as e:
                logger.warning(f"Failed to load user settings: {e}")
                self._user_data = {}
        else:
            self._user_data = {}

    @staticmethod
    def _get_from_dict(data: dict, dotted_key: str) -> Any:
        """Traverse nested dict by dotted key. Returns None if not found."""
        parts = dotted_key.split(".")
        node = data
        for p in parts:
            if not isinstance(node, dict):
                return None
            node = node.get(p)
            if node is None:
                return None
        return node

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

    def save_user_setting(self, dotted_key: str, value: Any):
        """Set a value in both memory and user-settings.yaml (persists across restarts)."""
        # Update in-memory merged data
        self._set_dotted(dotted_key, value)
        # Update user_data and save to file
        parts = dotted_key.split(".")
        node = self._user_data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
        self._save_user_settings()

    def _save_user_settings(self):
        """Write user-settings.yaml to disk."""
        if not self._user_settings_path:
            return
        try:
            self._user_settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._user_settings_path, "w", encoding="utf-8") as f:
                yaml.dump(self._user_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning(f"Failed to save user settings: {e}")


def get_config() -> AppConfig:
    """Return the singleton AppConfig instance."""
    global _instance
    if _instance is None:
        _instance = AppConfig()
    return _instance
