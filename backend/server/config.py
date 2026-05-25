"""
Server configuration — reads from config.yaml server section.
"""

import logging
import os
import secrets
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)

# Project root (Imagine/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

_DEFAULT_SECRET = "change-this-secret-in-production"
_cached_jwt_secret = None  # In-memory cache — survives across calls
_jwt_lock = __import__('threading').Lock()


def get_server_config() -> dict:
    """Load server config from config.yaml."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        server_cfg = dict(cfg.get("server", {}) or {})
    except Exception:
        server_cfg = {}

    host = os.getenv("IMAGINE_SERVER_HOST")
    if host:
        server_cfg["host"] = host

    port = os.getenv("IMAGINE_SERVER_PORT")
    if port:
        try:
            server_cfg["port"] = int(port)
        except ValueError:
            logger.warning("Invalid IMAGINE_SERVER_PORT=%r; using config/default port", port)

    cors_allow_all = os.getenv("IMAGINE_CORS_ALLOW_ALL")
    if cors_allow_all is not None:
        server_cfg["cors_allow_all"] = cors_allow_all.strip().lower() in {
            "1", "true", "yes", "on"
        }

    return server_cfg


def _save_jwt_secret(secret: str) -> None:
    """Persist auto-generated JWT secret to config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        data.setdefault("server", {}).setdefault("auth", {})["jwt_secret"] = secret

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False,
                           allow_unicode=True, sort_keys=False)
        logger.info("JWT secret auto-generated and saved to config.yaml")
    except Exception as e:
        logger.warning(f"Could not save JWT secret to config.yaml: {e}")


def get_jwt_secret() -> str:
    """Get JWT secret. Thread-safe, cached in memory."""
    global _cached_jwt_secret

    # Fast path: already cached
    if _cached_jwt_secret:
        return _cached_jwt_secret

    with _jwt_lock:
        # Double-check after acquiring lock
        if _cached_jwt_secret:
            return _cached_jwt_secret

        # 1. Environment variable (highest priority)
        env = os.getenv("IMAGINE_JWT_SECRET")
        if env:
            _cached_jwt_secret = env
            return env

        # 2. config.yaml (dotted key to avoid user_data partial override)
        secret = None
        try:
            from backend.utils.config import get_config
            secret = get_config().get("server.auth.jwt_secret")
        except Exception:
            pass

        # 3. Auto-generate if missing or insecure default
        if not secret or secret == _DEFAULT_SECRET:
            secret = secrets.token_urlsafe(32)
            logger.warning("JWT secret auto-generated (first run)")
            _save_jwt_secret(secret)
            try:
                from backend.utils.config import get_config
                get_config()._set_dotted("server.auth.jwt_secret", secret)
            except Exception:
                pass

        _cached_jwt_secret = secret
        return secret


def get_access_token_minutes() -> int:
    cfg = get_server_config()
    return cfg.get("auth", {}).get("access_token_expire_minutes", 15)


def get_refresh_token_days() -> int:
    cfg = get_server_config()
    return cfg.get("auth", {}).get("refresh_token_expire_days", 7)


def get_cors_origins() -> List[str]:
    cfg = get_server_config()
    # Wildcard CORS is opt-in. Binding to 0.0.0.0 alone is not enough.
    if cfg.get("cors_allow_all", False):
        return ["*"]
    origins = cfg.get("cors_origins", ["http://localhost:9274", "http://localhost:8000"])
    # Electron built apps use file:// protocol which sends Origin: null
    if "null" not in origins:
        origins.append("null")
    return origins


def get_storage_config() -> dict:
    cfg = get_server_config()
    return cfg.get("storage", {
        "mode": "server_upload",
        "upload_dir": str(PROJECT_ROOT / "uploads"),
        "thumbnail_dir": str(PROJECT_ROOT / "thumbnails"),
        "max_file_size_mb": 500,
    })


def get_queue_config() -> dict:
    cfg = get_server_config()
    return cfg.get("queue", {
        "assignment_timeout_minutes": 30,
        "max_retries": 3,
        "claim_batch_size": 10,
    })


def get_db_path() -> str:
    """Get SQLite database path."""
    return str(PROJECT_ROOT / "imageparser.db")


def get_relay_config() -> dict:
    """Phase 5: relay connector settings (env vars override config.yaml)."""
    try:
        from backend.utils.config import get_config

        cfg = get_config()
        relay_cfg = dict(cfg.get("relay", {}) or {})
    except Exception:
        relay_cfg = {}

    env_endpoint = os.getenv("IMAGINE_RELAY_ENDPOINT")
    if env_endpoint:
        relay_cfg["endpoint"] = env_endpoint
    env_secret = os.getenv("IMAGINE_RELAY_SERVER_SECRET")
    if env_secret:
        relay_cfg["server_secret"] = env_secret
    env_enabled = os.getenv("IMAGINE_RELAY_ENABLED")
    if env_enabled is not None:
        relay_cfg["enabled"] = env_enabled.strip().lower() in {"1", "true", "yes", "on"}

    relay_cfg.setdefault("enabled", False)
    relay_cfg.setdefault("auto_connect", True)
    relay_cfg.setdefault("heartbeat_interval_seconds", 60)
    relay_cfg.setdefault("idle_timeout_seconds", 600)
    return relay_cfg
