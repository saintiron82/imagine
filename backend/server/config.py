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


def get_server_config() -> dict:
    """Load server config from config.yaml."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        return cfg.get("server", {})
    except Exception:
        return {}


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
    """Get JWT secret. Auto-generates if missing or default."""
    # 1. Environment variable (highest priority)
    env = os.getenv("IMAGINE_JWT_SECRET")
    if env:
        return env

    # 2. config.yaml
    cfg = get_server_config()
    secret = cfg.get("auth", {}).get("jwt_secret")

    # 3. Auto-generate if missing or insecure default
    if not secret or secret == _DEFAULT_SECRET:
        secret = secrets.token_urlsafe(32)
        logger.warning(
            "JWT secret was missing or using insecure default. "
            "Auto-generated a secure secret."
        )
        _save_jwt_secret(secret)

    return secret


def get_access_token_minutes() -> int:
    cfg = get_server_config()
    return cfg.get("auth", {}).get("access_token_expire_minutes", 15)


def get_refresh_token_days() -> int:
    cfg = get_server_config()
    return cfg.get("auth", {}).get("refresh_token_expire_days", 7)


def get_cors_origins() -> List[str]:
    cfg = get_server_config()
    # Allow all origins when cors_allow_all is true (server mode, JWT-protected)
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
