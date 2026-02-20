"""
Server configuration — reads from config.yaml server section.
"""

import os
from pathlib import Path
from typing import List

# Project root (Imagine/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_server_config() -> dict:
    """Load server config from config.yaml."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        return cfg.get("server", {})
    except Exception:
        return {}


def get_jwt_secret() -> str:
    """Get JWT secret, preferring environment variable over config."""
    env = os.getenv("IMAGINE_JWT_SECRET")
    if env:
        return env
    cfg = get_server_config()
    return cfg.get("auth", {}).get("jwt_secret", "change-this-secret-in-production")


def get_access_token_minutes() -> int:
    cfg = get_server_config()
    return cfg.get("auth", {}).get("access_token_expire_minutes", 15)


def get_refresh_token_days() -> int:
    cfg = get_server_config()
    return cfg.get("auth", {}).get("refresh_token_expire_days", 7)


def get_cors_origins() -> List[str]:
    cfg = get_server_config()
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
