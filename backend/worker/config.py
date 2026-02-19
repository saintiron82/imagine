"""
Worker configuration — server URL, auth credentials, processing settings.
"""

import os
from pathlib import Path

# Project root (Imagine/)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_worker_config() -> dict:
    """Load worker config from config.yaml or environment."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        return cfg.get("worker", {})
    except Exception:
        return {}


def get_server_url() -> str:
    """Get server URL. Environment variable takes priority."""
    env = os.getenv("IMAGINE_SERVER_URL")
    if env:
        return env.rstrip("/")
    cfg = get_worker_config()
    return cfg.get("server_url", "http://localhost:8000").rstrip("/")


def get_worker_credentials() -> dict:
    """Get worker login credentials from env or config."""
    email = os.getenv("IMAGINE_WORKER_EMAIL")
    password = os.getenv("IMAGINE_WORKER_PASSWORD")
    if email and password:
        return {"email": email, "password": password}
    cfg = get_worker_config()
    return {
        "email": cfg.get("email", ""),
        "password": cfg.get("password", ""),
    }


def get_claim_batch_size() -> int:
    cfg = get_worker_config()
    return cfg.get("claim_batch_size", 5)


def get_poll_interval() -> int:
    """Seconds to wait when no jobs are available."""
    cfg = get_worker_config()
    return cfg.get("poll_interval_seconds", 30)


def get_storage_mode() -> str:
    """Get storage mode: 'server_upload' or 'shared_fs'."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        return cfg.get("server", {}).get("storage", {}).get("mode", "shared_fs")
    except Exception:
        return "shared_fs"
