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
    username = os.getenv("IMAGINE_WORKER_USERNAME")
    email = os.getenv("IMAGINE_WORKER_EMAIL")
    password = os.getenv("IMAGINE_WORKER_PASSWORD")
    if (username or email) and password:
        creds = {"password": password}
        if username:
            creds["username"] = username
        if email:
            creds["email"] = email
        return creds
    cfg = get_worker_config()
    return {
        "username": cfg.get("username", ""),
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


def get_batch_capacity() -> int:
    """Worker's actual batch processing capacity (file count).
    Used to calculate prefetch pool size (capacity * 2)."""
    cfg = get_worker_config()
    return cfg.get("batch_capacity", 5)


def get_heartbeat_interval() -> int:
    """Heartbeat interval in seconds."""
    cfg = get_worker_config()
    return cfg.get("heartbeat_interval", 30)


def get_rest_after_batch_s() -> int:
    """Seconds to rest after each batch completes. 0 = no rest."""
    cfg = get_worker_config()
    return cfg.get("rest_after_batch_s", 0)


def get_storage_mode() -> str:
    """Get storage mode: 'server_upload' or 'shared_fs'."""
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        return cfg.get("server", {}).get("storage", {}).get("mode", "shared_fs")
    except Exception:
        return "shared_fs"
