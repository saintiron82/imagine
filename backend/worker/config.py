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


def get_claim_batch_size() -> int:
    cfg = get_worker_config()
    return cfg.get("claim_batch_size", 5)


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


def get_temp_buffer_config() -> dict:
    """Get temp buffer settings for WebDAV file downloads.

    Returns:
        max_files: Max files to keep in temp folder (bounded buffer).
        download_workers: Number of parallel download threads.
    """
    cfg = get_worker_config()
    buf = cfg.get("temp_buffer", {})
    return {
        "max_files": buf.get("max_files", 10),
        "download_workers": buf.get("download_workers", 3),
    }


def get_storage_mode() -> str:
    """Get storage mode: 'server_upload' or 'shared_fs'.

    Auto-detection: if server URL points to a remote host (not localhost),
    force 'server_upload' mode since remote workers cannot access local files.
    """
    # Check explicit config first
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        explicit = cfg.get("server", {}).get("storage", {}).get("mode")
        if explicit:
            return explicit
    except Exception:
        pass

    # Auto-detect: remote server → server_upload
    server_url = get_server_url()
    from urllib.parse import urlparse
    parsed = urlparse(server_url)
    hostname = parsed.hostname or ""
    if hostname not in ("localhost", "127.0.0.1", "::1", ""):
        return "server_upload"

    return "shared_fs"
