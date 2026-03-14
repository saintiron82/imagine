"""
LRU download cache for remote files (WebDAV, server).

Preserves downloaded originals in a local cache directory so re-processing
the same file does not require re-downloading.  When the cache exceeds the
configured capacity, the least-recently-used files are evicted.

Usage:
    cache = DownloadCache.from_config()   # reads config.yaml
    hit = cache.get(remote_url)           # returns Path or None
    cache.put(remote_url, local_path)     # move temp file into cache
"""

import hashlib
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_instance: Optional["DownloadCache"] = None
_lock = threading.Lock()


class DownloadCache:
    """Capacity-limited LRU cache for downloaded files."""

    def __init__(self, cache_dir: str, limit_gb: float = 50.0, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.limit_bytes = int(limit_gb * 1024 ** 3) if limit_gb > 0 else 0
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, remote_url: str) -> Optional[Path]:
        """Look up *remote_url* in cache.  Returns local Path or None.

        On hit the file's mtime is touched (LRU refresh).
        """
        if not self.enabled:
            return None
        key = self._url_key(remote_url)
        candidates = list(self.cache_dir.glob(f"{key}_*"))
        if not candidates:
            return None
        cached = candidates[0]
        if not cached.is_file():
            return None
        # LRU touch
        try:
            os.utime(cached, None)
        except OSError:
            pass
        log.info("[Cache HIT] %s → %s", remote_url[:80], cached.name)
        return cached

    def put(self, remote_url: str, local_path, *, move: bool = True) -> Optional[Path]:
        """Store *local_path* in the cache under *remote_url*.

        If *move* is True (default) the file is moved; otherwise copied.
        Returns the cache path, or None if caching is disabled / fails.
        """
        if not self.enabled:
            return None
        src = Path(local_path)
        if not src.is_file():
            return None
        key = self._url_key(remote_url)
        dest = self.cache_dir / f"{key}_{src.name}"
        try:
            if move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))
            log.info("[Cache PUT] %s (%s)", dest.name, _human_size(dest.stat().st_size))
            self._evict_if_needed()
            return dest
        except OSError as e:
            log.warning("[Cache PUT FAIL] %s: %s", remote_url[:80], e)
            return None

    def evict_if_needed(self):
        """Public wrapper for eviction."""
        self._evict_if_needed()

    def stats(self) -> dict:
        """Return cache statistics."""
        if not self.enabled or not self.cache_dir.is_dir():
            return {"enabled": False, "total_mb": 0, "file_count": 0, "limit_gb": 0}
        total = 0
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                total += f.stat().st_size
                count += 1
        return {
            "enabled": True,
            "total_mb": round(total / (1024 ** 2), 1),
            "file_count": count,
            "limit_gb": round(self.limit_bytes / (1024 ** 3), 1) if self.limit_bytes else 0,
        }

    def cleanup(self) -> dict:
        """Delete ALL cached files.  Returns stats before deletion."""
        before = self.stats()
        if not self.enabled or not self.cache_dir.is_dir():
            return before
        deleted = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                    deleted += 1
                except OSError:
                    pass
        log.info("[Cache CLEANUP] deleted %d files (was %.1f MB)", deleted, before["total_mb"])
        return before

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _url_key(remote_url: str) -> str:
        """Derive a short, filesystem-safe key from *remote_url*."""
        return hashlib.sha256(remote_url.encode("utf-8")).hexdigest()[:16]

    def _evict_if_needed(self):
        """Remove oldest files until total size <= limit."""
        if not self.limit_bytes or not self.cache_dir.is_dir():
            return
        files = []
        total = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                st = f.stat()
                files.append((f, st.st_mtime, st.st_size))
                total += st.st_size
        if total <= self.limit_bytes:
            return
        # Sort by mtime ascending (oldest first)
        files.sort(key=lambda x: x[1])
        evicted = 0
        for path, _, size in files:
            if total <= self.limit_bytes:
                break
            try:
                path.unlink()
                total -= size
                evicted += 1
            except OSError:
                pass
        if evicted:
            log.info("[Cache EVICT] removed %d files, now %.1f MB", evicted, total / (1024 ** 2))

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "DownloadCache":
        """Return a singleton instance configured from config.yaml."""
        global _instance
        if _instance is not None:
            return _instance
        with _lock:
            if _instance is not None:
                return _instance
            cfg = _load_cache_config()
            _instance = cls(
                cache_dir=cfg["path"],
                limit_gb=cfg["limit_gb"],
                enabled=cfg["enabled"],
            )
            return _instance


def get_download_cache() -> DownloadCache:
    """Module-level convenience accessor."""
    return DownloadCache.from_config()


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------

def _load_cache_config() -> dict:
    """Read download_cache section from config.yaml."""
    import yaml

    defaults = {
        "enabled": True,
        "path": "output/download_cache",
        "limit_gb": 50,
    }
    try:
        cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
        if not cfg_path.is_file():
            return defaults
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        section = data.get("download_cache", {})
        return {
            "enabled": section.get("enabled", defaults["enabled"]),
            "path": section.get("path", defaults["path"]),
            "limit_gb": float(section.get("limit_gb", defaults["limit_gb"])),
        }
    except Exception as e:
        log.warning("Failed to load download_cache config: %s", e)
        return defaults


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
