"""
WebDAV Sync Engine — mirrors remote WebDAV folder to local cache.
Tracks sync state via JSON manifest file.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from backend.remote.webdav_client import RemoteFileInfo, WebDAVClient

logger = logging.getLogger("WebDAVSync")


@dataclass
class SyncManifest:
    """Persistent sync state for a WebDAV source."""
    source_id: str
    last_sync_iso: str = ""
    files: Dict[str, dict] = field(default_factory=dict)
    # files: { "relative/path.psd": { "size": N, "last_modified": "...", "etag": "..." } }


@dataclass
class SyncResult:
    """Result of a sync operation."""
    source_id: str
    downloaded: int = 0
    unchanged: int = 0
    deleted_remote: int = 0
    failed: int = 0
    total_remote: int = 0
    errors: List[str] = field(default_factory=list)
    cache_path: str = ""


class WebDAVSyncEngine:
    """
    Orchestrates sync between WebDAV remote and local cache.

    Cache structure:
      ~/.imagine-cache/webdav/<source_id>/
        _manifest.json
        <mirrored directory tree>
    """

    MANIFEST_FILENAME = "_manifest.json"

    def __init__(self, cache_base: Optional[Path] = None):
        self.cache_base = cache_base or self._default_cache_base()

    @staticmethod
    def _default_cache_base() -> Path:
        return Path.home() / ".imagine-cache" / "webdav"

    def get_cache_path(self, source_id: str) -> Path:
        return self.cache_base / source_id

    def sync(self, source_config: dict,
             progress_callback: Optional[Callable] = None) -> SyncResult:
        """
        Full sync: list remote -> compare manifest -> download changes -> cleanup.

        Args:
            source_config: {id, url, username, password, remote_path, verify_ssl}
            progress_callback: fn(event_type: str, data: dict)
                Events: "listing", "comparing", "downloading", "cleaning", "complete"
        """
        source_id = source_config['id']
        cache_path = self.get_cache_path(source_id)
        cache_path.mkdir(parents=True, exist_ok=True)

        result = SyncResult(source_id=source_id, cache_path=str(cache_path))

        # 1. Connect and list remote files
        if progress_callback:
            progress_callback("listing", {"message": "Listing remote files..."})

        client = WebDAVClient(
            base_url=source_config['url'],
            username=source_config['username'],
            password=source_config['password'],
            remote_path=source_config.get('remote_path', '/'),
            verify_ssl=source_config.get('verify_ssl', True),
        )

        try:
            remote_files = client.list_files_recursive()
        except Exception as e:
            msg = f"Failed to list remote files: {e}"
            logger.error(msg)
            result.errors.append(msg)
            client.close()
            return result

        result.total_remote = len(remote_files)
        logger.info(f"Remote: {len(remote_files)} files")

        # 2. Load manifest and compare
        if progress_callback:
            progress_callback("comparing", {"message": "Comparing with local cache..."})

        manifest = self._load_manifest(source_id)
        to_download, unchanged, to_delete = self._compare(remote_files, manifest)

        result.unchanged = len(unchanged)
        logger.info(
            f"Sync plan: {len(to_download)} download, "
            f"{len(unchanged)} unchanged, {len(to_delete)} delete"
        )

        # 3. Download changed/new files
        total_dl = len(to_download)
        for idx, file_info in enumerate(to_download):
            if progress_callback:
                progress_callback("downloading", {
                    "file": file_info.relative_path,
                    "index": idx + 1,
                    "total": total_dl,
                })

            local_path = cache_path / file_info.relative_path
            success = client.download_file(
                file_info.remote_path,
                local_path,
                expected_size=file_info.size,
            )

            if success:
                result.downloaded += 1
                # Update manifest entry
                manifest.files[file_info.relative_path] = {
                    "size": file_info.size,
                    "last_modified": file_info.last_modified,
                    "etag": file_info.etag,
                }
            else:
                result.failed += 1
                result.errors.append(f"Failed: {file_info.relative_path}")

        # 4. Clean up files deleted on remote
        if to_delete:
            if progress_callback:
                progress_callback("cleaning", {
                    "message": f"Removing {len(to_delete)} deleted files..."
                })

            for rel_path in to_delete:
                local_path = cache_path / rel_path
                if local_path.exists():
                    local_path.unlink()
                    # Remove empty parent dirs
                    self._cleanup_empty_parents(local_path.parent, cache_path)
                del manifest.files[rel_path]
                result.deleted_remote += 1

        # 5. Save manifest
        from datetime import datetime
        manifest.last_sync_iso = datetime.now().isoformat()
        self._save_manifest(manifest, source_id)

        client.close()

        if progress_callback:
            progress_callback("complete", asdict(result))

        logger.info(
            f"Sync complete: {result.downloaded} downloaded, "
            f"{result.unchanged} unchanged, "
            f"{result.deleted_remote} deleted, "
            f"{result.failed} failed"
        )

        return result

    def cleanup_source(self, source_id: str):
        """Remove cache directory and manifest for a source."""
        cache_path = self.get_cache_path(source_id)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info(f"Cleaned up cache: {cache_path}")

    # ── Internal helpers ──────────────────────────────────────

    def _load_manifest(self, source_id: str) -> SyncManifest:
        manifest_path = self.get_cache_path(source_id) / self.MANIFEST_FILENAME
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding='utf-8'))
                return SyncManifest(
                    source_id=data.get('source_id', source_id),
                    last_sync_iso=data.get('last_sync_iso', ''),
                    files=data.get('files', {}),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupt manifest, starting fresh: {e}")

        return SyncManifest(source_id=source_id)

    def _save_manifest(self, manifest: SyncManifest, source_id: str):
        manifest_path = self.get_cache_path(source_id) / self.MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _compare(
        self,
        remote_files: List[RemoteFileInfo],
        manifest: SyncManifest,
    ) -> Tuple[List[RemoteFileInfo], List[RemoteFileInfo], List[str]]:
        """
        Compare remote file list with manifest.

        Returns:
            (to_download, unchanged, to_delete_rel_paths)
        """
        to_download = []
        unchanged = []

        remote_paths = set()
        for rf in remote_files:
            remote_paths.add(rf.relative_path)
            cached = manifest.files.get(rf.relative_path)

            if cached is None:
                # New file
                to_download.append(rf)
                continue

            # Compare: ETag first, then last_modified + size
            if rf.etag and cached.get('etag'):
                if rf.etag != cached['etag']:
                    to_download.append(rf)
                else:
                    unchanged.append(rf)
            elif rf.last_modified != cached.get('last_modified') or rf.size != cached.get('size'):
                to_download.append(rf)
            else:
                unchanged.append(rf)

        # Files in manifest but not on remote = deleted remotely
        to_delete = [
            rel for rel in manifest.files
            if rel not in remote_paths
        ]

        return to_download, unchanged, to_delete

    @staticmethod
    def _cleanup_empty_parents(dir_path: Path, stop_at: Path):
        """Remove empty parent directories up to stop_at."""
        current = dir_path
        while current != stop_at and current.is_dir():
            try:
                # Only remove if empty (no files, only _manifest.json doesn't count)
                children = list(current.iterdir())
                if not children:
                    current.rmdir()
                    current = current.parent
                else:
                    break
            except OSError:
                break
