"""
Download-ahead pool — pre-downloads WebDAV files to a temp folder.

Monitors job_queue for pending WebDAV jobs and downloads original files
to a bounded temp folder. ParseAhead then uses these local copies for
Phase P (parsing + thumbnail generation).

Producer-Consumer pattern:
  - Producer: This pool downloads WebDAV originals (parallel threads)
  - Buffer:   Temp folder with max_files limit (semaphore-controlled)
  - Consumer: ParseAhead reads from temp folder, pipeline processes,
              complete_job() deletes temp file and releases slot
"""

import json
import logging
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.base_ahead_pool import BaseAheadPool

logger = logging.getLogger(__name__)

# In-memory cache of WebDAV source configs.
# Populated by the server API when Electron registers sources.
# Key: source_id (str), Value: dict with url, username, password, etc.
_webdav_sources: Dict[str, dict] = {}
_sources_lock = threading.Lock()


def register_webdav_source(source_config: dict):
    """Register a WebDAV source config for download-ahead access.

    Called by server API when Electron adds/updates a source.
    """
    source_id = source_config.get("id")
    if not source_id:
        logger.warning("register_webdav_source: missing source id")
        return
    with _sources_lock:
        _webdav_sources[source_id] = source_config
        logger.info(f"WebDAV source registered: {source_id}")


def unregister_webdav_source(source_id: str):
    """Remove a WebDAV source config."""
    with _sources_lock:
        _webdav_sources.pop(source_id, None)


def get_webdav_source(source_id: str) -> Optional[dict]:
    """Get a registered WebDAV source config."""
    with _sources_lock:
        return _webdav_sources.get(source_id)


def parse_webdav_path(file_path: str) -> tuple:
    """Parse 'webdav://source-id/remote/path' into (source_id, remote_path).

    Returns:
        (source_id, remote_path) or (None, None) if not a webdav path.
    """
    if not file_path.startswith("webdav://"):
        return None, None
    rest = file_path[len("webdav://"):]
    # source_id is the first path segment
    parts = rest.split("/", 1)
    source_id = parts[0]
    remote_path = "/" + parts[1] if len(parts) > 1 else "/"
    return source_id, remote_path


class DownloadAheadPool(BaseAheadPool):
    """Background daemon that pre-downloads WebDAV files to a temp folder.

    Bounded buffer: at most max_files originals on disk at any time.
    Parallel downloads: download_workers concurrent threads.

    Downloaded files are stored as:
        {temp_dir}/{file_id}_{filename}

    The temp_local_path is recorded in job_queue.parsed_metadata so
    ParseAhead can find the local copy for Phase P.
    """

    def __init__(self, db: SQLiteDB):
        super().__init__(db)
        from backend.worker.config import get_temp_buffer_config
        buf_cfg = get_temp_buffer_config()
        self._max_files = buf_cfg["max_files"]
        self._download_workers = buf_cfg["download_workers"]

        # Bounded buffer semaphore — limits files on disk
        self._buffer_sem = threading.Semaphore(self._max_files)
        # Track active temp files: file_id -> temp_local_path
        self._active_files: Dict[int, str] = {}
        self._active_lock = threading.Lock()

        # Temp directory — created on start, cleaned on stop
        self._temp_dir: Optional[Path] = None

        # Download thread pool
        self._executor: Optional[ThreadPoolExecutor] = None
        # Track in-flight downloads: file_id -> Future
        self._in_flight: Dict[int, Future] = {}

    @property
    def temp_dir(self) -> Optional[Path]:
        return self._temp_dir

    def start(self):
        """Start the download daemon and create temp directory."""
        self._temp_dir = Path(tempfile.mkdtemp(prefix="imagine_dl_"))
        logger.info(
            f"DownloadAheadPool: temp_dir={self._temp_dir}, "
            f"max_files={self._max_files}, workers={self._download_workers}"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._download_workers,
            thread_name_prefix="dl-ahead",
        )
        super().start()

    def stop(self):
        """Stop downloads and clean up temp directory."""
        super().stop()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.info(f"DownloadAheadPool: cleaned temp_dir {self._temp_dir}")
            except Exception as e:
                logger.warning(f"DownloadAheadPool: temp cleanup failed: {e}")
        self._temp_dir = None

    def release_slot(self, file_id: int):
        """Release a buffer slot after job completion.

        Called by JobQueueManager.complete_job() to delete the temp file
        and free a semaphore slot for new downloads.
        """
        with self._active_lock:
            temp_path = self._active_files.pop(file_id, None)
        if temp_path:
            try:
                p = Path(temp_path)
                if p.exists():
                    p.unlink()
                    logger.debug(f"DownloadAhead: deleted temp file {p.name}")
            except Exception as e:
                logger.warning(f"DownloadAhead: failed to delete {temp_path}: {e}")
        self._buffer_sem.release()

    def get_temp_path(self, file_id: int) -> Optional[str]:
        """Get the temp local path for a downloaded file."""
        with self._active_lock:
            return self._active_files.get(file_id)

    def _loop(self):
        """Main loop: find pending WebDAV jobs, download originals."""
        poll_interval = self._get_config_value(
            "server.parse_ahead.poll_interval_s", 2
        )
        while self._running:
            try:
                downloaded = self._download_batch()
                if downloaded == 0:
                    time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"DownloadAhead loop error: {e}")
                time.sleep(5)

    def _download_batch(self) -> int:
        """Find pending WebDAV jobs without temp files and start downloads.

        Returns number of downloads started.
        """
        cursor = self.db.conn.cursor()

        # Find WebDAV jobs needing download (pending or assigned, no temp file yet)
        cursor.execute(
            """SELECT jq.id, jq.file_id, jq.file_path, jq.parsed_metadata
               FROM job_queue jq
               WHERE jq.status IN ('pending', 'assigned')
                 AND jq.file_path LIKE 'webdav://%'
               ORDER BY jq.priority DESC, jq.created_at ASC
               LIMIT ?""",
            (self._max_files,),
        )
        rows = cursor.fetchall()
        if not rows:
            return 0

        started = 0
        for job_id, file_id, file_path, parsed_metadata_str in rows:
            if not self._running:
                break

            # Skip if already downloaded or in-flight
            with self._active_lock:
                if file_id in self._active_files:
                    continue
            if file_id in self._in_flight:
                continue

            # Check if parsed_metadata already has temp_local_path
            if parsed_metadata_str:
                try:
                    pm = json.loads(parsed_metadata_str)
                    tlp = pm.get("temp_local_path")
                    if tlp and Path(tlp).exists():
                        # Already downloaded (maybe from previous session recovery)
                        with self._active_lock:
                            self._active_files[file_id] = tlp
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

            # Try to acquire buffer slot (non-blocking)
            if not self._buffer_sem.acquire(blocking=False):
                break  # Buffer full, wait for slots to free up

            # Parse webdav:// path
            source_id, remote_path = parse_webdav_path(file_path)
            if not source_id:
                self._buffer_sem.release()
                continue

            source_config = get_webdav_source(source_id)
            if not source_config:
                logger.warning(
                    f"DownloadAhead: no config for source '{source_id}', "
                    f"skipping job {job_id}"
                )
                self._buffer_sem.release()
                continue

            # Submit download to thread pool
            future = self._executor.submit(
                self._download_one, job_id, file_id, file_path,
                source_config, remote_path,
            )
            self._in_flight[file_id] = future
            future.add_done_callback(
                lambda f, fid=file_id: self._in_flight.pop(fid, None)
            )
            started += 1

        return started

    def _download_one(
        self, job_id: int, file_id: int, file_path: str,
        source_config: dict, remote_path: str,
    ):
        """Download a single WebDAV file to temp folder (runs in thread pool)."""
        from backend.remote.webdav_client import WebDAVClient

        filename = PurePosixPath(remote_path).name
        local_path = self._temp_dir / f"{file_id}_{filename}"

        try:
            client = WebDAVClient(
                base_url=source_config["url"],
                username=source_config["username"],
                password=source_config["password"],
                remote_path="/",
                verify_ssl=source_config.get("verify_ssl", True),
                timeout=300,
            )
            success = client.download_file(remote_path, local_path)
            client.close()

            if not success:
                logger.error(
                    f"DownloadAhead: download failed for job {job_id}: {file_path}"
                )
                self._buffer_sem.release()
                return

            # Record temp path in active files
            with self._active_lock:
                self._active_files[file_id] = str(local_path)

            # Update job_queue.parsed_metadata with temp_local_path
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "SELECT parsed_metadata FROM job_queue WHERE id = ?",
                    (job_id,),
                )
                row = cursor.fetchone()
                pm = {}
                if row and row[0]:
                    try:
                        pm = json.loads(row[0])
                    except (json.JSONDecodeError, TypeError):
                        pass
                pm["temp_local_path"] = str(local_path)
                cursor.execute(
                    "UPDATE job_queue SET parsed_metadata = ? WHERE id = ?",
                    (json.dumps(pm, ensure_ascii=False, default=str), job_id),
                )
                self.db.conn.commit()
            except Exception as e:
                logger.warning(
                    f"DownloadAhead: failed to update parsed_metadata "
                    f"for job {job_id}: {e}"
                )

            logger.info(
                f"DownloadAhead: downloaded {filename} "
                f"({local_path.stat().st_size} bytes) for job {job_id}"
            )

        except Exception as e:
            logger.error(f"DownloadAhead: error downloading job {job_id}: {e}")
            self._buffer_sem.release()
            # Clean up partial file
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass

    def _unload_models(self):
        """No models to unload — this pool only downloads files."""
        pass

    def get_stats(self) -> dict:
        """Return current download pool statistics."""
        with self._active_lock:
            active_count = len(self._active_files)
        return {
            "active_files": active_count,
            "max_files": self._max_files,
            "in_flight": len(self._in_flight),
            "temp_dir": str(self._temp_dir) if self._temp_dir else None,
        }
