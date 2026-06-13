"""
Download-ahead pool — pre-downloads WebDAV files to a temp folder.

Monitors file_tasks for pending WebDAV downloads and downloads originals
to a bounded temp folder. Workers then use these local copies for
Phase P (parsing + thumbnail generation).

Producer-Consumer pattern:
  - Producer: This pool downloads WebDAV originals (parallel threads)
  - Buffer:   Temp folder with max_files limit (semaphore-controlled)
  - Consumer: Workers (embedded or external) access temp files via get_temp_path()
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


def _sources_file() -> "Path":
    """Persistence path for runtime-registered WebDAV sources.

    Override with IMAGINE_WEBDAV_SOURCES_FILE; defaults to project root.
    Contains plaintext credentials (same trust model as IMAGINE_WEBDAV_SOURCES
    env) — file is created 0600 and gitignored, never committed.
    """
    from pathlib import Path
    import os as _os
    override = _os.environ.get("IMAGINE_WEBDAV_SOURCES_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "webdav_sources.json"


def _persist_sources() -> None:
    """Write the current in-memory sources to disk (best-effort, 0600)."""
    import os as _os
    try:
        with _sources_lock:
            snapshot = list(_webdav_sources.values())
        path = _sources_file()
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        try:
            _os.chmod(tmp, 0o600)
        except OSError:
            pass
        _os.replace(tmp, path)
    except Exception as e:  # persistence is best-effort, never fatal
        logger.warning(f"Failed to persist WebDAV sources: {e}")


def load_persisted_sources() -> int:
    """Load runtime-registered sources from disk at startup. Returns count."""
    path = _sources_file()
    if not path.exists():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load persisted WebDAV sources: {e}")
        return 0
    n = 0
    for src in sources or []:
        register_webdav_source(src, persist=False)
        n += 1
    return n


def register_webdav_source(source_config: dict, persist: bool = False):
    """Register a WebDAV source config for download-ahead access.

    Called by server API when a source is added/updated. When persist=True,
    the updated source set is written to disk so it survives restart.
    """
    source_id = source_config.get("id")
    if not source_id:
        logger.warning("register_webdav_source: missing source id")
        return
    with _sources_lock:
        _webdav_sources[source_id] = source_config
        logger.info(f"WebDAV source registered: {source_id}")
    if persist:
        _persist_sources()


def remove_webdav_source(source_id: str, persist: bool = True) -> bool:
    """Remove a registered WebDAV source. Returns True if it existed."""
    with _sources_lock:
        existed = _webdav_sources.pop(source_id, None) is not None
    if existed and persist:
        _persist_sources()
    return existed


def list_webdav_sources() -> list:
    """Return a copy of all registered WebDAV source configs."""
    with _sources_lock:
        return [dict(s) for s in _webdav_sources.values()]


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

    Workers access downloads via get_temp_path(file_id).
    Scans file_tasks (Analysis Job System v1) for download_status='pending'.
    """

    def __init__(self, db: SQLiteDB):
        super().__init__(db)
        from backend.worker.config import get_temp_buffer_config
        buf_cfg = get_temp_buffer_config()
        self._max_files = buf_cfg["max_files"]
        self._download_workers = buf_cfg["download_workers"]
        # Track warned source IDs to avoid log spam
        self._warned_sources: set = set()

        # Bounded buffer semaphore — limits files on disk
        self._buffer_sem = threading.Semaphore(self._max_files)
        # Track active temp files: file_id -> temp_local_path
        self._active_files: Dict[int, str] = {}
        self._active_lock = threading.Lock()
        # Consecutive download failure counter for network detection
        self._dl_fail_streak = 0
        self._dl_success_streak = 0
        self._network_paused = False

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
        # Clean up stale temp folders from previous sessions
        self._cleanup_old_temp_dirs()

        self._temp_dir = Path(tempfile.mkdtemp(prefix="imagine_dl_"))
        logger.info(
            f"DownloadAheadPool: temp_dir={self._temp_dir}, "
            f"max_files={self._max_files}, workers={self._download_workers}"
        )
        # Reset incomplete WebDAV tasks whose temp files no longer exist
        # (e.g. server restart cleared temp dir).
        self._reset_stale_downloads()

        self._executor = ThreadPoolExecutor(
            max_workers=self._download_workers,
            thread_name_prefix="dl-ahead",
        )
        super().start()

    def _cleanup_old_temp_dirs(self):
        """Remove temp folders from previous server sessions.

        Each server start creates a new imagine_dl_* folder in the system
        temp directory. Previous sessions' folders are never cleaned if the
        server was killed (no graceful shutdown). This can waste 10s of GBs.
        """
        try:
            tmp_root = Path(tempfile.gettempdir())
            cleaned = 0
            freed = 0
            for d in tmp_root.glob("imagine_dl_*"):
                if d == self._temp_dir:
                    continue  # skip current session
                if d.is_dir():
                    try:
                        # Approximate size before deletion
                        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                        shutil.rmtree(d)
                        cleaned += 1
                        freed += size
                    except Exception:
                        pass
            if cleaned > 0:
                freed_mb = freed / (1024 * 1024)
                logger.info(
                    f"DownloadAhead: cleaned {cleaned} old temp dirs "
                    f"({freed_mb:.0f} MB freed)"
                )
        except Exception as e:
            logger.warning(f"DownloadAhead: old temp cleanup failed: {e}")

    def _reset_stale_downloads(self):
        """Reset download_status='done' → 'pending' for tasks where temp file is gone.

        On server restart, temp files are deleted. Tasks that were marked 'done'
        but not yet parsed need to be re-downloaded.
        Only resets unparsed tasks — parsed tasks already have thumbnails.
        """
        try:
            cursor = self.db.conn.cursor()
            # Find file_tasks that claim download is done but parse hasn't started
            cursor.execute(
                """SELECT ft.id, ft.file_id FROM file_tasks ft
                   JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                   WHERE ft.file_path LIKE 'webdav://%'
                     AND ft.download_status = 'done'
                     AND ft.parse_status = 'pending'
                     AND aj.status = 'active'"""
            )
            rows = cursor.fetchall()
            reset_ids = []
            for task_id, file_id in rows:
                # Check if temp file still exists
                with self._active_lock:
                    temp = self._active_files.get(file_id)
                if temp and Path(temp).exists():
                    continue  # temp file still exists
                reset_ids.append(task_id)

            if reset_ids:
                placeholders = ",".join("?" * len(reset_ids))
                cursor.execute(
                    f"""UPDATE file_tasks SET download_status = 'pending', updated_at = ?
                        WHERE id IN ({placeholders})""",
                    [time.strftime("%Y-%m-%d %H:%M:%S")] + reset_ids,
                )
                self.db.conn.commit()
                logger.info(
                    f"DownloadAhead: reset {len(reset_ids)} stale download tasks → pending"
                )
            else:
                self.db.conn.commit()
        except Exception as e:
            logger.warning(f"DownloadAhead: failed to reset stale downloads: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _mark_download_failed(self, task_id: int, file_path: str):
        """Mark a file_task download as failed.

        The periodic recovery loop resets them back to 'pending'
        after a cooldown, so they re-enter the download queue.
        """
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """UPDATE file_tasks SET download_status = 'failed',
                   error_message = 'download failed', updated_at = ?
                   WHERE id = ?""",
                (time.strftime("%Y-%m-%d %H:%M:%S"), task_id),
            )
            self.db.conn.commit()
            logger.info(f"DownloadAhead: marked task {task_id} as download-failed")
        except Exception as e:
            logger.warning(f"DownloadAhead: failed to mark task {task_id}: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _recover_failed_downloads(self):
        """Reset failed download tasks back to pending after 5 min cooldown.

        Called periodically from the download loop so failed tasks
        get another chance after a cooldown.
        """
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """UPDATE file_tasks SET download_status = 'pending',
                   error_message = NULL, updated_at = ?
                   WHERE download_status = 'failed'
                     AND datetime(updated_at) < datetime('now', '-5 minutes')""",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            recovered = cursor.rowcount
            self.db.conn.commit()
            if recovered > 0:
                logger.info(
                    f"DownloadAhead: recovered {recovered} failed downloads → pending"
                )
        except Exception as e:
            logger.warning(f"DownloadAhead: recovery check failed: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _cleanup_stale_temp_files(self):
        """Remove temp files for tasks that no longer need originals.

        Frees buffer slots for: parsed files (only thumbnail needed),
        failed tasks, cancelled/archived jobs.
        """
        if not self._temp_dir or not self._temp_dir.exists():
            return
        with self._active_lock:
            active_ids = set(self._active_files.keys())
        if not active_ids:
            return

        try:
            cursor = self.db.conn.cursor()
            placeholders = ",".join("?" * len(active_ids))
            cursor.execute(
                f"""SELECT ft.file_id, ft.parse_status, ft.download_status, aj.status
                    FROM file_tasks ft
                    JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                    WHERE ft.file_id IN ({placeholders})""",
                list(active_ids),
            )
            stale = set()
            for row in cursor.fetchall():
                fid, parse_st, dl_st, job_st = row
                # Remove if: already parsed, download failed, or job done/cancelled
                if parse_st == "done" or dl_st == "failed" or job_st in ("cancelled", "archived", "completed"):
                    stale.add(fid)

            for fid in stale:
                with self._active_lock:
                    path = self._active_files.pop(fid, None)
                if path:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._buffer_sem.release()

            if stale:
                logger.info(
                    f"DownloadAhead: cleaned {len(stale)} stale temp files "
                    f"(parsed/failed/job-done)"
                )
        except Exception as e:
            logger.warning(f"DownloadAhead: stale cleanup failed: {e}")

    def _check_network_health(self) -> bool:
        """Quick health check on all registered WebDAV sources.

        Returns True if at least one source is reachable.
        """
        import requests as _req
        with _sources_lock:
            sources = dict(_webdav_sources)
        if not sources:
            return True  # No sources = nothing to check

        for sid, cfg in sources.items():
            try:
                r = _req.request(
                    "HEAD", cfg["url"] + "/",
                    timeout=5,
                    verify=cfg.get("verify_ssl", True),
                )
                # Any HTTP response = network OK (even 401/404)
                return True
            except Exception:
                continue
        return False

    def recover_all_failed(self) -> int:
        """Force-recover all failed download tasks immediately. Called by API button."""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """UPDATE file_tasks SET download_status = 'pending',
                   error_message = NULL, updated_at = ?
                   WHERE download_status = 'failed'""",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            recovered = cursor.rowcount
            self.db.conn.commit()
            if recovered > 0:
                logger.info(f"DownloadAhead: force-recovered {recovered} failed downloads")
            self._cleanup_stale_temp_files()
            return recovered
        except Exception as e:
            logger.warning(f"DownloadAhead: force recovery failed: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass
            return 0

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
        """Release a buffer slot after parse completion.

        Deletes the temp file and frees a semaphore slot for new downloads.
        No LRU cache — temp files are deleted after parsing.
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

    def request_redownload(self, file_id: int, file_path: str):
        """Request re-download of a WebDAV file.

        If the file is already cached locally, reuses it.
        Otherwise, resets download_status to pending for the download loop.
        """
        if not file_path or not file_path.startswith("webdav://"):
            return

        with self._active_lock:
            existing = self._active_files.get(file_id)
            if existing and Path(existing).exists():
                logger.debug(f"Re-download: cache hit for file_id={file_id}")
                return

        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                """UPDATE file_tasks SET download_status = 'pending',
                   error_message = NULL, updated_at = ?
                   WHERE file_id = ?
                     AND file_path LIKE 'webdav://%'
                     AND download_status IN ('failed', 'done')""",
                (time.strftime("%Y-%m-%d %H:%M:%S"), file_id),
            )
            self.db.conn.commit()
            if cursor.rowcount > 0:
                logger.info(f"Re-download queued: file_id={file_id}")
        except Exception as e:
            logger.warning(f"Re-download request failed for file_id={file_id}: {e}")
            try:
                self.db.conn.rollback()
            except Exception:
                pass

    def _loop(self):
        """Main loop: find pending WebDAV jobs, download originals.

        Runs independently — downloads at its own pace, bounded by
        max_files buffer. FileTaskParsePool parses downloaded files
        as their download_status becomes done.
        """
        poll_interval = self._get_config_value(
            "server.parse_ahead.poll_interval_s", 2
        )
        recovery_counter = 0

        while self._running:
            try:
                # Network pause: wait for recovery (set by _download_one on 3+ failures)
                if self._network_paused:
                    if self._check_network_health():
                        logger.info("DownloadAhead: network recovered — resuming")
                        self._network_paused = False
                        self._dl_fail_streak = 0
                        self._recover_failed_downloads()
                        self._cleanup_stale_temp_files()
                    else:
                        time.sleep(30)
                        continue

                downloaded = self._download_batch()
                if downloaded == 0:
                    time.sleep(poll_interval)

                # Periodic maintenance
                recovery_counter += 1
                if recovery_counter >= 30:
                    self._recover_failed_downloads()
                    self._cleanup_stale_temp_files()
                    recovery_counter = 0

            except Exception as e:
                logger.error(f"DownloadAhead loop error: {e}")
                time.sleep(5)

    def _download_batch(self) -> int:
        """Find pending WebDAV files and start downloads.

        Scans file_tasks (Analysis Job System v1) for download_status='pending'.
        Returns number of downloads started.
        """
        cursor = self.db.conn.cursor()

        # Check if downloads are paused
        try:
            from backend.server.routers.analysis import get_paused_phases
            if get_paused_phases(self.db).get("dl"):
                return 0
        except Exception:
            pass

        # Query file_tasks for pending WebDAV downloads
        try:
            cursor.execute(
                """SELECT id, file_id, file_path FROM (
                       SELECT ft.id, ft.file_id, ft.file_path,
                              ft.priority, ft.created_at,
                              ROW_NUMBER() OVER (
                                  PARTITION BY ft.analysis_job_id
                                  ORDER BY ft.created_at ASC
                              ) AS job_rn
                       FROM file_tasks ft
                       JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                       WHERE aj.status = 'active'
                         AND ft.download_status = 'pending'
                         AND ft.file_path LIKE 'webdav://%'
                   )
                   ORDER BY priority DESC, job_rn ASC, created_at ASC
                   LIMIT ?""",
                (self._max_files,),
            )
            rows = cursor.fetchall()
        except Exception as e:
            logger.warning(f"DownloadAhead: file_tasks query failed: {e}")
            rows = []

        if not rows:
            return 0

        started = 0
        for task_id, file_id, file_path in rows:
            if not self._running:
                break

            # Skip if already downloaded or in-flight
            with self._active_lock:
                if file_id in self._active_files:
                    # Already downloaded — mark task as done
                    self._mark_task_downloaded(task_id)
                    continue
            if file_id in self._in_flight:
                continue

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
                if source_id not in self._warned_sources:
                    self._warned_sources.add(source_id)
                    logger.warning(
                        f"DownloadAhead: no config for source '{source_id}' "
                        f"— register via API or IMAGINE_WEBDAV_SOURCES env"
                    )
                self._buffer_sem.release()
                continue

            # Submit download to thread pool
            future = self._executor.submit(
                self._download_one, task_id, file_id, file_path,
                source_config, remote_path,
            )
            self._in_flight[file_id] = future
            future.add_done_callback(
                lambda f, fid=file_id: self._in_flight.pop(fid, None)
            )
            started += 1

        return started

    def _mark_task_downloaded(self, task_id: int):
        """Mark a file_task download as done (file already in active_files)."""
        for _retry in range(5):
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "UPDATE file_tasks SET download_status = 'done', updated_at = ? WHERE id = ?",
                    (time.strftime("%Y-%m-%d %H:%M:%S"), task_id),
                )
                self.db.conn.commit()
                return
            except Exception as e:
                if "locked" in str(e) and _retry < 4:
                    time.sleep(1)
                    continue
                logger.warning(f"DownloadAhead: mark task {task_id} done failed: {e}")
                try:
                    self.db.conn.rollback()
                except Exception:
                    pass

    def _download_one(
        self, task_id: int, file_id: int, file_path: str,
        source_config: dict, remote_path: str,
    ):
        """Download a single WebDAV file to temp folder (runs in thread pool).

        Args:
            task_id: file_tasks.id
            file_id: files.id
            file_path: webdav://source-id/remote/path
            source_config: WebDAV connection config
            remote_path: Remote path on WebDAV server
        """
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
                    f"DownloadAhead: download failed for task {task_id}: {file_path}"
                )
                self._mark_download_failed(task_id, file_path)
                self._dl_fail_streak += 1
                self._dl_success_streak = 0
                if self._dl_fail_streak >= 3 and not self._network_paused:
                    if not self._check_network_health():
                        logger.warning("DownloadAhead: network unreachable — pausing")
                        self._network_paused = True
                self._buffer_sem.release()
                # Remove partially-written file so it can't be mistaken
                # for a completed download later
                try:
                    local_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return

            # Success — reset failure streak
            self._dl_fail_streak = 0
            self._dl_success_streak += 1

            # Record temp path in active files
            with self._active_lock:
                self._active_files[file_id] = str(local_path)

            # Mark file_task download as done → FileTaskParsePool picks it up
            self._mark_task_downloaded(task_id)

            logger.info(
                f"DownloadAhead: downloaded {filename} "
                f"({local_path.stat().st_size} bytes) for task {task_id}"
            )

        except Exception as e:
            logger.error(f"DownloadAhead: error downloading task {task_id}: {e}")
            self._mark_download_failed(task_id, file_path)
            self._buffer_sem.release()
            if local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass

    def _unload_models(self):
        """No models to unload — this pool only downloads files."""
        pass

    def get_stats(self) -> dict:
        """Return current download pool statistics including disk usage."""
        with self._active_lock:
            active_count = len(self._active_files)
        # Calculate temp dir size
        temp_size_bytes = 0
        temp_file_count = 0
        if self._temp_dir and self._temp_dir.exists():
            try:
                for f in self._temp_dir.rglob("*"):
                    if f.is_file():
                        temp_size_bytes += f.stat().st_size
                        temp_file_count += 1
            except Exception:
                pass
        return {
            "active_files": active_count,
            "max_files": self._max_files,
            "in_flight": len(self._in_flight),
            "temp_dir": str(self._temp_dir) if self._temp_dir else None,
            "disk_usage_mb": round(temp_size_bytes / (1024 * 1024), 1),
            "disk_file_count": temp_file_count,
            "network_paused": self._network_paused,
            "fail_streak": self._dl_fail_streak,
        }
