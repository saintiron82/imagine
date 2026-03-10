"""
FileContainer — Source-agnostic file supply buffer for the pipeline.

Producer-Consumer pattern:
  - Suppliers (Local/WebDAV/...) put batches into the container
  - Pipeline takes batches out and processes them
  - After processing, cleanup callback removes temp files

Usage:
    container = FileContainer(buffer_size=2)
    supplier = WebDAVSupplier(source_config)
    supplier.start(container)   # BG thread starts downloading

    while True:
        batch = container.take_batch()
        if batch is None:
            break
        process_batch_phased(batch.file_infos, ...)
        container.cleanup_batch(batch)
"""

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("FileContainer")


@dataclass
class BatchInfo:
    """A batch of files ready for pipeline processing."""
    file_infos: List[Tuple[Path, Optional[str], int, List[str]]]
    # Maps local temp path (str) -> canonical path for DB storage.
    # Empty dict for local files (no remapping needed).
    path_remap: Dict[str, str] = field(default_factory=dict)
    # Temp directory to delete after processing (None for local files).
    temp_dir: Optional[Path] = None
    # Source identifier (e.g., "webdav://nas-1")
    source_id: Optional[str] = None


class FileContainer:
    """
    Thread-safe file supply buffer.

    Suppliers put BatchInfo into the queue.
    Pipeline takes BatchInfo out for processing.
    buffer_size controls how many batches can be pre-fetched.
    """

    def __init__(self, buffer_size: int = 2):
        self._queue: Queue[Optional[BatchInfo]] = Queue(maxsize=buffer_size)
        self._done = Event()
        self._error: Optional[str] = None

    def put_batch(self, batch: BatchInfo):
        """Supplier calls this to add a ready batch."""
        self._queue.put(batch)

    def take_batch(self, timeout: float = 30.0) -> Optional[BatchInfo]:
        """
        Pipeline calls this to get the next batch.
        Returns None when all batches are done.
        """
        while True:
            try:
                item = self._queue.get(timeout=timeout)
                if item is None:
                    # Sentinel: supplier is done
                    return None
                return item
            except Empty:
                if self._done.is_set():
                    # Supplier finished and queue is empty
                    return None
                # Still waiting for supplier, retry
                continue

    def mark_done(self):
        """Supplier calls this when all files have been supplied."""
        self._done.set()
        self._queue.put(None)  # Sentinel to unblock take_batch

    def mark_error(self, message: str):
        """Supplier calls this on fatal error."""
        self._error = message
        self._done.set()
        self._queue.put(None)

    @property
    def has_error(self) -> bool:
        return self._error is not None

    @property
    def error_message(self) -> Optional[str]:
        return self._error

    @staticmethod
    def cleanup_batch(batch: BatchInfo):
        """Delete temp files after batch processing completes."""
        if batch.temp_dir and batch.temp_dir.exists():
            try:
                shutil.rmtree(batch.temp_dir)
                logger.debug(f"Cleaned up temp dir: {batch.temp_dir}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temp dir {batch.temp_dir}: {e}")


class LocalSupplier:
    """
    Supplies local files directly into the container.
    No download needed — just wraps discover_files() output into BatchInfo.
    """

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size

    def start(
        self,
        container: FileContainer,
        file_infos: List[Tuple[Path, Optional[str], int, List[str]]],
    ):
        """
        Feed local file_infos into the container in batches.
        Runs synchronously (local files need no background work).
        """
        for i in range(0, len(file_infos), self.batch_size):
            batch_slice = file_infos[i:i + self.batch_size]
            container.put_batch(BatchInfo(file_infos=batch_slice))
        container.mark_done()


class WebDAVSupplier:
    """
    Supplies WebDAV files by downloading them to temp directories.
    Runs in a background thread to overlap download with pipeline processing.
    """

    def __init__(self, source_config: dict, batch_size: int = 5):
        """
        Args:
            source_config: {id, url, username, password, remote_path, verify_ssl}
            batch_size: Number of files per batch download
        """
        self.source_config = source_config
        self.batch_size = batch_size
        self._thread: Optional[Thread] = None
        self._stop_event = Event()

    def start(self, container: FileContainer,
              progress_callback: Optional[Callable] = None):
        """
        Start background download thread.
        1. PROPFIND to discover all files
        2. Download in batches to temp dirs
        3. Put each ready batch into container
        """
        self._thread = Thread(
            target=self._run,
            args=(container, progress_callback),
            daemon=True,
            name="WebDAVSupplier",
        )
        self._thread.start()

    def stop(self):
        """Signal the supplier to stop."""
        self._stop_event.set()

    def join(self, timeout: float = 30.0):
        """Wait for supplier thread to finish."""
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self, container: FileContainer,
             progress_callback: Optional[Callable]):
        """Background thread: discover → download batches → put into container."""
        from backend.remote.webdav_client import WebDAVClient

        source_id = self.source_config.get('id', 'webdav')

        try:
            client = WebDAVClient(
                base_url=self.source_config['url'],
                username=self.source_config['username'],
                password=self.source_config['password'],
                remote_path=self.source_config.get('remote_path', '/'),
                verify_ssl=self.source_config.get('verify_ssl', True),
            )

            # 1. Discover all remote files
            if progress_callback:
                progress_callback("listing", {"message": "Listing remote files..."})

            remote_files = client.list_files_recursive()
            total = len(remote_files)

            if progress_callback:
                progress_callback("discover_total", {
                    "total": total,
                    "message": f"Found {total} files on remote",
                })

            if total == 0:
                container.mark_done()
                client.close()
                return

            # 2. Download in batches
            for batch_start in range(0, total, self.batch_size):
                if self._stop_event.is_set():
                    logger.info("WebDAVSupplier: stop requested")
                    break

                batch_files = remote_files[batch_start:batch_start + self.batch_size]
                batch_num = batch_start // self.batch_size + 1
                total_batches = (total + self.batch_size - 1) // self.batch_size

                if progress_callback:
                    progress_callback("batch_download", {
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "files": len(batch_files),
                        "message": f"Downloading batch {batch_num}/{total_batches}...",
                    })

                # Create temp dir for this batch
                temp_dir = Path(tempfile.mkdtemp(prefix=f"imagine_webdav_{source_id}_"))

                file_infos = []
                path_remap = {}
                download_ok = True

                for rf in batch_files:
                    if self._stop_event.is_set():
                        break

                    local_path = temp_dir / rf.relative_path
                    success = client.download_file(
                        rf.remote_path, local_path, expected_size=rf.size
                    )

                    if success:
                        # Extract folder metadata from relative path
                        rel_parts = Path(rf.relative_path).parts
                        if len(rel_parts) > 1:
                            folder_path = str(Path(*rel_parts[:-1]))
                            folder_depth = len(rel_parts) - 2
                            folder_tags = list(rel_parts[:-1])
                        else:
                            folder_path = ""
                            folder_depth = 0
                            folder_tags = []

                        file_infos.append((
                            local_path, folder_path, folder_depth, folder_tags
                        ))

                        # Map temp local path → canonical webdav:// path for DB
                        canonical = f"webdav://{source_id}/{rf.relative_path}"
                        path_remap[str(local_path)] = canonical
                    else:
                        logger.warning(f"Download failed: {rf.relative_path}")

                if file_infos:
                    container.put_batch(BatchInfo(
                        file_infos=file_infos,
                        path_remap=path_remap,
                        temp_dir=temp_dir,
                        source_id=source_id,
                    ))
                else:
                    # No files downloaded in this batch, clean up empty temp dir
                    shutil.rmtree(temp_dir, ignore_errors=True)

            client.close()
            container.mark_done()

            if progress_callback:
                progress_callback("supply_complete", {
                    "total": total,
                    "message": "All files supplied",
                })

        except Exception as e:
            logger.error(f"WebDAVSupplier error: {e}")
            container.mark_error(str(e))
