"""
BufferPool — Source-agnostic file supply buffer for the pipeline.

Multiple suppliers put individual items into one BufferPool:
- Source-agnostic: local + NAS files mix naturally in one pool.
- Backpressure: maxsize prevents excess downloads.
- Streaming: only batch_capacity × 2 files on disk at any time.

Usage (BufferPool):
    pool = BufferPool(capacity=10)
    local_supplier.start_pool(pool)    # instant put
    webdav_supplier.start_pool(pool)   # background download + put

    while True:
        items = pool.take_batch(5)
        if not items:
            break
        runner.run_all(items)
        pool.cleanup(items)
"""

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event, Lock
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("BufferPool")


@dataclass
class PoolItem:
    """A single file unit in a BufferPool (individual-item supply)."""
    file_path: Path
    folder_path: str = ""
    folder_depth: int = 0
    folder_tags: List[str] = field(default_factory=list)
    # WebDAV: canonical_path for DB storage (e.g. "webdav://nas-1/foo.psd")
    canonical_path: Optional[str] = None
    # WebDAV: temp directory to delete after processing
    temp_dir: Optional[Path] = None
    # Source identifier (e.g., "webdav://nas-1")
    source_id: Optional[str] = None


class LocalSupplier:
    """
    Supplies local files directly into a BufferPool.
    """

    def start_pool(
        self,
        pool: "BufferPool",
        file_infos: List[Tuple[Path, Optional[str], int, List[str]]],
    ):
        """
        Feed local file_infos into a BufferPool as individual PoolItems.
        Runs synchronously (local files need no background work).
        """
        try:
            for fp, folder, depth, tags in file_infos:
                pool.put(PoolItem(
                    file_path=fp,
                    folder_path=folder or "",
                    folder_depth=depth,
                    folder_tags=tags or [],
                ))
        finally:
            pool.supplier_done()


class WebDAVSupplier:
    """
    Supplies WebDAV files by downloading them to temp directories.
    Runs in a background thread to overlap download with pipeline processing.
    """

    def __init__(self, source_config: dict):
        """
        Args:
            source_config: {id, url, username, password, remote_path, verify_ssl}
        """
        self.source_config = source_config
        self._thread: Optional[Thread] = None
        self._stop_event = Event()

    def start_pool(self, pool: "BufferPool",
                   progress_callback: Optional[Callable] = None):
        """
        Start background thread that downloads files individually
        into a BufferPool.
        """
        self._thread = Thread(
            target=self._run_pool,
            args=(pool, progress_callback),
            daemon=True,
            name="WebDAVSupplier-pool",
        )
        self._thread.start()

    def stop(self):
        """Signal the supplier to stop."""
        self._stop_event.set()

    def join(self, timeout: float = 30.0):
        """Wait for supplier thread to finish."""
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run_pool(self, pool: "BufferPool",
                  progress_callback: Optional[Callable]):
        """Background thread: discover → download individual files → put into pool."""
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
                client.close()
                return

            for idx, rf in enumerate(remote_files):
                if self._stop_event.is_set():
                    logger.info("WebDAVSupplier: stop requested")
                    break

                # Each file gets its own temp dir for independent cleanup
                temp_dir = Path(tempfile.mkdtemp(
                    prefix=f"imagine_wdav_{source_id}_"
                ))
                local_path = temp_dir / rf.relative_path

                success = client.download_file(
                    rf.remote_path, local_path, expected_size=rf.size
                )

                if success:
                    rel_parts = Path(rf.relative_path).parts
                    if len(rel_parts) > 1:
                        folder_path = str(Path(*rel_parts[:-1]))
                        folder_depth = len(rel_parts) - 2
                        folder_tags = list(rel_parts[:-1])
                    else:
                        folder_path = ""
                        folder_depth = 0
                        folder_tags = []

                    pool.put(PoolItem(
                        file_path=local_path,
                        folder_path=folder_path,
                        folder_depth=folder_depth,
                        folder_tags=folder_tags,
                        canonical_path=f"webdav://{source_id}/{rf.relative_path}",
                        temp_dir=temp_dir,
                        source_id=source_id,
                    ))
                else:
                    logger.warning(f"Download failed: {rf.relative_path}")
                    shutil.rmtree(temp_dir, ignore_errors=True)

            client.close()

            if progress_callback:
                progress_callback("supply_complete", {
                    "total": total,
                    "message": "All files supplied",
                })

        except Exception as e:
            logger.error(f"WebDAVSupplier error: {e}")
            pool.supplier_error(str(e))
        finally:
            pool.supplier_done()

# ---------------------------------------------------------------------------
# BufferPool — Source-agnostic individual-item buffer
# ---------------------------------------------------------------------------

class BufferPool:
    """
    Source-agnostic file buffer pool for streaming pipeline.

    Multiple suppliers (Local/WebDAV) put individual items.
    Pipeline takes batch_capacity items at a time for processing.
    After processing, cleanup deletes temp files.

    Key properties:
    - Source mixing: local + NAS items coexist naturally
    - Backpressure: maxsize blocks suppliers when pool is full
    - Streaming: only capacity items on disk at any time
    - Auto-refill: take() frees slots → suppliers unblock → auto refill

    Usage:
        pool = BufferPool(capacity=10)  # batch_capacity × 2

        local_supplier.start_pool(pool)     # puts items immediately
        webdav_supplier.start_pool(pool)    # puts after download

        while True:
            items = pool.take_batch(5)      # blocks until 5 ready
            if not items:
                break
            runner.run_all(items)
            pool.cleanup(items)
    """

    def __init__(self, capacity: int = 10):
        """
        Args:
            capacity: Maximum items in the pool (batch_capacity × 2).
                      Suppliers block when pool is full.
        """
        self._queue: Queue = Queue(maxsize=capacity)
        self._lock = Lock()
        self._suppliers_total = 0
        self._suppliers_done = 0
        self._error: Optional[str] = None
        self._stats = {"put": 0, "taken": 0, "cleaned": 0}

    def register_supplier(self) -> None:
        """Register a supplier. Call before supplier.start()."""
        with self._lock:
            self._suppliers_total += 1

    def put(self, item) -> None:
        """
        Supplier adds one ready item to the pool.
        Blocks if pool is full (backpressure).

        Args:
            item: PhaseItem instance
        """
        self._queue.put(item)
        with self._lock:
            self._stats["put"] += 1

    def take_batch(self, size: int, timeout: float = 60.0) -> list:
        """
        Pipeline takes up to `size` items from the pool.

        Collects items until:
        - `size` items gathered, OR
        - all suppliers done and pool empty, OR
        - timeout reached

        Returns:
            List of PhaseItems. Empty list = all work complete.
        """
        items = []
        deadline = time.time() + timeout

        while len(items) < size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            try:
                item = self._queue.get(timeout=min(remaining, 2.0))
                items.append(item)
            except Empty:
                # Check if all suppliers are done
                with self._lock:
                    if (self._suppliers_done >= self._suppliers_total
                            and self._suppliers_total > 0
                            and self._queue.empty()):
                        break
                # Otherwise keep waiting
                continue

        with self._lock:
            self._stats["taken"] += len(items)

        return items

    def cleanup(self, items: list) -> None:
        """
        Clean up temp files after processing.
        Deletes item.temp_dir if set.
        """
        cleaned = 0
        for item in items:
            if hasattr(item, "temp_dir") and item.temp_dir:
                temp_dir = Path(item.temp_dir)
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        cleaned += 1
                    except OSError as e:
                        logger.warning(f"Cleanup failed {temp_dir}: {e}")

        with self._lock:
            self._stats["cleaned"] += cleaned

    def supplier_done(self) -> None:
        """Supplier signals completion. Called by each supplier when finished."""
        with self._lock:
            self._suppliers_done += 1
            done = self._suppliers_done
            total = self._suppliers_total
        logger.info(f"Supplier done ({done}/{total})")

    def supplier_error(self, message: str) -> None:
        """Supplier signals fatal error."""
        with self._lock:
            self._error = message
            self._suppliers_done += 1
        logger.error(f"Supplier error: {message}")

    @property
    def all_done(self) -> bool:
        with self._lock:
            return (self._suppliers_done >= self._suppliers_total
                    and self._suppliers_total > 0
                    and self._queue.empty())

    @property
    def has_error(self) -> bool:
        with self._lock:
            return self._error is not None

    @property
    def error_message(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def pool_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
