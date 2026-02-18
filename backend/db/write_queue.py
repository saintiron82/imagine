"""
Async DB write queue with batch transaction support.

Purpose:
- Decouple GPU/CPU inference loops from SQLite write latency.
- Reduce commit overhead by grouping multiple writes in one transaction.
"""

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.db.sqlite_client import SQLiteDB

logger = logging.getLogger(__name__)


@dataclass
class _WriteItem:
    op: str
    payload: Dict[str, Any]
    future: Future


class DBWriteQueue:
    """Single-writer queue for SQLite operations with micro-batch commit."""

    _OP_METADATA = "metadata"
    _OP_VISION = "vision"
    _OP_VECTORS = "vectors"
    _OP_FLUSH = "__flush__"
    _OP_STOP = "__stop__"

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = 500,
        flush_interval_s: float = 0.15,
        max_queue_size: int = 1024,
    ):
        self.db = SQLiteDB(db_path)
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_s = max(0.01, float(flush_interval_s))
        self._q: queue.Queue[_WriteItem] = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="DBWriteQueue", daemon=True)
        self._thread.start()
        logger.info(
            f"[DBQ] started (batch_size={self.batch_size}, flush_interval_s={self.flush_interval_s}, "
            f"max_queue={self._q.maxsize})"
        )

    def submit_metadata(self, file_path: str, metadata: Dict[str, Any]) -> Future:
        return self._submit(self._OP_METADATA, {"file_path": file_path, "metadata": metadata})

    def submit_vision(self, file_path: str, fields: Dict[str, Any]) -> Future:
        return self._submit(self._OP_VISION, {"file_path": file_path, "fields": fields})

    def submit_vectors(self, file_id: int, vv_vec=None, mv_vec=None, structure_vec=None) -> Future:
        return self._submit(
            self._OP_VECTORS,
            {
                "file_id": file_id,
                "vv_vec": vv_vec,
                "mv_vec": mv_vec,
                "structure_vec": structure_vec,
            },
        )

    def flush(self, timeout: Optional[float] = None) -> bool:
        fut = self._submit(self._OP_FLUSH, {})
        return bool(fut.result(timeout=timeout))

    def close(self, timeout: Optional[float] = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        fut = self._submit(self._OP_STOP, {})
        try:
            fut.result(timeout=timeout)
        finally:
            self._thread.join(timeout=timeout)
            self.db.close()
            logger.info("[DBQ] stopped")

    def _submit(self, op: str, payload: Dict[str, Any]) -> Future:
        if self._closed and op != self._OP_STOP:
            fut = Future()
            fut.set_exception(RuntimeError("DBWriteQueue is closed"))
            return fut

        fut: Future = Future()
        self._q.put(_WriteItem(op=op, payload=payload, future=fut))
        return fut

    def _run(self) -> None:
        pending: list[_WriteItem] = []
        last_flush = time.monotonic()

        while True:
            timeout = max(0.0, self.flush_interval_s - (time.monotonic() - last_flush))
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is not None:
                if item.op == self._OP_FLUSH:
                    self._flush_pending(pending)
                    last_flush = time.monotonic()
                    if not item.future.cancelled():
                        item.future.set_result(True)
                    continue

                if item.op == self._OP_STOP:
                    self._flush_pending(pending)
                    if not item.future.cancelled():
                        item.future.set_result(True)
                    break

                pending.append(item)
                if len(pending) >= self.batch_size:
                    self._flush_pending(pending)
                    last_flush = time.monotonic()
            else:
                if pending:
                    self._flush_pending(pending)
                    last_flush = time.monotonic()

    def _flush_pending(self, pending: list[_WriteItem]) -> None:
        if not pending:
            return

        batch = pending[:]
        pending.clear()

        try:
            self.db.conn.execute("BEGIN IMMEDIATE")
            results: list[Any] = []
            for item in batch:
                if item.future.cancelled():
                    results.append(None)
                    continue
                results.append(self._execute(item, commit=False))
            self.db.conn.commit()

            for item, result in zip(batch, results):
                if not item.future.cancelled():
                    item.future.set_result(result)
        except Exception as batch_error:
            try:
                self.db.conn.rollback()
            except Exception:
                pass

            logger.warning(f"[DBQ] batch transaction failed, fallback per-item: {batch_error}")
            for item in batch:
                if item.future.cancelled():
                    continue
                try:
                    result = self._execute(item, commit=True)
                    item.future.set_result(result)
                except Exception as item_error:
                    item.future.set_exception(item_error)

    def _execute(self, item: _WriteItem, commit: bool):
        if item.op == self._OP_METADATA:
            return self.db.upsert_metadata(
                file_path=item.payload["file_path"],
                metadata=item.payload["metadata"],
                commit=commit,
            )
        if item.op == self._OP_VISION:
            return self.db.update_vision_fields(
                file_path=item.payload["file_path"],
                fields=item.payload["fields"],
                commit=commit,
            )
        if item.op == self._OP_VECTORS:
            return self.db.upsert_vectors(
                file_id=item.payload["file_id"],
                vv_vec=item.payload.get("vv_vec"),
                mv_vec=item.payload.get("mv_vec"),
                structure_vec=item.payload.get("structure_vec"),
                commit=commit,
            )
        raise ValueError(f"Unknown DB queue op: {item.op}")
