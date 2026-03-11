"""
storage_server.py — StorageBackend for distributed worker mode.

Accumulates phase results on PhaseItem, then uploads all at once
via complete_job() HTTP API call.

Used by worker_daemon.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.pipeline.protocols import PhaseItem, StorageBackend

logger = logging.getLogger("pipeline.storage_server")


class ServerAPIStorage:
    """
    HTTP-based storage via worker uploader.

    Phase results are accumulated on PhaseItem objects during processing.
    flush() calls complete_job() for each item to upload results to server.

    This differs from LocalDBStorage/DirectSQLStorage which save
    incrementally per-phase — here we batch all results and upload once.
    """

    def __init__(self, uploader):
        """
        Args:
            uploader: ResultUploader instance from worker_daemon
                      Must have complete_job(job_id, result_data) method
        """
        self._uploader = uploader
        self._pending: List[PhaseItem] = []

    def save_vision(self, item: PhaseItem, vision_result: dict) -> None:
        """Accumulate vision result on item (no upload yet)."""
        item.vision_result = vision_result
        if item not in self._pending:
            self._pending.append(item)

    def save_vv(self, item: PhaseItem, vv_vec: Any,
                structure_vec: Any = None) -> None:
        """Accumulate VV vector on item (no upload yet)."""
        item.vv_embedding = vv_vec
        item.structure_embedding = structure_vec
        if item not in self._pending:
            self._pending.append(item)

    def save_mv(self, item: PhaseItem, mv_vec: Any, text: str) -> None:
        """Accumulate MV vector on item (no upload yet)."""
        item.mv_embedding = mv_vec
        item.mv_text = text
        if item not in self._pending:
            self._pending.append(item)

    def flush(self) -> None:
        """Upload all accumulated results to server via complete_job()."""
        for item in self._pending:
            if item.job_id is None:
                continue
            try:
                result_data = self._build_result(item)
                self._uploader.complete_job(item.job_id, result_data)
                logger.debug(f"Uploaded job {item.job_id}: {item.file_name}")
            except Exception as e:
                logger.error(
                    f"[FAIL:upload] job {item.job_id} {item.file_name}: {e}")
                item.error = f"Upload failed: {e}"

        self._pending.clear()

    def flush_item(self, item: PhaseItem) -> bool:
        """Upload a single item immediately. Returns True on success."""
        if item.job_id is None:
            return False
        try:
            result_data = self._build_result(item)
            self._uploader.complete_job(item.job_id, result_data)
            if item in self._pending:
                self._pending.remove(item)
            return True
        except Exception as e:
            logger.error(
                f"[FAIL:upload] job {item.job_id} {item.file_name}: {e}")
            item.error = f"Upload failed: {e}"
            return False

    def _build_result(self, item: PhaseItem) -> dict:
        """Build the result payload for complete_job()."""
        result = {}

        # Vision results
        if item.vision_result:
            result["vision"] = item.vision_result

        # VV embedding
        if item.vv_embedding is not None:
            result["vv_embedding"] = item.vv_embedding
            if item.structure_embedding is not None:
                result["structure_embedding"] = item.structure_embedding

        # MV embedding
        if item.mv_embedding is not None:
            result["mv_embedding"] = item.mv_embedding
            if item.mv_text:
                result["mv_text"] = item.mv_text

        return result
