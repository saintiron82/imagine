"""
protocols.py — Unified pipeline data types and strategy protocols.

PhaseItem is the single unit of work that flows through the pipeline,
regardless of file source (local, WebDAV, server job queue).

Strategy protocols (StorageBackend, BatchStrategy, ProgressReporter)
allow the same PhaseRunner to work in all deployment modes.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger("pipeline.protocols")


# ---------------------------------------------------------------------------
# PhaseItem — source-agnostic file unit
# ---------------------------------------------------------------------------

@dataclass
class PhaseItem:
    """
    A single file unit flowing through V/VV/MV phases.

    Created by Suppliers (local pipeline / server file_tasks) and consumed
    by PhaseRunner. After processing, temp resources are cleaned up.
    """

    # Identity
    job_id: Any = None                       # file_tasks.id (server/worker), None (local)
    file_id: Optional[int] = None            # files.id in DB
    file_path: str = ""                      # local path to process (may be temp)
    canonical_path: Optional[str] = None     # DB-stored path (e.g. webdav://nas-1/...)

    # Inputs for phases
    thumb_path: Optional[str] = None         # thumbnail path for VLM / VV
    mc_raw: Optional[dict] = None            # existing MC data (skip vision if present)
    analysis_profile: Optional[dict] = None  # list-level analysis prior

    # Phase skip flags
    skip_vision: bool = False
    skip_vv: bool = False
    skip_mv: bool = False

    # Folder metadata (from Phase P / supplier)
    folder_path: str = ""
    folder_depth: int = 0
    folder_tags: List[str] = field(default_factory=list)

    # Temp resource (cleaned up after processing)
    temp_dir: Optional[Path] = None

    # Phase results (accumulated by PhaseRunner, consumed by StorageBackend)
    vision_result: Optional[dict] = None     # MC fields from VLM
    vv_embedding: Any = None                 # numpy array
    structure_embedding: Any = None          # DINOv2 structure vector (optional)
    mv_embedding: Any = None                 # numpy array
    mv_text: Optional[str] = None            # composed MC text for MV

    # Status
    error: Optional[str] = None              # set on failure

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name if self.file_path else ""

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Strategy Protocols
# ---------------------------------------------------------------------------

class StorageBackend(Protocol):
    """Where to save phase results — Local DB / HTTP upload / Direct SQL."""

    def save_vision(self, item: PhaseItem, vision_result: dict) -> None:
        """Save VLM results (mc_caption, ai_tags, image_type, etc.)."""
        ...

    def save_vv(self, item: PhaseItem, vv_vec: Any,
                structure_vec: Any = None) -> None:
        """Save SigLIP2 visual vector."""
        ...

    def save_mv(self, item: PhaseItem, mv_vec: Any, text: str) -> None:
        """Save Qwen3-Embedding meaning vector + source text."""
        ...

    def flush(self) -> None:
        """Flush any buffered writes."""
        ...


class BatchStrategy(Protocol):
    """How to decide sub-batch sizes for each phase."""

    def get_batch_size(self, phase: str) -> int:
        """Return sub-batch size for the given phase (vision/vv/mv)."""
        ...

    def after_sub_batch(self, phase: str, elapsed_s: float, count: int) -> None:
        """Called after a sub-batch completes — for adaptive adjustment."""
        ...


class ProgressReporter(Protocol):
    """How to report progress to the UI/caller."""

    def phase_start(self, phase: str, count: int) -> None:
        """Phase is starting with `count` items."""
        ...

    def file_done(self, phase: str, index: int, count: int,
                  file_name: str, success: bool) -> None:
        """One file completed within a phase."""
        ...

    def phase_complete(self, phase: str, elapsed_s: float) -> None:
        """Phase finished."""
        ...


# ---------------------------------------------------------------------------
# Default implementations
# ---------------------------------------------------------------------------

class FixedBatchStrategy:
    """Fixed sub-batch sizes per phase."""

    def __init__(self, vision: int = 1, vv: int = 8, mv: int = 16):
        self._sizes = {"vision": vision, "vv": vv, "mv": mv}

    def get_batch_size(self, phase: str) -> int:
        return self._sizes.get(phase, 1)

    def after_sub_batch(self, phase: str, elapsed_s: float, count: int) -> None:
        pass  # fixed — no adjustment


class AdaptiveBatchStrategy:
    """Adapter wrapping AdaptiveBatchController as BatchStrategy.

    Maps PhaseRunner phase names (vision/vv/mv) to controller phase names (vlm/vv/mv).
    """

    # PhaseRunner → controller phase name mapping
    _PHASE_MAP = {"vision": "vlm", "vv": "vv", "mv": "mv"}

    def __init__(self, controller):
        """
        Args:
            controller: AdaptiveBatchController instance
        """
        self._controller = controller

    def get_batch_size(self, phase: str) -> int:
        mapped = self._PHASE_MAP.get(phase, phase)
        return self._controller.get_batch_size(mapped)

    def after_sub_batch(self, phase: str, elapsed_s: float, count: int) -> None:
        mapped = self._PHASE_MAP.get(phase, phase)
        self._controller.after_sub_batch(mapped, count, elapsed_s)


class NullProgressReporter:
    """No-op progress reporter (logging only)."""

    def phase_start(self, phase: str, count: int) -> None:
        logger.info(f"[PHASE:{phase}] starting {count} items")

    def file_done(self, phase: str, index: int, count: int,
                  file_name: str, success: bool) -> None:
        pass

    def phase_complete(self, phase: str, elapsed_s: float) -> None:
        logger.info(f"[PHASE:{phase}] complete in {elapsed_s:.1f}s")
