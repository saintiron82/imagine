"""Phase A: absolute-score confidence classification."""
from __future__ import annotations

import enum
from dataclasses import dataclass


class ConfidenceLevel(str, enum.Enum):
    EMPTY = "empty"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ConfidenceThresholds:
    # Calibrated 2026-05-28 from precision_20260528_frozen_run1_sprint1
    # LLM-judge dataset (150 aligned samples). Targets:
    # precision-at-confidence >= 0.5 / 0.7 / 0.85.
    # NOTE: current calibration uses a rank-position proxy because the
    # bench JSON does not yet persist per-result raw axis scores; the
    # mid/high cuts here reflect that proxy's coarse 5-bucket
    # distribution and should be revisited once the bench schema is
    # widened. See tools/calibrate_confidence.py.
    low: float = 0.20
    mid: float = 1.0
    high: float = 1.0


def classify(score: float, thresholds: ConfidenceThresholds) -> ConfidenceLevel:
    if score >= thresholds.high:
        return ConfidenceLevel.HIGH
    if score >= thresholds.mid:
        return ConfidenceLevel.MEDIUM
    if score >= thresholds.low:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.EMPTY


def classify_topk(
    *,
    vector_score: float,
    text_vec_score: float,
    fts_hit: bool,
    thresholds: ConfidenceThresholds,
) -> ConfidenceLevel:
    """Pick the most generous signal for the top-1 result."""
    best = max(vector_score or 0.0, text_vec_score or 0.0)
    level = classify(best, thresholds)
    if level is ConfidenceLevel.EMPTY and fts_hit:
        return ConfidenceLevel.LOW
    return level


def thresholds_from_config(cfg) -> ConfidenceThresholds:
    return ConfidenceThresholds(
        low=float(cfg.get("search.confidence_thresholds.low", 0.20)),
        mid=float(cfg.get("search.confidence_thresholds.mid", 1.0)),
        high=float(cfg.get("search.confidence_thresholds.high", 1.0)),
    )
