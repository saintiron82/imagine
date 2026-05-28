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
    # Original guesses retained. A calibration tool exists at
    # tools/calibrate_confidence.py and was run on the 2026-05-28
    # LLM-judge dataset, but the bench currently only saves the
    # ranked file_id list per query (not per-result raw axis scores).
    # The rank-position proxy used in the tool produces only 5
    # quantized score buckets which cannot satisfy precision-at-
    # threshold >= 0.85 for any candidate, yielding degenerate values
    # (mid=high=1.0). Once bench_precision saves per-result raw
    # scores, re-run the calibration tool and update these literals.
    low: float = 0.20
    mid: float = 0.35
    high: float = 0.55


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
        mid=float(cfg.get("search.confidence_thresholds.mid", 0.35)),
        high=float(cfg.get("search.confidence_thresholds.high", 0.55)),
    )
