"""Phase A: confidence level mapping from raw scores."""
from __future__ import annotations

import pytest

from backend.search.confidence import (
    ConfidenceLevel,
    ConfidenceThresholds,
    classify,
)


DEFAULT = ConfidenceThresholds(low=0.20, mid=0.35, high=0.55)


@pytest.mark.parametrize("score, expected", [
    (0.05, ConfidenceLevel.EMPTY),
    (0.19, ConfidenceLevel.EMPTY),
    (0.20, ConfidenceLevel.LOW),
    (0.34, ConfidenceLevel.LOW),
    (0.35, ConfidenceLevel.MEDIUM),
    (0.54, ConfidenceLevel.MEDIUM),
    (0.55, ConfidenceLevel.HIGH),
    (0.99, ConfidenceLevel.HIGH),
])
def test_classify_uses_inclusive_lower_bounds(score, expected):
    assert classify(score, DEFAULT) is expected


def test_classify_uses_max_of_vector_and_text_scores():
    """The relevant scalar is the BEST raw score across axes, not a sum."""
    from backend.search.confidence import classify_topk

    # vector strong, text weak — should be MEDIUM
    assert classify_topk(
        vector_score=0.40, text_vec_score=0.05, fts_hit=False,
        thresholds=DEFAULT,
    ) is ConfidenceLevel.MEDIUM


def test_classify_returns_low_when_only_fts_hits():
    """FTS keyword hit alone keeps us out of EMPTY even if vector low."""
    from backend.search.confidence import classify_topk

    assert classify_topk(
        vector_score=0.05, text_vec_score=0.05, fts_hit=True,
        thresholds=DEFAULT,
    ) is ConfidenceLevel.LOW


def test_thresholds_from_config_uses_defaults_when_missing():
    from backend.search.confidence import thresholds_from_config

    cfg = {}
    t = thresholds_from_config(cfg)
    # Defaults reflect the 2026-05-28 calibration (see backend/search/confidence.py).
    assert t.low == 0.20 and t.mid == 1.0 and t.high == 1.0


def test_thresholds_from_config_reads_overrides():
    from backend.search.confidence import thresholds_from_config

    cfg = {"search.confidence_thresholds.low": 0.10,
           "search.confidence_thresholds.mid": 0.30,
           "search.confidence_thresholds.high": 0.50}
    t = thresholds_from_config(cfg)
    assert t.low == 0.10 and t.mid == 0.30 and t.high == 0.50


@pytest.mark.parametrize("vector_score, fts_hit, expected", [
    (0.60, False, ConfidenceLevel.HIGH),
    (0.60, True, ConfidenceLevel.HIGH),   # FTS hit must NOT demote HIGH
    (0.40, True, ConfidenceLevel.MEDIUM), # FTS hit must NOT demote MEDIUM
    (0.25, True, ConfidenceLevel.LOW),    # FTS hit must NOT promote LOW
])
def test_fts_hit_only_floors_empty_does_not_change_other_levels(
    vector_score, fts_hit, expected,
):
    """The FTS upgrade only applies when classification would otherwise be EMPTY."""
    from backend.search.confidence import classify_topk

    assert classify_topk(
        vector_score=vector_score, text_vec_score=0.0, fts_hit=fts_hit,
        thresholds=DEFAULT,
    ) is expected
