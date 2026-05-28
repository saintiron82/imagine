"""Sprint 3 S3.2: boost spatial_score rows on spatial-intent queries."""
from __future__ import annotations

import pytest

from backend.search.sqlite_search import apply_spatial_intent_boost


def test_no_op_when_query_type_not_spatial():
    rows = [{"id": 1, "rrf_score": 0.5, "spatial_score": 0.9}]
    out = apply_spatial_intent_boost(rows, query_type="balanced")
    assert out == rows


def test_no_op_when_rows_empty():
    out = apply_spatial_intent_boost([], query_type="spatial")
    assert out == []


def test_boosts_rrf_score_for_spatial_rows():
    rows = [
        {"id": 1, "rrf_score": 0.5, "spatial_score": 0.8},
        {"id": 2, "rrf_score": 0.5, "spatial_score": 0.0},
    ]
    out = apply_spatial_intent_boost(rows, query_type="spatial", boost=0.10)
    # id=1: 0.5 + 0.10*0.8 = 0.58 ; id=2: 0.5 unchanged
    by_id = {r["id"]: r for r in out}
    assert by_id[1]["rrf_score"] == pytest.approx(0.58)
    assert by_id[2]["rrf_score"] == pytest.approx(0.5)


def test_resort_after_boost_so_spatial_rises():
    rows = [
        {"id": 1, "rrf_score": 0.55, "spatial_score": 0.0},  # no spatial
        {"id": 2, "rrf_score": 0.50, "spatial_score": 0.8},  # spatial → +0.08
    ]
    out = apply_spatial_intent_boost(rows, query_type="spatial", boost=0.10)
    # After boost: id=1 still 0.55, id=2 → 0.58 → id=2 first
    assert out[0]["id"] == 2
    assert out[1]["id"] == 1


def test_preserves_cross_encoder_ordering_when_present():
    """If cross_encoder_score is set, boost both ce and rrf, and sort by ce."""
    rows = [
        {"id": 1, "rrf_score": 0.5, "spatial_score": 0.0, "cross_encoder_score": 0.7},
        {"id": 2, "rrf_score": 0.5, "spatial_score": 0.8, "cross_encoder_score": 0.6},
    ]
    out = apply_spatial_intent_boost(rows, query_type="spatial", boost=0.20)
    # id=1: ce stays 0.7. id=2: ce → 0.6 + 0.20*0.8 = 0.76. id=2 wins.
    assert out[0]["id"] == 2
    assert out[0]["cross_encoder_score"] == pytest.approx(0.76)
    assert out[1]["cross_encoder_score"] == pytest.approx(0.7)


def test_handles_missing_spatial_score_as_zero():
    rows = [{"id": 1, "rrf_score": 0.5}]  # no spatial_score key
    out = apply_spatial_intent_boost(rows, query_type="spatial", boost=0.10)
    # Default to 0 → no change
    assert out[0]["rrf_score"] == pytest.approx(0.5)


def test_negative_spatial_score_ignored():
    """Defensive: spatial_score < 0 shouldn't penalise."""
    rows = [{"id": 1, "rrf_score": 0.5, "spatial_score": -0.3}]
    out = apply_spatial_intent_boost(rows, query_type="spatial", boost=0.10)
    assert out[0]["rrf_score"] == pytest.approx(0.5)
