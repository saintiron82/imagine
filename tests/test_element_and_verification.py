"""Sprint 1 β1: multi-element AND verification."""
from __future__ import annotations

import pytest

from backend.search.sqlite_search import apply_element_verification


def test_no_op_when_elements_empty():
    rows = [{"id": 1, "mc_caption": "anything"}]
    out = apply_element_verification(rows, elements=[])
    assert out == rows


def test_keeps_rows_with_all_elements_in_caption():
    rows = [
        {"id": 1, "mc_caption": "character in a room with window", "rrf_score": 0.5},
        {"id": 2, "mc_caption": "only character here", "rrf_score": 0.5},
    ]
    out = apply_element_verification(rows, elements=["character", "room"], penalty=0.10)
    # Both kept, but the one missing 'room' is penalised → lower rrf_score → ranks lower.
    assert (out[0].get("rrf_score") or 0.0) >= (out[1].get("rrf_score") or 0.0)
    # Specifically: id=1 is full match (no penalty), id=2 misses room (one penalty).
    full_match = next(r for r in out if r["id"] == 1)
    partial = next(r for r in out if r["id"] == 2)
    assert full_match["element_miss_count"] == 0
    assert partial["element_miss_count"] == 1


def test_uses_tags_and_spatial_objects_as_fallback_text():
    rows = [
        {"id": 1, "mc_caption": "", "ai_tags": "room, window",
         "spatial_objects": ["character", "lamp"], "rrf_score": 0.7},
    ]
    out = apply_element_verification(rows, elements=["character", "room"], penalty=0.10)
    assert out[0]["element_match_count"] == 2
    assert out[0]["element_miss_count"] == 0
    # No penalty → rrf_score unchanged
    assert out[0]["rrf_score"] == pytest.approx(0.7)


def test_applies_per_missing_element_penalty():
    rows = [
        {"id": 1, "mc_caption": "character", "rrf_score": 1.0},  # misses 'room'
    ]
    out = apply_element_verification(rows, elements=["character", "room"], penalty=0.10)
    assert out[0]["rrf_score"] == pytest.approx(0.90)
    assert out[0]["element_miss_count"] == 1


def test_korean_and_english_element_both_match():
    rows = [
        {"id": 1, "mc_caption": "character in a room",
         "ai_tags": "캐릭터, 방", "rrf_score": 0.5},
    ]
    out = apply_element_verification(rows, elements=["캐릭터", "방"], penalty=0.10)
    assert out[0]["element_match_count"] == 2
    assert out[0]["element_miss_count"] == 0


def test_resort_after_penalty_so_full_matches_rise():
    rows = [
        {"id": 1, "mc_caption": "character only", "rrf_score": 0.80},
        {"id": 2, "mc_caption": "character in a room", "rrf_score": 0.75},
    ]
    out = apply_element_verification(rows, elements=["character", "room"], penalty=0.10)
    # After penalty: id=1 -> 0.70, id=2 -> 0.75 → id=2 first
    assert out[0]["id"] == 2
    assert out[1]["id"] == 1


def test_handles_missing_text_fields_gracefully():
    """A row with no caption/tags/spatial misses every element."""
    rows = [{"id": 1, "rrf_score": 0.5}]
    out = apply_element_verification(rows, elements=["a", "b"], penalty=0.10)
    assert out[0]["element_match_count"] == 0
    assert out[0]["element_miss_count"] == 2
    assert out[0]["rrf_score"] == pytest.approx(0.3)


def test_empty_string_elements_are_dropped():
    rows = [{"id": 1, "mc_caption": "x", "rrf_score": 0.5}]
    out = apply_element_verification(rows, elements=["", "  ", None], penalty=0.10)
    # All "elements" stripped → empty list → no-op
    assert out == rows
