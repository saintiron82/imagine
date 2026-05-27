"""Phase C: per-query benchmark outcome classification."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from bench_classify_outcome import classify, Outcome  # noqa: E402


def test_top1_in_gt_is_found():
    assert classify(top_k=[1, 2, 3], gt={1, 4}, system_confidence="medium") is Outcome.FOUND


def test_top_k_intersecting_gt_anywhere_is_found():
    assert classify(top_k=[7, 8, 1], gt={1, 4}, system_confidence="low") is Outcome.FOUND


def test_no_overlap_but_gt_present_is_missed():
    assert classify(top_k=[5, 6, 7], gt={1, 2}, system_confidence="medium") is Outcome.MISSED


def test_empty_response_and_no_gt_is_honest_empty():
    assert classify(top_k=[], gt=set(), system_confidence="empty") is Outcome.HONEST_EMPTY


def test_answer_when_gt_empty_is_false_answer():
    assert classify(top_k=[5, 6], gt=set(), system_confidence="medium") is Outcome.FALSE_ANSWER


def test_empty_confidence_with_present_gt_is_still_missed():
    # System said "no" but there were relevant items — that's a miss.
    assert classify(top_k=[], gt={1, 2}, system_confidence="empty") is Outcome.MISSED


def test_outcome_values_are_lower_snake_for_json_logging():
    """Outcome must serialize to the documented strings."""
    assert Outcome.FOUND.value == "found"
    assert Outcome.MISSED.value == "missed"
    assert Outcome.HONEST_EMPTY.value == "honest_empty"
    assert Outcome.FALSE_ANSWER.value == "false_answer"


def test_classify_treats_none_confidence_as_non_empty():
    """Defensive: if system_confidence is missing/None, fall back to 'medium-ish'."""
    # With GT present and no overlap → missed (not affected by confidence)
    assert classify(top_k=[5], gt={1}, system_confidence=None) is Outcome.MISSED
    # With GT empty and a returned result → false_answer (confidence != empty)
    assert classify(top_k=[5], gt=set(), system_confidence=None) is Outcome.FALSE_ANSWER
