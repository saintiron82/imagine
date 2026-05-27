"""Phase B: ConstraintPlan validation."""
from __future__ import annotations

import pytest

from backend.search.constraint_plan import (
    ConstraintPlan,
    ConstraintPlanError,
    from_decomposer_output,
)


def test_valid_payload_roundtrips():
    raw = {
        "folder": "#07",
        "elements": ["캐릭터", "방"],
        "negatives": [],
        "vector_query": "character in a room",
        "query_type": "balanced",
    }
    plan = from_decomposer_output(raw)
    assert isinstance(plan, ConstraintPlan)
    assert plan.folder == "#07"
    assert list(plan.elements) == ["캐릭터", "방"]
    assert list(plan.negatives) == []


def test_missing_required_field_raises():
    raw = {"folder": "#07"}  # no vector_query
    with pytest.raises(ConstraintPlanError):
        from_decomposer_output(raw)


def test_unknown_query_type_rejected():
    raw = {
        "folder": "",
        "elements": [],
        "negatives": [],
        "vector_query": "x",
        "query_type": "nonsense",
    }
    with pytest.raises(ConstraintPlanError):
        from_decomposer_output(raw)


def test_elements_strip_empty_strings():
    raw = {
        "folder": "",
        "elements": ["a", "", " "],
        "negatives": ["", "x"],
        "vector_query": "x",
        "query_type": "visual",
    }
    plan = from_decomposer_output(raw)
    assert list(plan.elements) == ["a"]
    assert list(plan.negatives) == ["x"]


def test_to_dict_is_stable_for_logging():
    plan = ConstraintPlan(
        folder="", elements=("a",), negatives=(),
        vector_query="x", query_type="visual", confidence=0.7,
    )
    d = plan.to_dict()
    assert set(d.keys()) == {
        "folder", "elements", "negatives", "vector_query", "query_type", "confidence",
    }
    assert d["elements"] == ["a"]   # to_dict returns lists, not tuples
    assert d["folder"] == ""


def test_empty_vector_query_rejected():
    raw = {
        "folder": "",
        "elements": [],
        "negatives": [],
        "vector_query": "   ",  # whitespace only
        "query_type": "visual",
    }
    with pytest.raises(ConstraintPlanError):
        from_decomposer_output(raw)


def test_confidence_clamped_to_unit_interval():
    raw = {
        "folder": "",
        "elements": [],
        "negatives": [],
        "vector_query": "x",
        "query_type": "visual",
        "confidence": 1.5,  # out of range
    }
    plan = from_decomposer_output(raw)
    assert plan.confidence == 1.0

    raw["confidence"] = -0.2
    plan = from_decomposer_output(raw)
    assert plan.confidence == 0.0


def test_confidence_falls_back_to_zero_when_non_numeric():
    raw = {
        "folder": "",
        "elements": [],
        "negatives": [],
        "vector_query": "x",
        "query_type": "visual",
        "confidence": "not-a-number",
    }
    plan = from_decomposer_output(raw)
    assert plan.confidence == 0.0


def test_confidence_nan_or_inf_falls_back_to_zero():
    for bad in (float("nan"), float("inf"), float("-inf")):
        plan = from_decomposer_output({
            "vector_query": "x", "query_type": "visual",
            "folder": "", "elements": [], "negatives": [],
            "confidence": bad,
        })
        assert plan.confidence == 0.0, f"failed for {bad!r}"


def test_confidence_boolean_falls_back_to_zero():
    for bad in (True, False):
        plan = from_decomposer_output({
            "vector_query": "x", "query_type": "visual",
            "folder": "", "elements": [], "negatives": [],
            "confidence": bad,
        })
        assert plan.confidence == 0.0, f"failed for {bad!r}"
