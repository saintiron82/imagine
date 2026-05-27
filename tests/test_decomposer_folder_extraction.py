"""Phase B: Decomposer normalises to ConstraintPlan."""
from __future__ import annotations

import json

import pytest


def _decomposer(monkeypatch, llm_raw):
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)
    monkeypatch.setattr(decomp, "_generate_llm", lambda q: llm_raw)
    return decomp


def test_folder_prefix_in_korean_query_is_extracted_into_constraint_plan(monkeypatch):
    llm_raw = json.dumps({
        "folder": "#07",
        "elements": ["캐릭터", "방"],
        "negatives": [],
        "vector_query": "character in a room",
        "query_type": "balanced",
    })
    decomp = _decomposer(monkeypatch, llm_raw)

    plan = decomp.decompose_plan("#07에서 캐릭터과 방 있는 이미지")
    assert plan.folder == "#07"
    assert set(plan.elements) == {"캐릭터", "방"}


def test_decomposer_retries_once_on_schema_failure(monkeypatch):
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)

    bad = "not-json"
    good = json.dumps({
        "folder": "",
        "elements": ["x"],
        "negatives": [],
        "vector_query": "x",
        "query_type": "visual",
    })
    calls = {"n": 0}

    def fake(_q):
        calls["n"] += 1
        return good if calls["n"] > 1 else bad

    monkeypatch.setattr(decomp, "_generate_llm", fake)

    plan = decomp.decompose_plan("x")
    assert calls["n"] == 2
    assert plan.vector_query == "x"


def test_decomposer_falls_back_when_retry_also_fails(monkeypatch):
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)
    monkeypatch.setattr(decomp, "_generate_llm", lambda q: "still-bad")

    plan = decomp.decompose_plan("아무 쿼리")
    # fallback never raises — it returns a degraded but valid ConstraintPlan
    assert plan.vector_query  # non-empty
    assert plan.query_type in {"visual", "keyword", "semantic", "balanced"}


def test_decomposer_existing_decompose_method_still_works(monkeypatch):
    """B2 must NOT break the legacy decompose() method that other callers use."""
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)
    monkeypatch.setattr(
        decomp, "_generate_llm",
        lambda q: json.dumps({
            "vector_query": "x",
            "fts_keywords": ["a"],
            "exclude_keywords": [],
            "negative_query": "",
            "filters": {},
            "query_type": "visual",
        }),
    )

    result = decomp.decompose("아무 쿼리")
    # Legacy dict shape preserved
    assert "vector_query" in result
    assert "fts_keywords" in result
    assert result["decomposed"] is True
