import json
from pathlib import Path

import pytest

from tools.build_background_complex_queryset import (
    NON_BACKGROUND_TERMS,
    QUERY_SPECS,
    QuerySpec,
    VALID_DIFFICULTIES,
    build_rows,
    main,
    validate_specs,
)
from tools.evaluate_search_quality import load_queries


def test_background_complex_queryset_is_valid_for_evaluator(tmp_path: Path):
    rows = build_rows(created_at="2026-05-06T00:00:00+09:00")
    path = tmp_path / "queryset.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    loaded = load_queries(path)

    assert len(rows) == 40
    assert len(loaded) == 40
    assert all(row["query_type"] == "complex" for row in rows)
    assert all(row["locale"] == "ko-KR" for row in rows)


def test_background_complex_queryset_uses_diverse_complex_conditions():
    rows = build_rows()
    intents = {tuple(sorted(row["must_terms"])) for row in rows}
    scopes = {row["scope"] for row in rows}
    positive_terms = {
        term.lower()
        for row in rows
        for term in (*row["must_terms"], *row["soft_terms"])
    }

    assert len(intents) == len(rows)
    assert len(scopes) >= 20
    assert all(len(row["must_terms"]) >= 2 for row in rows)
    assert not (positive_terms & NON_BACKGROUND_TERMS)


def test_background_complex_queryset_has_difficulty_distribution():
    rows = build_rows()
    difficulties = {row["difficulty"] for row in rows}

    assert difficulties == VALID_DIFFICULTIES
    assert sum(1 for row in rows if row["difficulty"] == "extreme") >= 5
    assert all(row["exclude_terms"] for row in rows if row["difficulty"] == "extreme")


def test_background_complex_queryset_rejects_duplicate_intent():
    specs = (
        QuerySpec("q1", "밤하늘과 달이 있는 배경", "a", ("밤하늘", "달")),
        QuerySpec("q2", "달과 밤하늘이 있는 배경", "b", ("달", "밤하늘")),
    )

    with pytest.raises(ValueError, match="duplicate intent"):
        validate_specs(specs)


def test_background_complex_queryset_rejects_non_background_terms():
    specs = (
        QuerySpec("q1", "하늘과 인물이 있는 배경", "a", ("하늘", "인물")),
    )

    with pytest.raises(ValueError, match="non-background"):
        validate_specs(specs)


def test_background_complex_queryset_allows_non_background_exclusions():
    specs = (
        QuerySpec(
            "q1",
            "하늘과 성이 있지만 캐릭터는 없는 배경",
            "a",
            ("하늘", "성"),
            exclude_terms=("캐릭터",),
        ),
    )

    validate_specs(specs)


def test_background_complex_queryset_rejects_simple_queries():
    specs = (
        QuerySpec("q1", "하늘이 있는 배경", "a", ("하늘",)),
    )

    with pytest.raises(ValueError, match="at least 2 must_terms"):
        validate_specs(specs)


def test_background_complex_queryset_cli_writes_artifacts(tmp_path: Path):
    output_dir = tmp_path / "background_complex"

    assert main(["--output-dir", str(output_dir)]) == 0

    queryset_path = output_dir / "queryset.jsonl"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    questions = (output_dir / "questions.md").read_text(encoding="utf-8")

    assert manifest["query_count"] == len(QUERY_SPECS)
    assert manifest["query_type"] == "complex"
    assert len(load_queries(queryset_path)) == len(QUERY_SPECS)
    assert "bg-complex-q0001" in questions
