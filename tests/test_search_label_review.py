import json
from pathlib import Path

import pytest

from tools.finalize_search_label_review import build_label_rows


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_finalize_search_label_review_skips_unreviewed_rows(tmp_path: Path):
    review_path = tmp_path / "review.jsonl"
    write_jsonl(review_path, [
        {
            "query_id": "q1",
            "item_id": "10",
            "reviewer_relevance": 2,
            "reviewer_id": "r1",
        },
        {
            "query_id": "q1",
            "item_id": "11",
            "reviewer_relevance": None,
            "reviewer_id": "r1",
        },
    ])

    rows = build_label_rows(review_path, label_version="scoped_gold_v1")

    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["item_id"] == "10"
    assert rows[0]["relevance"] == 2
    assert rows[0]["label_source"] == "human"
    assert rows[0]["label_version"] == "scoped_gold_v1"
    assert rows[0]["reviewer_id"] == "r1"


def test_finalize_search_label_review_supports_csv(tmp_path: Path):
    review_path = tmp_path / "review.csv"
    review_path.write_text(
        "query_id,item_id,reviewer_relevance,reviewer_id\n"
        "q1,10,1,\n",
        encoding="utf-8",
    )

    rows = build_label_rows(
        review_path,
        label_version="scoped_gold_v1",
        reviewer_id="fallback-reviewer",
    )

    assert len(rows) == 1
    assert rows[0]["relevance"] == 1
    assert rows[0]["reviewer_id"] == "fallback-reviewer"


def test_finalize_search_label_review_can_require_all_rows(tmp_path: Path):
    review_path = tmp_path / "review.jsonl"
    write_jsonl(review_path, [
        {"query_id": "q1", "item_id": "10", "reviewer_relevance": ""},
    ])

    with pytest.raises(ValueError, match="reviewer_relevance is required"):
        build_label_rows(review_path, label_version="scoped_gold_v1", require_all=True)
