import csv
from pathlib import Path

import pytest

from tools.serve_search_label_review import build_gallery_payload, update_review_fields


def write_review_csv(path: Path):
    path.write_text(
        "query_id,item_id,reviewer_relevance,reviewer_id\n"
        "q1,10,,\n"
        "q1,11,2,r1\n",
        encoding="utf-8",
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_update_review_fields_adds_caption_alignment_columns(tmp_path: Path):
    csv_path = tmp_path / "review.csv"
    write_review_csv(csv_path)

    result = update_review_fields(
        csv_path,
        query_id="q1",
        item_id="10",
        updates={"caption_alignment": "2"},
        reviewer_id="meta-reviewer",
    )
    rows = read_rows(csv_path)

    assert result["caption_alignment"] == "2"
    assert result["caption_reviewed_count"] == 1
    assert rows[0]["caption_alignment"] == "2"
    assert rows[0]["caption_alignment_reviewer_id"] == "meta-reviewer"
    assert rows[0]["reviewer_relevance"] == ""


def test_update_review_fields_can_update_search_and_caption_labels(tmp_path: Path):
    csv_path = tmp_path / "review.csv"
    write_review_csv(csv_path)

    result = update_review_fields(
        csv_path,
        query_id="q1",
        item_id="10",
        updates={"reviewer_relevance": "1", "caption_alignment": "0"},
        reviewer_id="manual",
    )
    payload = build_gallery_payload(csv_path)

    assert result["reviewer_relevance"] == "1"
    assert result["caption_alignment"] == "0"
    assert payload["reviewed_count"] == 2
    assert payload["caption_reviewed_count"] == 1


def test_update_review_fields_rejects_unknown_fields(tmp_path: Path):
    csv_path = tmp_path / "review.csv"
    write_review_csv(csv_path)

    with pytest.raises(ValueError, match="unsupported review field"):
        update_review_fields(
            csv_path,
            query_id="q1",
            item_id="10",
            updates={"unknown": "2"},
        )
