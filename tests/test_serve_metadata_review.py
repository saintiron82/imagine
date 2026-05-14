import csv
from pathlib import Path

import pytest

from tools.serve_metadata_review import build_gallery_payload, update_review_fields


def write_sample(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "item_id",
                "file_name",
                "thumbnail_url",
                "caption_alignment",
                "tag_alignment",
                "overall_alignment",
                "issue_types",
                "reviewer_notes",
                "reviewer_id",
                "reviewed_at",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "sample_id": "metadata-quality-v1-0001",
            "item_id": "10",
            "file_name": "a.psd",
            "thumbnail_url": "",
            "caption_alignment": "",
            "tag_alignment": "",
            "overall_alignment": "",
            "issue_types": "",
            "reviewer_notes": "",
            "reviewer_id": "",
            "reviewed_at": "",
        })


def test_update_review_fields_writes_scores_and_text(tmp_path: Path):
    csv_path = tmp_path / "review.csv"
    write_sample(csv_path)

    result = update_review_fields(
        csv_path,
        sample_id="metadata-quality-v1-0001",
        item_id="10",
        updates={
            "caption_alignment": "2",
            "tag_alignment": "1",
            "issue_types": "tag_missing_key",
            "reviewer_notes": "missing moon",
        },
        reviewer_id="tester",
    )

    assert result["ok"] is True
    assert result["caption_reviewed_count"] == 1
    assert result["tag_reviewed_count"] == 1

    with csv_path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert row["caption_alignment"] == "2"
    assert row["tag_alignment"] == "1"
    assert row["issue_types"] == "tag_missing_key"
    assert row["reviewer_notes"] == "missing moon"
    assert row["reviewer_id"] == "tester"
    assert row["reviewed_at"]


def test_update_review_fields_rejects_bad_score(tmp_path: Path):
    csv_path = tmp_path / "review.csv"
    write_sample(csv_path)

    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        update_review_fields(
            csv_path,
            sample_id="metadata-quality-v1-0001",
            item_id="10",
            updates={"overall_alignment": "9"},
        )


def test_build_gallery_payload_counts_reviewed(tmp_path: Path):
    csv_path = tmp_path / "review.csv"
    write_sample(csv_path)
    update_review_fields(
        csv_path,
        sample_id="metadata-quality-v1-0001",
        item_id="10",
        updates={"overall_alignment": "2"},
    )

    payload = build_gallery_payload(csv_path)

    assert payload["ok"] is True
    assert payload["total_count"] == 1
    assert payload["overall_reviewed_count"] == 1
    assert payload["rows"][0]["_csv_line"] == 2
