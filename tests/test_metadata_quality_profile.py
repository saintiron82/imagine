import csv
import json
from pathlib import Path

from tools.build_metadata_quality_profile import (
    build_profile,
    build_signals,
    read_csv,
    write_jsonl,
)


def _write_review_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "caption_alignment",
                "tag_alignment",
                "issue_types",
                "ai_tags",
                "analysis_status",
                "source_group",
                "reviewer_id",
                "reviewed_at",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "item_id": "1",
            "caption_alignment": "2",
            "tag_alignment": "2",
            "issue_types": "",
            "ai_tags": '["window", "room"]',
            "analysis_status": "ok",
            "source_group": "work/a",
            "reviewer_id": "manual_review",
            "reviewed_at": "2026-05-13T10:00:00",
        })
        writer.writerow({
            "item_id": "2",
            "caption_alignment": "1",
            "tag_alignment": "0",
            "issue_types": "tag_false_positive,caption_wrong",
            "ai_tags": '["monster", "room"]',
            "analysis_status": "legacy_warning",
            "source_group": "work/b",
            "reviewer_id": "manual_review",
            "reviewed_at": "2026-05-13T10:01:00",
        })
        writer.writerow({
            "item_id": "3",
            "caption_alignment": "",
            "tag_alignment": "",
            "issue_types": "",
            "ai_tags": '["unused"]',
            "analysis_status": "ok",
            "source_group": "work/c",
            "reviewer_id": "",
            "reviewed_at": "",
        })


def test_build_profile_summarizes_reviewed_rows_as_reliability(tmp_path: Path):
    csv_path = tmp_path / "metadata_review_sample.csv"
    _write_review_csv(csv_path)

    rows = read_csv(csv_path)
    profile = build_profile(rows, csv_path=csv_path)

    assert profile["total_count"] == 3
    assert profile["reviewed_count"] == 2
    assert profile["global"]["caption_reliability"] == 0.75
    assert profile["global"]["tag_reliability"] == 0.5
    assert profile["issues"]["tag_false_positive"]["count"] == 1
    assert profile["analysis_status"]["legacy_warning"]["tag_reliability"] == 0.0
    assert profile["tags"]["monster"]["low_count"] == 1
    assert profile["tags"]["room"]["reviewed_count"] == 2


def test_build_signals_writes_item_level_review_strength(tmp_path: Path):
    csv_path = tmp_path / "metadata_review_sample.csv"
    out_path = tmp_path / "metadata_quality_signals.jsonl"
    _write_review_csv(csv_path)

    signals = build_signals(read_csv(csv_path))
    write_jsonl(out_path, signals)

    assert [signal["item_id"] for signal in signals] == ["1", "2"]
    assert signals[0]["metadata_reliability_score"] == 1.0
    assert signals[1]["metadata_reliability_score"] == 0.25
    assert "tag_false_positive" in signals[1]["issue_types"]

    written = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert written == signals
