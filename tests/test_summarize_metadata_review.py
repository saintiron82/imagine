import csv
from pathlib import Path

from tools.summarize_metadata_review import build_summary, read_csv


def test_build_summary_counts_scores_issues_and_tags(tmp_path: Path):
    path = tmp_path / "review.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "overall_alignment",
                "caption_alignment",
                "tag_alignment",
                "issue_types",
                "ai_tags",
                "analysis_status",
                "source_group",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "overall_alignment": "2",
            "caption_alignment": "2",
            "tag_alignment": "1",
            "issue_types": "tag_missing_key,too_generic",
            "ai_tags": '["classroom", "school"]',
            "analysis_status": "ok",
            "source_group": "A",
        })
        writer.writerow({
            "overall_alignment": "",
            "caption_alignment": "0",
            "tag_alignment": "0",
            "issue_types": "caption_wrong",
            "ai_tags": "forest, night",
            "analysis_status": "partial",
            "source_group": "B",
        })

    rows = read_csv(path)
    summary = build_summary(rows, csv_path=path)

    assert summary["total_count"] == 2
    assert summary["reviewed_counts"]["overall"] == 1
    assert summary["score_counts"]["tag_alignment"]["0"] == 1
    assert summary["score_counts"]["tag_alignment"]["1"] == 1
    assert summary["issue_counts"]["tag_missing_key"] == 1
    assert summary["issue_counts"]["caption_wrong"] == 1
    assert ("classroom", 1) in summary["top_tags"]
    assert ("forest", 1) in summary["top_low_score_tags"]
