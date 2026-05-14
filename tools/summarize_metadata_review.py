#!/usr/bin/env python3
"""Summarize human metadata quality review results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_SCORE_VALUES = {"0", "1", "2"}
SCORE_FIELDS = ("overall_alignment", "caption_alignment", "tag_alignment")


def parse_tags(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass
    return [
        item.strip().strip("\"'")
        for item in text.strip("[]").split(",")
        if item.strip().strip("\"'")
    ]


def parse_issue_types(raw: Any) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score_counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "")).strip() or "unreviewed" for row in rows)
    return {key: counts.get(key, 0) for key in ["0", "1", "2", "unreviewed"]}


def reviewed_rows(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get(field, "")).strip() in VALID_SCORE_VALUES]


def build_summary(rows: list[dict[str, str]], *, csv_path: Path) -> dict[str, Any]:
    total = len(rows)
    field_counts = {field: score_counts(rows, field) for field in SCORE_FIELDS}
    issue_counts = Counter()
    tag_counts = Counter()
    low_tag_counts = Counter()
    analysis_status_counts = Counter(row.get("analysis_status", "") or "unknown" for row in rows)
    source_counts = Counter(row.get("source_group", "") or "unknown" for row in rows)

    for row in rows:
        tags = parse_tags(row.get("ai_tags"))
        tag_counts.update(tags)
        if str(row.get("tag_alignment", "")).strip() in {"0", "1"}:
            low_tag_counts.update(tags)
        issue_counts.update(parse_issue_types(row.get("issue_types")))

    reviewed_overall = reviewed_rows(rows, "overall_alignment")
    reviewed_caption = reviewed_rows(rows, "caption_alignment")
    reviewed_tag = reviewed_rows(rows, "tag_alignment")

    def average(field: str, subset: list[dict[str, str]]) -> float | None:
        if not subset:
            return None
        return round(sum(int(row[field]) for row in subset) / len(subset), 4)

    return {
        "schema_version": "metadata_review_summary_v1",
        "csv": str(csv_path),
        "total_count": total,
        "reviewed_counts": {
            "overall": len(reviewed_overall),
            "caption": len(reviewed_caption),
            "tag": len(reviewed_tag),
        },
        "score_counts": field_counts,
        "score_averages": {
            "overall": average("overall_alignment", reviewed_overall),
            "caption": average("caption_alignment", reviewed_caption),
            "tag": average("tag_alignment", reviewed_tag),
        },
        "issue_counts": dict(issue_counts.most_common()),
        "analysis_status_counts": dict(analysis_status_counts.most_common()),
        "source_group_count": len(source_counts),
        "top_source_groups": source_counts.most_common(20),
        "top_tags": tag_counts.most_common(50),
        "top_low_score_tags": low_tag_counts.most_common(50),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Metadata Review Summary",
        "",
        f"- CSV: `{summary['csv']}`",
        f"- total_count: {summary['total_count']}",
        f"- reviewed_overall: {summary['reviewed_counts']['overall']}",
        f"- reviewed_caption: {summary['reviewed_counts']['caption']}",
        f"- reviewed_tag: {summary['reviewed_counts']['tag']}",
        "",
        "## Score Counts",
        "",
    ]
    for field, counts in summary["score_counts"].items():
        lines.append(f"- {field}: {counts}")
    lines.extend([
        "",
        "## Score Averages",
        "",
        f"- overall: {summary['score_averages']['overall']}",
        f"- caption: {summary['score_averages']['caption']}",
        f"- tag: {summary['score_averages']['tag']}",
        "",
        "## Issue Counts",
        "",
    ])
    if summary["issue_counts"]:
        for key, count in summary["issue_counts"].items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Top Tags",
        "",
    ])
    for tag, count in summary["top_tags"][:30]:
        lines.append(f"- {tag}: {count}")
    lines.extend([
        "",
        "## Top Low-Score Tags",
        "",
    ])
    if summary["top_low_score_tags"]:
        for tag, count in summary["top_low_score_tags"][:30]:
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.csv)
    summary = build_summary(rows, csv_path=args.csv)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(args.output_md, summary)
    if not args.output_json and not args.output_md:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Summarized {summary['total_count']} rows from {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
