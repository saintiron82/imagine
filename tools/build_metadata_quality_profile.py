#!/usr/bin/env python3
"""Build search/extraction reliability signals from metadata review rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "benchmarks" / "reviews" / "metadata_quality_v1_20260510"
DEFAULT_CSV = DEFAULT_REVIEW_DIR / "metadata_review_sample.csv"
DEFAULT_PROFILE = DEFAULT_REVIEW_DIR / "metadata_quality_profile.json"
DEFAULT_SIGNALS = DEFAULT_REVIEW_DIR / "metadata_quality_signals.jsonl"
VALID_SCORE_VALUES = {"0", "1", "2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_tags(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]
    except Exception:
        pass
    return [
        item.strip().strip("\"'").lower()
        for item in text.strip("[]").split(",")
        if item.strip().strip("\"'")
    ]


def parse_issues(raw: Any) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def score_value(raw: Any) -> int | None:
    text = str(raw or "").strip()
    if text not in VALID_SCORE_VALUES:
        return None
    return int(text)


def score_reliability(raw: Any) -> float | None:
    value = score_value(raw)
    if value is None:
        return None
    return round(value / 2.0, 4)


def is_reviewed(row: dict[str, Any]) -> bool:
    return (
        score_value(row.get("caption_alignment")) is not None
        or score_value(row.get("tag_alignment")) is not None
    )


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part / total, 4)


def _profile_created_at(rows: list[dict[str, Any]]) -> str:
    reviewed_at_values = sorted(
        str(row.get("reviewed_at") or "").strip()
        for row in rows
        if str(row.get("reviewed_at") or "").strip()
    )
    if reviewed_at_values:
        return reviewed_at_values[-1]
    return datetime.now(timezone.utc).isoformat()


def _score_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field) or "").strip() or "unreviewed" for row in rows)
    return {key: counts.get(key, 0) for key in ["0", "1", "2", "unreviewed"]}


def build_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for row in rows:
        if not is_reviewed(row):
            continue
        caption_rel = score_reliability(row.get("caption_alignment"))
        tag_rel = score_reliability(row.get("tag_alignment"))
        reliabilities = [value for value in [caption_rel, tag_rel] if value is not None]
        signal = {
            "schema_version": "metadata_quality_signal_v1",
            "item_id": str(row.get("item_id") or ""),
            "caption_reliability": caption_rel,
            "tag_reliability": tag_rel,
            "metadata_reliability_score": _avg(reliabilities),
            "issue_types": parse_issues(row.get("issue_types")),
            "analysis_status": str(row.get("analysis_status") or "unknown"),
            "source_group": str(row.get("source_group") or ""),
            "reviewer_id": str(row.get("reviewer_id") or ""),
            "reviewed_at": str(row.get("reviewed_at") or ""),
        }
        signals.append(signal)
    return signals


def _field_reliability(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        score_reliability(row.get(field))
        for row in rows
        if score_reliability(row.get(field)) is not None
    ]
    return _avg([value for value in values if value is not None])


def _metadata_reliability(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        row_values = [
            value
            for value in [
                score_reliability(row.get("caption_alignment")),
                score_reliability(row.get("tag_alignment")),
            ]
            if value is not None
        ]
        if row_values:
            values.append(sum(row_values) / len(row_values))
    return _avg(values)


def _group_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = [row for row in rows if is_reviewed(row)]
    caption_reviewed = sum(1 for row in rows if score_value(row.get("caption_alignment")) is not None)
    tag_reviewed = sum(1 for row in rows if score_value(row.get("tag_alignment")) is not None)
    return {
        "total_count": len(rows),
        "reviewed_count": len(reviewed),
        "caption_reviewed_count": caption_reviewed,
        "tag_reviewed_count": tag_reviewed,
        "caption_reliability": _field_reliability(rows, "caption_alignment"),
        "tag_reliability": _field_reliability(rows, "tag_alignment"),
        "metadata_reliability": _metadata_reliability(rows),
        "reviewed_ratio": _ratio(len(reviewed), len(rows)),
    }


def build_profile(rows: list[dict[str, Any]], *, csv_path: Path) -> dict[str, Any]:
    reviewed_rows = [row for row in rows if is_reviewed(row)]
    issues = Counter()
    status_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tag_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    low_tag_counts = Counter()
    false_positive_tags = Counter()

    for row in rows:
        status_rows[str(row.get("analysis_status") or "unknown")].append(row)
        if not is_reviewed(row):
            continue
        row_issues = parse_issues(row.get("issue_types"))
        issues.update(row_issues)
        tag_low = score_value(row.get("tag_alignment")) in {0, 1}
        tag_false_positive = "tag_false_positive" in row_issues
        for tag in parse_tags(row.get("ai_tags")):
            tag_rows[tag].append(row)
            if tag_low:
                low_tag_counts[tag] += 1
            if tag_false_positive:
                false_positive_tags[tag] += 1

    tags: dict[str, Any] = {}
    for tag, tag_review_rows in sorted(tag_rows.items()):
        tags[tag] = {
            **_group_profile(tag_review_rows),
            "low_count": low_tag_counts[tag],
            "low_ratio": _ratio(low_tag_counts[tag], len(tag_review_rows)),
            "false_positive_count": false_positive_tags[tag],
            "false_positive_ratio": _ratio(false_positive_tags[tag], len(tag_review_rows)),
        }

    profile = {
        "schema_version": "metadata_quality_profile_v1",
        "created_at": _profile_created_at(rows),
        "csv": str(csv_path),
        "total_count": len(rows),
        "reviewed_count": len(reviewed_rows),
        "global": _group_profile(rows),
        "score_counts": {
            "caption_alignment": _score_counts(rows, "caption_alignment"),
            "tag_alignment": _score_counts(rows, "tag_alignment"),
            "overall_alignment": _score_counts(rows, "overall_alignment"),
        },
        "issues": {
            issue: {
                "count": count,
                "reviewed_ratio": _ratio(count, len(reviewed_rows)),
            }
            for issue, count in issues.most_common()
        },
        "analysis_status": {
            status: _group_profile(group_rows)
            for status, group_rows in sorted(status_rows.items())
        },
        "tags": tags,
        "limits": {
            "runtime_tag_min_reviewed": 30,
            "runtime_status_min_reviewed": 20,
            "recommended_initial_rank_weight": 0.0,
            "recommended_max_rank_weight_after_validation": 0.05,
        },
    }
    return profile


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output-signals", type=Path, default=DEFAULT_SIGNALS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.csv)
    profile = build_profile(rows, csv_path=args.csv)
    signals = build_signals(rows)
    write_json(args.output_profile, profile)
    write_jsonl(args.output_signals, signals)
    print(
        f"Wrote metadata quality profile for {profile['reviewed_count']}/"
        f"{profile['total_count']} reviewed rows"
    )
    print(f"Profile: {args.output_profile}")
    print(f"Signals: {args.output_signals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
