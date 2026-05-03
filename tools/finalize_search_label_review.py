#!/usr/bin/env python3
"""Convert completed search review tasks into a Search Evaluation LabelSet."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_search_quality import read_jsonl, require_fields


def _read_review_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported review file type: {path.suffix}; expected .jsonl or .csv")


def _parse_relevance(value: Any, *, require_all: bool, row_id: str) -> int | None:
    if value is None or str(value).strip() == "":
        if require_all:
            raise ValueError(f"{row_id}: reviewer_relevance is required")
        return None
    relevance = int(str(value).strip())
    if relevance not in (0, 1, 2):
        raise ValueError(f"{row_id}: reviewer_relevance must be 0, 1, or 2")
    return relevance


def build_label_rows(
    review_path: Path,
    *,
    label_version: str,
    label_source: str = "human",
    reviewer_id: str = "",
    require_all: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for idx, row in enumerate(_read_review_rows(review_path), start=1):
        require_fields(row, ("query_id", "item_id", "reviewer_relevance"), review_path, idx)
        row_id = f"{review_path}:{idx}"
        relevance = _parse_relevance(row["reviewer_relevance"], require_all=require_all, row_id=row_id)
        if relevance is None:
            continue
        rows.append({
            "query_id": str(row["query_id"]),
            "item_id": str(row["item_id"]),
            "relevance": relevance,
            "label_source": label_source,
            "label_version": label_version,
            "reviewer_id": str(row.get("reviewer_id") or reviewer_id),
            "reviewed_at": reviewed_at,
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="completed review JSONL or CSV")
    parser.add_argument("--output-labels", type=Path, required=True, help="LabelSet JSONL output path")
    parser.add_argument("--label-version", required=True, help="label_version value, e.g. scoped_gold_v1")
    parser.add_argument("--label-source", default="human", choices=("human", "adjudicated"))
    parser.add_argument("--reviewer-id", default="")
    parser.add_argument("--require-all", action="store_true", help="fail if any row lacks reviewer_relevance")
    args = parser.parse_args(argv)

    try:
        rows = build_label_rows(
            args.review,
            label_version=args.label_version,
            label_source=args.label_source,
            reviewer_id=args.reviewer_id,
            require_all=args.require_all,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_jsonl(args.output_labels, rows)
    print(f"Wrote {len(rows)} labels: {args.output_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
