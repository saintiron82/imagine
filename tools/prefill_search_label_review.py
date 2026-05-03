#!/usr/bin/env python3
"""Prefill search review relevance with conservative metadata heuristics.

The output is not a gold LabelSet. It is an assisted review sheet that a human
must verify before finalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_search_label_review import write_csv, write_jsonl
from tools.evaluate_search_quality import read_jsonl


TERM_ALIASES = {
    "소파": ("sofa", "couch", "leather sofa", "tufted sofa"),
    "창문": ("window", "windows", "window frame", "window light", "blinds", "glass pane", "sliding glass"),
    "하늘": ("sky", "cloud", "clouds", "star", "stars", "moon"),
    "밤": ("night", "dark", "starry", "stars", "moon"),
    "주방": ("kitchen", "refrigerator", "sink", "cabinet", "cabinets", "countertop", "utensils", "stove"),
    "커튼": ("curtain", "curtains", "drape", "drapes", "blinds"),
    "숲": ("forest", "trees", "tree", "woods", "woodland"),
    "fog": ("fog", "foggy", "mist", "misty", "shrouded"),
}

SCOPE_RE = re.compile(r"^\s*(?P<scope>.+?)\s*(?:중에서|중에|에서)\s*(?P<body>.+)$")


def read_review_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported review file type: {path.suffix}; expected .jsonl or .csv")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _path_parts(*values: Any) -> set[str]:
    parts: set[str] = set()
    for value in values:
        text = _normalize_text(value).replace("\\", "/")
        for part in text.split("/"):
            part = part.strip()
            if part:
                parts.add(part)
    return parts


def parse_query(query_text: str) -> tuple[str, list[str]]:
    match = SCOPE_RE.match(query_text)
    if not match:
        return "", []
    scope = match.group("scope").strip(" \t\r\n\"'`“”‘’[](){}")
    body = match.group("body")
    body = re.sub(r"\s*있는\s*이미지.*$", "", body).strip()
    terms = [
        term.strip(" \t\r\n\"'`“”‘’[](){}")
        for term in re.split(r"\s*(?:과|와|and|,|/)\s*", body)
        if term.strip(" \t\r\n\"'`“”‘’[](){}")
    ]
    return scope, terms


def scope_status(row: dict[str, Any], scope: str) -> str:
    if not scope:
        return "none"
    scope_key = scope.lower()
    parts = _path_parts(row.get("folder_path"), row.get("relative_path"), row.get("file_path"))
    if scope_key in parts:
        return "exact"

    path_text = " ".join(
        _normalize_text(row.get(field))
        for field in ("folder_path", "relative_path", "file_path")
    )
    if scope_key and scope_key in path_text:
        return "substring"
    return "none"


def _metadata_text(row: dict[str, Any]) -> str:
    return " ".join(
        _normalize_text(row.get(field))
        for field in ("mc_caption", "ai_tags", "image_type", "scene_type", "time_of_day", "weather", "file_name")
    )


def matched_terms(row: dict[str, Any], terms: list[str]) -> list[str]:
    text = _metadata_text(row)
    matched = []
    for term in terms:
        aliases = TERM_ALIASES.get(term, (term,))
        if any(alias.lower() in text for alias in aliases):
            matched.append(term)
    return matched


def prefill_row(row: dict[str, Any], *, reviewer_id: str) -> dict[str, Any]:
    scope, terms = parse_query(str(row.get("query_text", "")))
    scope_match = scope_status(row, scope)
    hits = matched_terms(row, terms)

    if scope and scope_match != "exact":
        relevance = 0
    elif terms and len(hits) == len(terms):
        relevance = 2
    elif hits:
        relevance = 1
    else:
        relevance = 0

    note = (
        f"assisted_prefill; scope={scope or '-'}:{scope_match}; "
        f"matched_terms={','.join(hits) or '-'}; "
        f"missing_terms={','.join(term for term in terms if term not in hits) or '-'}"
    )
    existing_note = str(row.get("review_notes") or "").strip()
    if existing_note:
        note = f"{existing_note} | {note}"

    return {
        **row,
        "reviewer_relevance": relevance,
        "reviewer_id": reviewer_id,
        "review_notes": note,
    }


def prefill_rows(rows: list[dict[str, Any]], *, reviewer_id: str) -> list[dict[str, Any]]:
    return [prefill_row(row, reviewer_id=reviewer_id) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="review task JSONL or CSV")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--reviewer-id", default="assisted_prefill_v1")
    args = parser.parse_args(argv)

    try:
        rows = prefill_rows(read_review_rows(args.review), reviewer_id=args.reviewer_id)
        write_jsonl(args.output_jsonl, rows)
        write_csv(args.output_csv, rows)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} prefilled review tasks: {args.output_jsonl}")
    print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
