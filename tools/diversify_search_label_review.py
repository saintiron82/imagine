#!/usr/bin/env python3
"""Create a source-diverse search label review sheet from existing review rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_search_label_review import write_jsonl  # noqa: E402


DIVERSITY_FIELDS = [
    "source_group",
    "visual_group",
    "diversity_note",
]
NON_BACKGROUND_QUERY_TERMS = {
    "캐릭터",
    "character",
    "characters",
    "portrait",
    "인물",
    "사람",
    "검",
    "sword",
    "armor",
    "갑옷",
}
NEGATION_TERMS = ("없는", "없이", "아닌", "제외", "배제")
RANK_STRATA = ((1, 5), (6, 10), (11, 20), (21, 50), (51, 999999))


def _has_bad_caption(row: dict[str, Any]) -> bool:
    caption = str(row.get("mc_caption") or "").strip()
    return not caption or caption.lower() == "unknown"


def _has_bad_tags(row: dict[str, Any]) -> bool:
    tags = str(row.get("ai_tags") or "").strip()
    return not tags or tags == "[]"


def is_repair_required(row: dict[str, Any]) -> bool:
    status = str(row.get("metadata_status") or "").strip()
    if status:
        return status == "repair_required"
    return _has_bad_caption(row) and _has_bad_tags(row)


def read_review_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported review file: {path}")


def _rank_key(row: dict[str, Any]) -> tuple[int, str]:
    try:
        rank = int(str(row.get("best_rank") or "999999"))
    except ValueError:
        rank = 999999
    return rank, str(row.get("item_id") or "")


_SCOPE_QUERY_RE = re.compile(r"^\s*(?P<scope>.+?)(?:중에서|중에|에서)\s*(?P<body>.+)$")


def query_terms(row: dict[str, Any]) -> list[str]:
    query_text = str(row.get("query_text") or "").strip()
    match = _SCOPE_QUERY_RE.match(query_text)
    body = match.group("body") if match else query_text
    body = re.sub(r"\s*있는\s*이미지.*$", "", body).strip()
    return [
        term.strip(" \t\r\n\"'`“”‘’[](){}").lower()
        for term in re.split(r"\s*(?:과|와|and|,|/)\s*", body)
        if term.strip(" \t\r\n\"'`“”‘’[](){}")
    ]


def query_intent_key(row: dict[str, Any]) -> str:
    """Group queries by requested condition terms, ignoring scope wording."""
    terms = query_terms(row)
    if not terms:
        return str(row.get("query_text") or "").strip().lower()
    return "|".join(sorted(terms))


def is_background_query(row: dict[str, Any]) -> bool:
    query_text = str(row.get("query_text") or "").lower()
    for blocked in NON_BACKGROUND_QUERY_TERMS:
        start = 0
        while True:
            index = query_text.find(blocked, start)
            if index < 0:
                break
            window = query_text[max(0, index - 8): index + len(blocked) + 16]
            if not any(marker in window for marker in NEGATION_TERMS):
                return False
            start = index + len(blocked)
    return True


def source_group(row: dict[str, Any]) -> str:
    source = str(row.get("folder_path") or "").strip()
    if source:
        return source.replace("\\", "/")
    relative = str(row.get("relative_path") or "").replace("\\", "/")
    if relative:
        return str(Path(relative).parent).replace("\\", "/")
    path = str(row.get("file_path") or "").replace("\\", "/")
    if path:
        return str(Path(path).parent).replace("\\", "/")
    return "unknown"


def _average_hash(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        from PIL import Image

        image = Image.open(path).convert("L").resize((8, 8))
        pixels = list(image.getdata())
    except Exception:
        return None
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return bits


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def rank_stratified_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave top, middle, and lower-ranked candidates for harder review sets."""
    buckets: list[list[dict[str, Any]]] = []
    for low, high in RANK_STRATA:
        bucket = [
            row for row in rows
            if low <= _rank_key(row)[0] <= high
        ]
        buckets.append(sorted(bucket, key=_rank_key))

    ordered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    while any(buckets):
        for bucket in buckets:
            if not bucket:
                continue
            row = bucket.pop(0)
            key = (str(row.get("query_id")), str(row.get("item_id")))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(row)
    return ordered


def annotate_visual_groups(
    rows: list[dict[str, Any]],
    *,
    hamming_threshold: int,
) -> list[dict[str, Any]]:
    """Assign visual groups within each query using thumbnail average hashes."""
    annotated = [{**row, "source_group": source_group(row)} for row in rows]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        by_query[str(row.get("query_id"))].append(row)

    hash_cache: dict[str, int | None] = {}
    for query_rows in by_query.values():
        groups: list[tuple[str, int, str]] = []
        for row in sorted(query_rows, key=_rank_key):
            if row.get("visual_group"):
                continue
            thumbnail = str(row.get("thumbnail_url") or "")
            if thumbnail not in hash_cache:
                hash_cache[thumbnail] = _average_hash(Path(thumbnail)) if thumbnail else None
            image_hash = hash_cache[thumbnail]
            if image_hash is None:
                visual_group = f"item:{row.get('item_id')}"
            else:
                visual_group = ""
                for group_id, group_hash, _group_item in groups:
                    if _hamming(image_hash, group_hash) <= hamming_threshold:
                        visual_group = group_id
                        break
                if not visual_group:
                    visual_group = f"visual:{row.get('query_id')}:{row.get('item_id')}"
                    groups.append((visual_group, image_hash, str(row.get("item_id"))))
            row["visual_group"] = visual_group
    return annotated


def diversify_rows(
    rows: list[dict[str, Any]],
    *,
    target_per_query: int,
    min_per_query: int,
    max_per_source: int,
    hamming_threshold: int,
    max_per_item: int = 1,
    max_per_intent: int = 1,
    background_only: bool = False,
    exclude_repair_required: bool = True,
    rank_strata: bool = False,
) -> list[dict[str, Any]]:
    if target_per_query < 1:
        raise ValueError("target_per_query must be >= 1")
    if min_per_query < 1:
        raise ValueError("min_per_query must be >= 1")
    if max_per_source < 1:
        raise ValueError("max_per_source must be >= 1")
    if max_per_item < 0:
        raise ValueError("max_per_item must be >= 0")
    if max_per_intent < 0:
        raise ValueError("max_per_intent must be >= 0")

    if exclude_repair_required:
        rows = [row for row in rows if not is_repair_required(row)]
    if background_only:
        rows = [row for row in rows if is_background_query(row)]
    annotated = annotate_visual_groups(rows, hamming_threshold=hamming_threshold)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        by_query[str(row.get("query_id"))].append(row)

    selected: list[dict[str, Any]] = []
    item_counts: dict[str, int] = defaultdict(int)
    intent_counts: dict[str, int] = defaultdict(int)
    for _query_id, query_rows in sorted(by_query.items()):
        intent = query_intent_key(query_rows[0]) if query_rows else ""
        if max_per_intent and intent_counts[intent] >= max_per_intent:
            continue
        ordered = rank_stratified_order(query_rows) if rank_strata else sorted(query_rows, key=_rank_key)
        picked: list[dict[str, Any]] = []
        picked_keys: set[tuple[str, str]] = set()
        visual_seen: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)

        def try_pick(row: dict[str, Any], *, enforce_source: bool) -> bool:
            key = (str(row.get("query_id")), str(row.get("item_id")))
            if key in picked_keys:
                return False
            item = str(row.get("item_id") or row.get("file_path") or "")
            if max_per_item and item_counts[item] >= max_per_item:
                return False
            visual = str(row.get("visual_group") or "")
            if visual in visual_seen:
                return False
            source = str(row.get("source_group") or "unknown")
            if enforce_source and source_counts[source] >= max_per_source:
                return False
            picked.append(row)
            picked_keys.add(key)
            visual_seen.add(visual)
            source_counts[source] += 1
            item_counts[item] += 1
            return True

        for row in ordered:
            if len(picked) >= target_per_query:
                break
            try_pick(row, enforce_source=True)

        if len(picked) < min_per_query:
            for row in ordered:
                if len(picked) >= min(target_per_query, min_per_query):
                    break
                try_pick(row, enforce_source=False)

        for index, row in enumerate(picked, start=1):
            row["diversity_note"] = (
                f"diverse_rank={index}; source_count={source_counts[str(row.get('source_group'))]}; "
                f"visual_group={row.get('visual_group')}"
            )
        if picked:
            intent_counts[intent] += 1
            selected.extend(picked)

    return sorted(selected, key=lambda row: (str(row.get("query_id")), _rank_key(row)))


def write_csv_preserve_fields(path: Path, rows: list[dict[str, Any]], source_fields: list[str]) -> None:
    fieldnames = list(source_fields)
    for field in DIVERSITY_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="input review CSV or JSONL")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--target-per-query", type=int, default=12)
    parser.add_argument("--min-per-query", type=int, default=8)
    parser.add_argument("--max-per-source", type=int, default=5)
    parser.add_argument("--max-per-item", type=int, default=1)
    parser.add_argument("--max-per-intent", type=int, default=1)
    parser.add_argument("--background-only", action="store_true", help="keep only background/scene-oriented queries")
    parser.add_argument(
        "--include-repair-required",
        action="store_true",
        help="include rows with missing/failed analysis metadata in review output",
    )
    parser.add_argument("--hamming-threshold", type=int, default=5)
    parser.add_argument(
        "--rank-strata",
        action="store_true",
        help="interleave rank buckets so review candidates include harder lower-ranked results",
    )
    args = parser.parse_args(argv)

    try:
        rows = read_review_rows(args.review)
        source_fields: list[str] = []
        if args.review.suffix.lower() == ".csv":
            with args.review.open("r", encoding="utf-8", newline="") as f:
                source_fields = list(csv.DictReader(f).fieldnames or [])
        else:
            for row in rows:
                for field in row:
                    if field not in source_fields:
                        source_fields.append(field)

        diversified = diversify_rows(
            rows,
            target_per_query=args.target_per_query,
            min_per_query=args.min_per_query,
            max_per_source=args.max_per_source,
            hamming_threshold=args.hamming_threshold,
            max_per_item=args.max_per_item,
            max_per_intent=args.max_per_intent,
            background_only=args.background_only,
            exclude_repair_required=not args.include_repair_required,
            rank_strata=args.rank_strata,
        )
        write_jsonl(args.output_jsonl, diversified)
        write_csv_preserve_fields(args.output_csv, diversified, source_fields)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(diversified)} diverse review tasks: {args.output_jsonl}")
    print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
