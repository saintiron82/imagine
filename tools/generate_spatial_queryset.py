#!/usr/bin/env python3
"""Generate a spatial-intent frozen queryset from populated file_objects.

For each (ko_name, primary_location) pair that appears in ≥2 files, emit
one Korean spatial-positioning query and mark the matching file_ids as
ground truth.

Usage:
    .venv/bin/python tools/generate_spatial_queryset.py \\
        --db imageparser.db \\
        --output benchmarks/querysets/frozen_spatial_30_v1.json \\
        --count 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Korean phrasing per location canonical.
_LOCATION_KO = {
    "top": "위쪽에",
    "top-left": "왼쪽 위에",
    "top-right": "오른쪽 위에",
    "left": "왼쪽에",
    "center": "중앙에",
    "right": "오른쪽에",
    "bottom": "아래쪽에",
    "bottom-left": "왼쪽 아래에",
    "bottom-right": "오른쪽 아래에",
}


def _location_phrase(loc: str) -> str:
    return _LOCATION_KO.get(loc, f"{loc}에")


def _ko_name(row_ko: str | None, row_en: str | None) -> str:
    """Prefer Korean name; fall back to English if missing."""
    if row_ko and row_ko.strip():
        return row_ko.strip()
    return (row_en or "").strip()


def _has_final_consonant(syllable: str) -> bool:
    """Return True if the final char is a Hangul syllable ending in a final consonant (batchim).

    Used to pick 이/가 (subject particle) correctly:
      - batchim → "이"  (벽 → 벽이)
      - no batchim → "가" (캐릭터 → 캐릭터가)
    """
    if not syllable:
        return False
    ch = syllable[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:  # Hangul syllable block
        return ((code - 0xAC00) % 28) != 0
    # Non-Hangul (English etc.): treat as no batchim → "가"
    return False


def _build_query(ko: str, location: str) -> str:
    particle = "이" if _has_final_consonant(ko) else "가"
    return f"{_location_phrase(location)} {ko}{particle} 있는 이미지"


def collect_pairs(db_path: Path, min_files_per_pair: int = 2) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT file_id, name, ko_name, primary_location
               FROM file_objects
               WHERE primary_location IS NOT NULL AND primary_location != ''"""
        ).fetchall()
    finally:
        conn.close()

    # Group by (ko_name, primary_location) → list of file_ids
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"file_ids": set(), "en_name": None}
    )
    for r in rows:
        ko = _ko_name(r["ko_name"], r["name"])
        loc = r["primary_location"]
        if not ko or not loc:
            continue
        key = (ko, loc)
        groups[key]["file_ids"].add(r["file_id"])
        if not groups[key]["en_name"]:
            groups[key]["en_name"] = r["name"]

    out = []
    for (ko, loc), info in groups.items():
        if len(info["file_ids"]) < min_files_per_pair:
            continue
        out.append(
            {
                "ko_name": ko,
                "en_name": info["en_name"] or ko,
                "location": loc,
                "file_ids": sorted(info["file_ids"]),
            }
        )
    # Sort by gt_count descending so we pick the most-supported pairs first.
    out.sort(key=lambda x: -len(x["file_ids"]))
    return out


def build_queryset(pairs: list[dict], count: int) -> dict:
    queries = []
    for p in pairs[:count]:
        q = _build_query(p["ko_name"], p["location"])
        queries.append(
            {
                "source_id": p["file_ids"][0],
                "file_name": None,
                "query": q,
                "elements_en": [p["en_name"]],
                "elements_ko": [p["ko_name"]],
                "folder": None,
                "spatial_location": p["location"],
                "gt_ids": p["file_ids"],
                "gt_count": len(p["file_ids"]),
                "scope_ground_truth": False,
            }
        )
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "count": len(queries),
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    parser.add_argument(
        "--output",
        default="benchmarks/querysets/frozen_spatial_30_v1.json",
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--min-files", type=int, default=2,
                        help="Each (object, location) pair must appear in ≥N files.")
    args = parser.parse_args()

    pairs = collect_pairs(Path(args.db), min_files_per_pair=args.min_files)
    print(f"pairs with ≥{args.min_files} supporting files: {len(pairs)}")
    if len(pairs) < args.count:
        print(
            f"WARNING: only {len(pairs)} eligible pairs available; "
            f"requested count={args.count}. Output will be smaller."
        )

    qs = build_queryset(pairs, args.count)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(qs['queries'])} queries to {args.output}")

    # Quick stats
    spatial_words = ("왼쪽", "오른쪽", "위", "아래", "중앙")
    matched = sum(
        1 for q in qs["queries"]
        if any(w in q["query"] for w in spatial_words)
    )
    print(f"queries with spatial language: {matched}/{len(qs['queries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
