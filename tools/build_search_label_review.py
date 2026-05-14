#!/usr/bin/env python3
"""Build human review tasks from Search Evaluation V1 artifacts.

This does not label data automatically. It turns QuerySet + RunResult rows into
deduplicated query-item review tasks that a reviewer can mark with relevance
0/1/2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_search_quality import load_queries, load_run, read_jsonl, require_fields

ITEM_METADATA_FIELDS = [
    "file_path",
    "file_name",
    "relative_path",
    "folder_path",
    "thumbnail_url",
    "format",
    "width",
    "height",
    "mc_caption",
    "ai_tags",
    "image_type",
    "scene_type",
    "time_of_day",
    "weather",
    "caption_model",
    "processing_status",
    "processing_error",
]

REVIEW_METADATA_FIELDS = [
    "metadata_status",
    "metadata_issue",
]


def load_label_metadata(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}

    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, row in enumerate(read_jsonl(path), start=1):
        require_fields(row, ("query_id", "item_id", "relevance", "label_source", "label_version"), path, idx)
        relevance = int(row["relevance"])
        if relevance not in (0, 1, 2):
            raise ValueError(f"{path}:{idx}: relevance must be 0, 1, or 2")
        key = (str(row["query_id"]), str(row["item_id"]))
        labels[key] = {
            "suggested_relevance": relevance,
            "suggested_label_source": str(row["label_source"]),
            "suggested_label_version": str(row["label_version"]),
        }
    return labels


def _empty_item_metadata() -> dict[str, Any]:
    return {field: None for field in ITEM_METADATA_FIELDS}


def _has_bad_caption(row: dict[str, Any]) -> bool:
    caption = str(row.get("mc_caption") or "").strip()
    return not caption or caption.lower() == "unknown"


def _has_bad_tags(row: dict[str, Any]) -> bool:
    tags = str(row.get("ai_tags") or "").strip()
    return not tags or tags == "[]"


def classify_metadata_status(row: dict[str, Any]) -> dict[str, str]:
    """Separate repair-target metadata gaps from normal relevance review."""
    bad_caption = _has_bad_caption(row)
    bad_tags = _has_bad_tags(row)
    processing_status = str(row.get("processing_status") or "").strip()
    processing_error = str(row.get("processing_error") or "").strip()
    caption_model = str(row.get("caption_model") or "").strip()

    legacy_parse_failure = (
        processing_status == "parse_fallback_legacy"
        or caption_model == "unknown_legacy"
        or "psd-tools failed" in processing_error
    )
    if legacy_parse_failure:
        return {
            "metadata_status": "repair_required",
            "metadata_issue": "parse_fallback_legacy",
        }

    if bad_caption and bad_tags:
        if caption_model:
            return {
                "metadata_status": "repair_required",
                "metadata_issue": "caption_model_marked_but_empty",
            }
        return {
            "metadata_status": "repair_required",
            "metadata_issue": "missing_caption_and_tags",
        }

    if bad_caption:
        return {
            "metadata_status": "repair_required",
            "metadata_issue": "missing_caption",
        }

    if bad_tags:
        return {
            "metadata_status": "metadata_partial",
            "metadata_issue": "missing_tags",
        }

    return {
        "metadata_status": "ok",
        "metadata_issue": "",
    }


def load_item_metadata(db_path: Path, item_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load file metadata needed for human relevance review."""
    if not item_ids:
        return {}
    if not db_path.exists():
        raise ValueError(f"database not found: {db_path}")

    metadata: dict[str, dict[str, Any]] = {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        if "id" not in columns:
            raise ValueError(f"{db_path}: files table is missing id column")
        selected = [field for field in ITEM_METADATA_FIELDS if field in columns]
        if not selected:
            return metadata

        id_values = sorted(
            item_ids,
            key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
        )
        for start in range(0, len(id_values), 500):
            chunk = id_values[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            fields_sql = ", ".join(["id", *selected])
            rows = conn.execute(
                f"SELECT {fields_sql} FROM files WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                item_id = str(row["id"])
                item_meta = _empty_item_metadata()
                for field in selected:
                    item_meta[field] = row[field]
                metadata[item_id] = item_meta
    finally:
        conn.close()
    return metadata


def enrich_rows_with_item_metadata(
    rows: list[dict[str, Any]],
    item_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        metadata = _empty_item_metadata()
        metadata.update(item_metadata.get(str(row["item_id"]), {}))
        merged = {**row, **metadata}
        enriched.append({**merged, **classify_metadata_status(merged)})
    return enriched


def build_review_rows(
    queries_path: Path,
    run_path: Path,
    labels_path: Path | None = None,
    top_k: int = 10,
    engines: set[str] | None = None,
    query_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    queries = load_queries(queries_path)
    query_metadata = {
        str(row.get("query_id")): row
        for row in read_jsonl(queries_path)
        if row.get("query_id")
    }
    run_groups = load_run(run_path)
    labels = load_label_metadata(labels_path)
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    engine_ranks: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    run_ranks: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)

    for (_run_id, engine_id), query_runs in run_groups.items():
        if engines is not None and engine_id not in engines:
            continue
        for query_id, records in query_runs.items():
            if query_filter is not None and query_id not in query_filter:
                continue
            query = queries.get(query_id)
            if query is None:
                continue
            for record in records:
                if record.rank > top_k:
                    continue
                key = (query_id, record.item_id)
                current_engine_rank = engine_ranks[key].get(engine_id)
                if current_engine_rank is None or record.rank < current_engine_rank:
                    engine_ranks[key][engine_id] = record.rank
                run_ranks[key][f"{record.run_id}:{engine_id}"] = record.rank
                if key not in candidates:
                    label = labels.get(key, {})
                    query_meta = query_metadata.get(query_id, {})
                    candidates[key] = {
                        "query_id": query_id,
                        "query_text": query.query_text,
                        "query_type": query.query_type,
                        "query_scope": str(query_meta.get("scope") or ""),
                        "item_id": record.item_id,
                        "suggested_relevance": label.get("suggested_relevance"),
                        "suggested_label_source": label.get("suggested_label_source"),
                        "suggested_label_version": label.get("suggested_label_version"),
                        "reviewer_relevance": None,
                        "reviewer_id": "",
                        "review_notes": "",
                    }

    rows = []
    for key, row in candidates.items():
        ranks = engine_ranks[key]
        ranked_runs = run_ranks[key]
        rows.append({
            **row,
            "engines": sorted(ranks),
            "engine_ranks": dict(sorted(ranks.items())),
            "run_ranks": dict(sorted(ranked_runs.items())),
            "best_rank": min(ranked_runs.values()),
        })

    return sorted(rows, key=lambda row: (row["query_id"], row["best_rank"], row["item_id"]))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _csv_engines(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _csv_json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "query_type",
        "query_scope",
        "item_id",
        "best_rank",
        "engines",
        "engine_ranks",
        "run_ranks",
        *ITEM_METADATA_FIELDS,
        *REVIEW_METADATA_FIELDS,
        "suggested_relevance",
        "suggested_label_source",
        "suggested_label_version",
        "reviewer_relevance",
        "reviewer_id",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "engines": _csv_engines(row.get("engines")),
                "engine_ranks": _csv_json_cell(row.get("engine_ranks")),
                "run_ranks": _csv_json_cell(row.get("run_ranks")),
            })


def parse_engines(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def parse_query_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def load_query_ids_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            values.add(value)
    return values or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="QuerySet JSONL path")
    parser.add_argument("--run", type=Path, required=True, help="RunResult JSONL path")
    parser.add_argument("--labels", type=Path, help="optional weak or existing LabelSet JSONL path")
    parser.add_argument("--top-k", type=int, default=10, help="candidate depth per engine")
    parser.add_argument("--engines", help="comma-separated engine filter, e.g. triaxis,mv")
    parser.add_argument("--query-ids", help="comma-separated query_id filter")
    parser.add_argument("--query-ids-file", type=Path, help="optional one-query-id-per-line filter")
    parser.add_argument("--db-path", type=Path, help="optional SQLite DB path for file metadata columns")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="review task JSONL output path")
    parser.add_argument("--output-csv", type=Path, help="optional review task CSV output path")
    args = parser.parse_args(argv)

    try:
        query_filter = set()
        for values in (parse_query_ids(args.query_ids), load_query_ids_file(args.query_ids_file)):
            if values:
                query_filter.update(values)
        query_filter_arg = query_filter or None

        rows = build_review_rows(
            queries_path=args.queries,
            run_path=args.run,
            labels_path=args.labels,
            top_k=args.top_k,
            engines=parse_engines(args.engines),
            query_filter=query_filter_arg,
        )
        if args.db_path:
            rows = enrich_rows_with_item_metadata(
                rows,
                load_item_metadata(args.db_path, {str(row["item_id"]) for row in rows}),
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    write_jsonl(args.output_jsonl, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)

    print(f"Wrote {len(rows)} review tasks: {args.output_jsonl}")
    if args.output_csv:
        print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
