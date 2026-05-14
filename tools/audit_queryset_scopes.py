#!/usr/bin/env python3
"""Audit whether QuerySet scopes map to meaningful database folders."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_search_label_review import classify_metadata_status  # noqa: E402


GENERIC_SCOPE_TOKENS = {
    "bg",
    "BG",
    "장소",
    "자료",
    "이미지",
    "image",
    "images",
    "file",
    "files",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def split_scope_segments(value: str) -> list[str]:
    normalized = str(value or "").replace("\\", "/").strip("/")
    return [part.strip().lower() for part in normalized.split("/") if part.strip()]


def path_has_scope_segments(path_value: str, scope_value: str) -> bool:
    path_segments = split_scope_segments(path_value)
    scope_segments = split_scope_segments(scope_value)
    if not path_segments or not scope_segments:
        return False
    width = len(scope_segments)
    if width > len(path_segments):
        return False
    return any(path_segments[index:index + width] == scope_segments for index in range(0, len(path_segments) - width + 1))


def scope_tokens(scope: str) -> list[str]:
    tokens = []
    for segment in split_scope_segments(scope):
        for token in re.split(r"[\s,]+", segment):
            cleaned = token.strip()
            if cleaned and cleaned not in GENERIC_SCOPE_TOKENS:
                tokens.append(cleaned)
    return tokens


def query_mentions_scope(query_text: str, scope: str) -> bool:
    text = str(query_text or "").lower()
    return any(token.lower() in text for token in scope_tokens(scope))


def load_files(db_path: Path) -> list[sqlite3.Row]:
    if not db_path.exists():
        raise ValueError(f"database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(files)")}
        required = {"id", "file_path", "relative_path", "folder_path"}
        missing = required - columns
        if missing:
            raise ValueError(f"{db_path}: files table missing columns: {', '.join(sorted(missing))}")
        selected = [
            name for name in (
                "id",
                "file_path",
                "relative_path",
                "folder_path",
                "image_type",
                "mc_caption",
                "ai_tags",
                "caption_model",
                "processing_status",
                "processing_error",
                "preview_only",
            )
            if name in columns
        ]
        return conn.execute(f"SELECT {', '.join(selected)} FROM files").fetchall()
    finally:
        conn.close()


def row_matches_scope(row: sqlite3.Row, scope: str) -> bool:
    if "preview_only" in row.keys() and int(row["preview_only"] or 0):
        return False
    return (
        path_has_scope_segments(str(row["folder_path"] or ""), scope)
        or path_has_scope_segments(str(row["file_path"] or ""), scope)
        or path_has_scope_segments(str(row["relative_path"] or ""), scope)
    )


def _metadata_status(row: sqlite3.Row) -> str:
    as_dict = {key: row[key] for key in row.keys()}
    return classify_metadata_status(as_dict)["metadata_status"]


def _folder_value(row: sqlite3.Row) -> str:
    folder = str(row["folder_path"] or "")
    return folder.replace("webdav://13730b09/", "").replace("webdav:/13730b09/", "")


def assess_scope(query: dict[str, Any], files: list[sqlite3.Row], *, broad_threshold: int) -> dict[str, Any]:
    scope = str(query.get("scope") or query.get("folder_scope") or query.get("folder") or "").strip()
    matched = [row for row in files if row_matches_scope(row, scope)] if scope else []
    folders = Counter(_folder_value(row) for row in matched)
    image_types = Counter(str(row["image_type"] or "-") if "image_type" in row.keys() else "-" for row in matched)
    metadata = Counter(_metadata_status(row) for row in matched)
    known_type_count = sum(count for image_type, count in image_types.items() if image_type != "-")
    background_count = image_types.get("background", 0)
    background_ratio = round(background_count / known_type_count, 4) if known_type_count else None
    metadata_ok_ratio = round(metadata.get("ok", 0) / len(matched), 4) if matched else None

    scope_flags = []
    data_flags = []
    if not scope:
        scope_flags.append("missing_scope")
    if scope and not matched:
        scope_flags.append("no_scope_match")
    if scope and not query_mentions_scope(str(query.get("query_text") or ""), scope):
        scope_flags.append("query_does_not_mention_scope")
    if len(matched) > broad_threshold:
        scope_flags.append("broad_scope")
    if len(split_scope_segments(scope)) <= 1 and len(folders) > 1:
        scope_flags.append("single_segment_matches_multiple_folders")

    if 0 < len(matched) < 8:
        data_flags.append("tiny_scope")
    if background_ratio is not None and known_type_count >= 10 and background_ratio < 0.2:
        data_flags.append("low_background_type_ratio")
    if metadata_ok_ratio is not None and metadata_ok_ratio < 0.2:
        data_flags.append("metadata_gap")

    scope_assessment = "ok" if not scope_flags else "review"
    if "missing_scope" in scope_flags or "no_scope_match" in scope_flags:
        scope_assessment = "bad"
    data_assessment = "ok" if not data_flags else "review"

    return {
        "query_id": query.get("query_id"),
        "query_text": query.get("query_text"),
        "scope": scope,
        "scope_assessment": scope_assessment,
        "scope_flags": scope_flags,
        "data_assessment": data_assessment,
        "data_flags": data_flags,
        "file_count": len(matched),
        "distinct_folder_count": len(folders),
        "top_folders": [{"folder": folder, "count": count} for folder, count in folders.most_common(5)],
        "image_type_counts": dict(image_types.most_common()),
        "metadata_status_counts": dict(metadata.most_common()),
        "background_ratio": background_ratio,
        "metadata_ok_ratio": metadata_ok_ratio,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query_id",
        "scope_assessment",
        "scope_flags",
        "data_assessment",
        "data_flags",
        "file_count",
        "distinct_folder_count",
        "background_ratio",
        "metadata_ok_ratio",
        "scope",
        "query_text",
        "top_folders",
        "image_type_counts",
        "metadata_status_counts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# QuerySet Scope Audit",
        "",
        "이 문서는 검색 결과 정답성이 아니라 QuerySet의 `scope`가 실제 DB 경로와 의미 있게 대응하는지 점검합니다.",
        "",
    ]
    for row in rows:
        flags = ", ".join(row["scope_flags"] or ["-"])
        data_flags = ", ".join(row["data_flags"] or ["-"])
        lines.extend([
            f"## {row['query_id']} · {row['scope_assessment']}",
            "",
            f"- scope: {row['scope']}",
            f"- files: {row['file_count']} / folders: {row['distinct_folder_count']}",
            f"- scope flags: {flags}",
            f"- data flags: {data_flags}",
            f"- query: {row['query_text']}",
            "- top folders:",
        ])
        for folder in row["top_folders"]:
            lines.append(f"  - {folder['count']}: {folder['folder']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=Path("imageparser.db"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--broad-threshold", type=int, default=300)
    args = parser.parse_args(argv)

    try:
        queries = read_jsonl(args.queries)
        files = load_files(args.db_path)
        rows = [assess_scope(query, files, broad_threshold=args.broad_threshold) for query in queries]
        write_jsonl(args.output_jsonl, rows)
        if args.output_csv:
            write_csv(args.output_csv, rows)
        if args.output_md:
            write_markdown(args.output_md, rows)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    scope_counts = Counter(row["scope_assessment"] for row in rows)
    data_counts = Counter(row["data_assessment"] for row in rows)
    print(f"Wrote {len(rows)} scope audit rows: {args.output_jsonl}")
    print(f"Scope assessment: {dict(sorted(scope_counts.items()))}")
    print(f"Data assessment: {dict(sorted(data_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
