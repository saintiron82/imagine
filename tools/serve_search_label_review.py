#!/usr/bin/env python3
"""Serve the search label review gallery with a small CSV update API."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "benchmarks" / "reviews" / "scoped_gold_v1_seed_20260502"
DEFAULT_REVIEW_CSV = DEFAULT_REVIEW_DIR / "review_tasks_prefilled.csv"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
VALID_REVIEW_VALUES = {"0", "1", "2"}
VALID_REVIEW_FIELDS = {"reviewer_relevance", "caption_alignment"}


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _write_json(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _backup_once(csv_path: Path) -> Path:
    backup_marker = csv_path.with_suffix(csv_path.suffix + ".review-server-backup")
    if backup_marker.exists():
        existing = backup_marker.read_text(encoding="utf-8").strip()
        if existing:
            return Path(existing)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_suffix(csv_path.suffix + f".bak.{timestamp}")
    shutil.copy2(csv_path, backup_path)
    backup_marker.write_text(str(backup_path), encoding="utf-8")
    return backup_path


def _load_csv(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _thumbnail_src(row: dict[str, Any]) -> str:
    thumbnail = str(row.get("thumbnail_url") or "")
    if not thumbnail:
        return ""
    path = Path(thumbnail)
    try:
        return "/" + path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return thumbnail


def build_gallery_payload(csv_path: Path) -> dict[str, Any]:
    rows, _fieldnames = _load_csv(csv_path)
    gallery_rows = []
    queries = []
    seen_queries = set()
    for line_number, row in enumerate(rows, start=2):
        query_id = str(row.get("query_id") or "")
        query_text = str(row.get("query_text") or "")
        if query_id and query_id not in seen_queries:
            seen_queries.add(query_id)
            queries.append({"query_id": query_id, "query_text": query_text})
        gallery_rows.append({
            **row,
            "_csv_line": line_number,
            "_thumb_src": _thumbnail_src(row),
        })

    reviewed_count = sum(
        1 for row in rows if str(row.get("reviewer_relevance", "")).strip() in VALID_REVIEW_VALUES
    )
    caption_reviewed_count = sum(
        1 for row in rows if str(row.get("caption_alignment", "")).strip() in VALID_REVIEW_VALUES
    )
    return {
        "ok": True,
        "rows": gallery_rows,
        "queries": queries,
        "reviewed_count": reviewed_count,
        "caption_reviewed_count": caption_reviewed_count,
        "total_count": len(rows),
        "csv": str(csv_path),
    }


def update_review_fields(
    csv_path: Path,
    *,
    query_id: str,
    item_id: str,
    updates: dict[str, str],
    reviewer_id: str = "manual_review",
) -> dict[str, Any]:
    cleaned_updates = {
        str(field).strip(): str(value).strip()
        for field, value in updates.items()
        if str(field).strip()
    }
    if not cleaned_updates:
        raise ValueError("no review fields to update")
    for field, value in cleaned_updates.items():
        if field not in VALID_REVIEW_FIELDS:
            raise ValueError(f"unsupported review field: {field}")
        if value not in VALID_REVIEW_VALUES:
            raise ValueError(f"{field} must be 0, 1, or 2")

    rows, fieldnames = _load_csv(csv_path)
    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")
    for field in ("query_id", "item_id"):
        if field not in fieldnames:
            raise ValueError(f"CSV missing required field: {field}")
    for field in cleaned_updates:
        if field not in fieldnames:
            fieldnames.append(field)
    reviewer_field_by_value = {
        "reviewer_relevance": "reviewer_id",
        "caption_alignment": "caption_alignment_reviewer_id",
    }
    for field in cleaned_updates:
        reviewer_field = reviewer_field_by_value[field]
        if reviewer_field not in fieldnames:
            fieldnames.append(reviewer_field)

    matches = [
        index for index, row in enumerate(rows)
        if str(row.get("query_id")) == query_id and str(row.get("item_id")) == item_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one matching row, found {len(matches)}")

    backup_path = _backup_once(csv_path)
    row = rows[matches[0]]
    previous = {field: row.get(field, "") for field in cleaned_updates}
    for field, value in cleaned_updates.items():
        row[field] = value
        row[reviewer_field_by_value[field]] = reviewer_id

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    reviewed_count = sum(
        1 for item in rows if str(item.get("reviewer_relevance", "")).strip() in VALID_REVIEW_VALUES
    )
    caption_reviewed_count = sum(
        1 for item in rows if str(item.get("caption_alignment", "")).strip() in VALID_REVIEW_VALUES
    )
    return {
        "ok": True,
        "query_id": query_id,
        "item_id": item_id,
        "previous": previous,
        "updates": cleaned_updates,
        **cleaned_updates,
        "reviewed_count": reviewed_count,
        "caption_reviewed_count": caption_reviewed_count,
        "total_count": len(rows),
        "backup": str(backup_path),
    }


def update_reviewer_relevance(
    csv_path: Path,
    *,
    query_id: str,
    item_id: str,
    reviewer_relevance: str,
    reviewer_id: str = "manual_review",
) -> dict[str, Any]:
    return update_review_fields(
        csv_path,
        query_id=query_id,
        item_id=item_id,
        updates={"reviewer_relevance": reviewer_relevance},
        reviewer_id=reviewer_id,
    )


class ReviewHandler(SimpleHTTPRequestHandler):
    review_csv: Path = DEFAULT_REVIEW_CSV

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/review/rows":
            _write_json(self, HTTPStatus.OK, build_gallery_payload(self.review_csv))
            return
        if self.path == "/api/review/status":
            rows, _ = _load_csv(self.review_csv)
            reviewed_count = sum(
                1 for row in rows if str(row.get("reviewer_relevance", "")).strip() in VALID_REVIEW_VALUES
            )
            caption_reviewed_count = sum(
                1 for row in rows if str(row.get("caption_alignment", "")).strip() in VALID_REVIEW_VALUES
            )
            _write_json(self, HTTPStatus.OK, {
                "ok": True,
                "reviewed_count": reviewed_count,
                "caption_reviewed_count": caption_reviewed_count,
                "total_count": len(rows),
                "csv": str(self.review_csv),
            })
            return
        if self.path in {"", "/"}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header(
                "Location",
                "/benchmarks/reviews/scoped_gold_v1_seed_20260502/review_gallery.html",
            )
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/review/update":
            _write_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            payload = _read_json_body(self)
            updates = {}
            for field in VALID_REVIEW_FIELDS:
                if field in payload:
                    updates[field] = str(payload.get(field, ""))
            result = update_review_fields(
                self.review_csv,
                query_id=str(payload.get("query_id", "")),
                item_id=str(payload.get("item_id", "")),
                updates=updates,
                reviewer_id=str(payload.get("reviewer_id") or "manual_review"),
            )
        except Exception as exc:
            _write_json(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return
        _write_json(self, HTTPStatus.OK, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_REVIEW_CSV)
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.exists():
        parser.error(f"review CSV does not exist: {csv_path}")

    handler = type(
        "ConfiguredReviewHandler",
        (ReviewHandler,),
        {"review_csv": csv_path},
    )
    server = ThreadingHTTPServer((args.host, args.port), lambda *a, **kw: handler(*a, directory=str(PROJECT_ROOT), **kw))
    url = f"http://{args.host}:{args.port}/benchmarks/reviews/scoped_gold_v1_seed_20260502/review_gallery.html"
    print(f"Serving review gallery: {url}")
    print(f"Writing review CSV: {csv_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
