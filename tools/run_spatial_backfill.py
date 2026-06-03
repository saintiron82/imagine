#!/usr/bin/env python3
"""Wrapper that invokes ingest_engine on a list of file_ids without
shell-quoting headaches (image-DB paths contain non-ASCII chars).

Usage:
    .venv/bin/python tools/run_spatial_backfill.py --limit 50 [--reason missing_objects]
    .venv/bin/python tools/run_spatial_backfill.py \\
        --path-prefix /Users/saintiron/imageDB/마캬베리즈무/ --limit 1000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.backfill_spatial_processing import select_reprocess_candidates  # noqa: E402

_REASON_TABLES = {
    "missing_objects": "file_objects",
    "missing_relations": "file_spatial_relations",
    "missing_depth_layers": "file_depth_layers",
}


def _select_by_path_prefix(db_path: Path, reason: str, prefix: str, limit: int) -> list[dict]:
    """Pick candidates whose file_path starts with `prefix` and lack the evidence row."""
    table = _REASON_TABLES[reason]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""SELECT f.id, f.file_path
                FROM files f
                WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                  AND f.file_path LIKE ?
                  AND NOT EXISTS (
                    SELECT 1 FROM {table} evidence WHERE evidence.file_id = f.id
                  )
                ORDER BY f.id
                LIMIT ?""",
            (prefix + "%", limit),
        ).fetchall()
        return [{"id": r["id"], "file_path": r["file_path"], "reason": reason} for r in rows]
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    parser.add_argument(
        "--reason",
        default="missing_objects",
        choices=tuple(_REASON_TABLES.keys()),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--path-prefix", default=None,
                        help="Only pick files whose file_path starts with this prefix.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the candidate list and exit (no ingest).")
    args = parser.parse_args()

    if args.path_prefix:
        rows = _select_by_path_prefix(
            Path(args.db), args.reason, args.path_prefix, args.limit,
        )
    else:
        rows = select_reprocess_candidates(Path(args.db), args.reason, args.limit)
    paths = [r["file_path"] for r in rows]
    print(f"[backfill] selected {len(paths)} candidates (reason={args.reason})", flush=True)
    for r in rows[:5]:
        print(f"  id={r['id']} path={r['file_path']}", flush=True)
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more", flush=True)

    if args.dry_run or not rows:
        return 0

    files_json = json.dumps(paths, ensure_ascii=False)
    cmd = [
        sys.executable,
        str(REPO / "backend/pipeline/ingest_engine.py"),
        "--files",
        files_json,
        "--no-skip",
    ]
    print(f"[backfill] invoking ingest_engine with {len(paths)} files", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.perf_counter() - t0
    n = max(1, len(paths))
    print(
        f"[backfill] ingest done: rc={proc.returncode} "
        f"total={elapsed:.1f}s avg={elapsed / n:.2f}s/file",
        flush=True,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
