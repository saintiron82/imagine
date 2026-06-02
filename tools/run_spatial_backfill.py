#!/usr/bin/env python3
"""Wrapper that invokes ingest_engine on a list of file_ids without
shell-quoting headaches (image-DB paths contain non-ASCII chars).

Usage:
    .venv/bin/python tools/run_spatial_backfill.py --limit 50 [--reason missing_objects]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    parser.add_argument(
        "--reason",
        default="missing_objects",
        choices=("missing_objects", "missing_relations", "missing_depth_layers"),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the candidate list and exit (no ingest).")
    args = parser.parse_args()

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
