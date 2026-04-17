"""
Re-ingest files marked as parse_fallback_legacy (P10 cleanup).

These 14,971 PSD files had empty layer_tree because of past psd-tools
parse failures. The current code stack handles them correctly (verified
on local sample 1.psd), so a forced re-ingest with --no-skip restores
layer_tree, structured_meta, perceptual_hash, width/height,
vec_structure, and processing_status in a single pass.

Scope summary (as of 2026-04-17):
  - local PSD: 670 files (~immediate, CPU + GPU only)
  - webdav PSD: 14,301 files (~1.1 TB download required, hours)

Usage:
    # Dry run — show how many files would be processed
    python tools/reingest_fallback.py --dry-run

    # Process local only (recommended first)
    python tools/reingest_fallback.py --scope local

    # Process WebDAV in 100-file batches, limit to 500 (test run)
    python tools/reingest_fallback.py --scope webdav --batch 100 --limit 500

    # Resume after partial run (skip files already re-processed)
    python tools/reingest_fallback.py --scope all --resume

After completion, run audit_null_ratios SQL to confirm column recovery.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reingest")


def _select_targets(conn, scope: str, limit: int, resume: bool):
    """Pull fallback file paths matching scope. Resume skips files whose
    structured_meta is already non-NULL (recently re-processed)."""
    where = ["processing_status = 'parse_fallback_legacy'"]
    if scope == "local":
        where.append("file_path NOT LIKE 'webdav://%'")
    elif scope == "webdav":
        where.append("file_path LIKE 'webdav://%'")
    if resume:
        where.append("(structured_meta IS NULL OR length(structured_meta) < 50)")

    sql = f"SELECT file_path FROM files WHERE {' AND '.join(where)} ORDER BY id"
    if limit > 0:
        sql += f" LIMIT {limit}"
    return [row[0] for row in conn.execute(sql).fetchall()]


def _run_single(path: str) -> tuple[bool, str]:
    """Invoke ingest_engine --file for a single file (--no-skip).

    Uses --file (not --files) because the single-file process_file() path
    correctly stores all columns including layer_tree, structured_meta,
    width/height. The batch --files path has a known issue with PhaseRunner
    vision result DB reflection.
    """
    cmd = [sys.executable, "-m", "backend.pipeline.ingest_engine",
           "--file", path, "--no-skip"]

    try:
        with open("/dev/zero", "rb") as devzero:
            r = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT),
                stdin=devzero,
                capture_output=True, text=True,
                timeout=600,  # 10 min per file
            )
        if r.returncode != 0:
            tail = "\n".join(r.stderr.splitlines()[-10:])
            return False, f"exit={r.returncode}\n{tail}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "timeout (>10min)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["local", "webdav", "all"],
                    default="local",
                    help="Which files to re-ingest")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all, otherwise process first N")
    ap.add_argument("--batch", type=int, default=50,
                    help="Files per ingest_engine invocation")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip files already re-processed (structured_meta non-NULL)")
    ap.add_argument("--db",
                    default=str(PROJECT_ROOT / "imageparser.db"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    if args.scope == "all":
        local = _select_targets(conn, "local", args.limit, args.resume)
        webdav = _select_targets(conn, "webdav",
                                 args.limit - len(local) if args.limit else 0,
                                 args.resume)
        targets = local + webdav
    else:
        targets = _select_targets(conn, args.scope, args.limit, args.resume)
    conn.close()

    logger.info(f"Scope: {args.scope}, batch: {args.batch}, total targets: {len(targets)}")

    if args.scope in ("webdav", "all"):
        # Estimate download size
        webdav_count = sum(1 for p in targets if p.startswith("webdav://"))
        if webdav_count:
            est_gb = webdav_count * 85 / 1024  # avg 85 MB/file
            logger.warning(
                f"WebDAV will download ~{est_gb:.0f} GB across {webdav_count} files. "
                f"Ensure NAS+disk capacity, then proceed."
            )

    if args.dry_run:
        logger.info(f"dry-run: would process {len(targets)} files in "
                    f"{(len(targets) + args.batch - 1) // args.batch} batches")
        if targets[:3]:
            logger.info(f"first 3 targets: {targets[:3]}")
        return

    if not targets:
        logger.info("No targets — nothing to do.")
        return

    succeeded = 0
    failed = 0
    t0 = time.time()
    total = len(targets)
    for i, path in enumerate(targets):
        idx = i + 1
        logger.info(f"[{idx}/{total}] {Path(path).name}")
        t_file = time.time()
        ok, err = _run_single(path)
        dt = time.time() - t_file
        if ok:
            succeeded += 1
            logger.info(f"  OK ({dt:.1f}s)")
        else:
            failed += 1
            logger.error(f"  FAIL ({dt:.1f}s): {err}")

        # ETA every 5 files
        if idx % 5 == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed else 0
            remaining = total - idx
            eta = remaining / rate if rate else 0
            logger.info(
                f"Progress: {idx}/{total} ok={succeeded} fail={failed} "
                f"({rate:.2f} f/s, ETA {eta / 60:.0f} min)"
            )

    logger.info(f"DONE. {succeeded} succeeded, {failed} failed "
                f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
