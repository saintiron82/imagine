"""
Register parse_fallback_legacy PSD files as a new analysis_job for
server-mode re-processing with 30 GB buffer.

This inserts:
  1. An analysis_job row (name="Fallback Re-parse {date}")
  2. file_task rows for each target file (parse_status='pending')

After registration, start the server — DownloadAheadPool will auto-download
(up to max_files=350 ≈ 30 GB), parse, vision, embed, then delete temp files.

Usage:
    # Register 10 files (test)
    python tools/register_fallback_job.py --limit 10

    # Register all local fallbacks (670, no download needed)
    python tools/register_fallback_job.py --scope local

    # Register all (14,971 — 1.2 TB download over time)
    python tools/register_fallback_job.py --scope all

    # Check status
    python tools/register_fallback_job.py --status
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("register-fallback")


def _create_job(conn, name: str, source_path: str, total: int) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files, started_at) "
        "VALUES (?, ?, 'active', ?, datetime('now'))",
        (name, source_path, total),
    )
    conn.commit()
    return cur.lastrowid


def _register_tasks(conn, job_id: int, files: list[tuple[int, str]]):
    cur = conn.cursor()
    rows = []
    for file_id, file_path in files:
        is_webdav = file_path.startswith("webdav://")
        dl_status = "pending" if is_webdav else "n/a"
        rows.append((
            job_id, file_id, file_path,
            dl_status,
            "pending",  # parse_status
            "pending",  # mc_status
            "pending",  # vv_status
            "pending",  # mv_status
        ))

    cur.executemany(
        "INSERT INTO file_tasks "
        "(analysis_job_id, file_id, file_path, "
        " download_status, parse_status, mc_status, vv_status, mv_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    logger.info(f"Registered {len(rows)} file_tasks for job #{job_id}")


def _show_status(conn):
    cur = conn.cursor()
    jobs = cur.execute(
        "SELECT id, name, status, total_files, completed_files, created_at "
        "FROM analysis_jobs ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if not jobs:
        print("No analysis jobs found.")
        return
    for jid, name, status, total, done, created in jobs:
        done = done or 0
        pct = (done / total * 100) if total else 0
        print(f"Job #{jid}: {name} [{status}] {done}/{total} ({pct:.0f}%) created={created}")

        # Phase breakdown
        phases = cur.execute(
            "SELECT "
            "  SUM(CASE WHEN parse_status='done' THEN 1 ELSE 0 END) AS p_done, "
            "  SUM(CASE WHEN mc_status='done' THEN 1 ELSE 0 END) AS mc_done, "
            "  SUM(CASE WHEN vv_status='done' THEN 1 ELSE 0 END) AS vv_done, "
            "  SUM(CASE WHEN mv_status='done' THEN 1 ELSE 0 END) AS mv_done, "
            "  SUM(CASE WHEN download_status='done' THEN 1 ELSE 0 END) AS dl_done, "
            "  COUNT(*) AS total "
            "FROM file_tasks WHERE analysis_job_id = ?",
            (jid,),
        ).fetchone()
        if phases:
            p, mc, vv, mv, dl, t = phases
            print(f"  DL:{dl}/{t} P:{p}/{t} MC:{mc}/{t} VV:{vv}/{t} MV:{mv}/{t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["local", "webdav", "all"], default="local")
    ap.add_argument("--limit", type=int, default=0, help="0 = all matching scope")
    ap.add_argument("--status", action="store_true", help="Show job status instead of registering")
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)

    if args.status:
        _show_status(conn)
        conn.close()
        return

    # Select target files
    where = ["processing_status = 'parse_fallback_legacy'"]
    if args.scope == "local":
        where.append("file_path NOT LIKE 'webdav://%'")
    elif args.scope == "webdav":
        where.append("file_path LIKE 'webdav://%'")

    sql = f"SELECT id, file_path FROM files WHERE {' AND '.join(where)} ORDER BY id"
    if args.limit > 0:
        sql += f" LIMIT {args.limit}"

    targets = conn.execute(sql).fetchall()
    if not targets:
        logger.info("No targets found.")
        conn.close()
        return

    n_webdav = sum(1 for _, p in targets if p.startswith("webdav://"))
    n_local = len(targets) - n_webdav
    logger.info(f"Scope: {args.scope}, targets: {len(targets)} (local={n_local}, webdav={n_webdav})")

    if n_webdav:
        est_gb = n_webdav * 85 / 1024
        logger.warning(f"WebDAV download estimated: ~{est_gb:.0f} GB (temp_buffer.max_files=350 ≈ 30 GB cap)")

    # Create job
    ts = time.strftime("%Y-%m-%d %H:%M")
    job_name = f"Fallback Re-parse ({args.scope}, {len(targets)} files, {ts})"
    source_path = f"fallback_{args.scope}"
    job_id = _create_job(conn, job_name, source_path, len(targets))
    logger.info(f"Created analysis_job #{job_id}: {job_name}")

    # Register file_tasks
    _register_tasks(conn, job_id, targets)

    logger.info(
        f"\nDone. Start the server to begin processing:\n"
        f"  source .venv/bin/activate\n"
        f"  python -m backend.server.app\n"
        f"\n"
        f"Monitor progress:\n"
        f"  python tools/register_fallback_job.py --status\n"
    )

    conn.close()


if __name__ == "__main__":
    main()
