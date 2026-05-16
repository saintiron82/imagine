"""
audit_null_ratios.py — DB hygiene snapshot helper.

Both a CLI (`python tools/audit_null_ratios.py`) and a library used by
`restore_fallback.py` to capture before/after baselines around the fallback
re-ingest run.

Reports:
  - total file rows
  - non-NULL counts + percentages for every column the fallback restore
    pipeline writes (structured_meta, perceptual_hash, dup_group_id,
    modified_at, folder_path, caption_model, dominant_color, ai_style,
    processing_status, width)
  - format breakdown
  - files_fts row count + caption/ai_tags/classification/spatial non-empty counts
  - parse_fallback_legacy remaining (split local vs webdav)
  - active analysis_jobs (so the orchestrator can see the registered
    re-ingest job's progress at a glance)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "imageparser.db"


_NULL_TARGET_COLS = [
    "structured_meta",
    "perceptual_hash",
    "dup_group_id",
    "modified_at",
    "folder_path",
    "caption_model",
    "dominant_color",
    "ai_style",
    "processing_status",
    "width",
]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def collect(db_path: Path) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        snapshot: Dict[str, Any] = {
            "db_path": str(db_path),
            "total_files": 0,
            "columns": {},
            "format_breakdown": {},
            "fts": {},
            "fallback_remaining": {"local": 0, "webdav": 0, "total": 0},
            "active_jobs": [],
        }

        total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        snapshot["total_files"] = total

        files_cols = _column_names(conn, "files")
        for col in _NULL_TARGET_COLS:
            if col not in files_cols:
                snapshot["columns"][col] = {"present": False}
                continue
            non_null = conn.execute(
                f"SELECT COUNT(*) FROM files WHERE {col} IS NOT NULL"
            ).fetchone()[0]
            pct = (non_null / total * 100) if total else 0
            snapshot["columns"][col] = {
                "present": True,
                "non_null": non_null,
                "pct": round(pct, 2),
            }

        rows = conn.execute(
            "SELECT COALESCE(format, '<NULL>'), COUNT(*) FROM files GROUP BY format"
        ).fetchall()
        snapshot["format_breakdown"] = {fmt: cnt for fmt, cnt in rows}

        if _table_exists(conn, "files_fts"):
            fts_total = conn.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
            snapshot["fts"]["rows"] = fts_total
            for col in ("caption", "ai_tags", "classification", "spatial"):
                try:
                    cnt = conn.execute(
                        f"SELECT COUNT(*) FROM files_fts WHERE {col} != ''"
                    ).fetchone()[0]
                    snapshot["fts"][col] = cnt
                except sqlite3.OperationalError:
                    snapshot["fts"][col] = None

        local_left = conn.execute(
            "SELECT COUNT(*) FROM files "
            "WHERE processing_status='parse_fallback_legacy' "
            "  AND file_path NOT LIKE 'webdav://%'"
        ).fetchone()[0]
        webdav_left = conn.execute(
            "SELECT COUNT(*) FROM files "
            "WHERE processing_status='parse_fallback_legacy' "
            "  AND file_path LIKE 'webdav://%'"
        ).fetchone()[0]
        snapshot["fallback_remaining"] = {
            "local": local_left,
            "webdav": webdav_left,
            "total": local_left + webdav_left,
        }

        if _table_exists(conn, "analysis_jobs"):
            jobs = conn.execute(
                "SELECT id, name, status, total_files, COALESCE(completed_files, 0) "
                "FROM analysis_jobs WHERE status IN ('active','paused') "
                "ORDER BY id DESC LIMIT 10"
            ).fetchall()
            snapshot["active_jobs"] = [
                {"id": jid, "name": name, "status": st,
                 "total": total_f, "done": done}
                for jid, name, st, total_f, done in jobs
            ]
        return snapshot
    finally:
        conn.close()


def render(snapshot: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"DB: {snapshot['db_path']}")
    lines.append(f"Total files: {snapshot['total_files']:,}")
    lines.append("")
    lines.append("Column non-NULL ratios:")
    total = snapshot["total_files"] or 1
    for col, info in snapshot["columns"].items():
        if not info.get("present"):
            lines.append(f"  {col:<22} (column missing)")
            continue
        lines.append(
            f"  {col:<22} {info['non_null']:>7,} / {total:,}  ({info['pct']:>5.1f}%)"
        )
    lines.append("")
    lines.append("Format breakdown:")
    for fmt, cnt in sorted(snapshot["format_breakdown"].items(),
                           key=lambda kv: -kv[1]):
        lines.append(f"  {fmt:<10} {cnt:>7,}")
    if snapshot["fts"]:
        lines.append("")
        lines.append("FTS:")
        for k, v in snapshot["fts"].items():
            lines.append(f"  {k:<22} {v}")
    rem = snapshot["fallback_remaining"]
    lines.append("")
    lines.append(f"parse_fallback_legacy remaining: "
                 f"local={rem['local']:,}, webdav={rem['webdav']:,}, "
                 f"total={rem['total']:,}")
    if snapshot["active_jobs"]:
        lines.append("")
        lines.append("Active analysis_jobs:")
        for job in snapshot["active_jobs"]:
            pct = (job["done"] / job["total"] * 100) if job["total"] else 0
            lines.append(
                f"  #{job['id']} [{job['status']}] {job['name']}  "
                f"{job['done']:,}/{job['total']:,} ({pct:.1f}%)"
            )
    return "\n".join(lines)


def diff(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    lines = ["Column delta (Before -> After):"]
    for col, after_info in after["columns"].items():
        if not after_info.get("present"):
            continue
        before_info = before["columns"].get(col, {})
        b = before_info.get("non_null", 0) if before_info.get("present") else 0
        a = after_info["non_null"]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {col:<22} {b:>7,} -> {a:>7,}   ({sign}{delta:,})")
    lines.append("")
    rem_b = before["fallback_remaining"]["total"]
    rem_a = after["fallback_remaining"]["total"]
    lines.append(f"parse_fallback_legacy: {rem_b:,} -> {rem_a:,} "
                 f"({rem_b - rem_a:,} resolved)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", action="store_true",
                    help="Emit raw JSON instead of human report")
    args = ap.parse_args()

    snapshot = collect(Path(args.db))
    if args.json:
        json.dump(snapshot, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(render(snapshot))
    return 0


if __name__ == "__main__":
    sys.exit(main())
