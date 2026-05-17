#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


_REASON_TABLES = {
    "missing_objects": "file_objects",
    "missing_relations": "file_spatial_relations",
    "missing_depth_layers": "file_depth_layers",
}


def select_reprocess_candidates(
    db_path: Path,
    reason: str,
    limit: int,
    file_ids: list[int] | None = None,
) -> list[dict]:
    table = _REASON_TABLES.get(reason)
    if not table:
        raise ValueError(f"unsupported reason: {reason}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        params: list[object] = []
        file_filter = ""
        if file_ids:
            placeholders = ", ".join("?" for _ in file_ids)
            file_filter = f" AND f.id IN ({placeholders})"
            params.extend(file_ids)
        params.append(limit)

        rows = conn.execute(
            f"""SELECT f.id, f.file_path
                FROM files f
                WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                  {file_filter}
                  AND NOT EXISTS (
                    SELECT 1 FROM {table} evidence WHERE evidence.file_id = f.id
                  )
                ORDER BY f.id
                LIMIT ?""",
            params,
        ).fetchall()
        return [
            {"id": row["id"], "file_path": row["file_path"], "reason": reason}
            for row in rows
        ]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    parser.add_argument("--reason", default="missing_relations", choices=sorted(_REASON_TABLES))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--file-id", type=int, action="append", dest="file_ids")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    rows = select_reprocess_candidates(
        Path(args.db),
        args.reason,
        args.limit,
        file_ids=args.file_ids,
    )
    payload = {
        "dry_run": not args.execute,
        "execute": bool(args.execute),
        "candidates": rows,
    }
    if args.execute:
        paths = [row["file_path"] for row in rows]
        payload["command"] = [
            ".venv/bin/python",
            "backend/pipeline/ingest_engine.py",
            "--files",
            json.dumps(paths, ensure_ascii=False),
            "--no-skip",
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
