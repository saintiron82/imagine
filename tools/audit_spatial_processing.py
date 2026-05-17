#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


def collect_spatial_processing_stats(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM files WHERE mc_caption IS NOT NULL AND mc_caption != ''"
        ).fetchone()[0]
        missing_objects = conn.execute(
            """SELECT COUNT(*) FROM files f
               WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                 AND NOT EXISTS (SELECT 1 FROM file_objects o WHERE o.file_id = f.id)"""
        ).fetchone()[0]
        missing_relations = conn.execute(
            """SELECT COUNT(*) FROM files f
               WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                 AND NOT EXISTS (SELECT 1 FROM file_spatial_relations r WHERE r.file_id = f.id)"""
        ).fetchone()[0]
        missing_depth_layers = conn.execute(
            """SELECT COUNT(*) FROM files f
               WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                 AND NOT EXISTS (SELECT 1 FROM file_depth_layers d WHERE d.file_id = f.id)"""
        ).fetchone()[0]
        return {
            "total_files_with_caption": total,
            "missing_objects": missing_objects,
            "missing_relations": missing_relations,
            "missing_depth_layers": missing_depth_layers,
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    args = parser.parse_args()
    print(
        json.dumps(
            collect_spatial_processing_stats(Path(args.db)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
