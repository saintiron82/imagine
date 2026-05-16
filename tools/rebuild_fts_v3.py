"""
FTS rebuild — v4 schema with VLM output and spatial object evidence.

Drops the existing FTS table and rebuilds with 6 columns,
backfilling from the files table. Idempotent — safe to re-run.

Usage:
    python tools/rebuild_fts_v3.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fts-rebuild")


_FTS_SCHEMA = """
DROP TRIGGER IF EXISTS files_fts_insert;
DROP TRIGGER IF EXISTS files_fts_update;
DROP TRIGGER IF EXISTS files_fts_delete;
DROP TABLE IF EXISTS files_fts;

CREATE VIRTUAL TABLE files_fts USING fts5(
    meta_strong,
    meta_weak,
    caption,
    ai_tags,
    classification,
    spatial
);

CREATE TRIGGER files_fts_insert AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, meta_strong, meta_weak, caption, ai_tags, classification, spatial)
    VALUES (new.id, '', '', '', '', '', '');
END;

CREATE TRIGGER files_fts_delete AFTER DELETE ON files BEGIN
    DELETE FROM files_fts WHERE rowid = old.id;
END;
"""

_FILE_OBJECTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    ko_name TEXT,
    primary_location TEXT NOT NULL,
    locations TEXT NOT NULL,
    extent TEXT,
    confidence TEXT,
    source TEXT NOT NULL DEFAULT 'vlm',
    spatial_text TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_file_objects_file_id ON file_objects(file_id);
CREATE INDEX IF NOT EXISTS idx_file_objects_name ON file_objects(name);
CREATE INDEX IF NOT EXISTS idx_file_objects_location ON file_objects(primary_location);
CREATE INDEX IF NOT EXISTS idx_file_objects_name_location ON file_objects(name, primary_location);
"""


def _build_caption(mc_caption):
    return str(mc_caption) if mc_caption else ""


def _build_ai_tags(raw):
    if not raw:
        return ""
    try:
        tags = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(tags, list):
            return " ".join(str(t) for t in tags if t)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(raw)


def _build_classification(image_type, scene_type, art_style,
                          character_type, item_type, time_of_day, weather):
    parts = [image_type, scene_type, art_style,
             character_type, item_type, time_of_day, weather]
    return " ".join(str(p) for p in parts if p)


def _replace_file_objects(cur, file_id, objects):
    cur.execute("DELETE FROM file_objects WHERE file_id = ?", (file_id,))
    for obj in objects or []:
        cur.execute(
            """INSERT INTO file_objects
               (file_id, name, ko_name, primary_location, locations, extent, confidence, source, spatial_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_id,
                obj.get("name") or "",
                obj.get("ko_name") or "",
                obj.get("primary_location") or "",
                json.dumps(obj.get("locations") or [], ensure_ascii=False),
                obj.get("extent") or "",
                obj.get("confidence") or "low",
                "vlm",
                SQLiteDB._build_fts_spatial([obj]),
            ),
        )


def _build_meta_strong(file_name, ocr_text, meta):
    """Mirror sqlite_client._build_fts_meta_strong (subset: drops user_tags/fonts)."""
    parts = []
    if file_name:
        parts.append(str(file_name))
    # layer_names from metadata.layer_tree
    tree = meta.get("layer_tree") if isinstance(meta, dict) else None
    if isinstance(tree, dict):
        names = []

        def _walk(node):
            n = node.get("name")
            if n and n != "Root":
                names.append(n)
            for child in node.get("children", []) or []:
                _walk(child)

        _walk(tree)
        if names:
            parts.append(" ".join(dict.fromkeys(names)))
    fonts = meta.get("used_fonts", []) if isinstance(meta, dict) else []
    if isinstance(fonts, list) and fonts:
        parts.append(" ".join(fonts))
    if ocr_text:
        parts.append(str(ocr_text))
    return " ".join(parts)


def _build_meta_weak(file_path, folder_path, relative_path, meta,
                     image_type, scene_type, art_style, folder_tags_raw):
    parts = []
    for p in (file_path, folder_path, relative_path):
        if p:
            parts.append(str(p))
    text_content = meta.get("text_content", []) if isinstance(meta, dict) else []
    if isinstance(text_content, list) and text_content:
        parts.append(" ".join(str(t) for t in text_content if t))
    if folder_tags_raw:
        try:
            ft = json.loads(folder_tags_raw)
            if isinstance(ft, list):
                parts.append(" ".join(str(t) for t in ft))
        except (json.JSONDecodeError, TypeError):
            pass
    for v in (image_type, scene_type, art_style):
        if v:
            parts.append(str(v))
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Schema rebuild
    logger.info("Dropping & recreating files_fts (6 columns)...")
    cur.executescript(_FILE_OBJECTS_SCHEMA)
    cur.executescript(_FTS_SCHEMA)
    conn.commit()

    # Backfill
    logger.info("Backfilling FTS rows from files...")
    rows = cur.execute(
        "SELECT id, file_path, file_name, mc_caption, ai_tags, "
        "metadata, ocr_text, folder_path, relative_path, "
        "image_type, scene_type, art_style, folder_tags, "
        "character_type, item_type, time_of_day, weather, structured_meta "
        "FROM files"
    ).fetchall()

    inserted = 0
    for r in rows:
        (fid, file_path, file_name, mc_caption, ai_tags_raw,
         metadata_str, ocr_text, folder_path, relative_path,
         image_type, scene_type, art_style, folder_tags_raw,
         character_type, item_type, time_of_day, weather, structured_meta) = r

        try:
            meta = json.loads(metadata_str) if metadata_str else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}

        meta_strong = _build_meta_strong(file_name, ocr_text, meta)
        meta_weak = _build_meta_weak(
            file_path, folder_path, relative_path, meta,
            image_type, scene_type, art_style, folder_tags_raw,
        )
        caption_col = _build_caption(mc_caption)
        ai_tags_col = _build_ai_tags(ai_tags_raw)
        classification_col = _build_classification(
            image_type, scene_type, art_style,
            character_type, item_type, time_of_day, weather,
        )
        spatial_objects = SQLiteDB._normalize_spatial_objects_from_meta(structured_meta)
        spatial_col = SQLiteDB._build_fts_spatial(spatial_objects)
        _replace_file_objects(cur, fid, spatial_objects)

        cur.execute(
            "INSERT INTO files_fts(rowid, meta_strong, meta_weak, caption, ai_tags, classification, spatial) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (fid, meta_strong, meta_weak, caption_col, ai_tags_col, classification_col, spatial_col),
        )
        inserted += 1

    conn.commit()
    logger.info(f"Inserted {inserted} FTS rows")

    # Bump system_meta version so SQLiteDB._ensure_fts() doesn't trigger another rebuild
    try:
        cur.execute(
            "INSERT OR REPLACE INTO system_meta(key, value) VALUES (?, ?)",
            ("fts_index_version", "4"),
        )
        conn.commit()
        logger.info("system_meta.fts_index_version = 4")
    except sqlite3.OperationalError:
        logger.warning("system_meta table missing — skipping version stamp")

    # Verify
    n_fts = cur.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
    n_caption = cur.execute("SELECT COUNT(*) FROM files_fts WHERE caption != ''").fetchone()[0]
    n_tags = cur.execute("SELECT COUNT(*) FROM files_fts WHERE ai_tags != ''").fetchone()[0]
    n_class = cur.execute("SELECT COUNT(*) FROM files_fts WHERE classification != ''").fetchone()[0]
    n_spatial = cur.execute("SELECT COUNT(*) FROM files_fts WHERE spatial != ''").fetchone()[0]
    n_objects = cur.execute("SELECT COUNT(*) FROM file_objects").fetchone()[0]
    logger.info(
        f"Verify: fts_rows={n_fts}, caption={n_caption}, ai_tags={n_tags}, "
        f"classification={n_class}, spatial={n_spatial}, file_objects={n_objects}"
    )

    conn.close()


if __name__ == "__main__":
    main()
