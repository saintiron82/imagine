"""
Migration: v3 P0 schema expansion.

Adds 15 columns to the files table for 2-Stage Vision Pipeline,
path abstraction, and embedding version tracking.
Rebuilds FTS5 with 4 additional columns (image_type, scene_type, art_style, folder_tags).

Usage:
    python -m backend.db.migrations.v3_p0_schema
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Columns to add: (name, type_with_default)
V3_COLUMNS = [
    # 2-Stage Vision classification
    ("image_type", "TEXT"),
    ("art_style", "TEXT"),
    ("color_palette", "TEXT"),
    ("scene_type", "TEXT"),
    ("time_of_day", "TEXT"),
    ("weather", "TEXT"),
    ("character_type", "TEXT"),
    ("item_type", "TEXT"),
    ("ui_type", "TEXT"),
    ("structured_meta", "TEXT"),
    # Path abstraction
    ("storage_root", "TEXT"),
    ("relative_path", "TEXT"),
    # Embedding version tracking
    ("embedding_model", "TEXT DEFAULT 'clip-ViT-L-14'"),
    ("embedding_version", "INTEGER DEFAULT 1"),
]

V3_INDEXES = [
    ("idx_image_type", "image_type"),
    ("idx_art_style", "art_style"),
    ("idx_scene_type", "scene_type"),
    ("idx_relative_path", "relative_path"),
]


def migrate():
    """Run v3 P0 schema migration."""
    db = SQLiteDB()
    cursor = db.conn.cursor()

    try:
        # ── Step 1: Add new columns ──
        existing = {row[1] for row in cursor.execute("PRAGMA table_info(files)").fetchall()}

        added = 0
        for col_name, col_def in V3_COLUMNS:
            if col_name not in existing:
                sql = f"ALTER TABLE files ADD COLUMN {col_name} {col_def}"
                cursor.execute(sql)
                added += 1
                logger.info(f"  + {col_name} ({col_def})")

        if added:
            db.conn.commit()
            logger.info(f"Added {added} columns to files table")
        else:
            logger.info("All v3 columns already exist — skipping ALTER TABLE")

        # ── Step 2: Create indexes ──
        for idx_name, col_name in V3_INDEXES:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON files({col_name})")
        db.conn.commit()
        logger.info(f"Ensured {len(V3_INDEXES)} indexes")

        # ── Step 3: FTS5 rebuild will happen automatically ──
        # sqlite_client._ensure_fts() checks column mismatch on every connect.
        # After updating _FTS_COLUMNS in sqlite_client.py (P0-4),
        # the next connect will auto-rebuild FTS5.
        logger.info("FTS5 rebuild will trigger on next SQLiteDB connect (after _FTS_COLUMNS update)")

        logger.info("v3 P0 schema migration complete!")

    except Exception as e:
        db.conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
