"""
Migration: Add ocr_text to FTS5 table.

Drops and recreates files_fts with ocr_text column,
then repopulates from existing data.

Usage:
    python -m backend.db.migrations.add_ocr_to_fts
"""

import sys
import logging
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def migrate():
    """Drop and recreate FTS5 table with ocr_text, repopulate data."""
    db = SQLiteDB()
    cursor = db.conn.cursor()

    try:
        logger.info("Starting FTS5 migration: adding ocr_text column...")

        # Drop existing FTS table and triggers
        cursor.executescript("""
            DROP TRIGGER IF EXISTS files_fts_insert;
            DROP TRIGGER IF EXISTS files_fts_update;
            DROP TRIGGER IF EXISTS files_fts_delete;
            DROP TABLE IF EXISTS files_fts;
        """)
        logger.info("Dropped old FTS5 table and triggers")

        # Recreate with ocr_text
        cursor.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                file_path,
                ai_caption,
                semantic_tags,
                ocr_text,
                user_note,
                user_tags
            );

            CREATE TRIGGER IF NOT EXISTS files_fts_insert AFTER INSERT ON files BEGIN
                INSERT INTO files_fts(rowid, file_path, ai_caption, semantic_tags, ocr_text, user_note, user_tags)
                VALUES (new.id, new.file_path, new.ai_caption,
                        json_extract(new.metadata, '$.semantic_tags'),
                        new.ocr_text,
                        new.user_note, new.user_tags);
            END;

            CREATE TRIGGER IF NOT EXISTS files_fts_update AFTER UPDATE ON files BEGIN
                DELETE FROM files_fts WHERE rowid = old.id;
                INSERT INTO files_fts(rowid, file_path, ai_caption, semantic_tags, ocr_text, user_note, user_tags)
                VALUES (new.id, new.file_path, new.ai_caption,
                        json_extract(new.metadata, '$.semantic_tags'),
                        new.ocr_text,
                        new.user_note, new.user_tags);
            END;

            CREATE TRIGGER IF NOT EXISTS files_fts_delete AFTER DELETE ON files BEGIN
                DELETE FROM files_fts WHERE rowid = old.id;
            END;
        """)
        logger.info("Created new FTS5 table with ocr_text")

        # Repopulate from existing data
        cursor.execute("""
            INSERT INTO files_fts(rowid, file_path, ai_caption, semantic_tags, ocr_text, user_note, user_tags)
            SELECT id, file_path, ai_caption,
                   json_extract(metadata, '$.semantic_tags'),
                   ocr_text,
                   user_note, user_tags
            FROM files
        """)
        db.conn.commit()

        count = cursor.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]
        logger.info(f"Repopulated FTS5 with {count} rows")
        logger.info("Migration complete!")

    except Exception as e:
        db.conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
