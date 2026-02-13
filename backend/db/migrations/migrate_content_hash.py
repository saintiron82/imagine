"""
Migration: Backfill content_hash for existing files in the database.

Scans all files with NULL content_hash, checks if the file still exists on disk,
and computes the SHA256 content hash.

Usage:
    python -m backend.db.migrations.migrate_content_hash
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.db.sqlite_client import SQLiteDB
from backend.utils.content_hash import compute_content_hash

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def migrate():
    db = SQLiteDB()
    cursor = db.conn.cursor()

    # Find all files with NULL content_hash
    cursor.execute("SELECT id, file_path FROM files WHERE content_hash IS NULL")
    rows = cursor.fetchall()

    if not rows:
        logger.info("All files already have content_hash. Nothing to do.")
        return

    logger.info(f"Found {len(rows)} files without content_hash. Computing...")

    updated = 0
    missing = 0
    errors = 0

    for row in rows:
        file_id = row[0]
        file_path = row[1]

        p = Path(file_path)
        if not p.exists():
            missing += 1
            continue

        try:
            ch = compute_content_hash(p)
            cursor.execute(
                "UPDATE files SET content_hash = ? WHERE id = ?",
                (ch, file_id)
            )
            updated += 1
        except Exception as e:
            logger.warning(f"  Error computing hash for {file_path}: {e}")
            errors += 1

    db.conn.commit()
    logger.info(
        f"Done. Updated: {updated}, Missing files: {missing}, Errors: {errors}"
    )


if __name__ == "__main__":
    migrate()
