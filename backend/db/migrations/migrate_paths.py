"""
Migration: Populate storage_root + relative_path from existing file_path.

All paths are normalized to POSIX style (forward slashes).
Does NOT use os.path.normpath (would revert to backslashes on Windows).

Usage:
    python -m backend.db.migrations.migrate_paths
    python -m backend.db.migrations.migrate_paths --root "C:/Projects/Assets"
"""

import sys
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Heuristic markers to split storage_root from relative_path
_MARKERS = ['/assets/', '/Assets/', '/resources/', '/Resources/',
            '/images/', '/Images/', '/sprites/', '/Sprites/']


def migrate(default_root: str = None):
    """Populate storage_root + relative_path for rows where relative_path IS NULL."""
    db = SQLiteDB()
    cursor = db.conn.cursor()

    try:
        cursor.execute("SELECT id, file_path FROM files WHERE relative_path IS NULL")
        rows = cursor.fetchall()

        if not rows:
            logger.info("No rows need path migration")
            db.close()
            return

        migrated = 0
        for row_id, file_path in rows:
            normalized = file_path.replace('\\', '/')

            root = None
            if default_root:
                root = default_root.replace('\\', '/')
            else:
                for marker in _MARKERS:
                    idx = normalized.find(marker)
                    if idx != -1:
                        root = normalized[:idx]
                        break

            if root is None:
                # Fallback: parent directory as root
                parts = normalized.rsplit('/', 1)
                root = parts[0] if len(parts) > 1 else ''

            relative = normalized[len(root):].lstrip('/')

            cursor.execute(
                "UPDATE files SET storage_root = ?, relative_path = ? WHERE id = ?",
                (root, relative, row_id)
            )
            migrated += 1

        db.conn.commit()
        logger.info(f"Migrated {migrated} paths (POSIX normalized)")

    except Exception as e:
        db.conn.rollback()
        logger.error(f"Path migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate file_path → storage_root + relative_path")
    parser.add_argument("--root", type=str, default=None, help="Default storage root path")
    args = parser.parse_args()
    migrate(default_root=args.root)
