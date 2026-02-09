"""
Migration: Recreate vec_files and vec_layers with 1152 dimensions for SigLIP 2.

sqlite-vec virtual tables cannot be ALTERed, so we DROP and recreate.
All existing embeddings are lost — run `tools/reindex_v3.py --embedding-only` after.

Usage:
    python -m backend.db.migrations.v3_p1_vec
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def migrate():
    """Recreate vec_files and vec_layers with new dimensions."""
    from backend.db.sqlite_client import SQLiteDB
    from backend.utils.config import get_config

    cfg = get_config()
    # Tier config takes precedence over global fallback
    try:
        from backend.utils.tier_config import get_active_tier
        _, tier_config = get_active_tier()
        dimensions = tier_config.get("visual", {}).get("dimensions") or cfg.get("embedding.visual.dimensions", 768)
        model_name = tier_config.get("visual", {}).get("model") or cfg.get("embedding.visual.model", "unknown")
    except Exception:
        dimensions = cfg.get("embedding.visual.dimensions", 768)
        model_name = cfg.get("embedding.visual.model", "unknown")

    db = SQLiteDB()
    cursor = db.conn.cursor()

    try:
        # Count existing embeddings for reference
        try:
            count = cursor.execute("SELECT COUNT(*) FROM vec_files").fetchone()[0]
            logger.info(f"Existing vec_files entries: {count} (will be dropped)")
        except Exception:
            logger.info("No existing vec_files table")

        # Recreate vec_files
        cursor.execute("DROP TABLE IF EXISTS vec_files")
        cursor.execute(f"""
            CREATE VIRTUAL TABLE vec_files USING vec0(
                file_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dimensions}]
            )
        """)
        logger.info(f"vec_files recreated with FLOAT[{dimensions}]")

        # Recreate vec_layers
        cursor.execute("DROP TABLE IF EXISTS vec_layers")
        cursor.execute(f"""
            CREATE VIRTUAL TABLE vec_layers USING vec0(
                layer_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dimensions}]
            )
        """)
        logger.info(f"vec_layers recreated with FLOAT[{dimensions}]")

        # Reset embedding_model and embedding_version for all files
        cursor.execute("""
            UPDATE files SET
                embedding_model = ?,
                embedding_version = 0
        """, (model_name,))
        db.conn.commit()

        total = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        logger.info(f"Reset embedding_version to 0 for {total} files")
        logger.info(f"Migration complete. Run: python tools/reindex_v3.py --embedding-only")

    except Exception as e:
        db.conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
