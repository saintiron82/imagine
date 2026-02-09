"""
Migration: Create vec_text virtual table for T-axis text embeddings.

Adds a sqlite-vec virtual table (1024-dim) for text embeddings
generated from ai_caption + ai_tags via Qwen3-Embedding-0.6B.

Usage:
    python -m backend.db.migrations.v3_p2_text_vec

After migration, run:
    python tools/reindex_v3.py --text-embedding
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def migrate():
    """Create vec_text virtual table for T-axis text embeddings."""
    from backend.db.sqlite_client import SQLiteDB
    from backend.utils.config import get_config

    cfg = get_config()
    # Tier config takes precedence over global fallback
    try:
        from backend.utils.tier_config import get_active_tier
        _, tier_config = get_active_tier()
        dimensions = tier_config.get("text_embed", {}).get("dimensions") or cfg.get("embedding.text.dimensions", 1024)
    except Exception:
        dimensions = cfg.get("embedding.text.dimensions", 1024)

    db = SQLiteDB()
    cursor = db.conn.cursor()

    try:
        # Check if vec_text already exists
        try:
            count = cursor.execute("SELECT COUNT(*) FROM vec_text").fetchone()[0]
            logger.info(f"vec_text already exists with {count} entries — skipping creation")
            db.close()
            return
        except Exception:
            pass  # Table doesn't exist, proceed with creation

        # Create vec_text virtual table
        cursor.execute(f"""
            CREATE VIRTUAL TABLE vec_text USING vec0(
                file_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dimensions}]
            )
        """)
        logger.info(f"vec_text created with FLOAT[{dimensions}]")

        db.conn.commit()

        total = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        logger.info(f"Migration complete. {total} files need T-axis embedding.")
        logger.info(f"Run: python tools/reindex_v3.py --text-embedding")

    except Exception as e:
        db.conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate()
