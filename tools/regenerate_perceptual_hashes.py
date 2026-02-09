"""
Regenerate perceptual_hash for files where it is NULL.

This script updates the perceptual_hash column for all files where it is
currently NULL, using the dHash64 algorithm on either thumbnails or original files.

Usage:
    python tools/regenerate_perceptual_hashes.py

Author: ImageParser v3.1
Date: 2026-02-09
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from backend.db.sqlite_client import SQLiteDB
from backend.utils.dhash import dhash64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Perceptual Hash Regeneration Tool")
    logger.info("=" * 60)

    # Connect to database
    try:
        db = SQLiteDB()
        logger.info(f"Connected to database: {db.db_path}")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return 1

    # Query files with NULL perceptual_hash
    cursor = db.conn.execute("""
        SELECT id, file_path, thumbnail_url
        FROM files
        WHERE perceptual_hash IS NULL
    """)

    rows = cursor.fetchall()
    total_count = len(rows)
    logger.info(f"Found {total_count} files with NULL perceptual_hash")

    if total_count == 0:
        logger.info("✅ All files already have perceptual_hash. Nothing to do.")
        return 0

    # Process each file
    success = 0
    failed = 0
    skipped = 0

    for file_id, file_path, thumb_url in rows:
        try:
            # Determine which image to use (priority: thumbnail → original)
            img_path = None

            if thumb_url and Path(thumb_url).exists():
                img_path = thumb_url
                source = "thumbnail"
            elif Path(file_path).exists():
                img_path = file_path
                source = "original"
            else:
                logger.warning(f"Skip {file_path}: neither thumbnail nor original file exists")
                skipped += 1
                continue

            # Load image and compute hash
            img = Image.open(img_path).convert("RGB")
            hash_val = dhash64(img)

            # Convert unsigned to signed (SQLite INTEGER is signed 64-bit)
            if hash_val >= 2**63:
                hash_val -= 2**64

            # Update database
            db.conn.execute(
                "UPDATE files SET perceptual_hash = ? WHERE id = ?",
                (hash_val, file_id)
            )

            logger.debug(f"✓ {Path(file_path).name}: hash={hash_val} (from {source})")
            success += 1

            # Commit every 10 files
            if success % 10 == 0:
                db.conn.commit()
                logger.info(f"  Progress: {success}/{total_count} ({success*100//total_count}%)")

        except Exception as e:
            logger.error(f"✗ {Path(file_path).name}: {e}")
            failed += 1

    # Final commit
    db.conn.commit()

    # Summary
    logger.info("=" * 60)
    logger.info("Regeneration Summary")
    logger.info("=" * 60)
    logger.info(f"Total files: {total_count}")
    logger.info(f"✅ Success: {success}")
    if failed > 0:
        logger.warning(f"❌ Failed: {failed}")
    if skipped > 0:
        logger.warning(f"⏭️  Skipped: {skipped} (files not found)")

    # Exit code
    if failed > 0 or skipped > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
