"""
Tier Migration Tool

Migrates the database to a new AI tier by:
1. Backing up current database
2. Recreating vec_files table with new dimension
3. Clearing tier metadata to force reprocessing

Usage:
    python backend/db/migrate_tier.py --tier standard
    python backend/db/migrate_tier.py --tier pro
    python backend/db/migrate_tier.py --tier ultra
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.sqlite_client import SQLiteDB
from backend.utils.tier_config import get_tier_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_database(db_path: Path) -> Path:
    """Create a backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".backup_{timestamp}.db")

    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    logger.info("Backup created successfully")

    return backup_path


def main():
    parser = argparse.ArgumentParser(description="Migrate database to a new AI tier")
    parser.add_argument(
        "--tier",
        required=True,
        choices=['standard', 'pro', 'ultra'],
        help="Target tier to migrate to"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip database backup (not recommended)"
    )
    parser.add_argument(
        "--db-path",
        help="Custom database path (default: ./imageparser.db)"
    )

    args = parser.parse_args()

    # Get tier configuration
    tier_config = get_tier_config(args.tier)
    if not tier_config:
        logger.error(f"Invalid tier: {args.tier}")
        sys.exit(1)

    visual_config = tier_config.get("visual", {})
    new_dimension = visual_config.get("dimensions")
    visual_model = visual_config.get("model")

    if not new_dimension:
        logger.error(f"Cannot determine embedding dimension for tier: {args.tier}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"TIER MIGRATION: {args.tier.upper()}")
    logger.info(f"  Visual Model: {visual_model}")
    logger.info(f"  Embedding Dimension: {new_dimension}")
    logger.info("=" * 60)

    # Backup database
    db_path = Path(args.db_path) if args.db_path else Path("imageparser.db")

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    if not args.no_backup:
        backup_path = backup_database(db_path)
        logger.info(f"Backup saved: {backup_path}")
    else:
        logger.warning("Skipping backup (--no-backup flag)")

    # Perform migration
    logger.info("\nStarting migration...")
    db = SQLiteDB(str(db_path))

    # Check current state
    current_tier = db.get_db_tier()
    current_dimension = db.get_db_embedding_dimension()

    logger.info(f"Current DB state:")
    logger.info(f"  Tier: {current_tier or 'empty'}")
    logger.info(f"  Dimension: {current_dimension}")

    if current_tier == args.tier and current_dimension == new_dimension:
        logger.warning(f"Database is already at tier {args.tier} with dimension {new_dimension}")
        logger.info("No migration needed.")
        db.close()
        return

    # Confirm migration
    logger.warning("\n⚠️  WARNING: This will delete all existing embeddings!")
    logger.warning("    You will need to reprocess all files after migration.")

    response = input("\nContinue with migration? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        logger.info("Migration cancelled.")
        db.close()
        return

    # Migrate
    success = db.migrate_tier(args.tier, new_dimension)
    db.close()

    if success:
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUCCESSFUL")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Update config.yaml:")
        logger.info(f"   ai_mode.override: {args.tier}")
        logger.info("\n2. Reprocess all files:")
        logger.info("   python backend/pipeline/ingest_engine.py --discover \"path/to/images\"")
        logger.info("\n3. Or reprocess via GUI: Settings → Reinstall/Verify")
    else:
        logger.error("\nMigration failed!")
        if not args.no_backup:
            logger.info(f"Restore backup: cp {backup_path} {db_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
