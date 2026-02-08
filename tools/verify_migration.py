"""
Verify PostgreSQL migration integrity.

This script checks that migrated data matches the original JSON files.

Usage:
    python tools/verify_migration.py
"""

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.pg_client import PostgresDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify():
    """Verify migration integrity."""
    json_dir = Path("output/json")

    if not json_dir.exists():
        logger.error(f"JSON directory not found: {json_dir}")
        return

    json_files = list(json_dir.glob("*.json"))
    total_files = len(json_files)

    logger.info(f"Verifying {total_files} files...")

    # Connect to PostgreSQL
    pg_db = PostgresDB()

    matched = 0
    missing = []
    mismatched = []

    for json_file in json_files:
        with open(json_file, encoding='utf-8') as f:
            json_data = json.load(f)

        file_path = json_data.get('file_path')
        if not file_path:
            continue

        # Get from PostgreSQL
        pg_data = pg_db.get_file_by_path(file_path)

        if not pg_data:
            missing.append(file_path)
            continue

        # Verify key fields
        checks_passed = True

        if pg_data['file_name'] != json_data.get('file_name'):
            mismatched.append((file_path, 'file_name', pg_data['file_name'], json_data.get('file_name')))
            checks_passed = False

        if pg_data['format'] != json_data.get('format'):
            mismatched.append((file_path, 'format', pg_data['format'], json_data.get('format')))
            checks_passed = False

        # Verify metadata JSONB contains nested data
        if pg_data['metadata']:
            if 'layer_tree' not in pg_data['metadata'] and json_data.get('layer_tree'):
                mismatched.append((file_path, 'layer_tree', None, 'Expected layer_tree'))
                checks_passed = False

        if checks_passed:
            matched += 1

        if (matched + len(missing) + len(mismatched)) % 10 == 0:
            logger.info(f"Progress: {matched + len(missing) + len(mismatched)}/{total_files}")

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"Verification complete!")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Matched: {matched}/{total_files}")
    logger.info(f"❌ Missing: {len(missing)}")
    logger.info(f"⚠️ Mismatched: {len(mismatched)}")

    if missing:
        logger.warning("\nMissing files (first 10):")
        for path in missing[:10]:
            logger.warning(f"  - {path}")

    if mismatched:
        logger.warning("\nMismatched fields (first 10):")
        for path, field, pg_val, json_val in mismatched[:10]:
            logger.warning(f"  - {path}")
            logger.warning(f"    Field: {field}")
            logger.warning(f"    PostgreSQL: {pg_val}")
            logger.warning(f"    JSON: {json_val}")

    # Database stats
    stats = pg_db.get_stats()
    logger.info(f"\nDatabase statistics:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Format distribution: {stats['format_distribution']}")

    pg_db.close()

    if matched == total_files:
        logger.info("\n✅ All files verified successfully!")
        return 0
    else:
        logger.error(f"\n❌ Verification failed: {total_files - matched} issues found")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
