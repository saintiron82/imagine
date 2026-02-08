"""
Migrate existing JSON + ChromaDB data to PostgreSQL.

This script transfers all image metadata from JSON files and CLIP embeddings
from ChromaDB into the new PostgreSQL + pgvector database.

Usage:
    python tools/migrate_to_postgres.py [--dry-run]
"""

import json
import logging
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.pg_client import PostgresDB
from backend.vector.searcher import VectorSearcher
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def migrate(dry_run: bool = False):
    """
    Migrate JSON + ChromaDB data to PostgreSQL.

    Args:
        dry_run: If True, only count files without inserting
    """
    json_dir = Path("output/json")

    if not json_dir.exists():
        logger.error(f"JSON directory not found: {json_dir}")
        logger.error("Please ensure output/json exists with metadata files")
        return

    json_files = list(json_dir.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        logger.warning("No JSON files found to migrate")
        return

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Found {total_files} JSON files to migrate")

    if dry_run:
        logger.info("Dry run mode - no data will be inserted")
        return

    # Initialize database
    logger.info("Connecting to PostgreSQL...")
    try:
        pg_db = PostgresDB()
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        logger.error("\nPlease ensure:")
        logger.error("1. PostgreSQL is running on localhost:5432")
        logger.error("2. Database 'imageparser' exists")
        logger.error("3. pgvector extension is installed")
        logger.error("\nInstallation guide:")
        logger.error("  - PostgreSQL: https://www.postgresql.org/download/")
        logger.error("  - pgvector: https://github.com/pgvector/pgvector")
        return

    # Initialize schema
    logger.info("Initializing database schema...")
    try:
        pg_db.init_schema()
    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return

    # Initialize ChromaDB searcher (optional - for backward compatibility)
    logger.info("Checking for ChromaDB (optional)...")
    vector_searcher = None
    try:
        # Try to import and use ChromaDB if available
        from backend.vector.searcher import VectorSearcher
        vector_searcher = VectorSearcher()
        logger.info("✅ ChromaDB found - will migrate existing embeddings")
    except ImportError:
        logger.info("ℹ️ ChromaDB not installed - will use zero embeddings")
        logger.info("   (This is fine for new installations)")
    except Exception as e:
        logger.warning(f"⚠️ ChromaDB connection failed: {e}")
        logger.info("   Migration will continue with zero embeddings")
        logger.info("   (Embeddings will be generated on next file processing)")

    # Migration statistics
    migrated = 0
    failed = []
    missing_embeddings = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting migration of {total_files} files...")
    logger.info(f"{'='*60}\n")

    for idx, json_file in enumerate(json_files, 1):
        try:
            # Load metadata from JSON
            with open(json_file, encoding='utf-8') as f:
                metadata = json.load(f)

            file_path = metadata.get('file_path')
            if not file_path:
                logger.warning(f"⚠️ Skipped {json_file.name}: missing file_path")
                failed.append((json_file.name, "Missing file_path"))
                continue

            # Get CLIP embedding from ChromaDB
            embedding = None
            if vector_searcher:
                try:
                    # ChromaDB uses file_path as ID (with forward slashes)
                    chroma_id = file_path.replace("\\", "/")
                    vector_data = vector_searcher.image_collection.get(
                        ids=[chroma_id],
                        include=["embeddings"]
                    )

                    if vector_data['embeddings'] and len(vector_data['embeddings']) > 0:
                        embedding = np.array(vector_data['embeddings'][0], dtype=np.float32)
                    else:
                        logger.warning(f"⚠️ No embedding found for: {file_path}")
                        missing_embeddings += 1
                        # Create zero embedding as placeholder
                        embedding = np.zeros(768, dtype=np.float32)

                except Exception as e:
                    logger.warning(f"⚠️ Failed to get embedding for {file_path}: {e}")
                    missing_embeddings += 1
                    embedding = np.zeros(768, dtype=np.float32)
            else:
                # No ChromaDB connection - use zero embedding
                embedding = np.zeros(768, dtype=np.float32)

            # Insert into PostgreSQL
            pg_db.insert_file(file_path, metadata, embedding)
            migrated += 1

            # Progress indicator
            if migrated % 10 == 0:
                logger.info(f"Progress: {migrated}/{total_files} ({migrated/total_files*100:.1f}%)")

        except Exception as e:
            failed.append((json_file.name, str(e)))
            logger.error(f"❌ Failed to migrate {json_file.name}: {e}")

    # Final statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Migration complete!")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successfully migrated: {migrated}/{total_files}")
    logger.info(f"❌ Failed: {len(failed)}")
    logger.info(f"⚠️ Missing embeddings: {missing_embeddings}")

    if failed:
        logger.warning("\nFailed files:")
        for fname, error in failed[:10]:  # Show first 10 errors
            logger.warning(f"  - {fname}: {error}")
        if len(failed) > 10:
            logger.warning(f"  ... and {len(failed) - 10} more")

    # Database statistics
    logger.info("\nDatabase statistics:")
    stats = pg_db.get_stats()
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Total layers: {stats['total_layers']}")
    logger.info(f"  Files with AI captions: {stats['files_with_ai_caption']}")
    logger.info(f"  Average layers per file: {stats['avg_layers_per_file']}")
    logger.info(f"  Format distribution: {stats['format_distribution']}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Next steps:")
    logger.info(f"{'='*60}")
    logger.info(f"1. Verify data:")
    logger.info(f"   python tools/verify_migration.py")
    logger.info(f"2. Test vector search:")
    logger.info(f"   python backend/cli_search_pg.py 'cartoon city'")
    logger.info(f"3. Backup JSON files (keep for 1 month):")
    logger.info(f"   mkdir output/json_backup")
    logger.info(f"   move output/json/*.json output/json_backup/")
    logger.info(f"4. Optional: Remove ChromaDB if satisfied")
    logger.info(f"   (Keep chroma_db/ folder for now as backup)")

    pg_db.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON + ChromaDB to PostgreSQL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files without inserting data"
    )
    args = parser.parse_args()

    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
