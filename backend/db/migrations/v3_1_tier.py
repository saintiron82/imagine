"""
Database Migration: v3.1 3-Tier AI Mode Metadata

Adds 5 columns to the files table for tier-aware metadata tracking.

Usage:
    python backend/db/migrations/v3_1_tier.py
    python backend/db/migrations/v3_1_tier.py --db-path /path/to/imageparser.db
"""

import sqlite3
import sys
import argparse
from pathlib import Path


def get_default_db_path() -> Path:
    """Get default database path (project root / imageparser.db)."""
    return Path(__file__).parent.parent.parent.parent / "imageparser.db"


def migrate_tier_columns(db_path: str):
    """
    Add v3.1 tier columns to existing database.

    Columns added:
    - mode_tier: AI tier (standard/pro/ultra)
    - caption_model: VLM model used
    - text_embed_model: Text embedding model used
    - runtime_version: Ollama/runtime version
    - preprocess_params: JSON preprocessing parameters

    Args:
        db_path: Path to SQLite database file
    """
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Create database first or run: python tools/db_reset.py")
        sys.exit(1)

    print(f"🔄 Migrating database: {db_path}")
    print(f"   Adding v3.1 tier metadata columns...")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(files)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    print(f"   Current columns: {len(existing_cols)}")

    # Define new columns
    new_columns = [
        ("mode_tier", "TEXT DEFAULT 'pro'"),
        ("caption_model", "TEXT"),
        ("text_embed_model", "TEXT"),
        ("runtime_version", "TEXT"),
        ("preprocess_params", "TEXT"),
    ]

    # Add columns if they don't exist
    added = 0
    for col_name, col_type in new_columns:
        if col_name not in existing_cols:
            try:
                sql = f"ALTER TABLE files ADD COLUMN {col_name} {col_type}"
                cursor.execute(sql)
                print(f"   ✅ Added: {col_name} {col_type}")
                added += 1
            except sqlite3.OperationalError as e:
                print(f"   ⚠️  Warning: Could not add {col_name}: {e}")
        else:
            print(f"   ⏭️  Skipped: {col_name} (already exists)")

    conn.commit()

    # Verify migration
    cursor.execute("PRAGMA table_info(files)")
    final_cols = {row[1] for row in cursor.fetchall()}
    print(f"\n📊 Migration summary:")
    print(f"   Columns before: {len(existing_cols)}")
    print(f"   Columns added: {added}")
    print(f"   Columns after: {len(final_cols)}")

    # Check if all v3.1 columns exist
    missing = []
    for col_name, _ in new_columns:
        if col_name not in final_cols:
            missing.append(col_name)

    if missing:
        print(f"\n⚠️  Warning: Some columns are missing: {', '.join(missing)}")
        sys.exit(1)
    else:
        print(f"\n✅ Migration complete! All v3.1 tier columns present.")

    # Show sample data
    cursor.execute("SELECT id, file_name, mode_tier, caption_model FROM files LIMIT 1")
    row = cursor.fetchone()
    if row:
        print(f"\n📝 Sample row:")
        print(f"   ID: {row[0]}")
        print(f"   File: {row[1]}")
        print(f"   Tier: {row[2] or '(null - will be set on next processing)'}")
        print(f"   Caption Model: {row[3] or '(null - will be set on next processing)'}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate database to v3.1 tier schema"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to imageparser.db (default: project root)"
    )

    args = parser.parse_args()

    db_path = args.db_path or get_default_db_path()

    print("=" * 60)
    print("🔧 ImageParser v3.1 Tier Migration")
    print("=" * 60)

    migrate_tier_columns(db_path)

    print("\n💡 Next steps:")
    print("   1. Re-process files to populate tier metadata:")
    print("      python backend/pipeline/ingest_engine.py --discover \"C:\\path\\to\\assets\"")
    print("   2. Or continue using existing data (tier metadata will be null)")
    print()


if __name__ == "__main__":
    main()
