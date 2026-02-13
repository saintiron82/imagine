"""
Export database and thumbnails as a portable zip archive.

Usage:
    python -m backend.api_export --output ./archive.zip
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def export_archive(output_path: str) -> dict:
    """
    Export imageparser.db + thumbnails as a zip archive.

    Returns:
        { success, file_count, tier, output_path, size_mb }
    """
    output_path = Path(output_path)
    db_path = PROJECT_ROOT / "imageparser.db"
    thumb_dir = PROJECT_ROOT / "output" / "thumbnails"

    if not db_path.exists():
        return {"success": False, "error": "Database not found"}

    # Get tier and file count from DB
    conn = sqlite3.connect(str(db_path))
    file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # Get tier from first file with mode_tier
    tier_row = conn.execute(
        "SELECT mode_tier FROM files WHERE mode_tier IS NOT NULL LIMIT 1"
    ).fetchone()
    tier = tier_row[0] if tier_row else "unknown"
    conn.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. VACUUM database into temp copy (smaller file)
        tmp_db = tmpdir / "imageparser.db"
        vacuumed = sqlite3.connect(str(tmp_db))
        conn = sqlite3.connect(str(db_path))
        conn.backup(vacuumed)
        vacuumed.execute("VACUUM")
        vacuumed.close()
        conn.close()

        # 2. Create manifest
        manifest = {
            "tier": tier,
            "file_count": file_count,
            "version": "3.5.1",
            "created_at": datetime.now().isoformat(),
            "platform": os.uname().sysname if hasattr(os, 'uname') else "unknown",
        }
        manifest_path = tmpdir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # 3. Package as zip
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp_db, "imageparser.db")
            zf.write(manifest_path, "manifest.json")

            # Add thumbnails
            thumb_count = 0
            if thumb_dir.exists():
                for thumb_file in thumb_dir.iterdir():
                    if thumb_file.is_file() and thumb_file.suffix.lower() == '.png':
                        zf.write(thumb_file, f"thumbnails/{thumb_file.name}")
                        thumb_count += 1

        size_mb = output_path.stat().st_size / (1024 * 1024)

        result = {
            "success": True,
            "file_count": file_count,
            "thumb_count": thumb_count,
            "tier": tier,
            "output_path": str(output_path),
            "size_mb": round(size_mb, 1),
        }
        logger.info(f"Export complete: {file_count} files, {thumb_count} thumbnails, {size_mb:.1f}MB")
        return result


def main():
    parser = argparse.ArgumentParser(description="Export DB + thumbnails archive")
    parser.add_argument("--output", required=True, help="Output zip path")
    args = parser.parse_args()

    result = export_archive(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
