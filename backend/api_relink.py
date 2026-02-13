"""
Import and relink a database archive to a local file folder.

Matches files by content_hash, updates paths, handles tier differences.

Usage:
    python -m backend.api_relink --package ./archive.zip --folder /path/to/images [--dry-run] [--delete-missing]
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SUPPORTED_EXTENSIONS = {'.psd', '.png', '.jpg', '.jpeg'}


def _scan_folder(root_dir: Path) -> dict:
    """
    DFS scan folder and compute content_hash for each file.

    Returns: { content_hash: [file_path, ...] }
    """
    from backend.utils.content_hash import compute_content_hash

    root_dir = Path(unicodedata.normalize('NFC', str(root_dir.resolve())))
    disk_map = defaultdict(list)
    skip_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vs'}

    file_count = 0

    def _dfs(current_dir: Path):
        nonlocal file_count
        try:
            entries = sorted(current_dir.iterdir(), key=lambda e: e.name.lower())
        except (PermissionError, OSError):
            return

        for entry in entries:
            if entry.is_dir():
                if entry.name.startswith('.') or entry.name in skip_dirs:
                    continue
                _dfs(entry)
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                nfc_path = Path(unicodedata.normalize('NFC', str(entry)))
                try:
                    ch = compute_content_hash(nfc_path)
                    disk_map[ch].append(nfc_path)
                    file_count += 1
                    if file_count % 500 == 0:
                        logger.info(f"  Scanned {file_count} files...")
                except Exception as e:
                    logger.warning(f"  Hash failed: {nfc_path.name}: {e}")

    logger.info(f"Scanning folder: {root_dir}")
    _dfs(root_dir)
    logger.info(f"  Found {file_count} files, {len(disk_map)} unique hashes")
    return disk_map


def _load_db_hashes(db_path: str) -> dict:
    """
    Load all content_hash entries from source DB.

    Returns: { content_hash: [{id, file_path, has_mc, has_vv, has_mv, mode_tier}, ...] }
    """
    import sqlite_vec

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    except Exception:
        pass
    conn.enable_load_extension(False)

    db_map = defaultdict(list)

    cursor = conn.execute("""
        SELECT id, file_path, content_hash, mc_caption, mode_tier
        FROM files
        WHERE content_hash IS NOT NULL
    """)

    for row in cursor.fetchall():
        file_id, file_path, content_hash, mc_caption, mode_tier = row
        mc = mc_caption or ""

        has_vv = conn.execute(
            "SELECT COUNT(*) FROM vec_files WHERE file_id = ?", (file_id,)
        ).fetchone()[0] > 0
        has_mv = conn.execute(
            "SELECT COUNT(*) FROM vec_text WHERE file_id = ?", (file_id,)
        ).fetchone()[0] > 0

        db_map[content_hash].append({
            "id": file_id,
            "file_path": file_path,
            "has_mc": len(mc.strip()) > 0,
            "has_vv": has_vv,
            "has_mv": has_mv,
            "mode_tier": mode_tier,
        })

    conn.close()

    total_files = sum(len(v) for v in db_map.values())
    logger.info(f"  DB: {total_files} files with content_hash, {len(db_map)} unique hashes")
    return db_map


def _compute_folder_meta(file_path: Path, root_dir: Path) -> dict:
    """Compute folder metadata for a file relative to root."""
    root_dir = Path(unicodedata.normalize('NFC', str(root_dir.resolve())))
    try:
        rel = file_path.parent.relative_to(root_dir)
        folder_str = str(rel).replace('\\', '/') if str(rel) != '.' else ''
        folder_tags = [p for p in folder_str.split('/') if p] if folder_str else []
        return {
            "folder_path": folder_str,
            "folder_depth": len(folder_tags),
            "folder_tags": json.dumps(folder_tags) if folder_tags else None,
            "storage_root": str(root_dir).replace('\\', '/'),
            "relative_path": str(file_path.relative_to(root_dir)).replace('\\', '/'),
        }
    except ValueError:
        return {
            "folder_path": None,
            "folder_depth": 0,
            "folder_tags": None,
            "storage_root": None,
            "relative_path": None,
        }


def relink(
    package_path: str,
    target_folder: str,
    dry_run: bool = False,
    delete_missing: bool = False,
) -> dict:
    """
    Import archive and relink files by content_hash.

    Args:
        package_path: Path to zip archive or .db file
        target_folder: B's image folder path
        dry_run: If True, only report matches without changes
        delete_missing: If True, delete DB rows with no matching disk file

    Returns:
        { matched, new_files, missing, tier_match, tier_source, tier_target, errors }
    """
    package_path = Path(package_path)
    target_folder = Path(target_folder)

    if not target_folder.exists():
        return {"success": False, "error": f"Folder not found: {target_folder}"}

    # Extract package
    tmpdir = None
    if package_path.suffix == '.zip':
        tmpdir = tempfile.mkdtemp(prefix="relink_")
        with zipfile.ZipFile(package_path, 'r') as zf:
            zf.extractall(tmpdir)
        source_db = Path(tmpdir) / "imageparser.db"
        source_thumbs = Path(tmpdir) / "thumbnails"
        manifest_path = Path(tmpdir) / "manifest.json"
    elif package_path.suffix == '.db':
        source_db = package_path
        source_thumbs = None
        manifest_path = None
    else:
        return {"success": False, "error": "Unsupported format (use .zip or .db)"}

    if not source_db.exists():
        return {"success": False, "error": "Database not found in package"}

    # Read manifest
    tier_source = "unknown"
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tier_source = manifest.get("tier", "unknown")

    # Get B's current tier
    try:
        from backend.utils.tier_config import get_active_tier
        tier_target, _ = get_active_tier()
    except Exception:
        tier_target = "unknown"

    tier_match = (tier_source == tier_target) and tier_source != "unknown"
    logger.info(f"Tier: source={tier_source}, target={tier_target}, match={tier_match}")

    # Phase 1: Scan B's folder
    disk_map = _scan_folder(target_folder)

    # Phase 2: Load A's DB hashes
    db_map = _load_db_hashes(str(source_db))

    # Phase 3: Match
    all_db_hashes = set(db_map.keys())
    all_disk_hashes = set(disk_map.keys())

    matched_hashes = all_db_hashes & all_disk_hashes
    new_hashes = all_disk_hashes - all_db_hashes
    missing_hashes = all_db_hashes - all_disk_hashes

    # Count files (not just unique hashes)
    matched_files = sum(len(disk_map[h]) for h in matched_hashes)
    new_files = sum(len(disk_map[h]) for h in new_hashes)
    missing_files = sum(len(db_map[h]) for h in missing_hashes)

    # For matched: count how many DB entries we'll use
    matched_db_entries = sum(
        min(len(disk_map[h]), len(db_map[h])) for h in matched_hashes
    )

    logger.info(f"\nMatch results:")
    logger.info(f"  Matched: {matched_db_entries} DB entries ↔ {matched_files} disk files")
    logger.info(f"  New files (disk only): {new_files}")
    logger.info(f"  Missing (DB only): {missing_files}")

    if dry_run:
        result = {
            "success": True,
            "dry_run": True,
            "matched": matched_db_entries,
            "new_files": new_files,
            "missing": missing_files,
            "tier_match": tier_match,
            "tier_source": tier_source,
            "tier_target": tier_target,
        }
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return result

    # Phase 4: Apply — copy A's DB as B's DB
    target_db = PROJECT_ROOT / "imageparser.db"

    # Backup B's existing DB
    if target_db.exists():
        backup_path = target_db.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy2(target_db, backup_path)
        logger.info(f"Backed up existing DB to: {backup_path.name}")

    # Copy source DB to target
    shutil.copy2(source_db, target_db)
    logger.info(f"Copied source DB to: {target_db}")

    # Phase 5: Open target DB and relink
    from backend.db.sqlite_client import SQLiteDB
    db = SQLiteDB(str(target_db))

    errors = 0
    relinked = 0
    thumb_dir = PROJECT_ROOT / "output" / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    for ch in matched_hashes:
        db_entries = db_map[ch]
        disk_paths = disk_map[ch]

        # Sort DB entries: most complete first (MC+VV+MV > MC+VV > MC > none)
        db_entries.sort(
            key=lambda e: (e["has_mc"], e["has_vv"], e["has_mv"]),
            reverse=True,
        )

        # Pair: 1 DB entry per 1 disk file
        for i, disk_path in enumerate(disk_paths):
            if i < len(db_entries):
                # Relink existing DB entry
                entry = db_entries[i]
                folder_meta = _compute_folder_meta(disk_path, target_folder)
                new_path = unicodedata.normalize('NFC', str(disk_path))

                # Compute thumbnail path
                stem = disk_path.stem
                thumb_path = f"output/thumbnails/{stem}_thumb.png"

                try:
                    db.conn.execute("""
                        UPDATE files SET
                            file_path = ?,
                            file_name = ?,
                            thumbnail_url = ?,
                            folder_path = ?,
                            folder_depth = ?,
                            folder_tags = ?,
                            storage_root = ?,
                            relative_path = ?,
                            parsed_at = datetime('now')
                        WHERE id = ?
                    """, (
                        new_path,
                        disk_path.name,
                        thumb_path,
                        folder_meta["folder_path"],
                        folder_meta["folder_depth"],
                        folder_meta["folder_tags"],
                        folder_meta["storage_root"],
                        folder_meta["relative_path"],
                        entry["id"],
                    ))
                    relinked += 1
                except Exception as e:
                    logger.warning(f"  Relink failed: {disk_path.name}: {e}")
                    errors += 1
            # else: extra disk files → will be discovered as new

        # Delete excess DB entries (DB N > disk M)
        if len(db_entries) > len(disk_paths):
            for extra in db_entries[len(disk_paths):]:
                try:
                    db.conn.execute("DELETE FROM files WHERE id = ?", (extra["id"],))
                except Exception as e:
                    logger.warning(f"  Delete excess failed: {e}")

    # Phase 6: Handle missing (DB only, no disk file)
    deleted_missing = 0
    if delete_missing:
        for ch in missing_hashes:
            for entry in db_map[ch]:
                try:
                    db.conn.execute("DELETE FROM files WHERE id = ?", (entry["id"],))
                    deleted_missing += 1
                except Exception:
                    pass

    # Phase 7: Tier mismatch — drop vec data
    dropped_vecs = 0
    if not tier_match:
        logger.info("Tier mismatch: clearing VV/MV vectors (MC preserved)")
        # Drop all vec data — B will regenerate on next discover
        db.conn.execute("DELETE FROM vec_files")
        db.conn.execute("DELETE FROM vec_text")
        dropped_vecs_row = db.conn.execute("SELECT changes()").fetchone()
        dropped_vecs = dropped_vecs_row[0] if dropped_vecs_row else 0

    db.conn.commit()

    # Phase 8: Rebuild FTS
    logger.info("Rebuilding FTS index...")
    db._rebuild_fts()

    # Phase 9: Copy thumbnails
    thumb_copied = 0
    if source_thumbs and source_thumbs.exists():
        for thumb_file in source_thumbs.iterdir():
            if thumb_file.is_file():
                dest = thumb_dir / thumb_file.name
                if not dest.exists():
                    shutil.copy2(thumb_file, dest)
                    thumb_copied += 1

    logger.info(f"\nRelink complete:")
    logger.info(f"  Relinked: {relinked}")
    logger.info(f"  New files (need processing): {new_files}")
    logger.info(f"  Missing deleted: {deleted_missing}")
    if not tier_match:
        logger.info(f"  Vectors cleared (tier mismatch): VV/MV will regenerate on discover")
    logger.info(f"  Thumbnails copied: {thumb_copied}")
    logger.info(f"  Errors: {errors}")

    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "success": True,
        "dry_run": False,
        "matched": relinked,
        "new_files": new_files,
        "missing": missing_files,
        "deleted_missing": deleted_missing,
        "tier_match": tier_match,
        "tier_source": tier_source,
        "tier_target": tier_target,
        "thumb_copied": thumb_copied,
        "dropped_vecs": not tier_match,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Import and relink database archive")
    parser.add_argument("--package", required=True, help="Path to .zip or .db file")
    parser.add_argument("--folder", required=True, help="Target image folder")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--delete-missing", action="store_true", help="Delete DB-only entries")
    args = parser.parse_args()

    result = relink(args.package, args.folder, args.dry_run, args.delete_missing)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
