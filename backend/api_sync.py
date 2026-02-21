"""
Sync registered folders with database — detect moved, missing, and new files.

Compares actual disk state against DB records using content_hash for
path-independent file matching.

Usage:
    python -m backend.api_sync --folder /path/to/images [--apply-moves] [--delete-missing file_id1,file_id2,...]
"""

import argparse
import json
import logging
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SUPPORTED_EXTENSIONS = {'.psd', '.png', '.jpg', '.jpeg'}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.idea', '.vs'}


def _scan_disk(root_dir: Path) -> Dict[str, List[str]]:
    """
    DFS scan folder and compute content_hash for each supported file.

    Returns: { content_hash: [absolute_path_str, ...] }
    """
    from backend.utils.content_hash import compute_content_hash

    root_dir = Path(unicodedata.normalize('NFC', str(root_dir.resolve())))
    disk_map = defaultdict(list)
    count = 0

    def _dfs(current_dir: Path):
        nonlocal count
        try:
            entries = sorted(current_dir.iterdir(), key=lambda e: e.name.lower())
        except (PermissionError, OSError):
            return

        for entry in entries:
            if entry.is_dir():
                if entry.name.startswith('.') or entry.name in SKIP_DIRS:
                    continue
                _dfs(entry)
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
                nfc_path = Path(unicodedata.normalize('NFC', str(entry)))
                try:
                    ch = compute_content_hash(nfc_path)
                    disk_map[ch].append(str(nfc_path))
                    count += 1
                    if count % 500 == 0:
                        logger.info(f"  Scanned {count} files...")
                except Exception as e:
                    logger.warning(f"  Hash failed: {nfc_path.name}: {e}")

    logger.info(f"Scanning folder: {root_dir}")
    _dfs(root_dir)
    logger.info(f"  Found {count} files, {len(disk_map)} unique hashes")
    return disk_map


def _load_db_files(db_path: str, storage_root: str) -> List[dict]:
    """
    Load all file records from DB whose file_path starts with storage_root.

    Returns list of dicts: { id, file_path, content_hash, file_name }
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Normalize storage root for matching
    storage_root_norm = unicodedata.normalize('NFC', str(Path(storage_root).resolve()))
    # Use LIKE prefix matching (with trailing /)
    prefix = storage_root_norm.rstrip('/') + '/'

    cursor = conn.execute("""
        SELECT id, file_path, content_hash, file_name
        FROM files
        WHERE file_path LIKE ? OR file_path = ?
    """, (prefix + '%', storage_root_norm))

    results = []
    for row in cursor.fetchall():
        results.append({
            "id": row["id"],
            "file_path": row["file_path"],
            "content_hash": row["content_hash"],
            "file_name": row["file_name"],
        })

    conn.close()
    logger.info(f"  DB has {len(results)} files under {storage_root_norm}")
    return results


def sync_folder(folder_path: str, db_path: Optional[str] = None) -> dict:
    """
    Compare disk state against DB records for a folder.

    Returns:
        {
            "success": True,
            "matched": int,           # path-matched (DB path == disk path)
            "moved": int,             # content_hash match but different path
            "missing": int,           # DB only (not on disk)
            "new_files": int,         # disk only (not in DB)
            "moved_list": [...],      # { id, old_path, new_path, file_name }
            "missing_list": [...],    # { id, file_path, file_name }
            "total_db": int,
            "total_disk": int,
        }
    """
    folder_path = Path(unicodedata.normalize('NFC', str(Path(folder_path).resolve())))
    if not folder_path.is_dir():
        return {"success": False, "error": f"Folder not found: {folder_path}"}

    if db_path is None:
        db_path = str(PROJECT_ROOT / "imageparser.db")

    if not Path(db_path).exists():
        return {"success": False, "error": "Database not found"}

    storage_root = str(folder_path)

    # Phase 1: Scan disk
    disk_map = _scan_disk(folder_path)  # { hash: [path, ...] }

    # Build reverse map: disk_path -> hash
    disk_paths = {}  # { path_str: hash }
    for ch, paths in disk_map.items():
        for p in paths:
            disk_paths[p] = ch

    # Phase 2: Load DB records
    db_files = _load_db_files(db_path, storage_root)

    # Build DB maps
    db_by_path = {}   # { path: db_record }
    db_by_hash = defaultdict(list)  # { hash: [db_record, ...] }
    for rec in db_files:
        db_by_path[rec["file_path"]] = rec
        if rec["content_hash"]:
            db_by_hash[rec["content_hash"]].append(rec)

    # Phase 3: Classify
    matched = 0
    moved_list = []
    missing_list = []
    matched_db_ids = set()
    matched_disk_paths = set()

    # 3a: Path-match (DB path exists on disk)
    for rec in db_files:
        if rec["file_path"] in disk_paths:
            matched += 1
            matched_db_ids.add(rec["id"])
            matched_disk_paths.add(rec["file_path"])

    # 3b: Hash-match for unmatched DB entries (detect moves)
    for rec in db_files:
        if rec["id"] in matched_db_ids:
            continue
        ch = rec["content_hash"]
        if not ch or ch not in disk_map:
            # No hash or hash not on disk -> missing
            missing_list.append({
                "id": rec["id"],
                "file_path": rec["file_path"],
                "file_name": rec["file_name"],
            })
            continue

        # Hash found on disk — find an unmatched disk path
        found_new_path = None
        for disk_path in disk_map[ch]:
            if disk_path not in matched_disk_paths:
                found_new_path = disk_path
                break

        if found_new_path:
            moved_list.append({
                "id": rec["id"],
                "old_path": rec["file_path"],
                "new_path": found_new_path,
                "file_name": rec["file_name"],
            })
            matched_db_ids.add(rec["id"])
            matched_disk_paths.add(found_new_path)
        else:
            # All disk copies already matched
            missing_list.append({
                "id": rec["id"],
                "file_path": rec["file_path"],
                "file_name": rec["file_name"],
            })

    # 3c: New files (on disk but not matched to any DB entry)
    new_count = 0
    for disk_path in disk_paths:
        if disk_path not in matched_disk_paths:
            new_count += 1

    result = {
        "success": True,
        "matched": matched,
        "moved": len(moved_list),
        "missing": len(missing_list),
        "new_files": new_count,
        "moved_list": moved_list,
        "missing_list": missing_list,
        "total_db": len(db_files),
        "total_disk": len(disk_paths),
    }

    logger.info(f"Sync result: matched={matched}, moved={len(moved_list)}, "
                f"missing={len(missing_list)}, new={new_count}")
    return result


def apply_moves(moves: List[dict], db_path: Optional[str] = None) -> dict:
    """
    Apply path updates for moved files.

    Args:
        moves: List of { id, new_path } dicts
        db_path: Optional DB path override

    Returns: { success, updated }
    """
    import sqlite3
    if db_path is None:
        db_path = str(PROJECT_ROOT / "imageparser.db")

    conn = sqlite3.connect(db_path)
    updated = 0

    for move in moves:
        file_id = move["id"]
        new_path = unicodedata.normalize('NFC', move["new_path"])
        new_name = Path(new_path).name

        # Compute folder metadata from new path
        parent = Path(new_path).parent
        folder_tags_list = [p for p in parent.parts if p != '/']

        conn.execute("""
            UPDATE files
            SET file_path = ?,
                file_name = ?,
                parsed_at = datetime('now')
            WHERE id = ?
        """, (new_path, new_name, file_id))
        updated += 1

    conn.commit()
    conn.close()

    logger.info(f"Applied {updated} path updates")
    return {"success": True, "updated": updated}


def delete_missing(file_ids: List[int], db_path: Optional[str] = None) -> dict:
    """
    Delete DB records for files no longer on disk.

    Cascade deletes clean up vec_files, vec_text, vec_structure, files_fts.

    Args:
        file_ids: List of file IDs to delete
        db_path: Optional DB path override

    Returns: { success, deleted }
    """
    import sqlite3
    if db_path is None:
        db_path = str(PROJECT_ROOT / "imageparser.db")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    deleted = 0

    for fid in file_ids:
        # Delete FTS entry (trigger handles this, but be explicit)
        conn.execute("DELETE FROM files_fts WHERE rowid = ?", (fid,))
        # Delete vectors
        conn.execute("DELETE FROM vec_files WHERE file_id = ?", (fid,))
        conn.execute("DELETE FROM vec_text WHERE file_id = ?", (fid,))
        conn.execute("DELETE FROM vec_structure WHERE file_id = ?", (fid,))
        # Delete file record
        conn.execute("DELETE FROM files WHERE id = ?", (fid,))
        deleted += 1

    conn.commit()
    conn.close()

    logger.info(f"Deleted {deleted} file records from DB")
    return {"success": True, "deleted": deleted}


def main():
    parser = argparse.ArgumentParser(description="Sync folder with database")
    parser.add_argument("--folder", required=True, help="Folder path to sync")
    parser.add_argument("--db", default=None, help="Database path (default: imageparser.db)")
    parser.add_argument("--apply-moves", action="store_true", help="Auto-apply moved file path updates")
    parser.add_argument("--delete-missing", default=None, help="Comma-separated file IDs to delete")

    args = parser.parse_args()

    if args.delete_missing:
        file_ids = [int(x.strip()) for x in args.delete_missing.split(",") if x.strip()]
        result = delete_missing(file_ids, db_path=args.db)
        print(json.dumps(result, ensure_ascii=False))
        return

    result = sync_folder(args.folder, db_path=args.db)
    print(json.dumps(result, ensure_ascii=False, default=str))

    if args.apply_moves and result.get("success") and result.get("moved_list"):
        move_result = apply_moves(result["moved_list"], db_path=args.db)
        print(json.dumps(move_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
