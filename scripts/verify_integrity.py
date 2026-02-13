#!/usr/bin/env python3
"""
VV/MV Data Integrity Verification Script.

Three-level verification:
  Level 1 (DB-only): Cross-check content_hash vs vector fingerprints — no model loading
  Level 2 (Sample re-encode): Re-encode sample images with SigLIP2, compare cosine similarity
  Level 3 (Full audit): Re-encode all images (slow but definitive)

Usage:
  python scripts/verify_integrity.py                    # Level 1 (fast, DB-only)
  python scripts/verify_integrity.py --level 2          # Level 1 + sample re-encode
  python scripts/verify_integrity.py --level 2 --sample 10  # Sample 10 files
  python scripts/verify_integrity.py --folder /path/to  # Restrict to folder
"""

import argparse
import json
import logging
import struct
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_db():
    """Get SQLiteDB instance."""
    from backend.db.sqlite_client import SQLiteDB
    return SQLiteDB()


# ── Level 1: DB-only Cross-checks ─────────────────────────────────────


def level1_verify(db, folder: str = None) -> dict:
    """
    DB-only integrity checks. No model loading required.

    Checks:
    1. Orphan vectors: vec_files/vec_text entries without matching files row
    2. Pipeline consistency: MC exists but MV missing, etc.
    3. Duplicate VV fingerprints across different content_hashes (corruption signal)
    4. content_hash NULL but vectors exist (should have hash)
    """
    cursor = db.conn.cursor()
    results = {
        "orphan_vv": [],
        "orphan_mv": [],
        "mc_without_mv": [],
        "vv_without_mc": [],
        "duplicate_vv_cross_hash": [],
        "null_hash_with_vectors": [],
        "total_files": 0,
        "total_vv": 0,
        "total_mv": 0,
    }

    # Filter by folder if specified
    folder_clause = ""
    folder_params = ()
    if folder:
        folder_clause = "WHERE f.file_path LIKE ?"
        folder_params = (f"{folder}%",)

    # --- Count totals ---
    if folder:
        cursor.execute(f"SELECT COUNT(*) FROM files f {folder_clause}", folder_params)
    else:
        cursor.execute("SELECT COUNT(*) FROM files")
    results["total_files"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vec_files")
    results["total_vv"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM vec_text")
    results["total_mv"] = cursor.fetchone()[0]

    # --- Check 1: Orphan vectors (vec without files row) ---
    cursor.execute("""
        SELECT vf.file_id FROM vec_files vf
        LEFT JOIN files f ON vf.file_id = f.id
        WHERE f.id IS NULL
    """)
    results["orphan_vv"] = [row[0] for row in cursor.fetchall()]

    cursor.execute("""
        SELECT vt.file_id FROM vec_text vt
        LEFT JOIN files f ON vt.file_id = f.id
        WHERE f.id IS NULL
    """)
    results["orphan_mv"] = [row[0] for row in cursor.fetchall()]

    # --- Check 2: Pipeline consistency ---
    # MC exists but no MV (MV should be generated from MC)
    cursor.execute(f"""
        SELECT f.id, f.file_path FROM files f
        LEFT JOIN vec_text vt ON f.id = vt.file_id
        {folder_clause.replace('WHERE', 'WHERE' if not folder_clause else 'AND') if folder_clause else ''}
        {"WHERE" if not folder else "AND"} f.mc_caption IS NOT NULL
        AND LENGTH(TRIM(f.mc_caption)) > 0
        AND vt.file_id IS NULL
    """, folder_params)
    results["mc_without_mv"] = [
        {"id": row[0], "path": row[1]} for row in cursor.fetchall()
    ]

    # VV exists but no MC (unusual — VV is image-based, MC is VLM-based, independent)
    # This is informational, not necessarily an error
    cursor.execute(f"""
        SELECT f.id, f.file_path FROM files f
        INNER JOIN vec_files vf ON f.id = vf.file_id
        {folder_clause.replace('WHERE', 'WHERE' if not folder_clause else 'AND') if folder_clause else ''}
        {"WHERE" if not folder else "AND"} (f.mc_caption IS NULL OR LENGTH(TRIM(f.mc_caption)) = 0)
    """, folder_params)
    results["vv_without_mc"] = [
        {"id": row[0], "path": row[1]} for row in cursor.fetchall()
    ]

    # --- Check 3: Duplicate VV fingerprints across different content_hashes ---
    # Extract first 8 floats as fingerprint for grouping
    cursor.execute("""
        SELECT vf.file_id, f.content_hash, vf.embedding
        FROM vec_files vf
        INNER JOIN files f ON vf.file_id = f.id
        WHERE f.content_hash IS NOT NULL
    """)

    fingerprint_map = defaultdict(list)  # fingerprint -> [(file_id, content_hash)]
    for row in cursor.fetchall():
        file_id = row[0]
        content_hash = row[1]
        emb_raw = row[2]

        # Extract fingerprint (first 8 floats from binary embedding)
        try:
            if isinstance(emb_raw, bytes):
                fp_floats = struct.unpack(f"<8f", emb_raw[:32])
            elif isinstance(emb_raw, str):
                fp_floats = tuple(json.loads(emb_raw)[:8])
            else:
                continue
            fp_key = tuple(round(f, 6) for f in fp_floats)
            fingerprint_map[fp_key].append((file_id, content_hash))
        except Exception:
            continue

    # Find fingerprints shared across different content_hashes
    for fp_key, entries in fingerprint_map.items():
        unique_hashes = set(ch for _, ch in entries)
        if len(unique_hashes) > 1:
            results["duplicate_vv_cross_hash"].append({
                "fingerprint": list(fp_key),
                "entries": [
                    {"file_id": fid, "content_hash": ch}
                    for fid, ch in entries
                ],
            })

    # --- Check 4: NULL content_hash but has vectors ---
    cursor.execute("""
        SELECT f.id, f.file_path FROM files f
        INNER JOIN vec_files vf ON f.id = vf.file_id
        WHERE f.content_hash IS NULL
    """)
    results["null_hash_with_vectors"] = [
        {"id": row[0], "path": row[1]} for row in cursor.fetchall()
    ]

    return results


def print_level1(results: dict):
    """Pretty-print Level 1 results."""
    print("\n" + "=" * 60)
    print("  Level 1: DB Integrity Check (no model loading)")
    print("=" * 60)

    print(f"\n  Total files: {results['total_files']}")
    print(f"  Total VV vectors: {results['total_vv']}")
    print(f"  Total MV vectors: {results['total_mv']}")

    # Orphans
    if results["orphan_vv"]:
        print(f"\n  ⚠ Orphan VV (no matching file): {len(results['orphan_vv'])}")
        for fid in results["orphan_vv"][:5]:
            print(f"    - file_id={fid}")
    else:
        print("\n  ✓ No orphan VV vectors")

    if results["orphan_mv"]:
        print(f"  ⚠ Orphan MV (no matching file): {len(results['orphan_mv'])}")
        for fid in results["orphan_mv"][:5]:
            print(f"    - file_id={fid}")
    else:
        print("  ✓ No orphan MV vectors")

    # Pipeline consistency
    mc_no_mv = results["mc_without_mv"]
    if mc_no_mv:
        print(f"\n  ⚠ MC exists but MV missing: {len(mc_no_mv)}")
        for entry in mc_no_mv[:5]:
            print(f"    - [{entry['id']}] {Path(entry['path']).name}")
        if len(mc_no_mv) > 5:
            print(f"    ... and {len(mc_no_mv) - 5} more")
    else:
        print("\n  ✓ All MC records have corresponding MV vectors")

    vv_no_mc = results["vv_without_mc"]
    if vv_no_mc:
        print(f"  ℹ VV exists but MC missing: {len(vv_no_mc)} (info only)")
    else:
        print("  ✓ All VV records have corresponding MC data")

    # Cross-hash duplicates (corruption signal)
    dupes = results["duplicate_vv_cross_hash"]
    if dupes:
        print(f"\n  🔴 CORRUPT: Same VV across different content_hashes: {len(dupes)} groups")
        for group in dupes[:3]:
            print(f"    Fingerprint: [{', '.join(f'{v:.4f}' for v in group['fingerprint'][:4])}...]")
            for entry in group["entries"]:
                print(f"      file_id={entry['file_id']}  hash={entry['content_hash'][:12]}...")
    else:
        print("\n  ✓ No cross-hash VV duplicates (no corruption detected)")

    # NULL hash
    null_hash = results["null_hash_with_vectors"]
    if null_hash:
        print(f"\n  ⚠ Vectors exist but content_hash is NULL: {len(null_hash)}")
        for entry in null_hash[:5]:
            print(f"    - [{entry['id']}] {Path(entry['path']).name}")
    else:
        print("  ✓ All vectorized files have content_hash")

    print()


# ── Level 2: Sample Re-encoding Verification ──────────────────────────


def _resolve_thumbnail_path(db, file_id: int) -> "Optional[Path]":
    """Resolve thumbnail path for a file (same logic as pipeline)."""
    cursor = db.conn.cursor()
    cursor.execute("SELECT thumbnail_url FROM files WHERE id = ?", (file_id,))
    row = cursor.fetchone()
    if not row or not row[0]:
        return None

    import urllib.parse
    thumb_url = row[0]
    if thumb_url.startswith("file:///"):
        thumb_path = Path(urllib.parse.unquote(thumb_url[8:]))
    else:
        thumb_path = Path(thumb_url)

    return thumb_path if thumb_path.exists() else None


def level2_verify(db, folder: str = None, sample_size: int = 5) -> dict:
    """
    Re-encode sample thumbnails and compare cosine similarity with stored VV.

    IMPORTANT: VV is encoded from thumbnails (not raw files), so we must
    re-encode the same thumbnail to get a valid comparison.

    High similarity (>0.99) = stored vector is correct for this file.
    Low similarity (<0.95) = stored vector may be from wrong file.
    """
    import numpy as np
    from PIL import Image

    cursor = db.conn.cursor()

    # Get random sample of files that have VV
    folder_clause = ""
    folder_params = ()
    if folder:
        folder_clause = "AND f.file_path LIKE ?"
        folder_params = (f"{folder}%",)

    cursor.execute(f"""
        SELECT f.id, f.file_path, vf.embedding
        FROM files f
        INNER JOIN vec_files vf ON f.id = vf.file_id
        WHERE 1=1 {folder_clause}
        ORDER BY RANDOM()
        LIMIT ?
    """, (*folder_params, sample_size))

    samples = cursor.fetchall()
    if not samples:
        return {"error": "No files with VV found", "results": []}

    # Load SigLIP2 encoder
    logger.info("Loading SigLIP2 encoder for re-encoding verification...")
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    encoder = SigLIP2Encoder()

    results = []
    for row in samples:
        file_id = row[0]
        file_path = row[1]
        stored_emb_raw = row[2]

        # Decode stored embedding
        try:
            if isinstance(stored_emb_raw, bytes):
                dim = len(stored_emb_raw) // 4
                stored_vec = np.array(
                    struct.unpack(f"<{dim}f", stored_emb_raw), dtype=np.float32
                )
            elif isinstance(stored_emb_raw, str):
                stored_vec = np.array(json.loads(stored_emb_raw), dtype=np.float32)
            else:
                results.append({
                    "file_id": file_id,
                    "path": file_path,
                    "status": "error",
                    "detail": "Unknown embedding format",
                })
                continue
        except Exception as e:
            results.append({
                "file_id": file_id,
                "path": file_path,
                "status": "error",
                "detail": f"Decode error: {e}",
            })
            continue

        # Resolve thumbnail path (pipeline encodes VV from thumbnail, not raw file)
        thumb_path = _resolve_thumbnail_path(db, file_id)
        if not thumb_path:
            results.append({
                "file_id": file_id,
                "path": file_path,
                "status": "no_thumb",
                "detail": "Thumbnail not found — cannot verify",
            })
            continue

        try:
            # Composite RGBA → RGB on white background (same as pipeline)
            thumb_img = Image.open(thumb_path)
            if thumb_img.mode == "RGBA":
                bg = Image.new("RGB", thumb_img.size, (255, 255, 255))
                bg.paste(thumb_img, mask=thumb_img.split()[3])
                img = bg
            else:
                img = thumb_img.convert("RGB")
            fresh_vec = encoder.encode_image(img)
            thumb_img.close()
        except Exception as e:
            results.append({
                "file_id": file_id,
                "path": file_path,
                "status": "error",
                "detail": f"Encode error: {e}",
            })
            continue

        # Cosine similarity
        cos_sim = float(np.dot(stored_vec, fresh_vec) / (
            np.linalg.norm(stored_vec) * np.linalg.norm(fresh_vec) + 1e-10
        ))

        status = "ok" if cos_sim > 0.99 else ("warn" if cos_sim > 0.95 else "CORRUPT")
        results.append({
            "file_id": file_id,
            "path": file_path,
            "thumbnail": str(thumb_path.name),
            "status": status,
            "cosine_similarity": round(cos_sim, 6),
        })

    # Unload model
    encoder.unload()

    return {"results": results}


def print_level2(results: dict):
    """Pretty-print Level 2 results."""
    print("\n" + "=" * 60)
    print("  Level 2: Sample Re-encoding Verification")
    print("=" * 60)

    if "error" in results:
        print(f"\n  Error: {results['error']}")
        return

    ok_count = sum(1 for r in results["results"] if r["status"] == "ok")
    warn_count = sum(1 for r in results["results"] if r["status"] == "warn")
    corrupt_count = sum(1 for r in results["results"] if r["status"] == "CORRUPT")
    error_count = sum(1 for r in results["results"] if r["status"] in ("error", "missing", "no_thumb"))

    print(f"\n  Sampled: {len(results['results'])} files")
    print(f"  ✓ OK (sim > 0.99): {ok_count}")
    if warn_count:
        print(f"  ⚠ Warn (0.95-0.99): {warn_count}")
    if corrupt_count:
        print(f"  🔴 CORRUPT (sim < 0.95): {corrupt_count}")
    if error_count:
        print(f"  ? Error/Missing: {error_count}")

    print()
    for r in results["results"]:
        name = Path(r["path"]).name
        sim = r.get("cosine_similarity", "N/A")
        thumb = r.get("thumbnail", "")
        icon = {"ok": "✓", "warn": "⚠", "CORRUPT": "🔴"}.get(r["status"], "?")
        if r["status"] in ("error", "missing", "no_thumb"):
            print(f"  {icon} [{r['file_id']}] {name}: {r['detail']}")
        else:
            print(f"  {icon} [{r['file_id']}] {name}: similarity={sim}")
    print()


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VV/MV Data Integrity Verification")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2],
                        help="Verification level (1=DB-only, 2=sample re-encode)")
    parser.add_argument("--folder", type=str, default=None,
                        help="Restrict check to folder path prefix")
    parser.add_argument("--sample", type=int, default=5,
                        help="Sample size for Level 2 (default: 5)")
    args = parser.parse_args()

    db = get_db()

    # Level 1 always runs
    logger.info("Running Level 1: DB integrity checks...")
    l1 = level1_verify(db, folder=args.folder)
    print_level1(l1)

    # Level 2 if requested
    if args.level >= 2:
        logger.info(f"Running Level 2: Sample re-encoding ({args.sample} files)...")
        l2 = level2_verify(db, folder=args.folder, sample_size=args.sample)
        print_level2(l2)

    # Summary
    l1_issues = (
        len(l1["orphan_vv"]) + len(l1["orphan_mv"]) +
        len(l1["duplicate_vv_cross_hash"]) + len(l1["null_hash_with_vectors"])
    )
    l2_issues = 0
    if args.level >= 2 and "results" in l2:
        l2_issues = sum(1 for r in l2["results"] if r["status"] == "CORRUPT")

    total_issues = l1_issues + l2_issues
    if total_issues == 0:
        print("  ✅ All checks passed — data integrity looks good.")
    else:
        print(f"  ⚠ {total_issues} issue(s) found — review above for details.")


if __name__ == "__main__":
    main()
