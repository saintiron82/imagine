#!/usr/bin/env python3
"""
DB Integrity Verification Script.

Three-level verification:
  Level 0 (Pipeline completeness): 3-axis completeness, data quality, counter sync
  Level 1 (DB-only): Cross-check content_hash vs vector fingerprints — no model loading
  Level 2 (Sample re-encode): Re-encode sample images with SigLIP2, compare cosine similarity

Usage:
  python scripts/verify_integrity.py                    # Level 0+1 (fast, DB-only)
  python scripts/verify_integrity.py --level 2          # Level 0+1+2 (sample re-encode)
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


# ── Level 0: Pipeline Completeness + Data Quality ────────────────────


def level0_verify(db) -> dict:
    """
    Pipeline completeness and data quality checks.

    Checks:
    0-1. 3-axis completeness: MC/VV/MV all present
    0-2. Data quality: empty captions, empty tags, missing image_type
    0-3. Counter sync: files vs vec_files vs vec_text vs files_fts
    0-4. Job queue state: incomplete jobs, failed jobs, error distribution
    0-5. Duplicates: file_path duplicates, content_hash duplicates
    """
    cursor = db.conn.cursor()

    # --- 0-1: 3-axis completeness ---
    cursor.execute("SELECT COUNT(*) FROM files")
    total_files = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM files
        WHERE mc_caption IS NOT NULL AND mc_caption != ''
    """)
    mc_complete = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM files f
        WHERE EXISTS (SELECT 1 FROM vec_files WHERE file_id = f.id)
    """)
    vv_complete = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM files f
        WHERE EXISTS (SELECT 1 FROM vec_text WHERE file_id = f.id)
    """)
    mv_complete = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM files f
        WHERE (f.mc_caption IS NOT NULL AND f.mc_caption != '')
          AND EXISTS (SELECT 1 FROM vec_files WHERE file_id = f.id)
          AND EXISTS (SELECT 1 FROM vec_text WHERE file_id = f.id)
    """)
    all_3axis = cursor.fetchone()[0]

    # MC but no VV
    cursor.execute("""
        SELECT COUNT(*) FROM files f
        WHERE (f.mc_caption IS NOT NULL AND f.mc_caption != '')
          AND NOT EXISTS (SELECT 1 FROM vec_files WHERE file_id = f.id)
    """)
    mc_no_vv = cursor.fetchone()[0]

    # MC but no MV
    cursor.execute("""
        SELECT COUNT(*) FROM files f
        WHERE (f.mc_caption IS NOT NULL AND f.mc_caption != '')
          AND NOT EXISTS (SELECT 1 FROM vec_text WHERE file_id = f.id)
    """)
    mc_no_mv = cursor.fetchone()[0]

    # --- 0-2: Data quality ---
    cursor.execute("""
        SELECT COUNT(*) FROM files
        WHERE ai_tags IS NULL OR ai_tags = '' OR ai_tags = '[]'
    """)
    empty_tags = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM files
        WHERE image_type IS NULL OR image_type = ''
    """)
    no_image_type = cursor.fetchone()[0]

    # --- 0-3: Counter sync ---
    cursor.execute("SELECT COUNT(*) FROM vec_files")
    vv_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM vec_text")
    mv_count = cursor.fetchone()[0]

    fts_count = 0
    try:
        cursor.execute("SELECT COUNT(*) FROM files_fts")
        fts_count = cursor.fetchone()[0]
    except Exception:
        fts_count = -1  # FTS table may not exist

    fts_synced = (fts_count == total_files) if fts_count >= 0 else None

    # --- 0-4: Job queue state ---
    job_stats = {}
    failed_errors = {}
    try:
        cursor.execute("""
            SELECT status, COUNT(*) FROM job_queue
            GROUP BY status
        """)
        job_stats = dict(cursor.fetchall())

        cursor.execute("""
            SELECT COALESCE(error_code, 'unknown'), COUNT(*) FROM job_queue
            WHERE status = 'failed'
            GROUP BY error_code
        """)
        failed_errors = dict(cursor.fetchall())
    except Exception:
        pass

    # --- 0-5: Duplicates ---
    cursor.execute("""
        SELECT file_path, COUNT(*) FROM files
        GROUP BY file_path HAVING COUNT(*) > 1
    """)
    dup_paths = cursor.fetchall()

    cursor.execute("""
        SELECT content_hash, COUNT(*) FROM files
        WHERE content_hash IS NOT NULL
        GROUP BY content_hash HAVING COUNT(*) > 1
    """)
    dup_hashes = cursor.fetchall()

    return {
        "total_files": total_files,
        "mc_complete": mc_complete,
        "vv_complete": vv_complete,
        "mv_complete": mv_complete,
        "all_3axis": all_3axis,
        "mc_no_vv": mc_no_vv,
        "mc_no_mv": mc_no_mv,
        "empty_tags": empty_tags,
        "no_image_type": no_image_type,
        "vv_count": vv_count,
        "mv_count": mv_count,
        "fts_count": fts_count,
        "fts_synced": fts_synced,
        "job_stats": job_stats,
        "failed_errors": failed_errors,
        "dup_paths": len(dup_paths),
        "dup_hashes": len(dup_hashes),
    }


def print_level0(r: dict):
    """Pretty-print Level 0 results."""
    total = r["total_files"] or 1

    print("\n" + "=" * 60)
    print("  Level 0: Pipeline Completeness + Data Quality")
    print("=" * 60)

    print(f"\n  3-Axis Completeness")
    print(f"  ───────────────────")
    print(f"  Total files:     {r['total_files']:,}")
    print(f"  MC complete:     {r['mc_complete']:,} ({r['mc_complete']/total*100:.1f}%)")
    print(f"  VV complete:     {r['vv_complete']:,} ({r['vv_complete']/total*100:.1f}%)")
    print(f"  MV complete:     {r['mv_complete']:,} ({r['mv_complete']/total*100:.1f}%)")
    print(f"  All 3-axis:      {r['all_3axis']:,} ({r['all_3axis']/total*100:.1f}%)")
    if r["mc_no_vv"]:
        print(f"  MC but no VV:    {r['mc_no_vv']:,}")
    if r["mc_no_mv"]:
        print(f"  MC but no MV:    {r['mc_no_mv']:,}")

    print(f"\n  Data Quality")
    print(f"  ────────────")
    print(f"  Empty ai_tags:   {r['empty_tags']:,}" + ("" if r["empty_tags"] == 0 else "  ⚠"))
    print(f"  No image_type:   {r['no_image_type']:,}" + ("" if r["no_image_type"] == 0 else "  ⚠"))

    fts_label = f"{r['fts_count']:,}" if r["fts_count"] >= 0 else "N/A"
    fts_ok = "  ✓" if r["fts_synced"] else "  ⚠ MISMATCH" if r["fts_synced"] is not None else ""
    print(f"  FTS count:       {fts_label} vs files {r['total_files']:,}{fts_ok}")

    if r["job_stats"]:
        print(f"\n  Job Queue")
        print(f"  ─────────")
        for status, count in sorted(r["job_stats"].items()):
            print(f"  {status:12}: {count:,}")
        if r["failed_errors"]:
            print(f"  Failed breakdown:")
            for code, count in sorted(r["failed_errors"].items(), key=lambda x: -x[1]):
                print(f"    {code}: {count}")

    if r["dup_paths"] or r["dup_hashes"]:
        print(f"\n  Duplicates")
        print(f"  ──────────")
        if r["dup_paths"]:
            print(f"  Duplicate paths:  {r['dup_paths']}  ⚠")
        if r["dup_hashes"]:
            print(f"  Duplicate hashes: {r['dup_hashes']}  (same file, different path)")

    print()


# ── Level 1: DB-only Cross-checks ─────────────────────────────────────


def level1_verify(db, folder: str = None) -> dict:
    """
    DB-only integrity checks. No model loading required.

    Checks:
    1. Dangling vectors: vec_files/vec_text entries without matching files row
    2. Pipeline consistency: MC exists but MV missing, etc.
    3. Duplicate VV fingerprints across different content_hashes (corruption signal)
    4. content_hash NULL but vectors exist (should have hash)
    """
    cursor = db.conn.cursor()
    results = {
        "dangling_vv": [],
        "dangling_mv": [],
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

    # --- Check 1: Dangling vectors (vec without files row) ---
    cursor.execute("""
        SELECT vf.file_id FROM vec_files vf
        LEFT JOIN files f ON vf.file_id = f.id
        WHERE f.id IS NULL
    """)
    results["dangling_vv"] = [row[0] for row in cursor.fetchall()]

    cursor.execute("""
        SELECT vt.file_id FROM vec_text vt
        LEFT JOIN files f ON vt.file_id = f.id
        WHERE f.id IS NULL
    """)
    results["dangling_mv"] = [row[0] for row in cursor.fetchall()]

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

    # Dangling references
    if results["dangling_vv"]:
        print(f"\n  ⚠ Dangling VV (no matching file): {len(results['dangling_vv'])}")
        for fid in results["dangling_vv"][:5]:
            print(f"    - file_id={fid}")
    else:
        print("\n  ✓ No dangling VV vectors")

    if results["dangling_mv"]:
        print(f"  ⚠ Dangling MV (no matching file): {len(results['dangling_mv'])}")
        for fid in results["dangling_mv"][:5]:
            print(f"    - file_id={fid}")
    else:
        print("  ✓ No dangling MV vectors")

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

    # Level 0 always runs
    logger.info("Running Level 0: Pipeline completeness + data quality...")
    l0 = level0_verify(db)
    print_level0(l0)

    # Level 1 always runs
    logger.info("Running Level 1: DB integrity checks...")
    l1 = level1_verify(db, folder=args.folder)
    print_level1(l1)

    # Level 2 if requested
    l2 = {}
    if args.level >= 2:
        logger.info(f"Running Level 2: Sample re-encoding ({args.sample} files)...")
        l2 = level2_verify(db, folder=args.folder, sample_size=args.sample)
        print_level2(l2)

    # Summary
    l0_issues = l0["mc_no_vv"] + l0["mc_no_mv"] + l0["empty_tags"] + l0["dup_paths"]
    if l0["fts_synced"] is not None and not l0["fts_synced"]:
        l0_issues += 1
    l1_issues = (
        len(l1["dangling_vv"]) + len(l1["dangling_mv"]) +
        len(l1["duplicate_vv_cross_hash"]) + len(l1["null_hash_with_vectors"])
    )
    l2_issues = 0
    if args.level >= 2 and "results" in l2:
        l2_issues = sum(1 for r in l2["results"] if r["status"] == "CORRUPT")

    total_issues = l0_issues + l1_issues + l2_issues
    print("=" * 60)
    if total_issues == 0:
        print("  All checks passed — data integrity OK.")
    else:
        print(f"  {total_issues} issue(s) found — review above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
