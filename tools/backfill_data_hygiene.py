"""
Backfill data hygiene columns (Sprint 1 cleanup).

Restores columns that storage refactor (PhaseRunner migration) left empty:
  - modified_at  (from filesystem stat)
  - folder_path / folder_depth / folder_tags  (derived from file_path)
  - structured_meta  (re-composed from existing scalar VLM columns)
  - perceptual_hash  (dHash64 over thumbnail/original)
  - dup_group_id  (Hamming-distance Union-Find clustering)
  - caption_model  (active tier metadata for legacy rows)

Each phase is idempotent and only touches NULL rows. Pass --phases to scope.

Usage:
    python tools/backfill_data_hygiene.py                # all phases
    python tools/backfill_data_hygiene.py --phases mtime,folder
    python tools/backfill_data_hygiene.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill")

PHASES = ("mtime", "folder", "structured", "phash", "dup", "caption_model",
          "format_norm", "resolution")


# ──────────────────────────────────────────────────────────────────────
# Phase: modified_at
# ──────────────────────────────────────────────────────────────────────


def backfill_modified_at(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, file_path FROM files WHERE modified_at IS NULL AND file_path IS NOT NULL"
    ).fetchall()
    logger.info(f"[mtime] {len(rows)} rows to update")

    updates: list[tuple[str, int]] = []
    missing = 0
    for fid, fp in rows:
        try:
            iso = datetime.fromtimestamp(os.stat(fp).st_mtime).isoformat()
            updates.append((iso, fid))
        except (FileNotFoundError, PermissionError, OSError):
            missing += 1
            continue

    if dry_run:
        logger.info(f"[mtime] dry-run: would update {len(updates)}, miss {missing}")
        return 0

    cur.executemany(
        "UPDATE files SET modified_at = ? WHERE id = ?", updates
    )
    conn.commit()
    logger.info(f"[mtime] updated {len(updates)} (missing files: {missing})")
    return len(updates)


# ──────────────────────────────────────────────────────────────────────
# Phase: folder_path / folder_depth / folder_tags
# ──────────────────────────────────────────────────────────────────────


def _derive_folder(file_path: str) -> tuple[str, int, list[str]]:
    p = Path(file_path)
    parent = p.parent
    folder_path = str(parent)
    folder_depth = len([s for s in folder_path.split(os.sep) if s])
    # Up to 3 ancestor directory names as tags
    parts = [s for s in parent.parts if s and s not in ("/", "\\")]
    folder_tags = parts[-3:] if parts else []
    return folder_path, folder_depth, folder_tags


def backfill_folder_path(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, file_path FROM files WHERE folder_path IS NULL AND file_path IS NOT NULL"
    ).fetchall()
    logger.info(f"[folder] {len(rows)} rows to update")

    updates = []
    for fid, fp in rows:
        fpth, fdepth, ftags = _derive_folder(fp)
        updates.append((fpth, fdepth, json.dumps(ftags), fid))

    if dry_run:
        logger.info(f"[folder] dry-run: would update {len(updates)}")
        return 0

    cur.executemany(
        "UPDATE files SET folder_path = ?, folder_depth = ?, folder_tags = ? WHERE id = ?",
        updates,
    )
    conn.commit()
    logger.info(f"[folder] updated {len(updates)}")
    return len(updates)


# ──────────────────────────────────────────────────────────────────────
# Phase: structured_meta (re-compose from scalar columns)
# ──────────────────────────────────────────────────────────────────────

_VLM_SCALAR_COLS = (
    "image_type", "art_style", "color_palette", "scene_type",
    "time_of_day", "weather", "character_type", "item_type", "ui_type",
    "dominant_color", "ai_style", "mc_caption", "ai_tags", "ocr_text",
)


def backfill_structured_meta(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    cols = ", ".join(["id"] + list(_VLM_SCALAR_COLS))
    rows = cur.execute(
        f"SELECT {cols} FROM files WHERE structured_meta IS NULL AND mc_caption IS NOT NULL"
    ).fetchall()
    logger.info(f"[structured] {len(rows)} rows to compose")

    updates = []
    for row in rows:
        fid = row[0]
        recomposed = {}
        for i, col in enumerate(_VLM_SCALAR_COLS, start=1):
            val = row[i]
            if val is None:
                continue
            if col == "ai_tags":
                # ai_tags is stored as JSON string
                try:
                    parsed = json.loads(val) if isinstance(val, str) else val
                except (json.JSONDecodeError, TypeError):
                    parsed = val
                recomposed["tags"] = parsed
            elif col == "mc_caption":
                recomposed["caption"] = val
            else:
                recomposed[col] = val
        if not recomposed:
            continue
        updates.append((json.dumps(recomposed, ensure_ascii=False), fid))

    if dry_run:
        logger.info(f"[structured] dry-run: would update {len(updates)}")
        return 0

    cur.executemany(
        "UPDATE files SET structured_meta = ? WHERE id = ?", updates
    )
    conn.commit()
    logger.info(f"[structured] updated {len(updates)}")
    return len(updates)


# ──────────────────────────────────────────────────────────────────────
# Phase: perceptual_hash
# ──────────────────────────────────────────────────────────────────────


def backfill_perceptual_hash(conn, dry_run: bool) -> int:
    from PIL import Image
    from backend.utils.dhash import dhash64

    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, file_path, thumbnail_url FROM files WHERE perceptual_hash IS NULL"
    ).fetchall()
    logger.info(f"[phash] {len(rows)} rows to hash")

    updates = []
    skipped = 0
    failed = 0
    for fid, fp, thumb in rows:
        img_path = None
        if thumb and Path(thumb).exists():
            img_path = thumb
        elif fp and Path(fp).exists():
            img_path = fp
        else:
            skipped += 1
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                h = dhash64(img)
            if h >= 2 ** 63:
                h -= 2 ** 64
            updates.append((h, fid))
        except Exception as e:
            failed += 1
            logger.debug(f"[phash] {fp}: {e}")

    if dry_run:
        logger.info(
            f"[phash] dry-run: would update {len(updates)} "
            f"(skip={skipped}, fail={failed})"
        )
        return 0

    cur.executemany(
        "UPDATE files SET perceptual_hash = ? WHERE id = ?", updates
    )
    conn.commit()
    logger.info(
        f"[phash] updated {len(updates)} (skip={skipped}, fail={failed})"
    )
    return len(updates)


# ──────────────────────────────────────────────────────────────────────
# Phase: dup_group_id (Union-Find on Hamming distance)
# ──────────────────────────────────────────────────────────────────────


def _popcount(x: int) -> int:
    return bin(x & 0xFFFFFFFFFFFFFFFF).count("1")


def backfill_dup_group(conn, dry_run: bool, threshold: int = 8) -> int:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, perceptual_hash FROM files WHERE perceptual_hash IS NOT NULL"
    ).fetchall()
    logger.info(f"[dup] {len(rows)} rows with phash")

    if not rows:
        return 0

    # Convert signed → unsigned for Hamming distance
    hashes = []
    for fid, h in rows:
        if h < 0:
            h = h + 2 ** 64
        hashes.append((fid, h))

    # Union-Find
    parent = {fid: fid for fid, _ in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # O(n^2) for now — 17k * 17k = 300M ops. Slow but one-shot.
    # If too slow, switch to hash bucketing on top byte.
    n = len(hashes)
    t0 = time.time()
    cmp = 0
    for i in range(n):
        a_id, a_h = hashes[i]
        for j in range(i + 1, n):
            cmp += 1
            b_id, b_h = hashes[j]
            if _popcount(a_h ^ b_h) < threshold:
                union(a_id, b_id)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - t0
            logger.info(f"[dup] {i}/{n} ({elapsed:.1f}s, {cmp/elapsed:.0f}/s)")

    # Count groups (>1 member only get a group id)
    from collections import defaultdict
    members = defaultdict(list)
    for fid, _ in hashes:
        members[find(fid)].append(fid)

    updates = []
    next_gid = 1
    for root, mems in members.items():
        if len(mems) < 2:
            continue
        gid = next_gid
        next_gid += 1
        for fid in mems:
            updates.append((gid, fid))

    n_groups = next_gid - 1
    logger.info(f"[dup] {n_groups} groups, {len(updates)} files")

    if dry_run:
        return 0

    cur.executemany(
        "UPDATE files SET dup_group_id = ? WHERE id = ?", updates
    )
    conn.commit()
    return n_groups


# ──────────────────────────────────────────────────────────────────────
# Phase: caption_model (legacy rows)
# ──────────────────────────────────────────────────────────────────────


def backfill_caption_model(conn, dry_run: bool) -> int:
    """Mark legacy rows with caption_model='unknown_legacy'.

    Cannot recover the actual model used pre-instrumentation, but lets us
    distinguish 'never tracked' from 'truly missing' for future audits.
    """
    cur = conn.cursor()
    n = cur.execute(
        "SELECT COUNT(*) FROM files WHERE caption_model IS NULL AND mc_caption IS NOT NULL"
    ).fetchone()[0]
    logger.info(f"[caption_model] {n} rows to mark as unknown_legacy")

    if dry_run:
        return 0

    cur.execute(
        "UPDATE files SET caption_model = 'unknown_legacy' "
        "WHERE caption_model IS NULL AND mc_caption IS NOT NULL"
    )
    conn.commit()
    logger.info(f"[caption_model] updated {cur.rowcount}")
    return cur.rowcount


# ──────────────────────────────────────────────────────────────────────
# Phase: format_norm — lowercase + recover NULL from file_name extension
# ──────────────────────────────────────────────────────────────────────


def backfill_format_norm(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    # Step 1: lowercase + strip leading dot
    rows = cur.execute(
        "SELECT id, format, file_name FROM files WHERE format IS NOT NULL"
    ).fetchall()
    norm_updates = []
    for fid, fmt, _ in rows:
        canonical = str(fmt).lower().lstrip(".")
        if canonical == "jpeg":
            canonical = "jpg"
        if canonical != fmt:
            norm_updates.append((canonical, fid))

    # Step 2: recover NULL format from file_name extension
    null_rows = cur.execute(
        "SELECT id, file_name FROM files WHERE format IS NULL AND file_name IS NOT NULL"
    ).fetchall()
    recover_updates = []
    for fid, name in null_rows:
        if "." not in name:
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        if ext == "jpeg":
            ext = "jpg"
        if ext in ("psd", "jpg", "png", "webp", "tiff", "gif", "bmp"):
            recover_updates.append((ext, fid))

    if dry_run:
        logger.info(
            f"[format_norm] dry-run: normalize {len(norm_updates)}, "
            f"recover {len(recover_updates)} of {len(null_rows)} NULL"
        )
        return 0

    if norm_updates:
        cur.executemany("UPDATE files SET format = ? WHERE id = ?", norm_updates)
    if recover_updates:
        cur.executemany("UPDATE files SET format = ? WHERE id = ?", recover_updates)
    conn.commit()
    logger.info(
        f"[format_norm] normalized {len(norm_updates)}, "
        f"recovered {len(recover_updates)} from extensions"
    )
    return len(norm_updates) + len(recover_updates)


# ──────────────────────────────────────────────────────────────────────
# Phase: resolution — width/height from file headers
# ──────────────────────────────────────────────────────────────────────


def backfill_resolution(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, file_path, format FROM files "
        "WHERE width IS NULL AND file_path IS NOT NULL"
    ).fetchall()
    logger.info(f"[resolution] {len(rows)} rows to read header")

    updates = []
    skipped = 0
    failed = 0
    for fid, fp, fmt in rows:
        try:
            if not Path(fp).exists():
                skipped += 1
                continue
            fmt_l = (fmt or "").lower().lstrip(".")
            if fmt_l == "psd":
                # Use psd-tools header read only (no thumbnail/composite)
                from psd_tools import PSDImage  # lazy
                psd = PSDImage.open(fp)
                w, h = psd.width, psd.height
            else:
                from PIL import Image
                with Image.open(fp) as img:
                    w, h = img.size
            updates.append((w, h, fid))
        except Exception as e:
            failed += 1
            logger.debug(f"[resolution] {fp}: {e}")

    if dry_run:
        logger.info(
            f"[resolution] dry-run: would update {len(updates)} "
            f"(skip={skipped}, fail={failed})"
        )
        return 0

    cur.executemany(
        "UPDATE files SET width = ?, height = ? WHERE id = ?", updates
    )
    conn.commit()
    logger.info(
        f"[resolution] updated {len(updates)} (skip={skipped}, fail={failed})"
    )
    return len(updates)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default=",".join(PHASES),
                    help=f"comma-separated subset of {PHASES}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db",
                    default=str(PROJECT_ROOT / "imageparser.db"),
                    help="SQLite DB path")
    args = ap.parse_args()

    selected = set(args.phases.split(","))
    unknown = selected - set(PHASES)
    if unknown:
        ap.error(f"Unknown phases: {unknown}. Choose from {PHASES}")

    import sqlite3
    conn = sqlite3.connect(args.db)

    logger.info(f"DB: {args.db}")
    logger.info(f"Phases: {sorted(selected)}")
    logger.info(f"Dry run: {args.dry_run}")

    if "mtime" in selected:
        backfill_modified_at(conn, args.dry_run)
    if "folder" in selected:
        backfill_folder_path(conn, args.dry_run)
    if "structured" in selected:
        backfill_structured_meta(conn, args.dry_run)
    if "phash" in selected:
        backfill_perceptual_hash(conn, args.dry_run)
    if "dup" in selected:
        backfill_dup_group(conn, args.dry_run)
    if "caption_model" in selected:
        backfill_caption_model(conn, args.dry_run)
    if "format_norm" in selected:
        backfill_format_norm(conn, args.dry_run)
    if "resolution" in selected:
        backfill_resolution(conn, args.dry_run)

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
