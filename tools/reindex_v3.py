"""
v3 Reindex Tool — Re-run 2-Stage Vision analysis on existing files.

Modes:
    --vision-only     Re-run Stage1+Stage2 classification (no embedding change)
    --embedding-only  Re-generate embeddings only (for model swap)
    --all             Both vision + embedding
    --limit N         Process only N files (for testing)
    --dry-run         Show what would be processed without actually doing it

Usage:
    python tools/reindex_v3.py --vision-only
    python tools/reindex_v3.py --vision-only --limit 5
    python tools/reindex_v3.py --embedding-only
    python tools/reindex_v3.py --all
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reindex_vision(limit: int = 0, dry_run: bool = False, unprocessed: bool = False):
    """Re-run 2-Stage Vision on files that have thumbnails."""
    from backend.db.sqlite_client import SQLiteDB
    from backend.vision.vision_factory import get_vision_analyzer
    from PIL import Image

    db = SQLiteDB()
    analyzer = get_vision_analyzer()
    cursor = db.conn.cursor()

    query = "SELECT id, file_path, thumbnail_url FROM files"
    if unprocessed:
        query += " WHERE image_type IS NULL"
    query += " ORDER BY id"
    if limit > 0:
        query += f" LIMIT {limit}"

    rows = cursor.execute(query).fetchall()
    total = len(rows)
    logger.info(f"Vision reindex: {total} files to process")

    if dry_run:
        for row in rows:
            logger.info(f"  [DRY] {row[1]}")
        return

    success = 0
    batch_start = time.perf_counter()
    for i, (file_id, file_path, thumbnail_url) in enumerate(rows, 1):
        if not thumbnail_url:
            logger.warning(f"  [{i}/{total}] No thumbnail: {file_path}")
            continue

        # Resolve thumbnail path
        thumb_path = Path(thumbnail_url)
        if not thumb_path.is_absolute():
            thumb_path = PROJECT_ROOT / thumbnail_url
        if thumbnail_url.startswith('file:///'):
            import urllib.parse
            thumb_path = Path(urllib.parse.unquote(thumbnail_url[8:]))

        if not thumb_path.exists():
            logger.warning(f"  [{i}/{total}] Thumbnail missing: {thumb_path}")
            continue

        try:
            file_start = time.perf_counter()
            image = Image.open(thumb_path).convert("RGB")
            result = analyzer.classify_and_analyze(image)

            # Update DB
            cursor.execute("""
                UPDATE files SET
                    image_type = ?,
                    art_style = ?,
                    color_palette = ?,
                    scene_type = ?,
                    time_of_day = ?,
                    weather = ?,
                    character_type = ?,
                    item_type = ?,
                    ui_type = ?,
                    structured_meta = ?,
                    ai_caption = COALESCE(?, ai_caption),
                    ai_tags = COALESCE(?, ai_tags),
                    ai_style = COALESCE(?, ai_style)
                WHERE id = ?
            """, (
                result.get("image_type"),
                result.get("art_style"),
                result.get("color_palette"),
                result.get("scene_type"),
                result.get("time_of_day"),
                result.get("weather"),
                result.get("character_type"),
                result.get("item_type"),
                result.get("ui_type"),
                json.dumps(result, ensure_ascii=False),
                result.get("caption") or None,
                json.dumps(result.get("tags", [])) if result.get("tags") else None,
                result.get("art_style") or None,
                file_id,
            ))
            db.conn.commit()

            file_elapsed = time.perf_counter() - file_start
            logger.info(f"  [{i}/{total}] {result.get('image_type', '?')} — {file_path} ({file_elapsed:.1f}s)")
            success += 1

        except Exception as e:
            file_elapsed = time.perf_counter() - file_start
            logger.error(f"  [{i}/{total}] Failed: {file_path} — {e} ({file_elapsed:.1f}s)")

    # Unload model after batch
    if hasattr(analyzer, 'unload_model'):
        analyzer.unload_model()

    batch_elapsed = time.perf_counter() - batch_start
    avg = batch_elapsed / total if total > 0 else 0
    logger.info(f"Vision reindex complete: {success}/{total} succeeded — total {batch_elapsed:.1f}s (avg {avg:.1f}s/file)")
    db.close()


def reindex_embedding(limit: int = 0, dry_run: bool = False):
    """Re-generate embeddings for all files using SigLIP 2."""
    from backend.db.sqlite_client import SQLiteDB
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    from PIL import Image
    import numpy as np

    db = SQLiteDB()
    cursor = db.conn.cursor()

    encoder = SigLIP2Encoder()
    model_name = encoder.model_name
    logger.info(f"Loading embedding model: {model_name}")

    query = "SELECT id, file_path, thumbnail_url FROM files ORDER BY id"
    if limit > 0:
        query += f" LIMIT {limit}"

    rows = cursor.execute(query).fetchall()
    total = len(rows)
    logger.info(f"Embedding reindex: {total} files to process")

    if dry_run:
        for row in rows:
            logger.info(f"  [DRY] {row[1]}")
        return

    success = 0
    for i, (file_id, file_path, thumbnail_url) in enumerate(rows, 1):
        if not thumbnail_url:
            continue

        thumb_path = Path(thumbnail_url)
        if not thumb_path.is_absolute():
            thumb_path = PROJECT_ROOT / thumbnail_url
        if thumbnail_url.startswith('file:///'):
            import urllib.parse
            thumb_path = Path(urllib.parse.unquote(thumbnail_url[8:]))

        if not thumb_path.exists():
            continue

        try:
            image = Image.open(thumb_path).convert("RGB")
            embedding = encoder.encode_image(image)
            embedding_list = embedding.astype(np.float32).tolist()

            cursor.execute("DELETE FROM vec_files WHERE file_id = ?", (file_id,))
            cursor.execute(
                "INSERT INTO vec_files (file_id, embedding) VALUES (?, ?)",
                (file_id, json.dumps(embedding_list))
            )
            cursor.execute(
                "UPDATE files SET embedding_model = ?, embedding_version = embedding_version + 1 WHERE id = ?",
                (model_name, file_id)
            )
            db.conn.commit()
            success += 1

            if i % 10 == 0:
                logger.info(f"  [{i}/{total}] processed")

        except Exception as e:
            logger.error(f"  [{i}/{total}] Failed: {file_path} — {e}")

    logger.info(f"Embedding reindex complete: {success}/{total} succeeded")
    db.close()


def reindex_text_embedding(limit: int = 0, dry_run: bool = False):
    """Re-generate T-axis text embeddings from ai_caption + ai_tags."""
    from backend.db.sqlite_client import SQLiteDB
    from backend.vector.text_embedding import get_text_embedding_provider, build_document_text
    import numpy as np

    db = SQLiteDB()
    cursor = db.conn.cursor()

    provider = get_text_embedding_provider()
    logger.info(f"Loading T-axis model: {provider.model} ({provider.dimensions}-dim)")

    # Ensure vec_text table exists
    try:
        cursor.execute("SELECT COUNT(*) FROM vec_text")
    except Exception:
        logger.info("Creating vec_text table...")
        from backend.db.migrations.v3_p2_text_vec import migrate
        migrate()
        # Reconnect after migration
        db = SQLiteDB()
        cursor = db.conn.cursor()

    query = "SELECT id, file_path, ai_caption, ai_tags FROM files ORDER BY id"
    if limit > 0:
        query += f" LIMIT {limit}"

    rows = cursor.execute(query).fetchall()
    total = len(rows)
    logger.info(f"T-axis reindex: {total} files to process")

    if dry_run:
        for row in rows:
            logger.info(f"  [DRY] {row[1]}")
        return

    success = 0
    batch_start = time.perf_counter()
    for i, (file_id, file_path, ai_caption, ai_tags_raw) in enumerate(rows, 1):
        # Parse ai_tags
        ai_tags = []
        if ai_tags_raw:
            try:
                ai_tags = json.loads(ai_tags_raw)
            except (json.JSONDecodeError, TypeError):
                ai_tags = []

        doc_text = build_document_text(ai_caption, ai_tags)
        if not doc_text:
            logger.warning(f"  [{i}/{total}] No caption/tags: {file_path}")
            continue

        try:
            embedding = provider.encode(doc_text)
            if not np.any(embedding):
                logger.warning(f"  [{i}/{total}] Zero embedding: {file_path}")
                continue

            embedding_list = embedding.astype(np.float32).tolist()

            cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (file_id,))
            cursor.execute(
                "INSERT INTO vec_text (file_id, embedding) VALUES (?, ?)",
                (file_id, json.dumps(embedding_list))
            )
            db.conn.commit()
            success += 1

            if i % 10 == 0 or i == total:
                elapsed = time.perf_counter() - batch_start
                logger.info(f"  [{i}/{total}] processed ({elapsed:.1f}s)")

        except Exception as e:
            logger.error(f"  [{i}/{total}] Failed: {file_path} — {e}")

    batch_elapsed = time.perf_counter() - batch_start
    avg = batch_elapsed / total if total > 0 else 0
    logger.info(
        f"T-axis reindex complete: {success}/{total} succeeded — "
        f"total {batch_elapsed:.1f}s (avg {avg:.2f}s/file)"
    )
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v3 Reindex Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--vision-only", action="store_true", help="Re-run 2-Stage Vision analysis")
    group.add_argument("--embedding-only", action="store_true", help="Re-generate V-axis embeddings")
    group.add_argument("--text-embedding", action="store_true", help="Re-generate T-axis text embeddings")
    group.add_argument("--all", action="store_true", help="Vision + V-axis + T-axis embedding")
    parser.add_argument("--limit", type=int, default=0, help="Limit files to process")
    parser.add_argument("--dry-run", action="store_true", help="Show without processing")
    parser.add_argument("--unprocessed", action="store_true", help="Only process files without image_type")

    args = parser.parse_args()

    if args.vision_only or args.all:
        reindex_vision(limit=args.limit, dry_run=args.dry_run, unprocessed=args.unprocessed)

    if args.embedding_only or args.all:
        reindex_embedding(limit=args.limit, dry_run=args.dry_run)

    if args.text_embedding or args.all:
        reindex_text_embedding(limit=args.limit, dry_run=args.dry_run)
