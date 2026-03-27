"""
Regenerate MV vectors for all files with valid MC captions.

Removes folder_path from MV input — MV should be pure semantic (image content only).
Folder paths belong to FTS (metadata axis).

Usage:
    .venv/bin/python scripts/regenerate_mv.py
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from backend.db.sqlite_client import SQLiteDB
from backend.vector.text_embedding import get_text_embedding_provider, build_document_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    db = SQLiteDB()
    cursor = db.conn.cursor()

    # Get all files with valid MC
    cursor.execute("""
        SELECT id, mc_caption, ai_tags, image_type, scene_type, art_style
        FROM files
        WHERE mc_caption IS NOT NULL AND mc_caption != '' AND mc_caption != 'unknown'
        ORDER BY id
    """)
    rows = cursor.fetchall()
    total = len(rows)
    logger.info(f"Regenerating MV for {total} files (folder_path removed from input)")

    # Load embedding provider
    provider = get_text_embedding_provider()
    if provider is None:
        logger.error("Failed to load text embedding provider")
        return

    batch_size = 16
    processed = 0
    t_start = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start:batch_start + batch_size]

        texts = []
        ids = []
        for row in batch:
            fid, caption, tags_raw, img_type, scene_type, art_style = row

            # Parse tags
            tags = []
            if tags_raw:
                try:
                    tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
                except (json.JSONDecodeError, TypeError):
                    tags = []

            # Build facts WITHOUT folder_path
            facts = {}
            if img_type:
                facts["image_type"] = img_type
            if scene_type:
                facts["scene_type"] = scene_type
            if art_style:
                facts["art_style"] = art_style

            text = build_document_text(caption, tags, facts=facts)
            texts.append(text)
            ids.append(fid)

        # Batch encode
        try:
            if hasattr(provider, 'encode_batch'):
                vectors = provider.encode_batch(texts)
            else:
                vectors = [provider.encode(t) for t in texts]
        except Exception as e:
            logger.error(f"Encode failed at batch {batch_start}: {e}")
            continue

        # Update DB
        for fid, vec in zip(ids, vectors):
            embedding_list = vec.astype(np.float32).tolist()
            cursor.execute("DELETE FROM vec_text WHERE file_id = ?", (fid,))
            cursor.execute(
                "INSERT INTO vec_text (file_id, embedding) VALUES (?, ?)",
                (fid, json.dumps(embedding_list))
            )

        db.conn.commit()
        processed += len(batch)

        if processed % 100 == 0 or processed == total:
            elapsed = time.perf_counter() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            logger.info(f"  {processed}/{total} ({rate:.1f} files/s, ETA {eta:.0f}s)")

    # Delete MV vectors for files with invalid MC
    cursor.execute("""
        DELETE FROM vec_text WHERE file_id IN (
            SELECT id FROM files WHERE mc_caption IS NULL OR mc_caption = '' OR mc_caption = 'unknown'
        )
    """)
    deleted = cursor.rowcount
    if deleted > 0:
        db.conn.commit()
        logger.info(f"Deleted {deleted} MV vectors for files with invalid MC")

    elapsed = time.perf_counter() - t_start
    logger.info(f"Done: {processed} MV vectors regenerated in {elapsed:.1f}s")
    db.close()


if __name__ == "__main__":
    main()
