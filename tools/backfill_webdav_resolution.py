"""
P09 잔여: WebDAV remote 파일에서 width/height 백필 (HTTP Range 요청).

PSD/JPG/PNG 헤더 첫 청크만 받아 dimension 추출. 파일 본체는 다운로드 안 함.

사용 전제:
  - WebDAV 서버가 실행 중이거나, 사용자가 base_url/user/pass를 환경변수로 제공
  - file_path가 webdav://{source_id}/{relative_path} 형태

실행:
    WEBDAV_BASE_URL=https://nas.example.com:5006 \
    WEBDAV_USER=imagine WEBDAV_PASS=secret \
    WEBDAV_SOURCE_ID=13730b09 \
    python tools/backfill_webdav_resolution.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import struct
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("webdav-resolution")


# ─── PSD header parser (first 26 bytes) ─────────────────────────────


def _parse_psd_header(buf: bytes):
    """8BPS magic at offset 0. Height@14 (4B), Width@18 (4B), big-endian."""
    if len(buf) < 26 or buf[:4] != b"8BPS":
        return None
    height = struct.unpack(">I", buf[14:18])[0]
    width = struct.unpack(">I", buf[18:22])[0]
    return width, height


def _parse_png_header(buf: bytes):
    """PNG signature 8B + IHDR chunk (Width@16, Height@20, big-endian)."""
    if len(buf) < 24 or buf[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = struct.unpack(">I", buf[16:20])[0]
    height = struct.unpack(">I", buf[20:24])[0]
    return width, height


def _parse_jpg_header(buf: bytes):
    """JPEG: scan markers until SOF0/SOF2 (0xFFC0/0xFFC2). PIL handles edge cases."""
    try:
        from PIL import Image
        with Image.open(BytesIO(buf)) as img:
            return img.size
    except Exception:
        return None


_PARSERS = {"psd": _parse_psd_header, "png": _parse_png_header,
            "jpg": _parse_jpg_header, "jpeg": _parse_jpg_header}


# ─── HTTP Range fetch ────────────────────────────────────────────────


def _fetch_range(session, url: str, n_bytes: int = 65536):
    """Fetch first n_bytes of url via Range header."""
    r = session.get(url, headers={"Range": f"bytes=0-{n_bytes - 1}"}, timeout=30)
    r.raise_for_status()
    return r.content


# ─── webdav:// URL → real URL ────────────────────────────────────────


_WEBDAV_RE = re.compile(r"^webdav://([^/]+)/(.+)$")


def _to_real_url(file_path: str, base_url: str, source_id: str):
    m = _WEBDAV_RE.match(file_path)
    if not m:
        return None
    if source_id and m.group(1) != source_id:
        return None
    rel = m.group(2)
    parts = rel.split("/")
    encoded = "/".join(quote(p, safe="") for p in parts)
    return f"{base_url.rstrip('/')}/{encoded}"


# ─── Main ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    ap.add_argument("--limit", type=int, default=0,
                    help="0=all, otherwise process first N rows")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_url = os.environ.get("WEBDAV_BASE_URL")
    user = os.environ.get("WEBDAV_USER")
    pwd = os.environ.get("WEBDAV_PASS")
    source_id = os.environ.get("WEBDAV_SOURCE_ID", "")

    if not base_url or not user or not pwd:
        logger.error(
            "Missing WEBDAV_BASE_URL / WEBDAV_USER / WEBDAV_PASS env vars. "
            "Run example:\n"
            "  WEBDAV_BASE_URL=https://nas:5006 WEBDAV_USER=u WEBDAV_PASS=p \\\n"
            "  WEBDAV_SOURCE_ID=13730b09 python tools/backfill_webdav_resolution.py"
        )
        sys.exit(2)

    import requests
    session = requests.Session()
    session.auth = (user, pwd)
    session.verify = True

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    sql = (
        "SELECT id, file_path, format FROM files "
        "WHERE width IS NULL AND file_path LIKE 'webdav://%'"
    )
    if args.limit:
        sql += f" LIMIT {args.limit}"
    rows = cur.execute(sql).fetchall()
    logger.info(f"{len(rows)} WebDAV rows missing width")

    updates = []
    skipped = 0
    failed = 0
    for fid, fp, fmt in rows:
        url = _to_real_url(fp, base_url, source_id)
        if not url:
            skipped += 1
            continue
        fmt_l = (fmt or fp.rsplit(".", 1)[-1] or "").lower().lstrip(".")
        parser = _PARSERS.get(fmt_l)
        if not parser:
            skipped += 1
            continue
        try:
            buf = _fetch_range(session, url)
            dim = parser(buf)
            if dim:
                w, h = dim
                updates.append((w, h, fid))
            else:
                failed += 1
        except Exception as e:
            failed += 1
            logger.debug(f"{fp}: {e}")

        if len(updates) and len(updates) % 500 == 0:
            logger.info(f"progress: {len(updates)} updates, {failed} fail")

    if args.dry_run:
        logger.info(f"dry-run: would update {len(updates)} (skip={skipped}, fail={failed})")
    else:
        cur.executemany("UPDATE files SET width = ?, height = ? WHERE id = ?", updates)
        conn.commit()
        logger.info(f"updated {len(updates)} (skip={skipped}, fail={failed})")

    conn.close()


if __name__ == "__main__":
    main()
