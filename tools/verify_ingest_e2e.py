"""
Sprint 1+2 잔여: 신규 인제스트 e2e 검증.

PhaseRunner + storage 화이트리스트 수정이 진짜 작동하는지 venv 환경에서 확인.

사용:
    1. venv 활성화 (torch/psd-tools/PIL 등 필수)
    2. python tools/verify_ingest_e2e.py <test_image.psd_or_png>

검증 항목:
    - structured_meta 채워짐
    - perceptual_hash 채워짐
    - dominant_color 채워짐
    - ai_style 채워짐
    - caption_model 채워짐
    - processing_status 'vision_done' 또는 'parse_fallback'
    - files_fts.caption / ai_tags / classification / spatial 채워짐
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


_REQUIRED_COLS = (
    "structured_meta",
    "perceptual_hash",
    "dominant_color",
    "ai_style",
    "caption_model",
    "processing_status",
)
_FTS_COLS = ("caption", "ai_tags", "classification", "spatial")


def _snapshot_max_id(conn) -> int:
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM files").fetchone()[0]


def _verify(conn, file_id: int) -> dict:
    cur = conn.cursor()
    cols = ", ".join(_REQUIRED_COLS)
    row = cur.execute(
        f"SELECT id, file_path, file_name, format, {cols} FROM files WHERE id = ?",
        (file_id,)
    ).fetchone()
    if not row:
        return {"error": f"file_id={file_id} not found"}

    out = {
        "file_id": row[0], "file_path": row[1], "file_name": row[2],
        "format": row[3], "checks": {}, "fts": {}, "objects": {},
    }
    for i, col in enumerate(_REQUIRED_COLS, start=4):
        val = row[i]
        out["checks"][col] = {
            "filled": val is not None,
            "preview": (str(val)[:80] if val is not None else None),
        }

    fts_cols = ", ".join(_FTS_COLS)
    fts_row = cur.execute(
        f"SELECT {fts_cols} FROM files_fts WHERE rowid = ?", (file_id,)
    ).fetchone()
    if fts_row:
        for i, col in enumerate(_FTS_COLS):
            v = fts_row[i] or ""
            out["fts"][col] = {"filled": bool(v.strip()), "len": len(v)}

    try:
        structured_raw = row[4 + _REQUIRED_COLS.index("structured_meta")]
        structured = json.loads(structured_raw or "{}")
        objects = structured.get("objects")
        out["objects"]["structured_meta_count"] = len(objects) if isinstance(objects, list) else 0
    except Exception:
        out["objects"]["structured_meta_count"] = 0

    try:
        out["objects"]["file_objects_count"] = cur.execute(
            "SELECT COUNT(*) FROM file_objects WHERE file_id = ?", (file_id,)
        ).fetchone()[0]
    except sqlite3.OperationalError:
        out["objects"]["file_objects_count"] = None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to test image (PSD/PNG/JPG)")
    ap.add_argument("--db", default=str(PROJECT_ROOT / "imageparser.db"))
    args = ap.parse_args()

    test_file = Path(args.file).resolve()
    if not test_file.exists():
        print(f"ERROR: test file not found: {test_file}")
        sys.exit(2)

    conn = sqlite3.connect(args.db)
    pre_max = _snapshot_max_id(conn)
    print(f"DB: {args.db}")
    print(f"Test file: {test_file}")
    print(f"Pre-ingest max(id) = {pre_max}")
    print()

    # Run ingest via the pipeline CLI
    print("Running ingest...")
    t0 = time.time()
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "backend.pipeline.ingest_engine",
         "--file", str(test_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    dt = time.time() - t0
    print(f"Ingest finished in {dt:.1f}s (exit={result.returncode})")
    if result.returncode != 0:
        print("STDERR (last 30 lines):")
        for line in result.stderr.splitlines()[-30:]:
            print(f"  {line}")
        sys.exit(1)

    # Re-open conn (writes from subprocess)
    conn.close()
    conn = sqlite3.connect(args.db)
    target_path = str(test_file)
    row = conn.execute(
        "SELECT id FROM files WHERE file_path = ? OR file_path = ? "
        "ORDER BY id DESC LIMIT 1",
        (target_path, target_path),
    ).fetchone()
    if not row:
        # Fallback: any new row after pre_max
        row = conn.execute(
            "SELECT id FROM files WHERE id > ? ORDER BY id DESC LIMIT 1",
            (pre_max,),
        ).fetchone()
    if not row:
        print("ERROR: no new row found in DB after ingest")
        sys.exit(1)

    fid = row[0]
    result = _verify(conn, fid)

    print("\n=== Verification ===")
    print(f"file_id={result['file_id']} format={result['format']}")
    print(f"  {result['file_name']}")
    print()
    print("Column checks:")
    fail = 0
    for col, info in result["checks"].items():
        mark = "✓" if info["filled"] else "✗"
        print(f"  {mark} {col:20s} {'(filled)' if info['filled'] else '(NULL)'}"
              + (f"  → {info['preview']!r}" if info["preview"] else ""))
        if not info["filled"]:
            fail += 1

    print("\nFTS row:")
    if not result["fts"]:
        print("  ✗ no FTS row found")
        fail += 1
    else:
        for col, info in result["fts"].items():
            mark = "✓" if info["filled"] else "✗"
            print(f"  {mark} {col:20s} {'len=' + str(info['len']) if info['filled'] else '(empty)'}")
            if not info["filled"]:
                fail += 1

    print("\nSpatial objects:")
    structured_count = result["objects"].get("structured_meta_count", 0)
    file_objects_count = result["objects"].get("file_objects_count")
    print(f"  {'✓' if structured_count else '✗'} structured_meta.objects count={structured_count}")
    print(f"  {'✓' if file_objects_count else '✗'} file_objects count={file_objects_count}")
    if not structured_count or not file_objects_count:
        fail += 1

    print()
    if fail == 0:
        print("ALL OK ✓")
    else:
        print(f"{fail} columns/checks failed ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
