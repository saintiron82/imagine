"""Fallback thumbnails must not leave residue in the user's asset folder
(M0, 2026-06-11).

The fallback path (full parse failed → PIL thumbnail-only) used to write
{stem}_thumb.png next to the original file and never delete it.
"""

import os
import tempfile

import pytest
from PIL import Image

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.file_task_parse_pool import FileTaskParsePool


@pytest.fixture()
def env(tmp_path, monkeypatch):
    # A valid PNG disguised as .psd: psd-tools fails → fallback PIL succeeds
    asset_dir = tmp_path / "user_assets"
    asset_dir.mkdir()
    fake_psd = asset_dir / "art.psd"
    Image.new("RGB", (64, 64), (200, 50, 50)).save(fake_psd, "PNG")

    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('t', ?, 'active', 1)", (str(asset_dir),))
    jid = cur.lastrowid
    cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                (str(fake_psd), fake_psd.name))
    cur.execute(
        "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
        "download_status, parse_status) VALUES (?,?,?,'n/a','pending')",
        (jid, cur.lastrowid, str(fake_psd)))
    db.conn.commit()

    # Keep server thumbnails inside tmp_path too
    pool = FileTaskParsePool(db)
    server_thumbs = tmp_path / "server_thumbs"
    monkeypatch.setattr(pool, "_get_thumbnail_dir",
                        lambda: server_thumbs.mkdir(exist_ok=True) or server_thumbs)
    return db, pool, asset_dir, fake_psd


def test_fallback_leaves_no_residue_next_to_original(env):
    db, pool, asset_dir, fake_psd = env
    task_id, file_id = db.conn.execute(
        "SELECT id, file_id FROM file_tasks").fetchone()

    error = pool._parse_single_task(task_id, file_id, str(fake_psd))
    assert error == "", error

    # The user's folder contains ONLY the original — no *_thumb.png residue
    assert sorted(os.listdir(asset_dir)) == ["art.psd"]

    # The server copy exists and files.thumbnail_url points at it
    row = db.conn.execute(
        "SELECT thumbnail_url FROM files WHERE id=?", (file_id,)).fetchone()
    assert row[0] and "server_thumbs" in row[0]
    assert os.path.exists(row[0])

    # No leftover temp dirs from the fallback either
    leftovers = [d for d in os.listdir(tempfile.gettempdir())
                 if d.startswith("imagine_fallback_thumb_")]
    assert leftovers == [], leftovers
