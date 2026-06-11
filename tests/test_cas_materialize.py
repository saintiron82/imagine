"""CAS M3: cache lookup + materialization at parse time (2026-06-11).

Identical content parsed again gets its MC/VV/MV served from the
derivation cache: search-plane rows are materialized, ledger phases are
marked done with cache_hit=1, and the file never reaches a worker.
"""

import json
import struct

import pytest
from PIL import Image

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.derivations import (
    apply_cache_hits, lookup_done, record_derivation,
)
from backend.server.queue.file_task_parse_pool import FileTaskParsePool
from backend.utils.content_hash import compute_content_hash


@pytest.fixture()
def db(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    return db


def _seed_task(db, file_path, content_hash=None):
    cur = db.conn.cursor()
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files) "
        "VALUES ('t','/x','active',1)")
    jid = cur.lastrowid
    cur.execute(
        "INSERT INTO files (file_path, file_name, content_hash) VALUES (?,?,?)",
        (file_path, file_path.rsplit("/", 1)[-1], content_hash))
    fid = cur.lastrowid
    cur.execute(
        "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
        "download_status, parse_status) VALUES (?,?,?,'n/a','pending')",
        (jid, fid, file_path))
    db.conn.commit()
    return jid, fid, cur.lastrowid


VV_VEC = struct.pack("<1152f", *([0.5] * 1152))   # vec_files dimension
MV_VEC = struct.pack("<1024f", *([0.5] * 1024))   # vec_text dimension


def _cache_all_phases(db, donor_file_id, caption="cached caption"):
    record_derivation(db, donor_file_id, "mc",
                      result_json=json.dumps({"mc_caption": caption,
                                              "ai_tags": ["cached"]}))
    record_derivation(db, donor_file_id, "vv", vector_blob=VV_VEC)
    record_derivation(db, donor_file_id, "mv", vector_blob=MV_VEC)
    db.conn.commit()


def test_apply_cache_hits_materializes_everything(db):
    _, donor_fid, _ = _seed_task(db, "/x/original.png", content_hash="h1")
    _cache_all_phases(db, donor_fid)

    # A duplicate file (same hash) in a new job
    jid, dup_fid, task_id = _seed_task(db, "/y/copy.png", content_hash="h1")

    applied = apply_cache_hits(db, task_id, dup_fid, "h1")
    db.conn.commit()
    assert set(applied) == {"mc", "vv", "mv"}

    # Ledger: done + cache_hit flags
    row = db.conn.execute(
        "SELECT mc_status, vv_status, mv_status, mc_cache_hit, vv_cache_hit, "
        "mv_cache_hit FROM file_tasks WHERE id=?", (task_id,)).fetchone()
    assert tuple(row) == ("done", "done", "done", 1, 1, 1)

    # Search plane: caption + vectors materialized for the duplicate
    cap = db.conn.execute(
        "SELECT mc_caption FROM files WHERE id=?", (dup_fid,)).fetchone()
    assert cap[0] == "cached caption"
    for table in ("vec_files", "vec_text"):
        n = db.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE file_id=?", (dup_fid,)
        ).fetchone()[0]
        assert n == 1, table


def test_mv_hit_requires_mc_hit(db):
    """MV embeds the MC caption — a cached MV without a cached MC would be
    incoherent with the fresh caption a worker will produce."""
    _, donor_fid, _ = _seed_task(db, "/x/o.png", content_hash="h2")
    record_derivation(db, donor_fid, "vv", vector_blob=VV_VEC)
    record_derivation(db, donor_fid, "mv", vector_blob=MV_VEC)
    db.conn.commit()

    _, dup_fid, task_id = _seed_task(db, "/y/c.png", content_hash="h2")
    applied = apply_cache_hits(db, task_id, dup_fid, "h2")
    db.conn.commit()

    assert set(applied) == {"vv"}
    row = db.conn.execute(
        "SELECT mv_status, mc_status FROM file_tasks WHERE id=?",
        (task_id,)).fetchone()
    assert row[0] == "pending" and row[1] == "pending"


def test_stale_model_version_misses(db):
    _, donor_fid, _ = _seed_task(db, "/x/o.png", content_hash="h3")
    _cache_all_phases(db, donor_fid)
    # Simulate a model upgrade: stored derivations now carry an old version
    db.conn.execute("UPDATE derivations SET model_version = 'old-model/v0'")
    db.conn.commit()

    assert lookup_done(db, "h3") == {}


def test_parse_pipeline_end_to_end_cache_hit(db, tmp_path, monkeypatch):
    """Full parse path: a duplicate PNG completes all phases at parse time."""
    img = tmp_path / "art.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
    dup = tmp_path / "art_copy.png"
    dup.write_bytes(img.read_bytes())

    # Donor: hashed + all phases cached
    _, donor_fid, _ = _seed_task(db, str(img), compute_content_hash(img))
    _cache_all_phases(db, donor_fid, caption="from donor")

    jid, dup_fid, task_id = _seed_task(db, str(dup))

    pool = FileTaskParsePool(db)
    server_thumbs = tmp_path / "thumbs"
    monkeypatch.setattr(pool, "_get_thumbnail_dir",
                        lambda: server_thumbs.mkdir(exist_ok=True) or server_thumbs)
    error = pool._parse_single_task(task_id, dup_fid, str(dup))
    assert error == "", error

    row = db.conn.execute(
        "SELECT mc_status, vv_status, mv_status FROM file_tasks WHERE id=?",
        (task_id,)).fetchone()
    assert tuple(row) == ("done", "done", "done")
    cap = db.conn.execute(
        "SELECT mc_caption FROM files WHERE id=?", (dup_fid,)).fetchone()
    assert cap[0] == "from donor"
