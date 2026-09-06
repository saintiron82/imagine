"""Content-hash backfill (CAS M2 prep, 2026-06-11).

Boundary hash = SHA256(size + first 8KB + last 8KB), so remote files
need only two Range reads. The parts-based assembly MUST be
byte-identical to the local file implementation.
"""

import os

import pytest

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue import hash_backfill
from backend.server.queue.download_ahead import (
    register_webdav_source,
)
from backend.utils.content_hash import (
    compute_content_hash,
    compute_content_hash_from_parts,
    split_points,
)


@pytest.mark.parametrize("size", [100, 8192, 12000, 16384, 50000])
def test_parts_hash_equivalent_to_local(tmp_path, size):
    data = os.urandom(size)
    p = tmp_path / "f.bin"
    p.write_bytes(data)

    head_r, tail_r = split_points(size)
    head = data[head_r[0]:head_r[1] + 1]
    tail = data[tail_r[0]:tail_r[1] + 1] if tail_r else b""

    assert compute_content_hash(p) == compute_content_hash_from_parts(
        size, head, tail)


class FakeRangeClient:
    """In-memory WebDAV client: read_range slices a known blob."""
    def __init__(self, blobs):
        self.blobs = blobs  # remote_path → bytes

    def read_range(self, remote_path, start, end):
        data = self.blobs.get(remote_path)
        if data is None:
            return None
        return data[start:end + 1]

    def close(self):
        pass


def test_hash_one_remote_matches_local(tmp_path, monkeypatch):
    data = os.urandom(40000)
    local = tmp_path / "art.psd"
    local.write_bytes(data)
    expected = compute_content_hash(local)

    register_webdav_source({
        "id": "nas1", "url": "http://nas", "username": "u",
        "password": "p", "remote_path": "/",
    })
    fake = FakeRangeClient({"/folder/art.psd": data})
    import backend.remote.webdav_client as wc
    monkeypatch.setattr(wc, "WebDAVClient", lambda **kw: fake)

    clients = {}
    got = hash_backfill._hash_one(
        "webdav://nas1/folder/art.psd", len(data), clients)
    assert got == expected


def test_hash_one_skips_unregistered_source_and_missing_local(tmp_path):
    assert hash_backfill._hash_one(
        "webdav://no-such-source/a.psd", 1000, {}) is None
    assert hash_backfill._hash_one(
        str(tmp_path / "gone.png"), 1000, {}) is None


def test_run_backfills_local_files(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    cur = db.conn.cursor()

    real = tmp_path / "a.png"
    real.write_bytes(os.urandom(20000))
    cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                (str(real), "a.png"))
    hashed_id = cur.lastrowid
    cur.execute("INSERT INTO files (file_path, file_name) VALUES (?,?)",
                (str(tmp_path / "missing.png"), "missing.png"))
    cur.execute(
        "INSERT INTO files (file_path, file_name, content_hash) VALUES (?,?,?)",
        ("/already.png", "already.png", "deadbeef"))
    db.conn.commit()

    hash_backfill._state["running"] = True
    hash_backfill._run(lambda: db)

    status = hash_backfill.get_status()
    assert status["running"] is False
    assert status["total"] == 2          # already-hashed row not selected
    assert status["done"] == 1
    assert status["skipped"] == 1        # missing file

    row = db.conn.execute(
        "SELECT content_hash FROM files WHERE id=?", (hashed_id,)).fetchone()
    assert row[0] == compute_content_hash(real)
