"""'전체 다시 분석' (force_reanalyze) — CAS 캐시 게이트 (2026-07-06, IMGV2 #266).

discover/scan 의 force_reanalyze 는 잡 프로필(analysis_profile_json)에
실리고, parse 단계의 캐시 적용(apply_cache_hits) 여부를
FileTaskParsePool._job_forces_reanalyze 가 결정한다.
게이트 판단 불가 시에는 캐시를 건너뛴다(재계산이 안전한 방향).
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.db.sqlite_client import SQLiteDB
from backend.server import deps
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.derivations import record_derivation
from backend.server.queue.file_task_parse_pool import FileTaskParsePool
from backend.server.routers import analysis
from backend.utils.content_hash import compute_content_hash


@pytest.fixture()
def db(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    return db


def _seed_task(db, file_path="/x/a.png", profile: dict = None,
               profile_raw: str = None, content_hash=None):
    cur = db.conn.cursor()
    profile_json = profile_raw if profile_raw is not None else (
        json.dumps(profile, ensure_ascii=False) if profile else None)
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files, "
        "analysis_profile_json) VALUES ('t','/x','active',1,?)",
        (profile_json,))
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


# ── 게이트 단위 판단 ─────────────────────────────────────────

def test_force_flag_true(db):
    _, _, task_id = _seed_task(db, profile={"force_reanalyze": True,
                                            "expected_types": ["character"]})
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is True


def test_no_profile_defaults_false(db):
    _, _, task_id = _seed_task(db, profile=None)
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is False


def test_profile_without_flag_defaults_false(db):
    _, _, task_id = _seed_task(db, profile={"expected_types": ["background"]})
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is False


def test_unknown_task_defaults_false(db):
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(999999) is False


def test_corrupt_profile_fails_toward_recompute(db):
    """판단 불가 시 캐시 적용(False)이 아니라 재계산(True)으로 기울어야 한다."""
    _, _, task_id = _seed_task(db, profile_raw="{not valid json")
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is True


# ── 통합: parse 8단계가 실제로 캐시를 건너뛰는지 ─────────────

from conftest import vv_blob, mv_blob

VV_VEC = vv_blob()
MV_VEC = mv_blob()


def _cache_all_phases(db, donor_file_id, caption="from donor"):
    record_derivation(db, donor_file_id, "mc",
                      result_json=json.dumps({"mc_caption": caption,
                                              "ai_tags": ["cached"]}))
    record_derivation(db, donor_file_id, "vv", vector_blob=VV_VEC)
    record_derivation(db, donor_file_id, "mv", vector_blob=MV_VEC)
    db.conn.commit()


def test_parse_pipeline_force_skips_cache(db, tmp_path, monkeypatch):
    """force 잡의 중복 파일은 캐시로 제공되지 않고 전 phase 가 pending 으로 남는다."""
    img = tmp_path / "art.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(img)
    dup = tmp_path / "art_copy.png"
    dup.write_bytes(img.read_bytes())

    # 도너: 해시 + 전 phase 캐시
    _, donor_fid, _ = _seed_task(db, str(img),
                                 content_hash=compute_content_hash(img))
    _cache_all_phases(db, donor_fid)

    # 동일 콘텐츠를 force 잡으로 등록
    _, dup_fid, task_id = _seed_task(db, str(dup),
                                     profile={"force_reanalyze": True})

    pool = FileTaskParsePool(db)
    server_thumbs = tmp_path / "thumbs"
    monkeypatch.setattr(pool, "_get_thumbnail_dir",
                        lambda: server_thumbs.mkdir(exist_ok=True) or server_thumbs)
    error = pool._parse_single_task(task_id, dup_fid, str(dup))
    assert error == "", error

    # 캐시 미적용 — AI phase 는 워커 몫으로 남는다
    row = db.conn.execute(
        "SELECT mc_status, vv_status, mv_status FROM file_tasks WHERE id=?",
        (task_id,)).fetchone()
    assert tuple(row) == ("pending", "pending", "pending")
    cap = db.conn.execute(
        "SELECT mc_caption FROM files WHERE id=?", (dup_fid,)).fetchone()
    assert cap[0] != "from donor"


# ── 라우터: discover/scan 의 force → profile 폴드 ────────────

def _scan_client(db, tmp_path, monkeypatch, captured):
    class _StubMgr:
        def create_job(self, **kw):
            captured.update(kw)
            return {"job_id": 1, "total_files": 1}

    app = FastAPI()
    app.include_router(analysis.router)
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = (
        lambda: {"id": 1, "username": "u"})
    monkeypatch.setattr(analysis, "_get_manager", lambda _db: _StubMgr())
    monkeypatch.setattr(analysis, "_auto_start_worker", lambda: None)

    folder = tmp_path / "assets"
    folder.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(folder / "a.png")
    return TestClient(app), str(folder)


def test_scan_folds_force_into_profile(db, tmp_path, monkeypatch):
    captured = {}
    client, folder = _scan_client(db, tmp_path, monkeypatch, captured)
    resp = client.post("/api/v1/discover/scan",
                       json={"folder_path": folder, "force_reanalyze": True})
    assert resp.status_code == 200
    assert captured["analysis_profile"] == {"force_reanalyze": True}


def test_scan_merges_force_with_existing_profile(db, tmp_path, monkeypatch):
    captured = {}
    client, folder = _scan_client(db, tmp_path, monkeypatch, captured)
    resp = client.post("/api/v1/discover/scan", json={
        "folder_path": folder,
        "force_reanalyze": True,
        "analysis_profile": {"expected_types": ["character"], "source": "user"},
    })
    assert resp.status_code == 200
    assert captured["analysis_profile"] == {
        "expected_types": ["character"], "source": "user",
        "force_reanalyze": True,
    }


def test_scan_without_force_keeps_profile_none(db, tmp_path, monkeypatch):
    captured = {}
    client, folder = _scan_client(db, tmp_path, monkeypatch, captured)
    resp = client.post("/api/v1/discover/scan", json={"folder_path": folder})
    assert resp.status_code == 200
    assert captured["analysis_profile"] is None
