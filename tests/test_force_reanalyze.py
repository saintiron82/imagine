"""'전체 다시 분석' (force_reanalyze) — CAS 캐시 게이트 (2026-07-06, IMGV2 #266).

discover/scan 의 force_reanalyze 는 잡 프로필(analysis_profile_json)에
실리고, parse 단계의 캐시 적용(apply_cache_hits) 여부를
FileTaskParsePool._job_forces_reanalyze 가 결정한다.
"""

import json

import pytest

from backend.db.sqlite_client import SQLiteDB
from backend.server.queue.analysis_manager import AnalysisJobManager
from backend.server.queue.file_task_parse_pool import FileTaskParsePool


@pytest.fixture()
def db(tmp_path):
    AnalysisJobManager._initialized = False
    db = SQLiteDB(str(tmp_path / "test.db"))
    AnalysisJobManager(db)
    return db


def _seed_task(db, profile: dict = None):
    cur = db.conn.cursor()
    profile_json = json.dumps(profile, ensure_ascii=False) if profile else None
    cur.execute(
        "INSERT INTO analysis_jobs (name, source_path, status, total_files, "
        "analysis_profile_json) VALUES ('t','/x','active',1,?)",
        (profile_json,))
    jid = cur.lastrowid
    cur.execute(
        "INSERT INTO files (file_path, file_name) VALUES ('/x/a.png','a.png')")
    fid = cur.lastrowid
    cur.execute(
        "INSERT INTO file_tasks (analysis_job_id, file_id, file_path, "
        "download_status, parse_status) VALUES (?,?,'/x/a.png','n/a','pending')",
        (jid, fid))
    db.conn.commit()
    return cur.lastrowid


def test_force_flag_true(db):
    task_id = _seed_task(db, {"force_reanalyze": True,
                              "expected_types": ["character"]})
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is True


def test_no_profile_defaults_false(db):
    task_id = _seed_task(db, None)
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is False


def test_profile_without_flag_defaults_false(db):
    task_id = _seed_task(db, {"expected_types": ["background"]})
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(task_id) is False


def test_unknown_task_defaults_false(db):
    pool = FileTaskParsePool(db)
    assert pool._job_forces_reanalyze(999999) is False
