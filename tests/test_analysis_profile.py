import sqlite3
from types import SimpleNamespace

from backend.server.queue.analysis_manager import AnalysisJobManager


def test_create_job_stores_analysis_profile_json():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE analysis_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            source_path TEXT,
            total_files INTEGER,
            created_by INTEGER,
            created_at TEXT,
            analysis_profile_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_job_id INTEGER,
            file_id INTEGER,
            file_path TEXT,
            download_status TEXT,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(analysis_job_id, file_id)
        )
    """)
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            file_name TEXT,
            relative_path TEXT,
            created_at TEXT
        )
    """)

    mgr = AnalysisJobManager.__new__(AnalysisJobManager)
    mgr.db = SimpleNamespace(conn=conn)

    profile = {
        "domain_id": "illustration",
        "expected_types": ["background", "effect"],
        "primary_type": "background",
        "source": "user",
    }
    result = mgr.create_job(
        name="BG batch",
        source_path="/tmp/bg",
        file_paths=["/tmp/bg/a.psd"],
        created_by=1,
        analysis_profile=profile,
    )

    assert result["total_files"] == 1
    stored = conn.execute(
        "SELECT analysis_profile_json FROM analysis_jobs WHERE id = ?",
        (result["job_id"],),
    ).fetchone()[0]
    assert '"expected_types": ["background", "effect"]' in stored
    assert '"primary_type": "background"' in stored


def test_claim_tasks_includes_analysis_profile():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE analysis_jobs (
            id INTEGER PRIMARY KEY,
            status TEXT,
            started_at TEXT,
            analysis_profile_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE file_tasks (
            id INTEGER PRIMARY KEY,
            analysis_job_id INTEGER,
            file_id INTEGER,
            file_path TEXT,
            parse_status TEXT,
            mc_status TEXT,
            mc_assigned_to INTEGER,
            mc_started_at TEXT,
            priority INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    profile_json = (
        '{"domain_id":"illustration","expected_types":["background","effect"],'
        '"primary_type":"background","source":"user"}'
    )
    conn.execute(
        "INSERT INTO analysis_jobs (id, status, analysis_profile_json) VALUES (1, 'active', ?)",
        (profile_json,),
    )
    conn.execute("""
        INSERT INTO file_tasks (
            id, analysis_job_id, file_id, file_path, parse_status,
            mc_status, priority, created_at
        ) VALUES (100, 1, 500, '/tmp/bg/a.psd', 'done', 'pending', 0, '2026-05-14')
    """)
    conn.commit()

    mgr = AnalysisJobManager.__new__(AnalysisJobManager)
    mgr.db = SimpleNamespace(conn=conn)

    tasks = mgr.claim_tasks("mc", worker_id=10, count=1)

    assert tasks[0]["analysis_profile"] == {
        "domain_id": "illustration",
        "expected_types": ["background", "effect"],
        "primary_type": "background",
        "source": "user",
    }


def test_classify_batch_uses_per_item_analysis_profiles():
    import sys
    import types

    sys.modules.setdefault(
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        ),
    )
    from backend.vision.analyzer import VisionAnalyzer

    analyzer = VisionAnalyzer.__new__(VisionAnalyzer)
    captured = {}

    def fake_generate_batch(_images, prompts, _system_prompt):
        captured["prompts"] = prompts
        return [
            '{"image_type":"background","confidence":"high"}',
            '{"image_type":"effect","confidence":"high"}',
        ]

    analyzer._generate_response_batch = fake_generate_batch

    analyzer.classify_batch(
        [object(), object()],
        analysis_profile=[
            {"expected_types": ["background"], "primary_type": "background"},
            {"expected_types": ["effect"], "primary_type": "effect"},
        ],
    )

    assert "Expected types: background" in captured["prompts"][0]
    assert "primary expected type: background" in captured["prompts"][0]
    assert "Expected types: effect" in captured["prompts"][1]
    assert "primary expected type: effect" in captured["prompts"][1]
