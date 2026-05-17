import json
import sqlite3
import types

from backend.db.sqlite_client import SQLiteDB


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "CREATE TABLE files(id INTEGER PRIMARY KEY, file_path TEXT, structured_meta TEXT)"
    )
    db = object.__new__(SQLiteDB)
    db._local = types.SimpleNamespace(conn=conn)
    return db, conn


def test_replace_vlm_raw_outputs_persists_latest_raw_payload():
    db, conn = make_db()
    conn.execute("INSERT INTO files(id, structured_meta) VALUES (1, '{}')")

    db._ensure_vlm_raw_outputs_table()
    db._replace_vlm_raw_output(
        conn.cursor(),
        file_id=1,
        stage="stage2",
        adapter="MLXVisionAnalyzer",
        model="Qwen/Qwen3.5-9B",
        prompt_version="spatial_v2",
        raw_text='{"caption":"x"}',
        parse_status="direct",
    )
    conn.commit()

    row = conn.execute(
        "SELECT file_id, stage, model, raw_text, parse_status FROM vlm_raw_outputs"
    ).fetchone()
    assert dict(row) == {
        "file_id": 1,
        "stage": "stage2",
        "model": "Qwen/Qwen3.5-9B",
        "raw_text": '{"caption":"x"}',
        "parse_status": "direct",
    }


def test_update_vision_fields_moves_vlm_raw_out_of_structured_meta():
    db, conn = make_db()
    conn.execute(
        "INSERT INTO files(id, file_path, structured_meta) VALUES (1, '/a.png', '{}')"
    )

    updated = db.update_vision_fields(
        "/a.png",
        {
            "structured_meta": {
                "caption": "x",
                "objects": [],
                "_vlm_raw": '{"caption":"x"}',
                "_vlm_provenance": {
                    "stage": "stage2",
                    "adapter": "MLXVisionAdapter",
                    "model": "mlx-community/Qwen3.5-9B",
                    "prompt_version": "spatial_v2",
                },
                "_parse_diagnostics": {"status": "direct", "repaired": False},
            }
        },
    )

    assert updated is True
    structured_meta = json.loads(
        conn.execute("SELECT structured_meta FROM files WHERE id = 1").fetchone()[0]
    )
    assert "_vlm_raw" not in structured_meta
    assert "_vlm_provenance" not in structured_meta
    assert "_parse_diagnostics" not in structured_meta

    row = conn.execute(
        "SELECT stage, adapter, model, prompt_version, raw_text, parse_status "
        "FROM vlm_raw_outputs WHERE file_id = 1"
    ).fetchone()
    assert dict(row) == {
        "stage": "stage2",
        "adapter": "MLXVisionAdapter",
        "model": "mlx-community/Qwen3.5-9B",
        "prompt_version": "spatial_v2",
        "raw_text": '{"caption":"x"}',
        "parse_status": "direct",
    }
