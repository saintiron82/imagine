import json
import sqlite3
from pathlib import Path

from tools.audit_queryset_scopes import (
    assess_scope,
    path_has_scope_segments,
    query_mentions_scope,
)


def make_db(path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            relative_path TEXT,
            folder_path TEXT,
            image_type TEXT,
            mc_caption TEXT,
            ai_tags TEXT,
            caption_model TEXT,
            processing_status TEXT,
            processing_error TEXT,
            preview_only INTEGER DEFAULT 0
        )
    """)
    rows = [
        (1, "/root/project/#08/bg/a.psd", "project/#08/bg/a.psd", "project/#08/bg", "background", "Moon sky", json.dumps(["moon"]), "model", "", "", 0),
        (2, "/root/project/#080/bg/b.psd", "project/#080/bg/b.psd", "project/#080/bg", "background", "Other", json.dumps(["other"]), "model", "", "", 0),
        (3, "/root/other/실내소품/c.psd", "other/실내소품/c.psd", "other/실내소품", "item", "Chair", json.dumps(["chair"]), "model", "", "", 0),
    ]
    conn.executemany(
        "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    files = conn.execute("SELECT * FROM files").fetchall()
    conn.close()
    return files


def test_path_has_scope_segments_uses_exact_segment_sequence():
    assert path_has_scope_segments("project/#08/bg/a.psd", "project/#08/bg")
    assert not path_has_scope_segments("project/#080/bg/a.psd", "project/#08/bg")


def test_query_mentions_scope_ignores_generic_scope_words():
    assert query_mentions_scope("기절용사와 암살공주 #08에서 달이 있는 배경", "기절용사와 암살공주/#08/bg")
    assert not query_mentions_scope("달이 있는 배경", "작품/bg")


def test_assess_scope_flags_broad_scope(tmp_path: Path):
    files = make_db(tmp_path / "scope.db")

    row = assess_scope(
        {
            "query_id": "q1",
            "query_text": "실내소품에서 의자가 있는 배경",
            "scope": "실내소품",
        },
        files,
        broad_threshold=0,
    )

    assert row["file_count"] == 1
    assert row["scope_assessment"] == "review"
    assert "broad_scope" in row["scope_flags"]
    assert row["data_assessment"] == "review"
    assert "tiny_scope" in row["data_flags"]
