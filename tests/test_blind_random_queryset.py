import json
import sqlite3
from pathlib import Path

from tools.build_blind_random_queryset import (
    ScopeCandidate,
    build_rows,
    load_scope_candidates,
    main,
)
from tools.evaluate_search_quality import load_queries


def test_blind_random_queryset_is_deterministic():
    scopes = [ScopeCandidate("작품/#01/bg", 20), ScopeCandidate("작품/#02/bg", 30)]

    first = build_rows(count=8, seed=20260507, scopes=scopes, scope_ratio=0.5)
    second = build_rows(count=8, seed=20260507, scopes=scopes, scope_ratio=0.5)

    assert first == second
    assert all(row["generation"] == "blind_random_v1" for row in first)
    assert all(row["query_type"] == "blind_random" for row in first)


def test_blind_random_queryset_can_generate_without_scopes():
    rows = build_rows(count=5, seed=1, scopes=[], scope_ratio=1.0)

    assert len(rows) == 5
    assert all(row["scope"] == "" for row in rows)
    assert all(len(row["must_terms"]) >= 2 for row in rows)


def test_load_scope_candidates_uses_only_folder_counts(tmp_path: Path):
    db_path = tmp_path / "files.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            folder_path TEXT,
            preview_only INTEGER DEFAULT 0,
            mc_caption TEXT,
            ai_tags TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO files (folder_path, preview_only, mc_caption, ai_tags) VALUES (?, ?, ?, ?)",
        [
            ("작품/#01/bg", 0, "caption-a", "[\"tag\"]"),
            ("작품/#01/bg", 0, "caption-b", "[\"tag\"]"),
            ("bg", 0, "caption-c", "[\"tag\"]"),
            ("작품/#02/bg", 1, "caption-d", "[\"tag\"]"),
        ],
    )
    conn.commit()
    conn.close()

    scopes = load_scope_candidates(db_path, min_files=2, max_files=10)

    assert scopes == [ScopeCandidate("작품/#01/bg", 2)]


def test_blind_random_queryset_cli_writes_evaluator_compatible_artifacts(tmp_path: Path):
    output_dir = tmp_path / "blind_random"

    assert main(["--output-dir", str(output_dir), "--count", "6", "--scope-ratio", "0"]) == 0

    queryset_path = output_dir / "queryset.jsonl"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["query_count"] == 6
    assert manifest["query_type"] == "blind_random"
    assert len(load_queries(queryset_path)) == 6
