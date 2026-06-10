"""Object-evidence recall guard — files whose structured objects satisfy
ALL condition groups must be findable even when no search axis surfaced
them (s19 diagnosis: object_evidence_present_but_not_top10)."""

import sqlite3
from types import SimpleNamespace

from backend.search.sqlite_search import SqliteVectorSearch


def _db_with_objects():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY, file_path TEXT, file_name TEXT,
            format TEXT, width INTEGER, height INTEGER, mc_caption TEXT,
            ai_tags TEXT, ocr_text TEXT, metadata TEXT, thumbnail_url TEXT,
            user_note TEXT, user_tags TEXT, user_category TEXT,
            user_rating INTEGER, folder_path TEXT, folder_depth INTEGER,
            folder_tags TEXT, storage_root TEXT, relative_path TEXT,
            image_type TEXT, art_style TEXT, color_palette TEXT,
            scene_type TEXT, time_of_day TEXT, weather TEXT,
            character_type TEXT, item_type TEXT, ui_type TEXT,
            structured_meta TEXT, preview_only INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE file_objects (
            id INTEGER PRIMARY KEY, file_id INTEGER, name TEXT, ko_name TEXT,
            primary_location TEXT, locations TEXT, extent TEXT,
            confidence TEXT, spatial_text TEXT
        )
    """)
    conn.execute("CREATE TABLE file_spatial_relations (id INTEGER PRIMARY KEY, file_id INTEGER, subject TEXT, relation TEXT, object TEXT, subject_location TEXT, object_location TEXT, confidence TEXT, spatial_text TEXT)")
    conn.execute("CREATE TABLE file_depth_layers (id INTEGER PRIMARY KEY, file_id INTEGER, name TEXT, layer TEXT, confidence TEXT)")

    # file 1: wall + curtain (full match)
    # file 2: wall only
    # file 3: curtain + window (partial for wall+curtain)
    # file 4: wall + curtain but preview_only
    for fid in (1, 2, 3, 4):
        conn.execute(
            "INSERT INTO files (id, file_path, file_name, preview_only) VALUES (?, ?, ?, ?)",
            (fid, f"/tmp/{fid}.psd", f"{fid}.psd", 1 if fid == 4 else 0),
        )
    objs = [
        (1, "wall", "벽"), (1, "curtain", "커튼"),
        (2, "wall", "벽"),
        (3, "curtain", "커튼"), (3, "window", "창문"),
        (4, "wall", "벽"), (4, "curtain", "커튼"),
    ]
    for fid, name, ko in objs:
        conn.execute(
            "INSERT INTO file_objects (file_id, name, ko_name, locations) VALUES (?, ?, ?, '[]')",
            (fid, name, ko),
        )
    conn.commit()
    return SimpleNamespace(conn=conn)


def _searcher():
    s = SqliteVectorSearch.__new__(SqliteVectorSearch)
    s.db = _db_with_objects()
    return s


def test_guard_returns_only_full_condition_matches():
    s = _searcher()
    ids = s._object_evidence_guard_ids(["벽|wall", "커튼|curtain"])
    assert 1 in ids
    assert 2 not in ids  # wall only
    assert 3 not in ids  # curtain only (no wall)


def test_guard_requires_at_least_two_groups():
    s = _searcher()
    assert s._object_evidence_guard_ids(["벽|wall"]) == []


def test_guard_returns_empty_when_any_group_unmatched():
    s = _searcher()
    assert s._object_evidence_guard_ids(["벽|wall", "용|dragon"]) == []


def test_guard_respects_limit():
    s = _searcher()
    ids = s._object_evidence_guard_ids(["벽|wall", "커튼|curtain"], limit=0)
    assert ids == []


def test_rows_for_file_ids_excludes_preview_only():
    s = _searcher()
    rows = s._rows_for_file_ids([1, 4])
    assert [r["id"] for r in rows] == [1]
    # Rows must carry structured object evidence for the evidence rerank
    assert rows[0]["spatial_objects"][0]["ko_name"] in ("벽", "커튼")
