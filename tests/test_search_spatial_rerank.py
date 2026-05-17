import sqlite3
import types

import numpy as np

from backend.search.rrf import get_weights
from backend.search.scoring import quality_rerank, rrf_merge_multi
from backend.search.sqlite_search import (
    SqliteVectorSearch,
    _extract_spatial_intent,
)


class _DummyDB:
    def __init__(self, conn):
        self.conn = conn


class _DummyEncoders:
    text_search_enabled = False


def _make_spatial_searcher():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            format TEXT,
            width INTEGER,
            height INTEGER,
            mc_caption TEXT,
            ai_tags TEXT,
            ocr_text TEXT,
            metadata TEXT,
            thumbnail_url TEXT,
            user_note TEXT,
            user_tags TEXT,
            user_category TEXT,
            user_rating INTEGER,
            folder_path TEXT,
            folder_depth INTEGER,
            folder_tags TEXT,
            storage_root TEXT,
            relative_path TEXT,
            image_type TEXT,
            art_style TEXT,
            color_palette TEXT,
            scene_type TEXT,
            time_of_day TEXT,
            weather TEXT,
            character_type TEXT,
            item_type TEXT,
            ui_type TEXT,
            structured_meta TEXT,
            preview_only INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE file_objects (
            id INTEGER PRIMARY KEY,
            file_id INTEGER,
            name TEXT,
            ko_name TEXT,
            primary_location TEXT,
            locations TEXT,
            extent TEXT,
            confidence TEXT,
            spatial_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE file_spatial_relations (
            id INTEGER PRIMARY KEY,
            file_id INTEGER,
            subject TEXT,
            relation TEXT,
            object TEXT,
            subject_location TEXT,
            object_location TEXT,
            confidence TEXT,
            spatial_text TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE file_depth_layers (
            id INTEGER PRIMARY KEY,
            file_id INTEGER,
            name TEXT,
            ko_name TEXT,
            layer TEXT,
            confidence TEXT,
            spatial_text TEXT
        )
    """)
    conn.executemany(
        """INSERT INTO files
           (id, file_path, file_name, format, mc_caption, ai_tags, metadata,
            folder_path, relative_path, preview_only)
           VALUES (?, ?, ?, 'PNG', ?, '[]', '{}', '', ?, 0)""",
        [
            (1, "/real.png", "real.png", "A cup sits on a foreground table.", "real.png"),
            (2, "/caption-only.png", "caption-only.png", "A cup and a table are visible.", "caption-only.png"),
        ],
    )
    conn.execute("""
        INSERT INTO file_spatial_relations
            (file_id, subject, relation, object, subject_location, object_location, confidence, spatial_text)
        VALUES
            (1, 'cup', 'on', 'table', 'center', 'bottom', 'high',
             'cup on table cup table 위 컵 테이블 위')
    """)
    conn.execute("""
        INSERT INTO file_depth_layers
            (file_id, name, ko_name, layer, confidence, spatial_text)
        VALUES
            (1, 'table', '테이블', 'foreground', 'high', 'table foreground 테이블 전경')
    """)
    conn.commit()

    searcher = object.__new__(SqliteVectorSearch)
    searcher.db = _DummyDB(conn)
    searcher.encoders = _DummyEncoders()
    return searcher


def test_extract_spatial_intent_detects_relation_depth_and_location():
    relation = _extract_spatial_intent("테이블 위 컵")
    assert relation["active"] is True
    assert "on" in relation["relations"]
    assert {"테이블", "컵"}.issubset(set(relation["terms"]))

    depth = _extract_spatial_intent("전경 테이블")
    assert depth["depth_layers"] == ["foreground"]
    assert "테이블" in depth["terms"]

    location = _extract_spatial_intent("오른쪽 달")
    assert location["locations"] == ["right"]
    assert "달" in location["terms"]


def test_extract_spatial_intent_strips_korean_particles_for_gold_queries():
    intent = _extract_spatial_intent("컵이 테이블 위에 있는 이미지", ["컵", "테이블", "위"])

    assert "컵" in intent["terms"]
    assert "컵이" not in intent["terms"]
    assert "테이블" in intent["terms"]


def test_extract_spatial_intent_does_not_treat_generic_background_as_depth():
    intent = _extract_spatial_intent("밤 도시 배경")

    assert intent["active"] is False
    assert intent["depth_layers"] == []


def test_spatial_evidence_search_returns_exact_relation_before_caption_only():
    searcher = _make_spatial_searcher()
    intent = _extract_spatial_intent("cup on table")

    results = searcher._spatial_evidence_search(intent, top_k=10)

    assert [row["id"] for row in results] == [1]
    assert results[0]["spatial_score"] > 0
    assert results[0]["spatial_matches"][0]["table"] == "file_spatial_relations"


def test_triaxis_search_merges_spatial_axis_ahead_of_caption_only(monkeypatch):
    searcher = _make_spatial_searcher()
    monkeypatch.setenv("SEARCH_DIAGNOSTIC", "0")
    monkeypatch.setattr(searcher, "encode_text", lambda query: np.array([1.0], dtype=np.float32))
    monkeypatch.setattr(searcher, "vector_search_by_embedding", lambda *args, **kwargs: [])
    monkeypatch.setattr(searcher, "_batch_similarity", lambda *args, **kwargs: {})
    monkeypatch.setattr(searcher, "_batch_fts_score", lambda *args, **kwargs: {})

    def fake_fts_search(self, keywords, top_k, exclude_keywords=None, file_ids=None):
        row = dict(self.db.conn.execute("SELECT * FROM files WHERE id = 2").fetchone())
        self._parse_json_fields(row)
        row["fts_rank"] = -1.0
        row["text_score"] = 1.0
        return [row]

    searcher.fts_search = types.MethodType(fake_fts_search, searcher)

    results, diag = searcher.triaxis_search(
        "cup on table",
        top_k=2,
        return_diagnostic=True,
        use_codex=False,
    )

    assert [row["id"] for row in results] == [1, 2]
    assert results[0]["spatial_score"] > 0
    assert diag["spatial_results"]["count"] == 1


def test_reranker_and_rrf_preserve_spatial_score_as_first_class_axis():
    spatial_hit = {
        "id": 1,
        "file_path": "/spatial.png",
        "file_name": "spatial.png",
        "spatial_score": 1.0,
        "mc_caption": "cup on table",
    }
    fts_only = {
        "id": 2,
        "file_path": "/fts.png",
        "file_name": "fts.png",
        "fts_rank": -1.0,
        "text_score": 1.0,
        "mc_caption": "cup and table",
    }

    merged = rrf_merge_multi(
        [("fts", [fts_only]), ("spatial", [spatial_hit])],
        k=60,
        weights={"fts": 0.2, "spatial": 0.8},
    )
    reranked = quality_rerank(
        merged,
        top_k=2,
        query="cup on table",
        axis_weights={"fts": 0.2, "spatial": 0.8},
        pool_size=2,
    )

    assert merged[0]["spatial_score"] == 1.0
    assert reranked[0]["id"] == 1
    assert reranked[0]["axes_present"] >= 1


def test_spatial_query_weights_prioritize_spatial_axis():
    weights = get_weights("spatial", ["fts", "text_vec", "spatial"])

    assert weights["spatial"] > weights["fts"]
    assert weights["spatial"] > weights["text_vec"]
