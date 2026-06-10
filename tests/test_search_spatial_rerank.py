import sqlite3
import types

import numpy as np

from backend.search.rrf import get_weights
from backend.search.scoring import quality_rerank, rrf_merge_multi
from backend.search import sqlite_search
from backend.search.sqlite_search import (
    SqliteVectorSearch,
    apply_evidence_matrix_rerank,
    apply_element_verification,
    _build_element_verification_groups,
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


def test_extract_spatial_intent_ignores_contradictory_decomposer_location_keywords():
    intent = _extract_spatial_intent(
        "왼쪽 위에 텍스트 오버레이가 있는 이미지",
        ["오른쪽", "위", "텍스트", "오버레이", "text", "overlay", "top left"],
    )

    assert "top-left" in intent["locations"]
    assert "right" not in intent["locations"]


def test_sanitize_spatial_fts_keywords_removes_contradictory_locations():
    sanitize = getattr(sqlite_search, "_sanitize_spatial_fts_keywords", None)
    assert sanitize is not None

    keywords = sanitize(
        "왼쪽 위에 텍스트 오버레이가 있는 이미지",
        ["오른쪽", "위", "텍스트", "오버레이", "text", "overlay", "top left"],
    )

    assert "오른쪽" not in keywords
    assert "텍스트" in keywords
    assert "top left" in keywords


def test_spatial_evidence_search_returns_exact_relation_before_caption_only():
    searcher = _make_spatial_searcher()
    intent = _extract_spatial_intent("cup on table")

    results = searcher._spatial_evidence_search(intent, top_k=10)

    assert [row["id"] for row in results] == [1]
    assert results[0]["spatial_score"] > 0
    assert results[0]["spatial_matches"][0]["table"] == "file_spatial_relations"


def test_spatial_evidence_search_ranks_primary_location_above_secondary_location():
    searcher = _make_spatial_searcher()
    conn = searcher.db.conn
    conn.executemany(
        """INSERT INTO files
           (id, file_path, file_name, format, mc_caption, ai_tags, metadata,
            folder_path, relative_path, preview_only)
           VALUES (?, ?, ?, 'PNG', ?, '[]', '{}', '', ?, 0)""",
        [
            (3, "/primary-right-wall.png", "primary-right-wall.png", "A wall on the right.", "primary-right-wall.png"),
            (4, "/secondary-right-wall.png", "secondary-right-wall.png", "A wide wall also spans right.", "secondary-right-wall.png"),
        ],
    )
    conn.executemany(
        """INSERT INTO file_objects
           (file_id, name, ko_name, primary_location, locations, extent, confidence, spatial_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (3, "wall", "벽", "right", '["right"]', "medium", "high", "wall right 벽 오른쪽"),
            (4, "wall", "벽", "top", '["top", "right"]', "wide", "high", "wall top right 벽 위 오른쪽"),
        ],
    )
    conn.commit()

    intent = _extract_spatial_intent("오른쪽 벽")
    results = searcher._spatial_evidence_search(intent, top_k=10)

    assert [row["id"] for row in results[:2]] == [3, 4]
    assert results[0]["spatial_score"] > results[1]["spatial_score"]
    assert results[0]["spatial_matches"][0]["match_strength"] == "primary"
    assert results[1]["spatial_matches"][0]["match_strength"] == "secondary"


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


def test_triaxis_search_adds_metadata_quality_shadow_signal(monkeypatch):
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

    def fake_annotate(results):
        for row in results:
            row["metadata_reliability_score"] = 0.25
            row["metadata_quality_source"] = "test_profile"

    searcher.fts_search = types.MethodType(fake_fts_search, searcher)
    monkeypatch.setattr("backend.search.sqlite_search.annotate_metadata_quality", fake_annotate)

    results, diag = searcher.triaxis_search(
        "cup on table",
        top_k=2,
        return_diagnostic=True,
        use_codex=False,
    )

    assert results[0]["metadata_quality_source"] == "test_profile"
    assert results[0]["metadata_reliability_score"] == 0.25
    assert diag["rerank"]["metadata_quality_weight"] == 0.0


def test_triaxis_search_attaches_evidence_matrix_before_trim(monkeypatch):
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

    assert results[0]["evidence_matrix"]["conditions"]["matched"] >= 1
    assert results[0]["evidence_score"] > 0
    assert diag["evidence_matrix"]["stage"] == "pre_trim"


def test_triaxis_search_reapplies_evidence_after_cross_encoder(monkeypatch):
    searcher = _make_spatial_searcher()
    conn = searcher.db.conn
    conn.executemany(
        """INSERT INTO files
           (id, file_path, file_name, format, mc_caption, ai_tags, metadata,
            folder_path, relative_path, preview_only)
           VALUES (?, ?, ?, 'PNG', ?, ?, '{}', '', ?, 0)""",
        [
            (
                5, "/caption-bottle-window.png", "caption-bottle-window.png",
                "A room with a bottle near a window.",
                '["bottle", "window"]',
                "caption-bottle-window.png",
            ),
            (
                6, "/object-bottle-window.png", "object-bottle-window.png",
                "A shop shelf.",
                '["shelf"]',
                "object-bottle-window.png",
            ),
        ],
    )
    conn.executemany(
        """INSERT INTO file_objects
           (file_id, name, ko_name, primary_location, locations, extent, confidence, spatial_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (5, "bottle", "병", "left", '["left"]', "medium", "high", "bottle 병"),
            (6, "bottle", "병", "left", '["left"]', "medium", "high", "bottle 병"),
            (6, "window", "창문", "right", '["right"]', "large", "high", "window 창문"),
        ],
    )
    conn.commit()

    class FakeDecomposer:
        def __init__(self, use_codex=True):
            pass

        def decompose(self, query):
            return {
                "_decomp_backend": "test",
                "find": {
                    "description": "bottle and window",
                    "keywords": ["병", "창문", "bottle", "window"],
                },
                "_legacy": {"query_type": "balanced"},
            }

    class FakeCrossEncoder:
        def score_pairs(self, pairs):
            return [1.0 if "near a window" in doc else 0.1 for _, doc in pairs]

    def fake_fts_search(self, keywords, top_k, exclude_keywords=None, file_ids=None):
        rows = []
        for file_id in [5, 6]:
            row = dict(self.db.conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone())
            self._parse_json_fields(row)
            row["fts_rank"] = -1.0
            row["text_score"] = 1.0
            rows.append(row)
        return rows

    monkeypatch.setenv("SEARCH_DIAGNOSTIC", "0")
    monkeypatch.setattr(sqlite_search, "QueryDecomposer", FakeDecomposer)
    monkeypatch.setattr("backend.search.cross_encoder.load_default_reranker", lambda: FakeCrossEncoder())
    monkeypatch.setattr(searcher, "encode_text", lambda query: np.array([1.0], dtype=np.float32))
    monkeypatch.setattr(searcher, "vector_search_by_embedding", lambda *args, **kwargs: [])
    monkeypatch.setattr(searcher, "_batch_similarity", lambda *args, **kwargs: {})
    monkeypatch.setattr(searcher, "_batch_fts_score", lambda *args, **kwargs: {})
    searcher.fts_search = types.MethodType(fake_fts_search, searcher)

    results, diag = searcher.triaxis_search(
        "병, 창문가 함께 있는 이미지",
        top_k=2,
        return_diagnostic=True,
        use_codex=False,
    )

    assert results[0]["id"] == 6
    assert results[0]["evidence_matrix"]["conditions"]["object_matched"] == 2
    assert diag["evidence_matrix"]["post_cross_encoder"] is True


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


def test_element_verification_promotes_full_multi_condition_evidence():
    partial = {
        "id": 1,
        "rrf_score": 1.0,
        "quality_score": 1.0,
        "mc_caption": "snowy mountain landscape",
        "ai_tags": ["snowy mountain", "landscape"],
        "spatial_objects": [
            {"name": "mountain", "ko_name": "산"},
        ],
    }
    full = {
        "id": 2,
        "rrf_score": 0.8,
        "quality_score": 0.8,
        "mc_caption": "snowy mountain with a bird flying in the sky",
        "ai_tags": ["snowy mountain", "bird", "flying", "sky"],
        "spatial_objects": [
            {"name": "mountain", "ko_name": "설산"},
            {"name": "bird", "ko_name": "새"},
        ],
    }

    rows = apply_element_verification(
        [partial, full],
        elements=["설산|snowy mountain", "새|bird", "날고|flying"],
        penalty=0.15,
    )

    assert rows[0]["id"] == 2
    assert rows[0]["element_match_count"] == 3
    assert rows[0]["element_match_ratio"] == 1.0
    assert rows[0]["element_missing"] == []
    assert rows[1]["id"] == 1
    assert rows[1]["element_missing"] == ["새|bird", "날고|flying"]


def test_evidence_matrix_rerank_uses_metadata_and_spatial_evidence():
    partial = {
        "id": 1,
        "rrf_score": 1.0,
        "quality_score": 1.0,
        "text_score": 1.0,
        "mc_caption": "snowy mountain landscape",
        "ai_tags": ["snowy mountain", "landscape"],
    }
    full = {
        "id": 2,
        "rrf_score": 0.8,
        "quality_score": 0.8,
        "text_score": 0.5,
        "text_vec_score": 0.7,
        "spatial_score": 0.9,
        "mc_caption": "snowy mountain landscape",
        "ai_tags": ["bird"],
        "metadata": {"semantic_tags": ["flying", "sky"]},
        "spatial_matches": [
            {"name": "bird", "ko_name": "새", "spatial_text": "bird flying 새 날고"}
        ],
    }

    rows = apply_evidence_matrix_rerank(
        [partial, full],
        elements=["설산|snowy mountain", "새|bird", "날고|flying"],
        boost=0.30,
    )

    assert rows[0]["id"] == 2
    assert rows[0]["evidence_score"] > rows[1]["evidence_score"]
    assert rows[0]["evidence_matrix"]["conditions"]["match_ratio"] == 1.0
    assert rows[0]["evidence_matrix"]["conditions"]["matches"]["새|bird"]
    assert rows[0]["evidence_matrix"]["axes"]["spatial"]["present"] is True
    assert rows[0]["evidence_matrix"]["axes"]["fts"]["present"] is True
    assert rows[1]["evidence_matrix"]["conditions"]["missing"] == ["새|bird", "날고|flying"]


def test_evidence_matrix_rerank_prioritizes_structured_object_cooccurrence():
    caption_only = {
        "id": 1,
        "rrf_score": 1.0,
        "quality_score": 1.0,
        "mc_caption": "A room with a bottle near a window.",
        "ai_tags": ["bottle", "window"],
        "spatial_objects": [
            {"name": "bottle", "ko_name": "병"},
        ],
    }
    object_cooccurrence = {
        "id": 2,
        "rrf_score": 0.8,
        "quality_score": 0.8,
        "mc_caption": "A shop shelf.",
        "ai_tags": ["shelf"],
        "spatial_objects": [
            {"name": "bottle", "ko_name": "병"},
            {"name": "window", "ko_name": "창문"},
        ],
    }

    rows = apply_evidence_matrix_rerank(
        [caption_only, object_cooccurrence],
        elements=["병|bottle", "창문|window"],
        boost=0.30,
    )

    assert rows[0]["id"] == 2
    assert rows[0]["evidence_matrix"]["conditions"]["object_matched"] == 2
    assert rows[1]["evidence_matrix"]["conditions"]["object_missing"] == ["창문|window"]


def test_evidence_matrix_rerank_promotes_object_evidence_even_when_caption_is_weak():
    caption_strong = {
        "id": 1,
        "rrf_score": 1.0,
        "quality_score": 1.0,
        "text_score": 1.0,
        "mc_caption": "A room with walls and curtains.",
        "ai_tags": ["wall", "curtain"],
        "spatial_objects": [
            {"name": "wall", "ko_name": "벽"},
        ],
    }
    object_strong = {
        "id": 2,
        "rrf_score": 0.75,
        "quality_score": 0.75,
        "text_score": 0.2,
        "mc_caption": "A dark room interior.",
        "ai_tags": ["dark", "interior"],
        "spatial_objects": [
            {"name": "wall", "ko_name": "벽", "confidence": "high"},
            {"name": "curtain", "ko_name": "커튼", "confidence": "medium"},
        ],
    }

    rows = apply_evidence_matrix_rerank(
        [caption_strong, object_strong],
        elements=["벽|wall", "커튼|curtain"],
        boost=0.35,
    )

    assert rows[0]["id"] == 2
    assert rows[0]["evidence_matrix"]["conditions"]["object_matched"] == 2
    assert rows[0]["evidence_matrix"]["conditions"]["object_missing"] == []


def test_build_element_verification_groups_keeps_three_scene_conditions():
    groups = _build_element_verification_groups([
        "설산", "새", "날고", "이미지",
        "snowy mountain", "bird", "flying", "image",
    ])

    assert groups == ["설산|snowy mountain", "새|bird", "날고|flying"]


def test_build_element_verification_groups_drops_query_glue_words():
    groups = _build_element_verification_groups([
        "벽", "와", "커튼", "가", "함께", "모두", "보이는", "이미지",
    ])

    assert groups == ["벽", "커튼"]


def test_build_element_verification_groups_pairs_after_dropping_glue_words():
    groups = _build_element_verification_groups([
        "벽", "와", "커튼", "가", "함께", "있는",
        "image", "wall", "curtain", "together", "with",
    ])

    assert groups == ["벽|wall", "커튼|curtain"]


def test_build_element_verification_groups_pairs_known_synonyms_when_english_order_differs():
    groups = _build_element_verification_groups([
        "벽", "커튼", "curtain", "wall",
    ])

    assert groups == ["벽|wall", "커튼|curtain"]


def test_build_element_verification_groups_drops_phrase_noise_from_decomposer():
    groups = _build_element_verification_groups([
        "병", "창문", "병과 창문", "병, 창문가 함께 있는 이미지",
        "bottle", "window", "window side", "window and bottle",
    ])

    assert groups == ["병|bottle", "창문|window"]
