import sqlite3

from backend.search.sqlite_search import (
    SqliteVectorSearch,
    _diagnostic_log_enabled,
    _extract_scope_hint_candidates,
    _path_has_scope_segments,
    _query_explicitly_requests_format,
    _relax_unmatched_scope,
)


class _DummyDB:
    def __init__(self, conn):
        self.conn = conn


class _DummyConfig:
    def __init__(self, value):
        self.value = value

    def get(self, dotted_key, default=None):
        if dotted_key == "search.diagnostic.enabled":
            return self.value
        return default


def _search_with_files(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            preview_only INTEGER DEFAULT 0,
            folder_path TEXT,
            file_path TEXT,
            image_type TEXT,
            format TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO files (id, preview_only, folder_path, file_path, image_type, format) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    search = object.__new__(SqliteVectorSearch)
    search.db = _DummyDB(conn)
    return search


def test_extract_scope_hint_from_korean_particle_query():
    assert _extract_scope_hint_candidates("만게츠의집에서 wooden floor 있는 이미지") == [
        "만게츠의집"
    ]


def test_extract_scope_hint_cleans_generic_suffixes():
    assert _extract_scope_hint_candidates("세일러문폴더 이미지 중에서 낮씬") == [
        "세일러문"
    ]


def test_extract_scope_hint_adds_path_variant_for_multi_segment_scope():
    assert _extract_scope_hint_candidates("마캬베리즈무 실내소품 중에서 어두운거") == [
        "마캬베리즈무 실내소품",
        "마캬베리즈무/실내소품",
    ]


def test_extract_scope_hint_ignores_unscoped_query():
    assert _extract_scope_hint_candidates("밤 도시 배경 이미지") == []


def test_path_scope_matching_uses_full_segments():
    assert _path_has_scope_segments("작품/#3/AT03_200.psd", "#3")
    assert not _path_has_scope_segments("작품/#30/226.psd", "#3")
    assert not _path_has_scope_segments("작품/#33/291.psd", "#3")
    assert _path_has_scope_segments("예비/크랑베르무/장소/홍콩사무실/grb.psd", "홍콩사무실")


def test_apply_plan_filter_prefers_exact_folder_segment_for_hash_scope():
    search = _search_with_files([
        (1, 0, "#3", "작품/#3/a.psd", "background", "PSD"),
        (2, 0, "발송작품/작품/세일러문/#30/1", "발송작품/작품/세일러문/#30/1/b.psd", "background", "PSD"),
        (3, 0, "발송작품/작품/세일러문/#33/1", "발송작품/작품/세일러문/#33/1/c.psd", "background", "PSD"),
    ])

    assert search._apply_plan_filter({"folder": "#3"}) == {1}
    ids, info = search._apply_plan_filter_with_info({"folder": "#3"})
    assert ids == {1}
    assert info["match_mode"] == "exact_segment"
    assert info["requested_folder"] == "#3"


def test_apply_plan_filter_falls_back_to_substring_when_no_exact_segment_exists():
    search = _search_with_files([
        (1, 0, "학교교실", "작품/학교교실/a.psd", "background", "PSD"),
        (2, 0, "도시", "작품/도시/b.psd", "background", "PSD"),
    ])

    assert search._apply_plan_filter({"folder": "학교"}) == {1}
    ids, info = search._apply_plan_filter_with_info({"folder": "학교"})
    assert ids == {1}
    assert info["match_mode"] == "substring"


def test_apply_plan_filter_keeps_code_like_substring_before_fuzzy_match():
    search = _search_with_files([
        (1, 0, "AGM_01_001_013_030 3DBG", "장소/AGM_01_001_013_030 3DBG/a.psd", "background", "PSD"),
        (2, 0, "bg", "작품/#01/bg/b.psd", "background", "PSD"),
    ])

    assert search._apply_plan_filter({"folder": "3DBG"}) == {1}
    ids, info = search._apply_plan_filter_with_info({"folder": "3DBG"})
    assert ids == {1}
    assert info["match_mode"] == "substring"


def test_scope_hint_can_recover_zero_padded_hash_scope_from_llm_normalization():
    search = _search_with_files([
        (1, 0, "#2", "작품/#2/a.psd", "background", "PSD"),
        (2, 0, "발송/#02/bg", "발송/#02/bg/b.psd", "background", "PSD"),
    ])

    folder, ids = search._scope_ids_from_query_hint(
        "#02에서 밤과 숲 있는 이미지",
        base_scope={"folder": "#2"},
        skip_folder="#2",
    )

    assert folder == "#02"
    assert ids == {2}


def test_diagnostic_log_enabled_uses_config_when_env_absent(monkeypatch):
    monkeypatch.delenv("SEARCH_DIAGNOSTIC", raising=False)

    assert _diagnostic_log_enabled(_DummyConfig(False)) is False
    assert _diagnostic_log_enabled(_DummyConfig(True)) is True


def test_diagnostic_log_enabled_env_overrides_config(monkeypatch):
    monkeypatch.setenv("SEARCH_DIAGNOSTIC", "0")
    assert _diagnostic_log_enabled(_DummyConfig(True)) is False

    monkeypatch.setenv("SEARCH_DIAGNOSTIC", "1")
    assert _diagnostic_log_enabled(_DummyConfig(False)) is True


def test_format_request_ignores_extension_inside_keyword():
    assert not _query_explicitly_requests_format(
        "로네느의집 실험실에서 occult과 thumb_33767.png 있는 이미지",
        "PNG",
    )


def test_format_request_detects_explicit_file_type():
    assert _query_explicitly_requests_format("로네느의집에서 PNG 파일 찾기", "PNG")


def test_relax_unmatched_scope_drops_non_explicit_format_only():
    relaxed, keys = _relax_unmatched_scope(
        {"folder": "로네느의집", "image_type": None, "format": "PNG"},
        "로네느의집 실험실에서 occult과 thumb_33767.png 있는 이미지",
    )

    assert relaxed == {"folder": "로네느의집", "image_type": None, "format": None}
    assert keys == {"format"}


def test_relax_unmatched_scope_keeps_explicit_format():
    relaxed, keys = _relax_unmatched_scope(
        {"folder": "로네느의집", "image_type": None, "format": "PNG"},
        "로네느의집에서 PNG 파일 찾기",
    )

    assert relaxed == {"folder": "로네느의집", "image_type": None, "format": "PNG"}
    assert keys == set()
