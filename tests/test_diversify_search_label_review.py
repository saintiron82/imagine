from tools.diversify_search_label_review import (
    diversify_rows,
    is_background_query,
    is_repair_required,
    query_intent_key,
)


def row(query_id, item_id, rank, folder, visual=None):
    return {
        "query_id": query_id,
        "query_text": "test",
        "item_id": item_id,
        "best_rank": rank,
        "folder_path": folder,
        "visual_group": visual or f"visual:{item_id}",
        "mc_caption": "A usable caption.",
        "ai_tags": '["usable"]',
    }


def qrow(query_id, item_id, rank, query_text):
    data = row(query_id, item_id, rank, query_id)
    data["query_text"] = query_text
    return data


def test_diversify_caps_source_before_relaxing_to_minimum():
    rows = [
        row("q1", "1", 1, "A"),
        row("q1", "2", 2, "A"),
        row("q1", "3", 3, "A"),
        row("q1", "4", 4, "B"),
        row("q1", "5", 5, "C"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=4,
        min_per_query=3,
        max_per_source=1,
        hamming_threshold=0,
    )

    assert [r["item_id"] for r in diversified] == ["1", "4", "5"]


def test_diversify_removes_visual_duplicates_within_query():
    rows = [
        row("q1", "1", 1, "A", visual="same"),
        row("q1", "2", 2, "B", visual="same"),
        row("q1", "3", 3, "C", visual="other"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=3,
        min_per_query=1,
        max_per_source=5,
        hamming_threshold=0,
    )

    assert [r["item_id"] for r in diversified] == ["1", "3"]


def test_diversify_caps_repeated_items_across_queries():
    rows = [
        row("q1", "same", 1, "A"),
        row("q1", "a", 2, "B"),
        row("q2", "same", 1, "C"),
        row("q2", "b", 2, "D"),
        row("q3", "same", 1, "E"),
        row("q3", "c", 2, "F"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=2,
        min_per_query=1,
        max_per_source=5,
        max_per_item=2,
        max_per_intent=0,
        hamming_threshold=0,
    )

    assert [r["item_id"] for r in diversified] == ["same", "a", "same", "b", "c"]


def test_diversify_defaults_to_one_review_row_per_item():
    rows = [
        row("q1", "same", 1, "A"),
        row("q2", "same", 1, "B"),
        row("q2", "other", 2, "C"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=2,
        min_per_query=1,
        max_per_source=5,
        max_per_intent=0,
        hamming_threshold=0,
    )

    assert [r["item_id"] for r in diversified] == ["same", "other"]


def test_query_intent_key_groups_same_terms_across_different_scopes():
    assert query_intent_key({"query_text": "#09에서 하늘과 캐릭터 있는 이미지"}) == "캐릭터|하늘"
    assert query_intent_key({"query_text": "#05에서 캐릭터와 하늘 있는 이미지"}) == "캐릭터|하늘"


def test_diversify_defaults_to_one_query_per_intent():
    rows = [
        qrow("q1", "a", 1, "#09에서 하늘과 캐릭터 있는 이미지"),
        qrow("q2", "b", 1, "#05에서 하늘과 캐릭터 있는 이미지"),
        qrow("q3", "c", 1, "#01에서 밤과 달 있는 이미지"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=1,
        min_per_query=1,
        max_per_source=5,
        hamming_threshold=0,
    )

    assert [(r["query_id"], r["item_id"]) for r in diversified] == [("q1", "a"), ("q3", "c")]


def test_background_query_filter_excludes_character_and_object_queries():
    assert is_background_query({"query_text": "#09에서 하늘과 캐릭터 있는 이미지"}) is False
    assert is_background_query({"query_text": "성에서 armor과 검 있는 이미지"}) is False
    assert is_background_query({"query_text": "#09에서 바다과 하늘 있는 이미지"}) is True


def test_background_query_filter_allows_negated_character_terms():
    assert is_background_query({"query_text": "성에서 달과 조명이 있고 캐릭터가 없는 밤 배경"}) is True
    assert is_background_query({"query_text": "교실에서 창문과 책상이 보이지만 인물은 없는 실내 배경"}) is True


def test_diversify_can_keep_background_queries_only():
    rows = [
        qrow("q1", "a", 1, "#09에서 하늘과 캐릭터 있는 이미지"),
        qrow("q2", "b", 1, "#09에서 바다과 하늘 있는 이미지"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=1,
        min_per_query=1,
        max_per_source=5,
        hamming_threshold=0,
        background_only=True,
    )

    assert [(r["query_id"], r["item_id"]) for r in diversified] == [("q2", "b")]


def test_diversify_rank_strata_interleaves_easy_and_hard_candidates():
    rows = [
        row("q1", "r1", 1, "A"),
        row("q1", "r2", 2, "B"),
        row("q1", "r6", 6, "C"),
        row("q1", "r11", 11, "D"),
        row("q1", "r21", 21, "E"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=4,
        min_per_query=1,
        max_per_source=5,
        max_per_intent=0,
        hamming_threshold=0,
        rank_strata=True,
    )

    assert [r["item_id"] for r in diversified] == ["r1", "r6", "r11", "r21"]


def test_diversify_excludes_repair_required_rows_by_default():
    rows = [
        {**row("q1", "repair", 1, "A"), "metadata_status": "repair_required"},
        row("q1", "ok", 2, "B"),
    ]

    diversified = diversify_rows(
        rows,
        target_per_query=2,
        min_per_query=1,
        max_per_source=5,
        hamming_threshold=0,
    )

    assert [r["item_id"] for r in diversified] == ["ok"]


def test_is_repair_required_falls_back_to_missing_caption_and_tags():
    assert is_repair_required({"mc_caption": "unknown", "ai_tags": "[]"}) is True
    assert is_repair_required({"mc_caption": "A room.", "ai_tags": "[]"}) is False
