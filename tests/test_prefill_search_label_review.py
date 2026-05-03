from tools.prefill_search_label_review import parse_query, prefill_row


def test_parse_query_extracts_scope_and_terms():
    scope, terms = parse_query("홍콩사무실에서 소파과 창문 있는 이미지")

    assert scope == "홍콩사무실"
    assert terms == ["소파", "창문"]


def test_prefill_requires_exact_hash_scope():
    row = {
        "query_text": "#3에서 하늘과 밤 있는 이미지",
        "folder_path": "발송작품/작품/세일러문/#30/1",
        "relative_path": "발송작품/작품/세일러문/#30/1/226.psd",
        "mc_caption": "A starry night sky overlooks a cityscape.",
        "ai_tags": '["night", "sky", "stars"]',
    }

    result = prefill_row(row, reviewer_id="test")

    assert result["reviewer_relevance"] == 0
    assert "scope=#3:substring" in result["review_notes"]


def test_prefill_scores_exact_scope_and_partial_terms():
    row = {
        "query_text": "#3에서 창문과 주방 있는 이미지",
        "folder_path": "#3",
        "relative_path": "작품/#3/example.psd",
        "mc_caption": "A room with a red sofa and windows showing daylight.",
        "ai_tags": '["windows", "indoor"]',
    }

    result = prefill_row(row, reviewer_id="test")

    assert result["reviewer_relevance"] == 1
    assert "matched_terms=창문" in result["review_notes"]
    assert "missing_terms=주방" in result["review_notes"]


def test_prefill_scores_exact_scope_and_all_terms():
    row = {
        "query_text": "늪지대에서 숲과 fog 있는 이미지",
        "folder_path": "장소/늪지대",
        "relative_path": "장소/늪지대/nfb09_152.psd",
        "mc_caption": "A misty, eerie forest path shrouded in fog.",
        "ai_tags": '["forest", "fog", "trees"]',
    }

    result = prefill_row(row, reviewer_id="test")

    assert result["reviewer_relevance"] == 2
    assert "matched_terms=숲,fog" in result["review_notes"]
