from backend.search.metadata_quality import (
    annotate_metadata_quality,
    metadata_quality_for_result,
)
from backend.search.scoring import quality_rerank


def _profile_bundle():
    return {
        "profile": {
            "global": {"metadata_reliability": 0.8},
            "analysis_status": {
                "ok": {"metadata_reliability": 0.9, "reviewed_count": 20},
            },
            "tags": {
                "monster": {
                    "tag_reliability": 0.2,
                    "metadata_reliability": 0.2,
                    "reviewed_count": 3,
                },
                "room": {
                    "tag_reliability": 1.0,
                    "metadata_reliability": 1.0,
                    "reviewed_count": 3,
                },
            },
            "limits": {
                "runtime_status_min_reviewed": 10,
                "runtime_tag_min_reviewed": 3,
            },
        },
        "signals_by_item_id": {
            "1": {
                "item_id": "1",
                "caption_reliability": 0.5,
                "tag_reliability": 0.0,
                "metadata_reliability_score": 0.25,
                "issue_types": ["caption_wrong", "tag_false_positive"],
            }
        },
    }


def test_metadata_quality_prefers_reviewed_item_signal():
    signal = metadata_quality_for_result(
        {"id": 1, "analysis_status": "ok", "ai_tags": ["room"]},
        bundle=_profile_bundle(),
    )

    assert signal["metadata_reliability_score"] == 0.25
    assert signal["metadata_quality_source"] == "item_review"
    assert signal["caption_reliability"] == 0.5
    assert "tag_false_positive" in signal["metadata_quality_issues"]


def test_metadata_quality_uses_status_and_tag_profile_for_unreviewed_items():
    signal = metadata_quality_for_result(
        {"id": 2, "analysis_status": "ok", "ai_tags": ["monster"]},
        bundle=_profile_bundle(),
    )

    assert signal["metadata_quality_source"] == "profile_inferred"
    assert signal["metadata_reliability_score"] == 0.55
    assert signal["tag_reliability"] == 0.2


def test_annotate_metadata_quality_mutates_search_results():
    results = [{"id": 1, "ai_tags": []}, {"id": 2, "analysis_status": "ok", "ai_tags": ["room"]}]

    annotate_metadata_quality(results, bundle=_profile_bundle())

    assert results[0]["metadata_quality_source"] == "item_review"
    assert results[1]["metadata_reliability_score"] == 0.95


def test_quality_rerank_keeps_metadata_quality_shadow_by_default():
    low_reliability_first = {
        "id": 1,
        "file_path": "/a.png",
        "file_name": "a.png",
        "rrf_score": 0.03,
        "vector_score": 1.0,
        "text_vec_score": 1.0,
        "text_score": 1.0,
        "mc_caption": "room",
        "ai_tags": ["room"],
        "metadata_reliability_score": 0.0,
    }
    high_reliability_second = {
        "id": 2,
        "file_path": "/b.png",
        "file_name": "b.png",
        "rrf_score": 0.02,
        "vector_score": 1.0,
        "text_vec_score": 1.0,
        "text_score": 1.0,
        "mc_caption": "room",
        "ai_tags": ["room"],
        "metadata_reliability_score": 1.0,
    }

    shadow = quality_rerank(
        [low_reliability_first.copy(), high_reliability_second.copy()],
        top_k=2,
        query="room",
        pool_size=2,
    )
    weighted = quality_rerank(
        [low_reliability_first.copy(), high_reliability_second.copy()],
        top_k=2,
        query="room",
        pool_size=2,
        metadata_quality_weight=0.2,
    )

    assert [row["id"] for row in shadow] == [1, 2]
    assert [row["id"] for row in weighted] == [2, 1]
    assert weighted[0]["metadata_quality_adjustment"] > 0
