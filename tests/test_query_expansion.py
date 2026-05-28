"""Sprint 3 S3.3: text-embedding query expansion."""
from __future__ import annotations

from backend.search.sqlite_search import expand_query_for_text_embedding


def test_no_keywords_returns_original():
    assert expand_query_for_text_embedding("balcony at night", []) == "balcony at night"
    assert expand_query_for_text_embedding("balcony at night", None) == "balcony at night"


def test_empty_query_returns_empty():
    assert expand_query_for_text_embedding("", ["x"]) == ""
    assert expand_query_for_text_embedding(None, ["x"]) == ""


def test_appends_distinct_synonyms():
    out = expand_query_for_text_embedding(
        "balcony at night",
        ["발코니", "밤", "night", "balcony", "BG"],
        max_extras=4,
    )
    # "night" and "balcony" are already in the base query (case-insensitive
    # word match), so they're skipped. We get the remaining 3 distinct
    # additions.
    assert "발코니" in out
    assert "밤" in out
    assert "BG" in out
    # original query preserved
    assert out.startswith("balcony at night")


def test_respects_max_extras_cap():
    out = expand_query_for_text_embedding(
        "x",
        ["a", "b", "c", "d", "e", "f"],
        max_extras=3,
    )
    # appended count = 3
    appended = out[len("x "):].split()
    assert len(appended) == 3
    assert appended == ["a", "b", "c"]


def test_skips_non_string_keywords():
    out = expand_query_for_text_embedding(
        "x",
        ["good", None, 123, "  ", "also-good"],
        max_extras=4,
    )
    assert "good" in out
    assert "also-good" in out
    assert "123" not in out
    assert "None" not in out


def test_case_insensitive_dedup():
    """If the base query already has 'balcony', don't append 'Balcony' or 'BALCONY'."""
    out = expand_query_for_text_embedding(
        "balcony at night",
        ["Balcony", "BALCONY", "balcony"],
        max_extras=4,
    )
    # No appended duplicates
    assert out == "balcony at night"
