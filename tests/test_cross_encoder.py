"""Sprint 1 α: cross-encoder rerank."""
from __future__ import annotations

import pytest

from backend.search.cross_encoder import rerank


def test_rerank_no_op_when_reranker_is_none():
    rows = [{"id": 1}, {"id": 2}]
    out = rerank(query="x", rows=rows, reranker=None)
    assert out is rows


def test_rerank_reorders_by_score_descending():
    class _Stub:
        def score_pairs(self, pairs):
            mapping = {"a": 0.9, "b": 0.2, "c": 0.5}
            return [mapping.get(doc, 0.0) for _, doc in pairs]

    rows = [
        {"id": 1, "mc_caption": "b"},
        {"id": 2, "mc_caption": "a"},
        {"id": 3, "mc_caption": "c"},
    ]
    out = rerank(query="q", rows=rows, reranker=_Stub())
    assert [r["id"] for r in out] == [2, 3, 1]


def test_rerank_stable_for_ties():
    class _Equal:
        def score_pairs(self, pairs):
            return [0.5] * len(pairs)

    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    out = rerank(query="q", rows=rows, reranker=_Equal())
    assert [r["id"] for r in out] == [1, 2, 3]


def test_rerank_uses_caption_then_tags_then_empty():
    class _Recorder:
        def __init__(self):
            self.seen = []

        def score_pairs(self, pairs):
            self.seen = list(pairs)
            return [0.0] * len(pairs)

    rec = _Recorder()
    rows = [
        {"id": 1, "mc_caption": "cap-1"},
        {"id": 2, "ai_tags": "tag-2"},
        {"id": 3},
    ]
    rerank(query="q", rows=rows, reranker=rec)
    assert rec.seen[0] == ("q", "cap-1")
    assert rec.seen[1] == ("q", "tag-2")
    assert rec.seen[2] == ("q", "")


def test_rerank_attaches_cross_encoder_score():
    class _Score:
        def score_pairs(self, pairs):
            return [0.3, 0.7]

    rows = [{"id": 1}, {"id": 2}]
    out = rerank(query="q", rows=rows, reranker=_Score())
    assert out[0]["id"] == 2
    assert out[0]["cross_encoder_score"] == 0.7
    assert out[1]["cross_encoder_score"] == 0.3


def test_rerank_handles_empty_rows():
    class _Boom:
        def score_pairs(self, pairs):
            raise AssertionError("should not be called with empty rows")

    assert rerank(query="q", rows=[], reranker=_Boom()) == []


def test_rerank_keeps_tail_beyond_pool_intact():
    """Wire pattern: rerank head pool, append untouched tail."""
    from backend.search.cross_encoder import rerank

    class _Reverse:
        def score_pairs(self, pairs):
            # Higher index -> higher score so first 5 reverse order.
            return list(range(len(pairs)))

    pool = [{"id": i} for i in range(5)]
    tail = [{"id": i} for i in range(5, 10)]
    out_pool = rerank(query="q", rows=pool, reranker=_Reverse())
    full = out_pool + tail
    assert [r["id"] for r in out_pool] == [4, 3, 2, 1, 0]
    assert [r["id"] for r in full[5:]] == [5, 6, 7, 8, 9]
