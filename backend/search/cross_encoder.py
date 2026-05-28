"""Sprint 1 α: cross-encoder rerank over a small candidate pool.

The reranker is a Protocol-shaped duck type so unit tests inject a
stub. The production implementation lazy-loads
`BAAI/bge-reranker-v2-m3` via transformers — `load_default_reranker()`.
"""
from __future__ import annotations

from typing import Protocol


class CrossEncoderReranker(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        ...


def rerank(*, query: str, rows: list[dict], reranker: object) -> list[dict]:
    """Return `rows` reordered by reranker.score_pairs((query, doc))
    descending. Stable for ties. When `reranker` is falsy or `rows` is
    empty, returns the input list unchanged. Mutates each kept row to
    add `cross_encoder_score`.
    """
    if reranker is None or not rows:
        return rows

    def _doc_text(row: dict) -> str:
        return row.get("mc_caption") or row.get("ai_tags") or ""

    pairs = [(query, _doc_text(r)) for r in rows]
    scores = reranker.score_pairs(pairs)
    for r, s in zip(rows, scores):
        r["cross_encoder_score"] = float(s)

    indexed = list(enumerate(rows))
    indexed.sort(key=lambda iv: -scores[iv[0]])
    return [r for _, r in indexed]


_default_reranker = None


def load_default_reranker():
    """Lazy-load BAAI/bge-reranker-v2-m3 via transformers (CPU OK).

    Returns None if transformers / the model is unavailable so callers
    gracefully skip reranking.
    """
    global _default_reranker
    if _default_reranker is not None:
        return _default_reranker
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception:
        return None
    try:
        model_id = "BAAI/bge-reranker-v2-m3"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        mdl.eval()

        class _BGEReranker:
            def score_pairs(self, pairs):
                if not pairs:
                    return []
                with torch.no_grad():
                    inputs = tok(
                        [p[0] for p in pairs],
                        [p[1] for p in pairs],
                        padding=True, truncation=True,
                        max_length=384, return_tensors="pt",
                    )
                    return mdl(**inputs).logits.view(-1).float().tolist()

        _default_reranker = _BGEReranker()
    except Exception:
        _default_reranker = None
    return _default_reranker
