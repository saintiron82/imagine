"""Phase C: classify each benchmark query into 4 buckets.

found          top_k ∩ gt ≠ ∅
missed         gt ≠ ∅ and top_k ∩ gt = ∅
honest_empty   gt = ∅ and system_confidence = empty
false_answer   gt = ∅ and system_confidence ≠ empty

The split lets a benchmark distinguish "system was wrong" from "system
was honest about having no good match" — invisible under a single P@K
number.
"""
from __future__ import annotations

import enum
from typing import Iterable, Optional


class Outcome(str, enum.Enum):
    FOUND = "found"
    MISSED = "missed"
    HONEST_EMPTY = "honest_empty"
    FALSE_ANSWER = "false_answer"


def classify(
    *,
    top_k: Iterable[int],
    gt: set[int],
    system_confidence: Optional[str],
) -> Outcome:
    """Return the bucket for a single query result."""
    top_set = set(top_k or [])
    has_gt = bool(gt)
    if has_gt:
        if top_set & gt:
            return Outcome.FOUND
        return Outcome.MISSED
    # gt is empty
    if system_confidence == "empty":
        return Outcome.HONEST_EMPTY
    return Outcome.FALSE_ANSWER
