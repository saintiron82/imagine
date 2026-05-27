"""Phase B: Decomposer output's structured representation.

ConstraintPlan is the single source of truth for the search pipeline.
LLM自由 텍스트 응답을 from_decomposer_output()이 이 객체로 정규화
하고, 정규화 실패는 ConstraintPlanError 로 명시한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_QUERY_TYPES = frozenset({"visual", "keyword", "semantic", "balanced"})


class ConstraintPlanError(ValueError):
    """LLM payload did not satisfy the ConstraintPlan schema."""


@dataclass(frozen=True)
class ConstraintPlan:
    folder: str
    elements: tuple[str, ...]
    negatives: tuple[str, ...]
    vector_query: str
    query_type: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "folder": self.folder,
            "elements": list(self.elements),
            "negatives": list(self.negatives),
            "vector_query": self.vector_query,
            "query_type": self.query_type,
            "confidence": self.confidence,
        }


def _clean_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ConstraintPlanError(f"expected list, got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return tuple(out)


def from_decomposer_output(payload: Mapping[str, Any]) -> ConstraintPlan:
    """Validate a decomposer payload and return a ConstraintPlan.

    Raises ConstraintPlanError on schema violation. Callers should
    catch and either retry (one shot) or fall back to a rule-based plan.
    """
    if not isinstance(payload, Mapping):
        raise ConstraintPlanError("payload must be a mapping")
    for required in ("vector_query", "query_type"):
        if required not in payload:
            raise ConstraintPlanError(f"missing required field: {required}")

    query_type = payload["query_type"]
    if not isinstance(query_type, str) or query_type not in ALLOWED_QUERY_TYPES:
        raise ConstraintPlanError(
            f"query_type must be one of {sorted(ALLOWED_QUERY_TYPES)}; got {query_type!r}"
        )

    vector_query = payload["vector_query"]
    if not isinstance(vector_query, str) or not vector_query.strip():
        raise ConstraintPlanError("vector_query must be a non-empty string")

    folder = payload.get("folder", "") or ""
    if not isinstance(folder, str):
        raise ConstraintPlanError("folder must be a string")

    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return ConstraintPlan(
        folder=folder.strip(),
        elements=_clean_list(payload.get("elements", [])),
        negatives=_clean_list(payload.get("negatives", [])),
        vector_query=vector_query.strip(),
        query_type=query_type,
        confidence=confidence,
    )
