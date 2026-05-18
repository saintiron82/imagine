"""Runtime metadata quality signals for search reranking diagnostics."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "benchmarks" / "reviews" / "metadata_quality_v1_20260510"
DEFAULT_PROFILE_PATH = DEFAULT_REVIEW_DIR / "metadata_quality_profile.json"
DEFAULT_SIGNALS_PATH = DEFAULT_REVIEW_DIR / "metadata_quality_signals.jsonl"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _parse_tags(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]
    except Exception:
        pass
    return [
        item.strip().strip("\"'").lower()
        for item in text.strip("[]").split(",")
        if item.strip().strip("\"'")
    ]


def _has_usable_caption(value: Any) -> bool:
    caption = str(value or "").strip()
    return bool(caption) and caption.lower() != "unknown"


def _has_usable_tags(value: Any) -> bool:
    raw = str(value or "").strip()
    return bool(raw) and raw != "[]"


def _infer_analysis_status(result: dict[str, Any]) -> str:
    explicit = str(result.get("analysis_status") or "").strip()
    if explicit:
        return explicit
    caption_ok = _has_usable_caption(result.get("mc_caption"))
    tags_ok = _has_usable_tags(result.get("ai_tags"))
    if not caption_ok and not tags_ok:
        return "missing"
    if not caption_ok or not tags_ok:
        return "partial"
    if str(result.get("processing_status") or "").strip() == "parse_fallback_legacy":
        return "legacy_warning"
    if str(result.get("caption_model") or "").strip() == "unknown_legacy":
        return "legacy_warning"
    return "ok"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_signals(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    signals: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row.get("item_id") or "").strip()
            if item_id:
                signals[item_id] = row
    return signals


@lru_cache(maxsize=4)
def load_metadata_quality_bundle(
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    signals_path: str | Path = DEFAULT_SIGNALS_PATH,
) -> dict[str, Any]:
    profile = _load_json(Path(profile_path))
    signals_by_item_id = _load_signals(Path(signals_path))
    return {
        "profile": profile,
        "signals_by_item_id": signals_by_item_id,
    }


def _item_signal(result: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any] | None:
    item_id = str(result.get("id") or result.get("item_id") or "").strip()
    signal = (bundle.get("signals_by_item_id") or {}).get(item_id)
    if not signal:
        return None
    return {
        "metadata_reliability_score": _as_float(signal.get("metadata_reliability_score")),
        "caption_reliability": _as_float(signal.get("caption_reliability")),
        "tag_reliability": _as_float(signal.get("tag_reliability")),
        "metadata_quality_source": "item_review",
        "metadata_quality_confidence": "high",
        "metadata_quality_issues": list(signal.get("issue_types") or []),
        "metadata_quality_basis": [f"item_id:{item_id}"],
    }


def metadata_quality_for_result(
    result: dict[str, Any],
    *,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = bundle or load_metadata_quality_bundle()
    profile = bundle.get("profile") or {}
    if not profile:
        return {}

    item_signal = _item_signal(result, bundle)
    if item_signal:
        return item_signal

    limits = profile.get("limits") or {}
    status_min = int(limits.get("runtime_status_min_reviewed") or 20)
    tag_min = int(limits.get("runtime_tag_min_reviewed") or 30)
    global_rel = _as_float((profile.get("global") or {}).get("metadata_reliability"))

    basis: list[str] = []
    score_parts: list[float] = []
    status = _infer_analysis_status(result)
    status_profile = (profile.get("analysis_status") or {}).get(status) or {}
    status_reviewed = int(status_profile.get("reviewed_count") or 0)
    status_rel = _as_float(status_profile.get("metadata_reliability"))
    caption_rel = _as_float(status_profile.get("caption_reliability"))
    tag_rel = _as_float(status_profile.get("tag_reliability"))
    if status_rel is not None and status_reviewed >= status_min:
        score_parts.append(status_rel)
        basis.append(f"analysis_status:{status}")
    elif global_rel is not None:
        score_parts.append(global_rel)
        basis.append("global")

    tag_rels: list[float] = []
    tag_profiles = profile.get("tags") or {}
    for tag in _parse_tags(result.get("ai_tags")):
        tag_profile = tag_profiles.get(tag)
        if not tag_profile:
            continue
        if int(tag_profile.get("reviewed_count") or 0) < tag_min:
            continue
        tag_value = _as_float(tag_profile.get("tag_reliability"))
        if tag_value is not None:
            tag_rels.append(tag_value)
            basis.append(f"tag:{tag}")
    if tag_rels:
        tag_rel = _avg(tag_rels)
        if tag_rel is not None:
            score_parts.append(tag_rel)

    score = _avg(score_parts)
    if score is None:
        return {}

    return {
        "metadata_reliability_score": score,
        "caption_reliability": caption_rel,
        "tag_reliability": tag_rel,
        "metadata_quality_source": "profile_inferred" if basis != ["global"] else "profile_global",
        "metadata_quality_confidence": "medium" if basis and basis != ["global"] else "low",
        "metadata_quality_issues": [],
        "metadata_quality_basis": basis,
    }


def annotate_metadata_quality(
    results: list[dict[str, Any]],
    *,
    bundle: dict[str, Any] | None = None,
) -> None:
    bundle = bundle or load_metadata_quality_bundle()
    for result in results:
        signal = metadata_quality_for_result(result, bundle=bundle)
        if signal:
            result.update(signal)
