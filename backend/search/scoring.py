"""
Scoring, reranking, filtering, and RRF merge functions.

Extracted from SqliteVectorSearch to separate scoring/ranking concerns
from DB access and encoder management.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_norm(val: Optional[float], low: float, high: float) -> Optional[float]:
    """Min-max normalize a score to [0, 1]."""
    if val is None:
        return None
    span = high - low
    if span <= 1e-12:
        return 1.0
    x = (float(val) - low) / span
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def query_tokens(query: str) -> List[str]:
    """Extract lightweight query tokens for soft intent matching."""
    if not query:
        return []
    tokens = []
    for raw in query.lower().split():
        t = raw.strip(" \t\r\n,.;:!?\"'()[]{}")
        if len(t) >= 2:
            tokens.append(t)
    # Preserve order, remove duplicates
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ---------------------------------------------------------------------------
# RRF merge
# ---------------------------------------------------------------------------

def rrf_merge(
    vector_results: List[Dict],
    fts_results: List[Dict],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to merge results from two sources.

    Preserves per-axis scores: vector_score (cosine similarity) and
    text_score (min-max normalized FTS rank).
    """
    scores = {}       # file_path -> rrf_score
    result_map = {}   # file_path -> result dict
    vector_scores = {}  # file_path -> cosine similarity
    fts_raw_ranks = {}  # file_path -> fts_rank (negative)

    for rank, result in enumerate(vector_results):
        fp = result["file_path"]
        scores[fp] = scores.get(fp, 0) + 1.0 / (k + rank + 1)
        vector_scores[fp] = result.get("similarity", 0)
        if fp not in result_map:
            result_map[fp] = result

    for rank, result in enumerate(fts_results):
        fp = result["file_path"]
        scores[fp] = scores.get(fp, 0) + 1.0 / (k + rank + 1)
        fts_raw_ranks[fp] = result.get("fts_rank", 0)
        if fp not in result_map:
            result_map[fp] = result

    # Normalize FTS ranks (negative -> 0~1)
    normalized_text = {}
    if fts_raw_ranks:
        ranks = list(fts_raw_ranks.values())
        best = min(ranks)
        worst = max(ranks)
        span = worst - best
        for fp, r in fts_raw_ranks.items():
            normalized_text[fp] = (worst - r) / span if span else 1.0

    sorted_paths = sorted(scores.keys(), key=lambda fp: scores[fp], reverse=True)

    merged = []
    for fp in sorted_paths:
        result = result_map[fp]
        result["rrf_score"] = scores[fp]
        result["vector_score"] = vector_scores.get(fp)
        result["text_score"] = normalized_text.get(fp)
        merged.append(result)

    return merged


def rrf_merge_multi(
    result_lists: list[tuple[str, list[dict]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Multi-axis RRF merge for 2+ result lists.

    Generalizes rrf_merge to handle V + T + F (+X) axes.
    Preserves per-axis scores: vector_score, text_vec_score, text_score,
    structure_score, spatial_score.

    Args:
        result_lists: List of (axis_name, results) tuples.
                      axis_name: "visual", "text_vec", "fts", "structure", or "spatial"
        k: RRF constant (default 60)
        weights: Per-axis weight dict. If None, uniform weighting (1.0 per axis).
    """
    scores = {}       # file_path -> cumulative rrf_score
    result_map = {}   # file_path -> result dict
    axis_scores = {}  # file_path -> {axis_name: raw_score}

    for axis_name, results in result_lists:
        w = weights.get(axis_name, 1.0) if weights else 1.0
        for rank, result in enumerate(results):
            fp = result["file_path"]
            scores[fp] = scores.get(fp, 0) + w / (k + rank + 1)

            if fp not in axis_scores:
                axis_scores[fp] = {}

            if axis_name == "visual":
                axis_scores[fp]["visual"] = result.get("similarity", 0)
            elif axis_name == "text_vec":
                axis_scores[fp]["text_vec"] = result.get("text_similarity", 0)
            elif axis_name == "fts":
                axis_scores[fp]["fts_rank"] = result.get("fts_rank", 0)
            elif axis_name == "structure":
                axis_scores[fp]["structure"] = result.get("structural_similarity", 0)
            elif axis_name == "spatial":
                axis_scores[fp]["spatial"] = result.get("spatial_score", 0)

            if fp not in result_map:
                result_map[fp] = result

    # Normalize FTS ranks to 0~1
    fts_raw = {fp: s.get("fts_rank") for fp, s in axis_scores.items() if "fts_rank" in s}
    normalized_fts = {}
    if fts_raw:
        ranks = list(fts_raw.values())
        best = min(ranks)
        worst = max(ranks)
        span = worst - best
        for fp, r in fts_raw.items():
            normalized_fts[fp] = (worst - r) / span if span else 1.0

    sorted_paths = sorted(scores.keys(), key=lambda fp: scores[fp], reverse=True)

    merged = []
    for fp in sorted_paths:
        result = result_map[fp]
        result["rrf_score"] = scores[fp]
        result["vector_score"] = axis_scores.get(fp, {}).get("visual")
        result["text_vec_score"] = axis_scores.get(fp, {}).get("text_vec")
        result["text_score"] = normalized_fts.get(fp)
        result["structure_score"] = axis_scores.get(fp, {}).get("structure")
        result["spatial_score"] = axis_scores.get(fp, {}).get("spatial", result.get("spatial_score"))
        merged.append(result)

    return merged


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_axis_scores(
    merged: List[Dict],
    v_embedding: Optional[np.ndarray],
    t_embedding: Optional[np.ndarray],
    fts_keywords: Optional[List[str]] = None,
    s_embedding: Optional[np.ndarray] = None,
    *,
    batch_similarity_fn: Optional[Callable] = None,
    batch_fts_fn: Optional[Callable] = None,
) -> None:
    """
    Enrich merged results with missing per-axis scores via direct DB lookup.

    Display-only: does NOT affect ranking (runs after RRF + trim).
    Computes actual V/S/M scores so all badges can be displayed in the UI.

    Args:
        merged: List of result dicts (mutated in-place).
        v_embedding: VV query embedding (SigLIP2).
        t_embedding: MV query embedding (Qwen3-Embedding).
        fts_keywords: Keywords for FTS scoring.
        s_embedding: Structure query embedding (DINOv2).
        batch_similarity_fn: Callable(table, query_embedding, file_ids) -> {fid: score}.
        batch_fts_fn: Callable(fts_keywords, file_ids) -> {fid: score}.
    """
    if not merged:
        return

    v_missing = [r["id"] for r in merged if r.get("vector_score") is None and r.get("id")]
    s_missing = [r["id"] for r in merged if r.get("text_vec_score") is None and r.get("id")]
    m_missing = [r["id"] for r in merged if r.get("text_score") is None and r.get("id")]
    st_missing = [r["id"] for r in merged if r.get("structure_score") is None and r.get("id")]

    if v_missing and v_embedding is not None and batch_similarity_fn:
        v_scores = batch_similarity_fn("vec_files", v_embedding, v_missing)
        for r in merged:
            if r.get("vector_score") is None and r.get("id") in v_scores:
                r["vector_score"] = v_scores[r["id"]]

    if s_missing and t_embedding is not None and batch_similarity_fn:
        s_scores = batch_similarity_fn("vec_text", t_embedding, s_missing)
        for r in merged:
            if r.get("text_vec_score") is None and r.get("id") in s_scores:
                r["text_vec_score"] = s_scores[r["id"]]

    if m_missing and fts_keywords and batch_fts_fn:
        m_scores = batch_fts_fn(fts_keywords, m_missing)
        for r in merged:
            if r.get("text_score") is None and r.get("id") in m_scores:
                r["text_score"] = m_scores[r["id"]]

    if st_missing and s_embedding is not None and batch_similarity_fn:
        st_scores = batch_similarity_fn("vec_structure", s_embedding, st_missing)
        for r in merged:
            if r.get("structure_score") is None and r.get("id") in st_scores:
                r["structure_score"] = st_scores[r["id"]]
            if r.get("structural_similarity") is None and r.get("id") in st_scores:
                r["structural_similarity"] = st_scores[r["id"]]


# ---------------------------------------------------------------------------
# Quality rerank
# ---------------------------------------------------------------------------

def quality_rerank(
    results: List[Dict[str, Any]],
    top_k: int,
    query: str,
    llm_filters: Optional[Dict[str, Any]] = None,
    user_filters: Optional[Dict[str, Any]] = None,
    axis_weights: Optional[Dict[str, float]] = None,
    pool_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Quality-focused rerank over top candidate pool.

    Goal:
    - Keep RRF recall benefits
    - Promote results with stronger cross-axis agreement
    - Prefer entries with richer stored metadata (caption/tags/structured fields)
    """
    if not results:
        return results

    llm_filters = llm_filters or {}
    user_filters = user_filters or {}

    pool_n = min(len(results), int(pool_size or max(top_k * 3, 80)))
    if pool_n <= 1:
        return results

    pool = results[:pool_n]
    tail = results[pool_n:]

    # Axis ranges for normalization
    def _axis_range(key: str):
        vals = [float(r[key]) for r in pool if r.get(key) is not None]
        if not vals:
            return 0.0, 1.0
        return min(vals), max(vals)

    v_low, v_high = _axis_range("vector_score")
    x_low, x_high = _axis_range("structure_score")
    s_low, s_high = _axis_range("text_vec_score")
    m_low, m_high = _axis_range("text_score")
    p_low, p_high = _axis_range("spatial_score")

    axis_w = {
        "visual": 0.24,
        "structure": 0.12,
        "text_vec": 0.29,
        "fts": 0.17,
        "spatial": 0.18,
    }
    if axis_weights:
        axis_w.update({k: float(v) for k, v in axis_weights.items() if k in axis_w})

    q_tokens = query_tokens(query)
    q_path = (query or "").replace("\\", "/").strip().lower()
    path_hint = ("/" in q_path) or bool(q_path.endswith((".psd", ".png", ".jpg", ".jpeg")))
    soft_filter_keys = {
        "format", "image_type", "scene_type", "art_style",
        "time_of_day", "weather", "folder_path"
    }
    all_filters = {}
    all_filters.update(llm_filters)
    all_filters.update(user_filters)

    rescored = []
    n = len(pool)
    for idx, r in enumerate(pool):
        rrf_prior = 1.0 if n <= 1 else 1.0 - (idx / (n - 1))

        v_norm = safe_norm(r.get("vector_score"), v_low, v_high)
        x_norm = safe_norm(r.get("structure_score"), x_low, x_high)
        s_norm = safe_norm(r.get("text_vec_score"), s_low, s_high)
        m_norm = safe_norm(r.get("text_score"), m_low, m_high)
        p_norm = safe_norm(r.get("spatial_score"), p_low, p_high)

        # Per-axis contribution. Missing axes contribute 0 to the numerator
        # and the FULL axis weight to the denominator — so a result that only
        # fires on one strong axis cannot post a high blended score on its own.
        # (Previously missing axes were excluded from the denominator, which
        # let a single-axis spike on the dominant axis tie or beat results
        # that matched on multiple axes weakly. Concrete failure: a query
        # for "mountain" surfaced an irrelevant garden image at 96% (vector
        # rank 5 + everything else missing) above a real mountain landscape
        # at 33% (text_vec rank 39, vector missing).)
        axis_num = 0.0
        axes_present = 0
        if v_norm is not None:
            axis_num += axis_w["visual"] * v_norm
            axes_present += 1
        if x_norm is not None:
            axis_num += axis_w["structure"] * x_norm
            axes_present += 1
        if s_norm is not None:
            axis_num += axis_w["text_vec"] * s_norm
            axes_present += 1
        if m_norm is not None:
            axis_num += axis_w["fts"] * m_norm
            axes_present += 1
        if p_norm is not None:
            axis_num += axis_w["spatial"] * p_norm
            axes_present += 1

        axis_total_weight = sum(axis_w.values())
        if axis_total_weight > 0:
            axis_blend = axis_num / axis_total_weight
        else:
            axis_blend = rrf_prior

        # Multi-axis agreement bonus: results that fire on 2+ axes signal
        # cross-channel relevance and should outrank single-axis spikes.
        # +5% / +10% / +15% for 2 / 3 / 4 axes (caps quality_score at ~1.15).
        multi_axis_bonus = 0.0
        if axes_present >= 2:
            multi_axis_bonus = 0.05 * (axes_present - 1)

        # Metadata completeness
        has_caption = 1.0 if (r.get("mc_caption") or "").strip() else 0.0
        tags = r.get("ai_tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        has_tags = 1.0 if isinstance(tags, list) and len(tags) > 0 else 0.0
        has_struct = 1.0 if (r.get("image_type") or r.get("scene_type") or r.get("art_style")) else 0.0
        has_user = 1.0 if (r.get("user_note") or r.get("user_tags") or r.get("user_category")) else 0.0
        meta_completeness = (has_caption * 0.35) + (has_tags * 0.25) + (has_struct * 0.30) + (has_user * 0.10)

        # Soft intent match
        hay_parts = [
            str(r.get("file_name") or ""),
            str(r.get("folder_path") or ""),
            str(r.get("relative_path") or ""),
            str(r.get("file_path") or ""),
            str(r.get("mc_caption") or ""),
            str(r.get("image_type") or ""),
            str(r.get("scene_type") or ""),
            str(r.get("art_style") or ""),
        ]
        if isinstance(tags, list):
            hay_parts.extend(str(t) for t in tags)
        spatial_matches = r.get("spatial_matches") or []
        if isinstance(spatial_matches, list):
            for match in spatial_matches:
                if isinstance(match, dict):
                    hay_parts.extend(str(v) for v in match.values() if v is not None)
        hay = " ".join(hay_parts).lower()
        token_hits = sum(1 for t in q_tokens if t in hay)
        token_score = (token_hits / max(1, len(q_tokens))) if q_tokens else 0.0

        filter_hits = 0
        filter_total = 0
        for fk, fv in all_filters.items():
            if fk not in soft_filter_keys or fv in (None, ""):
                continue
            filter_total += 1
            rv = str(r.get(fk) or "").lower()
            if rv and rv == str(fv).lower():
                filter_hits += 1
        filter_score = (filter_hits / filter_total) if filter_total else 0.0

        # Path intent boost
        path_score = 0.0
        if path_hint and q_path:
            cands = [
                str(r.get("relative_path") or "").replace("\\", "/").lower(),
                str(r.get("file_path") or "").replace("\\", "/").lower(),
                str(r.get("folder_path") or "").replace("\\", "/").lower(),
            ]
            for cp in cands:
                if not cp:
                    continue
                if cp == q_path or cp.endswith(q_path):
                    path_score = 1.0
                    break
                if q_path in cp:
                    path_score = 0.8
                    break

        intent_boost = (token_score * 0.55) + (filter_score * 0.25) + (path_score * 0.20)

        quality_score = (
            (0.62 * axis_blend) +
            (0.23 * rrf_prior) +
            (0.10 * meta_completeness) +
            (0.05 * intent_boost) +
            multi_axis_bonus
        )

        r["quality_score"] = quality_score
        r["axes_present"] = axes_present
        r["multi_axis_bonus"] = round(multi_axis_bonus, 4)
        rescored.append((r, quality_score, idx))

    rescored.sort(key=lambda x: (x[1], x[0].get("rrf_score", 0.0), -x[2]), reverse=True)
    return [x[0] for x in rescored] + tail


# ---------------------------------------------------------------------------
# User / metadata filters
# ---------------------------------------------------------------------------

def apply_user_filters(
    results: List[Dict],
    filters: Dict[str, Any],
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply metadata filters to results in-memory.

    Args:
        results: Search results to filter
        filters: Metadata filter dict
        strict: If True (user filters), exclude results missing the field.
                If False (LLM filters), pass results where the field is None/empty.
    """
    filtered = []

    for result in results:
        if "format" in filters and filters["format"]:
            if result.get("format", "").upper() != filters["format"].upper():
                continue

        if "user_category" in filters and filters["user_category"]:
            if result.get("user_category", "") != filters["user_category"]:
                continue

        if "min_rating" in filters and filters["min_rating"]:
            if (result.get("user_rating") or 0) < int(filters["min_rating"]):
                continue

        if "user_tags" in filters and filters["user_tags"]:
            result_tags = result.get("user_tags", [])
            if isinstance(result_tags, str):
                try:
                    result_tags = json.loads(result_tags)
                except Exception:
                    result_tags = []
            filter_tag = filters["user_tags"].lower()
            if not any(filter_tag in t.lower() for t in result_tags):
                continue

        if "folder_path" in filters and filters["folder_path"]:
            result_folder = result.get("folder_path") or ""
            if not result_folder.startswith(filters["folder_path"]):
                continue

        if "folder_tag" in filters and filters["folder_tag"]:
            result_ftags = result.get("folder_tags", [])
            if isinstance(result_ftags, str):
                try:
                    result_ftags = json.loads(result_ftags)
                except Exception:
                    result_ftags = []
            filter_ftag = filters["folder_tag"].lower()
            if not any(filter_ftag in t.lower() for t in result_ftags):
                continue

        # v3 P0: structured vision filters
        if "image_type" in filters and filters["image_type"]:
            result_val = (result.get("image_type") or "")
            if not strict and not result_val:
                pass
            elif result_val.lower() != filters["image_type"].lower():
                continue

        if "art_style" in filters and filters["art_style"]:
            result_val = (result.get("art_style") or "")
            if not strict and not result_val:
                pass
            elif result_val.lower() != filters["art_style"].lower():
                continue

        if "scene_type" in filters and filters["scene_type"]:
            result_val = (result.get("scene_type") or "")
            if not strict and not result_val:
                pass
            elif result_val.lower() != filters["scene_type"].lower():
                continue

        if "time_of_day" in filters and filters["time_of_day"]:
            result_val = (result.get("time_of_day") or "")
            if not strict and not result_val:
                pass
            elif result_val.lower() != filters["time_of_day"].lower():
                continue

        if "weather" in filters and filters["weather"]:
            result_val = (result.get("weather") or "")
            if not strict and not result_val:
                pass
            elif result_val.lower() != filters["weather"].lower():
                continue

        filtered.append(result)

    return filtered


# ---------------------------------------------------------------------------
# Negative filter
# ---------------------------------------------------------------------------

def apply_negative_filter(
    results: List[Dict],
    negative_query: str,
    neg_v_embedding: Optional[np.ndarray] = None,
    *,
    batch_similarity_fn: Optional[Callable] = None,
) -> List[Dict]:
    """
    Post-filter: demote results matching negative concepts via two layers:

    Layer 1 (Text): Check mc_caption + ai_tags for negative term text matches.
    Layer 2 (Visual): If neg_v_embedding is provided and batch_similarity_fn
                      is available, compute VV similarity and demote outliers.

    Args:
        results: Search results to filter
        negative_query: Space-separated negative terms
        neg_v_embedding: Optional SigLIP2 embedding of negative_query
        batch_similarity_fn: Callable(table, query_embedding, file_ids) -> {fid: score}
    """
    if not negative_query or not results:
        return results

    neg_terms = [t.lower().strip() for t in negative_query.split() if t.strip()]
    if not neg_terms:
        return results

    # Layer 2: Compute VV negative similarity scores
    neg_v_scores = {}
    if neg_v_embedding is not None and batch_similarity_fn:
        file_ids = [r["id"] for r in results if r.get("id")]
        if file_ids:
            neg_v_scores = batch_similarity_fn("vec_files", neg_v_embedding, file_ids)

    # Pre-compute visual negative outlier threshold
    v_outlier_flag = {}
    if neg_v_scores:
        all_neg_sims = list(neg_v_scores.values())
        if len(all_neg_sims) >= 3:
            mean_neg = sum(all_neg_sims) / len(all_neg_sims)
            variance = sum((x - mean_neg) ** 2 for x in all_neg_sims) / len(all_neg_sims)
            std_neg = variance ** 0.5
            outlier_thresh = mean_neg + 1.0 * std_neg
            for fid, sim in neg_v_scores.items():
                if sim > outlier_thresh:
                    v_outlier_flag[fid] = sim
            logger.debug(
                f"Negative VV stats: mean={mean_neg:.4f}, std={std_neg:.4f}, "
                f"outlier_thresh={outlier_thresh:.4f}, outliers={len(v_outlier_flag)}"
            )

    scored = []
    for r in results:
        caption = (r.get("mc_caption") or "").lower()
        tags = r.get("ai_tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        if not isinstance(tags, list):
            tags = []
        tags_text = " ".join(str(t).lower() for t in tags)
        combined_text = f"{caption} {tags_text}"

        text_neg = sum(1 for term in neg_terms if term in combined_text)

        fid = r.get("id")
        v_is_outlier = fid in v_outlier_flag

        neg_score = (text_neg * 1.0) + (0.6 if v_is_outlier else 0.0)

        scored.append((r, neg_score, text_neg, v_is_outlier))

    neg_threshold = 0.45
    filtered = [(r, ns) for r, ns, _, _ in scored if ns <= neg_threshold]
    demoted = [(r, ns) for r, ns, _, _ in scored if ns > neg_threshold]

    demoted.sort(key=lambda x: x[1], reverse=True)

    if demoted:
        demoted_info = [
            (r.get("file_name", "?"), round(ns, 3))
            for r, ns in demoted[:5]
        ]
        logger.info(
            f"Negative filter: demoted {len(demoted)} results "
            f"(threshold={neg_threshold}, terms={neg_terms[:5]}, "
            f"outlier_count={len(v_outlier_flag)}, "
            f"top_demoted={demoted_info})"
        )

    return [r for r, _ in filtered] + [r for r, _ in demoted]
