#!/usr/bin/env python3
"""
Triaxis Search Quality Benchmark
==================================
Measures retrieval quality of the 3-axis fusion (VV + MV + FTS → RRF)
compared to each individual axis.

Core idea: Use MC-generated tags/captions as "ground truth" queries for
each image. Then search with each axis and the combined Triaxis, measuring
whether the original image appears in top-K results.

This proves the project thesis: individually modest models, when combined
through RRF fusion, achieve search quality greater than any single axis.

Metrics per axis:
  - R@1:  Original image is rank 1
  - R@3:  Original image in top 3
  - R@5:  Original image in top 5
  - MRR:  Mean Reciprocal Rank (1/rank, averaged)

Key output: **Fusion Lift** = Triaxis R@K / best single-axis R@K

Prerequisites:
  - DB must have processed images (mc_caption, ai_tags, VV, MV, FTS populated)
  - Run pipeline first: python backend/pipeline/ingest_engine.py --discover <dir>

Usage:
    python tools/bench_triaxis_quality.py                 # all DB images (sample 50)
    python tools/bench_triaxis_quality.py --count 100     # sample 100
    python tools/bench_triaxis_quality.py --profile benchmarks/profiles/triaxis.yaml
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ── Query Builder ──────────────────────────────────────────────────────

import random


def _parse_tags(tags_raw: str) -> List[str]:
    """Parse tags from DB format (JSON array or comma-separated)."""
    if not tags_raw:
        return []
    try:
        parsed = json.loads(tags_raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in tags_raw.split(",") if t.strip()]


def _extract_keywords(tags: List[str], max_words: int = 10) -> List[str]:
    """Extract individual keywords from tags."""
    keywords = []
    for t in tags:
        for word in t.replace("_", " ").split():
            w = word.strip().lower()
            if len(w) > 2 and w not in keywords:
                keywords.append(w)
                if len(keywords) >= max_words:
                    return keywords
    return keywords


def build_queries_from_mc(row: dict) -> dict:
    """
    Build 3 difficulty levels of search queries from MC data.

    Simulates real user search behavior at different specificity levels.

    Returns:
        {
            "file_id": int,
            "file_name": str,
            "queries": {
                "exact":  full MC caption — ceiling reference (MV advantage),
                "sparse": 2-3 random tags — typical user search ("anime room sunset"),
                "novel":  short natural phrase from 2 key concepts — new wording,
            },
            "keywords_exact":  all keywords from tags,
            "keywords_sparse": keywords from sparse tags only,
        }
    """
    caption = row.get("mc_caption", "") or ""
    tags = _parse_tags(row.get("ai_tags", "") or "")

    # ── Exact: full caption + all tags (MV ceiling) ──
    exact = caption if caption else ", ".join(tags[:8])

    # ── Sparse: 2-3 random tags (realistic user query) ──
    if len(tags) >= 3:
        sparse_tags = random.sample(tags, 3)
    elif tags:
        sparse_tags = tags[:]
    else:
        sparse_tags = caption.split()[:3]
    sparse = " ".join(sparse_tags)

    # ── Novel: natural phrase from 2 key concepts (new wording) ──
    # Pick 2 tags and form a short query the user might type
    if len(tags) >= 2:
        picks = random.sample(tags, 2)
        novel = f"{picks[0]} with {picks[1]}"
    elif tags:
        novel = f"image of {tags[0]}"
    else:
        # Rephrase: take first 5 words of caption + rearrange
        words = caption.split()[:6]
        if len(words) >= 3:
            novel = " ".join(words[1:4])  # drop first word, take middle
        else:
            novel = caption[:40]

    return {
        "file_id": row["id"],
        "file_name": row.get("file_name", "?"),
        "queries": {
            "exact": exact,
            "sparse": sparse,
            "novel": novel,
        },
        "keywords_exact": _extract_keywords(tags),
        "keywords_sparse": _extract_keywords(sparse_tags),
    }


# ── Search Runners ─────────────────────────────────────────────────────


def search_vv(searcher, query: str, top_k: int) -> List[int]:
    """VV axis only. Returns list of file_ids in rank order."""
    try:
        results = searcher.vector_search(query, top_k=top_k, threshold=0.0)
        return [r["id"] for r in results]
    except Exception:
        return []


def search_mv(searcher, query: str, top_k: int) -> List[int]:
    """MV axis only. Returns list of file_ids in rank order."""
    try:
        results = searcher.text_vector_search(query, top_k=top_k, threshold=0.0)
        return [r["id"] for r in results]
    except Exception:
        return []


def search_fts(searcher, keywords: List[str], top_k: int) -> List[int]:
    """FTS axis only. Returns list of file_ids in rank order."""
    try:
        results = searcher.fts_search(keywords, top_k=top_k)
        return [r["id"] for r in results]
    except Exception:
        return []


def search_triaxis(searcher, query: str, top_k: int) -> List[int]:
    """Full Triaxis (VV + MV + FTS → RRF). Returns list of file_ids."""
    try:
        results = searcher.triaxis_search(
            query, top_k=top_k, threshold=0.0, use_codex=False
        )
        return [r["id"] for r in results]
    except Exception:
        return []


# ── Evaluation ─────────────────────────────────────────────────────────


def evaluate_axis(
    axis_name: str,
    search_fn,
    queries: List[dict],
    difficulty: str = "sparse",
    k_values: List[int] = [1, 3, 5],
) -> dict:
    """
    Evaluate one search axis across all queries at a given difficulty.

    Args:
        axis_name: "vv", "mv", "fts", "triaxis"
        search_fn: callable that takes (query_text_or_keywords) → [file_ids]
        queries: list from build_queries_from_mc()
        difficulty: "exact", "sparse", "novel"

    Returns: {recall_at_k, mrr, per_query_ranks}
    """
    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    per_query = []
    n = len(queries)

    for i, q in enumerate(queries):
        target_id = q["file_id"]
        qtext = q["queries"][difficulty]

        # Choose query input based on axis
        if axis_name == "fts":
            kw_key = "keywords_exact" if difficulty == "exact" else "keywords_sparse"
            ranked_ids = search_fn(q[kw_key])
        else:
            ranked_ids = search_fn(qtext)

        # Find rank of target
        if target_id in ranked_ids:
            rank = ranked_ids.index(target_id) + 1
        else:
            rank = -1  # Not found in top-K

        for k in k_values:
            if 0 < rank <= k:
                recall_hits[k] += 1

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        per_query.append({
            "file": q["file_name"],
            "rank": rank,
            "query": str(qtext)[:60],
        })

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"      [{i+1}/{n}]")

    recall = {f"R@{k}": round(recall_hits[k] / n, 4) for k in k_values}
    mrr = round(float(np.mean(reciprocal_ranks)), 4)

    return {
        "recall": recall,
        "mrr": mrr,
        "per_query": per_query,
    }


# ── Report Builder ─────────────────────────────────────────────────────


def score_grade(pct: float) -> str:
    if pct >= 85:
        return "A"
    elif pct >= 70:
        return "B"
    elif pct >= 55:
        return "C"
    elif pct >= 40:
        return "D"
    else:
        return "F"


def build_report(
    all_results: Dict[str, Dict[str, dict]],
    query_count: int,
    db_size: int,
    timing: dict,
    sample_queries: List[dict],
) -> str:
    """
    Build report. all_results[difficulty][axis] = {recall, mrr, per_query}
    """
    axes = ["vv", "mv", "fts", "triaxis"]
    difficulties = ["exact", "sparse", "novel"]

    lines = [
        f"# Triaxis Search Quality Benchmark",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries**: {query_count} (from MC tags/captions)",
        f"**DB Size**: {db_size} images",
        f"",
        f"## Query Difficulty Levels",
        f"",
        f"| Level | Description | Example |",
        f"|---|---|---|",
    ]
    if sample_queries:
        sq = sample_queries[0]["queries"]
        lines.extend([
            f"| **exact** | Full MC caption (MV ceiling) | {sq['exact'][:60]} |",
            f"| **sparse** | 2-3 random tags (typical user) | {sq['sparse'][:60]} |",
            f"| **novel** | New phrasing from 2 concepts | {sq['novel'][:60]} |",
        ])

    # Main comparison table per difficulty
    for diff in difficulties:
        results = all_results[diff]

        single_best_r1 = max(results[a]["recall"].get("R@1", 0) for a in ["vv", "mv", "fts"])
        single_best_r5 = max(results[a]["recall"].get("R@5", 0) for a in ["vv", "mv", "fts"])
        tri_r1 = results["triaxis"]["recall"].get("R@1", 0)
        tri_r5 = results["triaxis"]["recall"].get("R@5", 0)
        lift_r1 = (tri_r1 / single_best_r1 - 1) * 100 if single_best_r1 > 0 else 0
        lift_r5 = (tri_r5 / single_best_r5 - 1) * 100 if single_best_r5 > 0 else 0

        lines.extend([
            f"",
            f"---",
            f"",
            f"## {diff.upper()} Difficulty",
            f"",
            f"| Axis | R@1 | R@3 | R@5 | MRR |",
            f"|---|:-:|:-:|:-:|:-:|",
        ])
        for axis in axes:
            r = results[axis]
            recall = r["recall"]
            p = "**" if axis == "triaxis" else ""
            lines.append(
                f"| {p}{axis.upper()}{p} | "
                f"{p}{recall.get('R@1', 0):.3f}{p} | "
                f"{p}{recall.get('R@3', 0):.3f}{p} | "
                f"{p}{recall.get('R@5', 0):.3f}{p} | "
                f"{p}{r['mrr']:.3f}{p} |"
            )

        lines.extend([
            f"",
            f"**Fusion Lift (R@1)**: best single={single_best_r1:.3f} → "
            f"Triaxis={tri_r1:.3f} (**{lift_r1:+.1f}%**) | "
            f"**R@5**: {single_best_r5:.3f} → {tri_r5:.3f} (**{lift_r5:+.1f}%**)",
        ])

    # Summary table — all difficulties in one view
    lines.extend([
        f"",
        f"---",
        f"",
        f"## Summary — Fusion Lift by Difficulty",
        f"",
        f"| Difficulty | Best Single R@1 | Triaxis R@1 | Lift R@1 | Best Single R@5 | Triaxis R@5 | Lift R@5 |",
        f"|---|:-:|:-:|:-:|:-:|:-:|:-:|",
    ])
    for diff in difficulties:
        results = all_results[diff]
        sb1 = max(results[a]["recall"].get("R@1", 0) for a in ["vv", "mv", "fts"])
        sb5 = max(results[a]["recall"].get("R@5", 0) for a in ["vv", "mv", "fts"])
        t1 = results["triaxis"]["recall"].get("R@1", 0)
        t5 = results["triaxis"]["recall"].get("R@5", 0)
        l1 = (t1 / sb1 - 1) * 100 if sb1 > 0 else 0
        l5 = (t5 / sb5 - 1) * 100 if sb5 > 0 else 0
        lines.append(
            f"| **{diff}** | {sb1:.3f} | {t1:.3f} | **{l1:+.1f}%** | "
            f"{sb5:.3f} | {t5:.3f} | **{l5:+.1f}%** |"
        )

    # Per-query rank comparison (sparse, top 20)
    if "sparse" in all_results:
        results = all_results["sparse"]
        lines.extend([
            f"",
            f"---",
            f"",
            f"## Per-Query Rank Comparison (sparse)",
            f"",
            f"| File | Query | VV | MV | FTS | Triaxis |",
            f"|---|---|:-:|:-:|:-:|:-:|",
        ])
        for i in range(min(20, query_count)):
            fname = results["triaxis"]["per_query"][i]["file"]
            qtext = results["triaxis"]["per_query"][i]["query"][:35]
            ranks = []
            for axis in axes:
                r = results[axis]["per_query"][i]["rank"]
                ranks.append(str(r) if r > 0 else "-")
            lines.append(f"| {fname} | {qtext} | {' | '.join(ranks)} |")

    # Timing
    lines.extend([
        f"",
        f"---",
        f"",
        f"## Timing",
        f"",
        f"| Phase | Time |",
        f"|---|---|",
    ])
    for phase, t in timing.items():
        lines.append(f"| {phase} | {t:.1f}s |")

    lines.append("")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Triaxis Search Quality Benchmark — measures fusion lift"
    )
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50,
                        help="Number of images to sample from DB")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Search top-K for each query")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.profile:
        try:
            import yaml
            with open(args.profile, "r", encoding="utf-8") as f:
                profile = yaml.safe_load(f) or {}
        except ImportError:
            from ruamel.yaml import YAML
            with open(args.profile, "r", encoding="utf-8") as f:
                profile = YAML().load(f) or {}
        if "count" in profile and args.count == 50:
            args.count = profile["count"]
        if "top_k" in profile:
            args.top_k = profile["top_k"]

    timing = {}

    print("=" * 70)
    print("  Triaxis Search Quality Benchmark")
    print("=" * 70)

    # Load DB and search engine
    print("  Loading search engine...")
    t0 = time.perf_counter()
    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch

    db = SQLiteDB()
    searcher = SqliteVectorSearch(db=db)
    timing["init"] = round(time.perf_counter() - t0, 1)

    # Get images with MC data from DB
    print("  Loading images from DB...")
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT id, file_name, mc_caption, ai_tags
        FROM files
        WHERE mc_caption IS NOT NULL AND mc_caption != ''
          AND ai_tags IS NOT NULL AND ai_tags != ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (args.count,))
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    if not rows:
        print("  ERROR: No images with MC data in DB. Run pipeline first.")
        sys.exit(1)

    # Total DB size for context
    cursor.execute("SELECT COUNT(*) FROM files")
    db_size = cursor.fetchone()[0]

    print(f"  Sampled {len(rows)} images (DB total: {db_size})")

    # Build queries (reproducible random with seed)
    random.seed(42)
    queries = [build_queries_from_mc(row) for row in rows]
    valid_queries = [q for q in queries if q["queries"]["exact"]]
    print(f"  Valid queries: {len(valid_queries)}")

    if len(valid_queries) < 5:
        print("  ERROR: Too few valid queries. Need at least 5 images with MC data.")
        sys.exit(1)

    # Show sample queries
    print(f"\n  Sample queries (first image):")
    sq = valid_queries[0]["queries"]
    print(f"    exact:  {sq['exact'][:70]}")
    print(f"    sparse: {sq['sparse'][:70]}")
    print(f"    novel:  {sq['novel'][:70]}")

    # Run each axis × each difficulty
    top_k = args.top_k
    difficulties = ["exact", "sparse", "novel"]
    axes = ["vv", "mv", "fts", "triaxis"]
    all_results = {}  # all_results[difficulty][axis]

    for diff in difficulties:
        print(f"\n{'=' * 70}")
        print(f"  Difficulty: {diff.upper()}")
        print(f"{'=' * 70}")
        all_results[diff] = {}

        for axis in axes:
            label = {"vv": "VV (SigLIP2)", "mv": "MV (Qwen3-Embed)",
                     "fts": "FTS (BM25)", "triaxis": "TRIAXIS (RRF)"}[axis]
            print(f"\n  {label}:")
            t0 = time.perf_counter()

            if axis == "vv":
                fn = lambda q: search_vv(searcher, q, top_k)
            elif axis == "mv":
                fn = lambda q: search_mv(searcher, q, top_k)
            elif axis == "fts":
                fn = lambda q: search_fts(searcher, q, top_k)
            else:
                fn = lambda q: search_triaxis(searcher, q, top_k)

            result = evaluate_axis(axis, fn, valid_queries, difficulty=diff)
            elapsed = round(time.perf_counter() - t0, 1)
            timing[f"{diff}_{axis}"] = elapsed
            all_results[diff][axis] = result

            print(f"    R@1={result['recall']['R@1']:.3f}  "
                  f"R@5={result['recall']['R@5']:.3f}  "
                  f"MRR={result['mrr']:.3f}  ({elapsed}s)")

    # Compute key fusion lifts for console summary
    sparse_results = all_results["sparse"]
    sb1 = max(sparse_results[a]["recall"]["R@1"] for a in ["vv", "mv", "fts"])
    sb5 = max(sparse_results[a]["recall"]["R@5"] for a in ["vv", "mv", "fts"])
    tri1 = sparse_results["triaxis"]["recall"]["R@1"]
    tri5 = sparse_results["triaxis"]["recall"]["R@5"]
    lift1 = (tri1 / sb1 - 1) * 100 if sb1 > 0 else 0
    lift5 = (tri5 / sb5 - 1) * 100 if sb5 > 0 else 0

    # Report
    report_md = build_report(
        all_results, len(valid_queries), db_size, timing, valid_queries
    )

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"triaxis_quality_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "query_count": len(valid_queries),
        "db_size": db_size,
        "top_k": top_k,
        "difficulties": {
            diff: {
                axis: {"recall": r["recall"], "mrr": r["mrr"]}
                for axis, r in all_results[diff].items()
            }
            for diff in difficulties
        },
        "timing": timing,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    # Console summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — SPARSE difficulty (realistic user queries)")
    print(f"{'=' * 70}")
    print(f"  {'Axis':<12} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'MRR':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for axis in axes:
        r = sparse_results[axis]
        marker = " ★" if axis == "triaxis" else ""
        print(
            f"  {axis.upper():<12} "
            f"{r['recall']['R@1']:>7.3f} "
            f"{r['recall']['R@3']:>7.3f} "
            f"{r['recall']['R@5']:>7.3f} "
            f"{r['mrr']:>7.3f}{marker}"
        )
    print(f"{'─' * 70}")
    print(f"  Fusion Lift R@1: {lift1:+.1f}% | R@5: {lift5:+.1f}%  (vs best single axis)")
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"  JSON:   {json_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
