#!/usr/bin/env python3
"""
Triaxis Full-Pipeline Search Quality Benchmark
================================================
Measures retrieval quality of the COMPLETE Triaxis pipeline:
  LLM Decomposer → Scope Filter → 3-axis Search → Auto-Weight RRF
  → Negative Filter → Quality Rerank

Tests 3 query complexity levels that exercise progressively more
pipeline stages:

  Level 1 — SIMPLE: "anime room desk"
    Tests: VV + MV + FTS → RRF (basic 3-axis)

  Level 2 — SCOPED: "background 타입에서 anime room desk"
    Tests: + QueryDecomposer (scope extraction) + Scope Filter

  Level 3 — COMPLEX: "character 중에서 sword warrior, cute 제외"
    Tests: + Negative Filter + Exclude Keywords + full pipeline

This proves the project thesis: the full pipeline achieves higher
retrieval quality than any single axis, and each pipeline stage adds
measurable value.

Usage:
    python tools/bench_triaxis_quality.py                 # default
    python tools/bench_triaxis_quality.py --count 50
    python tools/bench_triaxis_quality.py --profile benchmarks/profiles/triaxis.yaml
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ── Query Generation ───────────────────────────────────────────────────


def _parse_tags(tags_raw: str) -> List[str]:
    if not tags_raw:
        return []
    try:
        parsed = json.loads(tags_raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in tags_raw.split(",") if t.strip()]


def _pick_negative_tag(all_tags: List[str], own_tags: List[str]) -> str:
    """Pick a tag from the corpus that this image does NOT have."""
    own_set = {t.lower() for t in own_tags}
    candidates = [t for t in all_tags if t.lower() not in own_set and len(t) > 2]
    return random.choice(candidates) if candidates else ""


def generate_queries(rows: List[dict], all_tags_pool: List[str]) -> List[dict]:
    """
    Generate 3 complexity levels of natural-language queries for each image.

    Level 1 SIMPLE:  "anime room desk" (2-3 tags as keywords)
    Level 2 SCOPED:  "background 중에서 anime room" (image_type scope)
    Level 3 COMPLEX: "character에서 sword warrior 찾아줘, cute 스타일 제외"
                     (scope + find + exclude)
    """
    queries = []

    for row in rows:
        tags = _parse_tags(row.get("ai_tags", ""))
        caption = row.get("mc_caption", "") or ""
        image_type = row.get("image_type", "") or "other"
        art_style = row.get("art_style", "") or ""

        if not tags and not caption:
            continue

        # Pick 2-3 descriptive tags (skip generic ones like "anime")
        desc_tags = [t for t in tags if t.lower() not in ("anime", "illustration", "sketch", "digital")]
        if len(desc_tags) < 2:
            desc_tags = tags[:3]
        sparse_tags = random.sample(desc_tags, min(3, len(desc_tags)))
        sparse_text = " ".join(sparse_tags)

        # SIMPLE: just keywords
        simple = sparse_text

        # SCOPED: image_type + keywords
        scoped = f"{image_type} 중에서 {sparse_text}"

        # COMPLEX: scope + keywords + negative
        neg_tag = _pick_negative_tag(all_tags_pool, tags)
        if neg_tag:
            complex_q = f"{image_type}에서 {sparse_text} 찾아줘, {neg_tag} 제외"
        else:
            complex_q = f"{image_type}에서 {sparse_text} 찾아줘"

        queries.append({
            "file_id": row["id"],
            "file_name": row.get("file_name", "?"),
            "image_type": image_type,
            "tags": tags,
            "queries": {
                "simple": simple,
                "scoped": scoped,
                "complex": complex_q,
            },
            "negative_tag": neg_tag,
        })

    return queries


# ── Search Functions ───────────────────────────────────────────────────


def search_vv(searcher, query: str, top_k: int) -> List[int]:
    try:
        results = searcher.vector_search(query, top_k=top_k, threshold=0.0)
        return [r["id"] for r in results]
    except Exception:
        return []


def search_mv(searcher, query: str, top_k: int) -> List[int]:
    try:
        results = searcher.text_vector_search(query, top_k=top_k, threshold=0.0)
        return [r["id"] for r in results]
    except Exception:
        return []


def search_fts(searcher, query: str, top_k: int) -> List[int]:
    keywords = [w.strip() for w in query.replace(",", " ").split() if len(w.strip()) > 2]
    if not keywords:
        return []
    try:
        results = searcher.fts_search(keywords[:8], top_k=top_k)
        return [r["id"] for r in results]
    except Exception:
        return []


def search_triaxis_full(searcher, query: str, top_k: int) -> List[int]:
    """Full pipeline: decomposer + scope + negative + rerank."""
    try:
        results = searcher.triaxis_search(
            query, top_k=top_k, threshold=0.0, use_codex=False
        )
        return [r["id"] for r in results]
    except Exception:
        return []


# ── Evaluation ─────────────────────────────────────────────────────────


def evaluate(
    name: str,
    search_fn,
    queries: List[dict],
    level: str,
    k_values: List[int] = [1, 3, 5, 10],
) -> dict:
    recall_hits = {k: 0 for k in k_values}
    reciprocal_ranks = []
    per_query = []
    n = len(queries)

    for i, q in enumerate(queries):
        target_id = q["file_id"]
        qtext = q["queries"][level]
        ranked_ids = search_fn(qtext)

        rank = ranked_ids.index(target_id) + 1 if target_id in ranked_ids else -1
        for k in k_values:
            if 0 < rank <= k:
                recall_hits[k] += 1

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        per_query.append({
            "file": q["file_name"],
            "rank": rank,
            "query": qtext[:70],
        })

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"      [{i+1}/{n}]")

    recall = {f"R@{k}": round(recall_hits[k] / n, 4) for k in k_values}
    mrr = round(float(np.mean(reciprocal_ranks)), 4)
    return {"recall": recall, "mrr": mrr, "per_query": per_query}


# ── Report ─────────────────────────────────────────────────────────────


def score_grade(pct: float) -> str:
    if pct >= 85: return "A"
    elif pct >= 70: return "B"
    elif pct >= 55: return "C"
    elif pct >= 40: return "D"
    else: return "F"


def build_report(
    all_results: Dict[str, Dict[str, dict]],
    query_count: int, db_size: int, timing: dict,
    sample_queries: List[dict],
) -> str:
    axes = ["vv", "mv", "fts", "triaxis"]
    levels = ["simple", "scoped", "complex"]

    lines = [
        "# Triaxis Full-Pipeline Search Quality Benchmark",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries**: {query_count}",
        f"**DB Size**: {db_size} images",
        "",
        "## Query Complexity Levels",
        "",
        "| Level | Pipeline Stages | Example |",
        "|---|---|---|",
    ]
    if sample_queries:
        sq = sample_queries[0]["queries"]
        lines.extend([
            f"| **simple** | VV + MV + FTS → RRF | `{sq['simple'][:55]}` |",
            f"| **scoped** | + LLM Decomposer + Scope Filter | `{sq['scoped'][:55]}` |",
            f"| **complex** | + Negative Filter + Exclude | `{sq['complex'][:55]}` |",
        ])

    # Per-level tables
    for level in levels:
        results = all_results[level]
        lines.extend([
            "", "---", "",
            f"## {level.upper()}",
            "",
            "| Axis | R@1 | R@3 | R@5 | R@10 | MRR |",
            "|---|:-:|:-:|:-:|:-:|:-:|",
        ])
        for axis in axes:
            r = results[axis]
            rc = r["recall"]
            p = "**" if axis == "triaxis" else ""
            lines.append(
                f"| {p}{axis.upper()}{p} | "
                f"{p}{rc.get('R@1',0):.3f}{p} | "
                f"{p}{rc.get('R@3',0):.3f}{p} | "
                f"{p}{rc.get('R@5',0):.3f}{p} | "
                f"{p}{rc.get('R@10',0):.3f}{p} | "
                f"{p}{r['mrr']:.3f}{p} |"
            )
        # Fusion lift
        sb5 = max(results[a]["recall"].get("R@5", 0) for a in ["vv", "mv", "fts"])
        t5 = results["triaxis"]["recall"].get("R@5", 0)
        lift5 = (t5 / sb5 - 1) * 100 if sb5 > 0 else 0
        sb1 = max(results[a]["recall"].get("R@1", 0) for a in ["vv", "mv", "fts"])
        t1 = results["triaxis"]["recall"].get("R@1", 0)
        lift1 = (t1 / sb1 - 1) * 100 if sb1 > 0 else 0
        lines.append(f"\n**Fusion Lift**: R@1 {sb1:.3f}→{t1:.3f} (**{lift1:+.1f}%**) | R@5 {sb5:.3f}→{t5:.3f} (**{lift5:+.1f}%**)")

    # Summary
    lines.extend(["", "---", "", "## Summary — Pipeline Stage Value", ""])
    lines.append("| Level | Triaxis R@1 | Triaxis R@5 | Triaxis R@10 | MRR | Stages Active |")
    lines.append("|---|:-:|:-:|:-:|:-:|---|")
    for level in levels:
        r = all_results[level]["triaxis"]
        rc = r["recall"]
        stages = {
            "simple": "VV+MV+FTS→RRF",
            "scoped": "+Decomposer+Scope",
            "complex": "+Negative+Exclude",
        }[level]
        lines.append(
            f"| **{level}** | {rc.get('R@1',0):.3f} | {rc.get('R@5',0):.3f} | "
            f"{rc.get('R@10',0):.3f} | {r['mrr']:.3f} | {stages} |"
        )

    # Scoped vs simple lift
    s_r5 = all_results["simple"]["triaxis"]["recall"].get("R@5", 0)
    sc_r5 = all_results["scoped"]["triaxis"]["recall"].get("R@5", 0)
    cx_r5 = all_results["complex"]["triaxis"]["recall"].get("R@5", 0)
    if s_r5 > 0:
        lines.append(f"\n**Scope Filter value**: simple→scoped R@5 {s_r5:.3f}→{sc_r5:.3f} (**{(sc_r5/s_r5-1)*100:+.1f}%**)")
    if sc_r5 > 0:
        lines.append(f"**Negative Filter value**: scoped→complex R@5 {sc_r5:.3f}→{cx_r5:.3f} (**{(cx_r5/sc_r5-1)*100:+.1f}%**)")

    # Per-query detail (scoped, top 20)
    if "scoped" in all_results:
        results = all_results["scoped"]
        lines.extend(["", "---", "", "## Per-Query Detail (scoped, top 20)", ""])
        lines.append("| File | Query | VV | MV | FTS | Triaxis |")
        lines.append("|---|---|:-:|:-:|:-:|:-:|")
        for i in range(min(20, query_count)):
            fname = results["triaxis"]["per_query"][i]["file"]
            qtext = results["triaxis"]["per_query"][i]["query"][:40]
            ranks = []
            for axis in axes:
                r = results[axis]["per_query"][i]["rank"]
                ranks.append(str(r) if r > 0 else "-")
            lines.append(f"| {fname} | {qtext} | {' | '.join(ranks)} |")

    # Timing
    lines.extend(["", "---", "", "## Timing", ""])
    lines.append("| Phase | Time |")
    lines.append("|---|---|")
    for phase, t in timing.items():
        lines.append(f"| {phase} | {t:.1f}s |")

    lines.append("")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Triaxis Full-Pipeline Search Quality Benchmark"
    )
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
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
    random.seed(42)

    print("=" * 70)
    print("  Triaxis Full-Pipeline Benchmark")
    print("=" * 70)

    # Load search engine
    print("  Loading search engine...")
    t0 = time.perf_counter()
    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    db = SQLiteDB()
    searcher = SqliteVectorSearch(db=db)
    timing["init"] = round(time.perf_counter() - t0, 1)

    # Sample images with MC data
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT id, file_name, image_type, folder_path, art_style, mc_caption, ai_tags
        FROM files
        WHERE mc_caption IS NOT NULL AND mc_caption != ''
          AND ai_tags IS NOT NULL AND ai_tags != ''
          AND image_type IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (args.count,))
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM files")
    db_size = cursor.fetchone()[0]

    if len(rows) < 5:
        print("  ERROR: Too few images with MC data. Run pipeline first.")
        sys.exit(1)

    print(f"  Sampled {len(rows)} images (DB: {db_size})")

    # Build tag pool for negative generation
    all_tags = set()
    for row in rows:
        for t in _parse_tags(row.get("ai_tags", "")):
            all_tags.add(t)
    all_tags_list = list(all_tags)

    # Generate queries
    queries = generate_queries(rows, all_tags_list)
    print(f"  Generated {len(queries)} queries")

    # Sample display
    sq = queries[0]
    print(f"\n  Sample query ({sq['file_name']}, type={sq['image_type']}):")
    for level, q in sq["queries"].items():
        print(f"    {level:>8}: {q[:65]}")

    # Run evaluations: 3 levels × 4 axes = 12
    levels = ["simple", "scoped", "complex"]
    axes_config = {
        "vv":      lambda q: search_vv(searcher, q, args.top_k),
        "mv":      lambda q: search_mv(searcher, q, args.top_k),
        "fts":     lambda q: search_fts(searcher, q, args.top_k),
        "triaxis": lambda q: search_triaxis_full(searcher, q, args.top_k),
    }
    all_results = {}

    for level in levels:
        print(f"\n{'=' * 70}")
        print(f"  Level: {level.upper()}")
        print(f"{'=' * 70}")
        all_results[level] = {}

        for axis_name, search_fn in axes_config.items():
            label = {"vv": "VV", "mv": "MV", "fts": "FTS", "triaxis": "TRIAXIS (full pipeline)"}[axis_name]
            print(f"\n  {label}:")
            t0 = time.perf_counter()
            result = evaluate(axis_name, search_fn, queries, level)
            elapsed = round(time.perf_counter() - t0, 1)
            timing[f"{level}_{axis_name}"] = elapsed
            all_results[level][axis_name] = result
            rc = result["recall"]
            print(f"    R@1={rc['R@1']:.3f}  R@5={rc['R@5']:.3f}  R@10={rc['R@10']:.3f}  MRR={result['mrr']:.3f}  ({elapsed}s)")

    # Report
    report_md = build_report(all_results, len(queries), db_size, timing, queries)

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"triaxis_fullpipe_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "query_count": len(queries),
        "db_size": db_size,
        "top_k": args.top_k,
        "levels": {
            level: {
                axis: {"recall": r["recall"], "mrr": r["mrr"]}
                for axis, r in all_results[level].items()
            }
            for level in levels
        },
        "timing": timing,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    # Console summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Pipeline Stage Value")
    print(f"{'=' * 70}")
    print(f"  {'Level':<10} {'Tri R@1':>8} {'Tri R@5':>8} {'Tri R@10':>9} {'MRR':>8}  {'Best Single R@5':>16} {'Lift':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*9} {'─'*8}  {'─'*16} {'─'*8}")
    for level in levels:
        tri = all_results[level]["triaxis"]
        sb5 = max(all_results[level][a]["recall"]["R@5"] for a in ["vv", "mv", "fts"])
        t5 = tri["recall"]["R@5"]
        lift = (t5 / sb5 - 1) * 100 if sb5 > 0 else 0
        print(
            f"  {level:<10} "
            f"{tri['recall']['R@1']:>7.3f} "
            f"{tri['recall']['R@5']:>7.3f} "
            f"{tri['recall']['R@10']:>8.3f} "
            f"{tri['mrr']:>7.3f}  "
            f"{sb5:>15.3f} "
            f"{lift:>+7.1f}%"
        )

    # Stage value
    s5 = all_results["simple"]["triaxis"]["recall"]["R@5"]
    sc5 = all_results["scoped"]["triaxis"]["recall"]["R@5"]
    cx5 = all_results["complex"]["triaxis"]["recall"]["R@5"]
    print(f"{'─' * 70}")
    if s5 > 0:
        print(f"  Scope Filter:    simple→scoped  R@5 {s5:.3f}→{sc5:.3f} ({(sc5/s5-1)*100:+.1f}%)")
    if sc5 > 0:
        print(f"  Negative Filter: scoped→complex R@5 {sc5:.3f}→{cx5:.3f} ({(cx5/sc5-1)*100:+.1f}%)")
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"  JSON:   {json_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
