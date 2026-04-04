#!/usr/bin/env python3
"""
Precision@K Benchmark — LLM-as-Judge
======================================
Measures REAL search quality: "Are the top-K results relevant?"

Instead of R@K (did we find the exact original file?), this measures
P@K (what fraction of top-K results are relevant to the query?).

This matches how users actually experience search:
  "교실에 창문이 있는 이미지" → top-5 results → 3 are classroom images = P@5=0.60

Ground truth generation:
  1. Generate natural Korean query from MC data
  2. Fast DB scan: find ALL images whose MC caption/tags match query keywords
  3. LLM judge: for borderline cases, use local LLM to judge relevance
  4. Compare Triaxis top-K results against ground truth set
  5. P@K = |relevant ∩ top-K| / K

Usage:
    python tools/bench_precision.py --count 30
    python tools/bench_precision.py --count 30 --judge-mode keyword  # fast, no LLM
    python tools/bench_precision.py --count 30 --judge-mode llm      # accurate, slower
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"

# ── Korean tag mapping (reuse from realworld bench) ────────────────────

TAG_KO_MAP = {
    "sky": "하늘", "mountain": "산", "river": "강", "lake": "호수",
    "window": "창문", "door": "문", "room": "방", "classroom": "교실",
    "tree": "나무", "forest": "숲", "flower": "꽃", "garden": "정원",
    "building": "건물", "house": "집", "city": "도시", "street": "거리",
    "bridge": "다리", "stairs": "계단", "wall": "벽",
    "desk": "책상", "chair": "의자", "bed": "침대", "sofa": "소파",
    "lamp": "램프", "table": "테이블", "shelf": "선반", "bookshelf": "책장",
    "night": "밤", "sunset": "일몰", "star": "별", "moon": "달",
    "anime": "애니메이션", "character": "캐릭터",
    "sword": "검", "park": "공원", "ocean": "바다", "cliff": "절벽",
    "kitchen": "주방", "bathroom": "욕실", "hallway": "복도",
    "curtain": "커튼", "painting": "그림", "mirror": "거울",
    "plant": "식물", "cloud": "구름", "fire": "불", "water": "물",
    "light": "빛", "rooftop": "옥상", "balcony": "발코니",
}

SKIP_TAGS = {
    # Style / medium
    "anime", "illustration", "sketch", "digital", "3d_render",
    "painterly", "realistic", "fantasy", "abstract", "cartoon",
    "pixel_art", "line_art", "lineart", "watercolor", "digital painting",
    # Palette / mood (abstract adjectives — users don't search these)
    "cool_palette", "warm_palette", "neutral_color_palette",
    "warm_color_palette", "cool_color_palette",
    "atmospheric", "dynamic", "dramatic", "serene", "ethereal",
    "mysterious", "mythical", "mystical", "surreal", "tranquil",
    "melancholic", "desolate", "contemplative", "futuristic",
    "translucent", "ghostly", "overlay", "glow", "glowing",
    # Colors alone (too generic)
    "blue", "red", "green", "yellow", "pink", "purple", "orange",
    "white", "black", "grey", "gray", "brown",
    # Meta / system
    "public_art", "close-up", "profile", "draft", "concept_art",
    "split-screen", "indoor_scene", "outdoor_scene",
}

# Only use tags that represent CONCRETE OBJECTS (things you can see)
CONCRETE_OBJECTS = set(TAG_KO_MAP.keys()) | {
    "car", "boat", "ship", "train", "bicycle", "bus",
    "cat", "dog", "bird", "horse", "fish",
    "sword", "shield", "armor", "weapon", "bow",
    "book", "bottle", "cup", "basket", "pillow",
    "statue", "fountain", "bench", "gate", "tower",
    "candle", "piano", "guitar", "drum",
    "tent", "campfire", "lantern", "sign",
    "snow", "rain", "fog", "lightning",
    "fence", "pillar", "column", "arch",
    "chandelier", "rug", "carpet", "vase",
}


def _parse_tags(tags_raw):
    if not tags_raw:
        return []
    try:
        parsed = json.loads(tags_raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
    except (json.JSONDecodeError, TypeError):
        pass
    return [t.strip() for t in tags_raw.split(",") if t.strip()]


def _tag_to_korean(tag):
    tag_lower = tag.lower().replace("_", " ")
    if tag_lower in TAG_KO_MAP:
        return TAG_KO_MAP[tag_lower]
    first_word = tag_lower.split()[0] if " " in tag_lower else tag_lower
    if first_word in TAG_KO_MAP:
        return TAG_KO_MAP[first_word]
    return tag


def _folder_short(folder_path):
    if not folder_path:
        return ""
    parts = [p for p in folder_path.split("/") if p]
    meaningful = [p for p in parts if len(p) > 2 and p not in ("bg", "발송", "외주 회사")]
    return meaningful[-1] if meaningful else (parts[-1] if parts else "")


# ── Query + Ground Truth Generation ───────────────────────────────────


def generate_queries_with_gt(db, count: int, judge_mode: str = "keyword") -> List[dict]:
    """
    Generate queries and find ground truth (relevant image sets) from DB.

    For each query:
      1. Pick a source image, extract 2 visual elements
      2. Build natural Korean query
      3. Find ALL images in DB whose MC matches those elements → ground truth set
    """
    cursor = db.conn.cursor()

    # Sample source images
    cursor.execute("""
        SELECT id, file_name, folder_path, image_type, mc_caption, ai_tags
        FROM files
        WHERE mc_caption IS NOT NULL AND mc_caption != ''
          AND ai_tags IS NOT NULL AND ai_tags != ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (count * 3,))
    columns = [d[0] for d in cursor.description]
    source_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    queries = []

    for row in source_rows:
        if len(queries) >= count:
            break

        tags = _parse_tags(row.get("ai_tags", ""))
        # Only pick concrete visible objects, not abstract adjectives
        concrete = [t for t in tags
                    if t.lower() not in SKIP_TAGS
                    and len(t) > 2
                    and not t.startswith("imagine_worker")  # system tags
                    and not any(c.isdigit() for c in t[:3])  # file codes
                    ]
        # Prefer tags that are in our known objects list
        known = [t for t in concrete if t.lower().replace("_", " ") in CONCRETE_OBJECTS
                 or t.lower().split("_")[0] in CONCRETE_OBJECTS]
        pool = known if len(known) >= 2 else concrete
        if len(pool) < 2:
            continue

        picks = random.sample(pool, 2)
        el1_en, el2_en = picks[0], picks[1]
        el1_ko = _tag_to_korean(el1_en)
        el2_ko = _tag_to_korean(el2_en)
        folder = _folder_short(row.get("folder_path", "") or "")

        # Build query
        if folder:
            query = f"{folder}에서 {el1_ko}과 {el2_ko} 있는 이미지"
        else:
            query = f"{el1_ko}이 있는 {el2_ko} 이미지"

        # Find ground truth: all images whose tags contain BOTH elements
        gt_ids = _find_ground_truth(cursor, el1_en, el2_en, judge_mode)

        if len(gt_ids) < 2:
            # Too few matches — try with just 1 element
            gt_ids = _find_ground_truth_single(cursor, el1_en)

        # Skip if GT too small or too large (>100 means query is too generic)
        if len(gt_ids) < 2 or len(gt_ids) > 100:
            continue

        queries.append({
            "source_id": row["id"],
            "file_name": row.get("file_name", "?"),
            "query": query,
            "elements_en": [el1_en, el2_en],
            "elements_ko": [el1_ko, el2_ko],
            "folder": folder,
            "gt_ids": gt_ids,
            "gt_count": len(gt_ids),
        })

    return queries


def _find_ground_truth(cursor, el1: str, el2: str, judge_mode: str) -> Set[int]:
    """Find all images whose MC tags/caption contain both elements."""
    # Keyword matching on ai_tags and mc_caption
    el1_lower = el1.lower()
    el2_lower = el2.lower()

    cursor.execute("""
        SELECT id, ai_tags, mc_caption
        FROM files
        WHERE mc_caption IS NOT NULL
          AND (LOWER(ai_tags) LIKE ? OR LOWER(mc_caption) LIKE ?)
          AND (LOWER(ai_tags) LIKE ? OR LOWER(mc_caption) LIKE ?)
    """, (
        f"%{el1_lower}%", f"%{el1_lower}%",
        f"%{el2_lower}%", f"%{el2_lower}%",
    ))

    gt_ids = set()
    for row_id, tags_raw, caption in cursor.fetchall():
        tags_text = (tags_raw or "").lower()
        caption_text = (caption or "").lower()
        combined = tags_text + " " + caption_text

        # Verify both elements are genuinely present (not substring false positives)
        has_el1 = el1_lower in combined
        has_el2 = el2_lower in combined

        if has_el1 and has_el2:
            gt_ids.add(row_id)

    return gt_ids


def _find_ground_truth_single(cursor, el: str) -> Set[int]:
    """Fallback: find images with at least one element."""
    el_lower = el.lower()
    cursor.execute("""
        SELECT id FROM files
        WHERE mc_caption IS NOT NULL
          AND (LOWER(ai_tags) LIKE ? OR LOWER(mc_caption) LIKE ?)
        LIMIT 200
    """, (f"%{el_lower}%", f"%{el_lower}%"))
    return {row[0] for row in cursor.fetchall()}


# ── Search Functions ───────────────────────────────────────────────────


def search_vv(searcher, query, top_k):
    try:
        return [r["id"] for r in searcher.vector_search(query, top_k=top_k, threshold=0.0)]
    except Exception:
        return []


def search_mv(searcher, query, top_k):
    try:
        return [r["id"] for r in searcher.text_vector_search(query, top_k=top_k, threshold=0.0)]
    except Exception:
        return []


def search_fts(searcher, query, top_k):
    keywords = [w.strip() for w in query.replace(",", " ").split() if len(w.strip()) > 1]
    if not keywords:
        return []
    try:
        return [r["id"] for r in searcher.fts_search(keywords[:10], top_k=top_k)]
    except Exception:
        return []


def search_triaxis(searcher, query, top_k):
    try:
        return [r["id"] for r in searcher.triaxis_search(
            query, top_k=top_k, threshold=0.0, use_codex=False
        )]
    except Exception:
        return []


# ── Evaluation ─────────────────────────────────────────────────────────


def evaluate_precision(
    name: str, search_fn, queries: List[dict],
    k_values: List[int] = [3, 5, 10],
) -> dict:
    """
    Evaluate Precision@K: what fraction of top-K results are relevant?
    """
    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}
    per_query = []
    n = len(queries)

    for i, q in enumerate(queries):
        gt_ids = q["gt_ids"]
        ranked_ids = search_fn(q["query"])

        pq = {"file": q["file_name"], "query": q["query"][:60], "gt_size": len(gt_ids)}

        for k in k_values:
            top_k_ids = set(ranked_ids[:k])
            relevant_in_k = len(top_k_ids & gt_ids)
            p_at_k = relevant_in_k / k if k > 0 else 0
            r_at_k = relevant_in_k / len(gt_ids) if gt_ids else 0
            precision_sums[k] += p_at_k
            recall_sums[k] += r_at_k
            pq[f"P@{k}"] = round(p_at_k, 3)
            pq[f"hits@{k}"] = relevant_in_k

        per_query.append(pq)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"      [{i+1}/{n}]")

    mean_precision = {f"P@{k}": round(precision_sums[k] / n, 4) for k in k_values}
    mean_recall = {f"Recall@{k}": round(recall_sums[k] / n, 4) for k in k_values}

    return {
        "precision": mean_precision,
        "recall": mean_recall,
        "per_query": per_query,
    }


# ── Report ─────────────────────────────────────────────────────────────


def score_grade(pct):
    if pct >= 85: return "A"
    elif pct >= 70: return "B"
    elif pct >= 55: return "C"
    elif pct >= 40: return "D"
    else: return "F"


def build_report(all_results, queries, db_size, timing):
    axes = ["vv", "mv", "fts", "triaxis"]

    # Key P@5 values
    tri_p5 = all_results["triaxis"]["precision"]["P@5"]
    best_single_p5 = max(all_results[a]["precision"]["P@5"] for a in ["vv", "mv", "fts"])
    lift = (tri_p5 / best_single_p5 - 1) * 100 if best_single_p5 > 0 else 0

    # GT stats
    gt_sizes = [q["gt_count"] for q in queries]
    avg_gt = np.mean(gt_sizes)
    median_gt = np.median(gt_sizes)

    lines = [
        "# Precision@K Benchmark — Real Search Quality",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries**: {len(queries)} (natural Korean)",
        f"**DB Size**: {db_size} images",
        f"**Avg ground truth set**: {avg_gt:.0f} images (median: {median_gt:.0f})",
        "",
        "## What This Measures",
        "",
        "Not \"did we find that exact file?\" but **\"are the search results relevant?\"**",
        "",
        "P@5 = 0.60 means: of the top 5 results, 3 are relevant to the query.",
        "This is how users experience search quality.",
        "",
        "## Results",
        "",
        "| Axis | P@3 | P@5 | P@10 | Role |",
        "|---|:-:|:-:|:-:|---|",
    ]

    role_map = {
        "vv": "Visual similarity (SigLIP2)",
        "mv": "Semantic meaning (Qwen3-Embed)",
        "fts": "Keyword match (FTS5 BM25)",
        "triaxis": "**VV + MV + FTS → RRF**",
    }

    for axis in axes:
        r = all_results[axis]
        p = r["precision"]
        bold = "**" if axis == "triaxis" else ""
        grade = score_grade(p["P@5"] * 100)
        lines.append(
            f"| {bold}{axis.upper()}{bold} | "
            f"{bold}{p['P@3']:.3f}{bold} | "
            f"{bold}{p['P@5']:.3f} ({grade}){bold} | "
            f"{bold}{p['P@10']:.3f}{bold} | "
            f"{role_map[axis]} |"
        )

    lines.extend([
        "",
        f"**Fusion Lift P@5**: {best_single_p5:.3f} → {tri_p5:.3f} (**{lift:+.1f}%**)",
        "",
        "## Interpretation",
        "",
        f"Triaxis P@5 = {tri_p5:.3f} means:",
        f"- Of top 5 results, **{tri_p5*5:.1f} images are relevant** on average",
        f"- User sees {tri_p5*100:.0f}% relevant results in the first page",
        "",
    ])

    # Production level
    p5_pct = tri_p5 * 100
    if p5_pct >= 70:
        level = "Production-ready (엔터프라이즈급)"
    elif p5_pct >= 50:
        level = "Production (주력 검색 엔진 가능)"
    elif p5_pct >= 30:
        level = "초기 Production (탐색/브라우징)"
    else:
        level = "연구 프로토타입"
    lines.append(f"**Production Level**: {level}")

    # Per-query detail (top 20)
    lines.extend([
        "", "---", "",
        "## Per-Query Detail (top 20)",
        "",
        "| Query | GT Size | P@5 | Hits@5 | VV P@5 | MV P@5 | FTS P@5 |",
        "|---|:-:|:-:|:-:|:-:|:-:|:-:|",
    ])
    for i in range(min(20, len(queries))):
        tri_pq = all_results["triaxis"]["per_query"][i]
        vv_pq = all_results["vv"]["per_query"][i]
        mv_pq = all_results["mv"]["per_query"][i]
        fts_pq = all_results["fts"]["per_query"][i]
        lines.append(
            f"| {tri_pq['query'][:40]} | {tri_pq['gt_size']} | "
            f"**{tri_pq.get('P@5', 0):.2f}** | {tri_pq.get('hits@5', 0)} | "
            f"{vv_pq.get('P@5', 0):.2f} | {mv_pq.get('P@5', 0):.2f} | "
            f"{fts_pq.get('P@5', 0):.2f} |"
        )

    # Query examples
    lines.extend(["", "---", "", "## Query Examples", ""])
    for q in queries[:5]:
        lines.append(f"- `{q['query']}` — GT: {q['gt_count']} images, "
                     f"elements: {q['elements_en']}")

    # Timing
    lines.extend(["", "---", "", "## Timing", ""])
    for phase, t in timing.items():
        lines.append(f"- {phase}: {t:.1f}s")

    lines.append("")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Precision@K Benchmark — LLM-as-Judge")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--judge-mode", choices=["keyword", "llm"], default="keyword",
                        help="keyword=fast tag matching, llm=LLM judges relevance")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    random.seed(42)
    timing = {}

    print("=" * 70)
    print("  Precision@K Benchmark — Real Search Quality")
    print(f"  Judge mode: {args.judge_mode}")
    print("=" * 70)

    # Load
    t0 = time.perf_counter()
    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    db = SQLiteDB()
    searcher = SqliteVectorSearch(db=db)
    timing["init"] = round(time.perf_counter() - t0, 1)

    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    db_size = cursor.fetchone()[0]

    # Generate queries + ground truth
    print("  Generating queries + ground truth...")
    t0 = time.perf_counter()
    queries = generate_queries_with_gt(db, args.count, args.judge_mode)
    timing["query_gen"] = round(time.perf_counter() - t0, 1)
    print(f"  Generated {len(queries)} queries")

    if len(queries) < 5:
        print("  ERROR: Too few valid queries.")
        sys.exit(1)

    # GT stats
    gt_sizes = [q["gt_count"] for q in queries]
    print(f"  Ground truth: avg={np.mean(gt_sizes):.0f}, "
          f"median={np.median(gt_sizes):.0f}, "
          f"min={min(gt_sizes)}, max={max(gt_sizes)}")

    # Sample
    print(f"\n  Sample queries:")
    for q in queries[:3]:
        print(f"    \"{q['query']}\"")
        print(f"    elements: {q['elements_en']} → GT: {q['gt_count']} images")

    # Evaluate
    axes = {
        "vv": lambda q: search_vv(searcher, q, args.top_k),
        "mv": lambda q: search_mv(searcher, q, args.top_k),
        "fts": lambda q: search_fts(searcher, q, args.top_k),
        "triaxis": lambda q: search_triaxis(searcher, q, args.top_k),
    }
    all_results = {}

    for axis_name, fn in axes.items():
        label = {"vv": "VV", "mv": "MV", "fts": "FTS", "triaxis": "TRIAXIS (full pipeline)"}[axis_name]
        print(f"\n  {label}:")
        t0 = time.perf_counter()
        all_results[axis_name] = evaluate_precision(axis_name, fn, queries)
        elapsed = round(time.perf_counter() - t0, 1)
        timing[axis_name] = elapsed
        p = all_results[axis_name]["precision"]
        print(f"    P@3={p['P@3']:.3f}  P@5={p['P@5']:.3f}  P@10={p['P@10']:.3f}  ({elapsed}s)")

    # Report
    report_md = build_report(all_results, queries, db_size, timing)

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"precision_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "query_count": len(queries),
        "db_size": db_size,
        "judge_mode": args.judge_mode,
        "axes": {
            axis: {"precision": r["precision"], "recall": r["recall"]}
            for axis, r in all_results.items()
        },
        "timing": timing,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    # Console summary
    tri_p5 = all_results["triaxis"]["precision"]["P@5"]
    best_p5 = max(all_results[a]["precision"]["P@5"] for a in ["vv", "mv", "fts"])
    lift = (tri_p5 / best_p5 - 1) * 100 if best_p5 > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Precision@K (search result relevance)")
    print(f"{'=' * 70}")
    print(f"  {'Axis':<12} {'P@3':>8} {'P@5':>8} {'P@10':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    for axis in ["vv", "mv", "fts", "triaxis"]:
        p = all_results[axis]["precision"]
        marker = " ★" if axis == "triaxis" else ""
        print(f"  {axis.upper():<12} {p['P@3']:>7.3f} {p['P@5']:>7.3f} {p['P@10']:>7.3f}{marker}")
    print(f"{'─' * 70}")
    print(f"  Triaxis P@5 = {tri_p5:.3f} → top-5 중 {tri_p5*5:.1f}개가 관련 이미지")
    print(f"  Fusion Lift: {lift:+.1f}% over best single axis")
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
