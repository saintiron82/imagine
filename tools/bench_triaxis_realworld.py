#!/usr/bin/env python3
"""
Triaxis Real-World Search Benchmark
=====================================
Tests the ACTUAL user search experience using natural Korean queries
that mirror how artists really search their asset libraries.

Real user patterns:
  "코난폴더에서 산과 강이 있는 이미지"
  "교실에 창문이 있는 이미지"
  "다윈즈게임 도시 배경중 밤하늘"

Query generation:
  1. Pick a folder from DB (= user knows which project folder)
  2. Pick 2 visual elements from MC caption/tags (= what user remembers)
  3. Generate natural Korean query: "{폴더} 폴더에서 {요소1}과 {요소2} 있는 이미지"

Evaluates: VV / MV / FTS / Triaxis(full pipeline) at each level.

Usage:
    python tools/bench_triaxis_realworld.py
    python tools/bench_triaxis_realworld.py --count 50
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ── Natural Korean Query Templates ────────────────────────────────────

# User says folder name naturally (last meaningful part of path)
# + describes 2 visual elements they remember

TEMPLATES_FOLDER_SCOPED = [
    "{folder}폴더에서 {el1}이 있는 {el2} 이미지",
    "{folder}에서 {el1}과 {el2} 있는 이미지",
    "{folder} 중에 {el1} {el2} 찾아줘",
    "{folder}폴더 {el1} {el2}",
]

TEMPLATES_CONTENT_ONLY = [
    "{el1}이 있는 {el2} 이미지",
    "{el1}과 {el2} 배경",
    "{el1} {el2} 찾아줘",
    "{el1} {el2}",
]

TEMPLATES_WITH_EXCLUDE = [
    "{folder}에서 {el1} {el2} 찾아줘 {neg} 빼고",
    "{el1} {el2} 이미지 {neg} 제외",
]

# Visual element mapping: English tag → natural Korean expression
TAG_KO_MAP = {
    "sky": "하늘", "mountain": "산", "river": "강", "lake": "호수",
    "window": "창문", "door": "문", "room": "방", "classroom": "교실",
    "tree": "나무", "forest": "숲", "flower": "꽃", "garden": "정원",
    "building": "건물", "house": "집", "city": "도시", "street": "거리",
    "car": "자동차", "bridge": "다리", "stairs": "계단", "wall": "벽",
    "desk": "책상", "chair": "의자", "bed": "침대", "sofa": "소파",
    "lamp": "램프", "table": "테이블", "shelf": "선반", "bookshelf": "책장",
    "night": "밤", "sunset": "일몰", "morning": "아침", "rain": "비",
    "snow": "눈", "star": "별", "moon": "달", "sun": "태양",
    "anime": "애니메이션", "sketch": "스케치", "character": "캐릭터",
    "girl": "소녀", "boy": "소년", "figure": "인물", "crowd": "군중",
    "sword": "검", "weapon": "무기", "armor": "갑옷",
    "park": "공원", "beach": "해변", "ocean": "바다", "cliff": "절벽",
    "kitchen": "주방", "bathroom": "욕실", "hallway": "복도",
    "curtain": "커튼", "painting": "그림", "mirror": "거울",
    "plant": "식물", "grass": "잔디", "cloud": "구름",
    "fire": "불", "water": "물", "light": "빛", "shadow": "그림자",
    "rooftop": "옥상", "balcony": "발코니", "fence": "울타리",
}

# Tags to skip (too generic to be a useful visual element)
SKIP_TAGS = {
    "anime", "illustration", "sketch", "digital", "3d_render",
    "painterly", "realistic", "fantasy", "abstract", "cartoon",
    "cool_palette", "warm_palette", "neutral_color_palette",
    "atmospheric", "dynamic", "dramatic", "serene", "ethereal",
}


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


def _folder_short(folder_path: str) -> str:
    """Extract the human-meaningful folder name."""
    if not folder_path:
        return ""
    parts = [p for p in folder_path.split("/") if p]
    # Skip generic parts like "발송", "외주 회사", "bg"
    meaningful = [p for p in parts if len(p) > 2 and p not in ("bg", "발송", "외주 회사")]
    if meaningful:
        return meaningful[-1]  # Last meaningful part
    return parts[-1] if parts else ""


def _tag_to_korean(tag: str) -> str:
    """Convert tag to natural Korean if mapping exists."""
    tag_lower = tag.lower().replace("_", " ")
    # Direct match
    if tag_lower in TAG_KO_MAP:
        return TAG_KO_MAP[tag_lower]
    # Partial match (first word)
    first_word = tag_lower.split()[0] if " " in tag_lower else tag_lower
    if first_word in TAG_KO_MAP:
        return TAG_KO_MAP[first_word]
    # Keep original (might be Korean already or specific enough)
    return tag


def _pick_visual_elements(tags: List[str], n: int = 2) -> List[str]:
    """Pick n concrete visual elements, translate to Korean."""
    concrete = [t for t in tags if t.lower() not in SKIP_TAGS and len(t) > 2]
    if len(concrete) < n:
        concrete = tags[:n]
    picks = random.sample(concrete, min(n, len(concrete)))
    return [_tag_to_korean(t) for t in picks]


def generate_realworld_queries(rows: List[dict]) -> List[dict]:
    """Generate natural Korean queries mimicking real user behavior."""
    queries = []

    for row in rows:
        tags = _parse_tags(row.get("ai_tags", ""))
        caption = row.get("mc_caption", "") or ""
        folder = _folder_short(row.get("folder_path", "") or "")

        if not tags or len(tags) < 2:
            continue

        elements = _pick_visual_elements(tags)
        if len(elements) < 2:
            continue

        el1, el2 = elements[0], elements[1]

        # Pick a negative element (from another image's tags)
        neg_candidates = [t for t in tags if t.lower() not in SKIP_TAGS
                          and _tag_to_korean(t) not in (el1, el2)]
        neg = _tag_to_korean(random.choice(neg_candidates)) if neg_candidates else ""

        # Level 1: content only (no folder)
        tmpl = random.choice(TEMPLATES_CONTENT_ONLY)
        content_q = tmpl.format(el1=el1, el2=el2)

        # Level 2: folder + content
        if folder:
            tmpl = random.choice(TEMPLATES_FOLDER_SCOPED)
            folder_q = tmpl.format(folder=folder, el1=el1, el2=el2)
        else:
            folder_q = content_q  # fallback

        # Level 3: folder + content + exclude
        if folder and neg:
            tmpl = random.choice(TEMPLATES_WITH_EXCLUDE)
            complex_q = tmpl.format(folder=folder, el1=el1, el2=el2, neg=neg)
        else:
            complex_q = folder_q

        queries.append({
            "file_id": row["id"],
            "file_name": row.get("file_name", "?"),
            "folder": folder,
            "elements": [el1, el2],
            "queries": {
                "content": content_q,
                "folder": folder_q,
                "complex": complex_q,
            },
        })

    return queries


# ── Search Functions ───────────────────────────────────────────────────


def search_vv(searcher, query: str, top_k: int) -> List[int]:
    try:
        return [r["id"] for r in searcher.vector_search(query, top_k=top_k, threshold=0.0)]
    except Exception:
        return []


def search_mv(searcher, query: str, top_k: int) -> List[int]:
    try:
        return [r["id"] for r in searcher.text_vector_search(query, top_k=top_k, threshold=0.0)]
    except Exception:
        return []


def search_fts(searcher, query: str, top_k: int) -> List[int]:
    keywords = [w.strip() for w in query.replace(",", " ").split() if len(w.strip()) > 1]
    if not keywords:
        return []
    try:
        return [r["id"] for r in searcher.fts_search(keywords[:10], top_k=top_k)]
    except Exception:
        return []


def search_triaxis(searcher, query: str, top_k: int) -> List[int]:
    try:
        return [r["id"] for r in searcher.triaxis_search(
            query, top_k=top_k, threshold=0.0, use_codex=False
        )]
    except Exception:
        return []


# ── Evaluation ─────────────────────────────────────────────────────────


def evaluate(name, search_fn, queries, level, k_values=[1, 3, 5, 10]):
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
        per_query.append({"file": q["file_name"], "rank": rank, "query": qtext[:70]})

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"      [{i+1}/{n}]")

    recall = {f"R@{k}": round(recall_hits[k] / n, 4) for k in k_values}
    mrr = round(float(np.mean(reciprocal_ranks)), 4)
    return {"recall": recall, "mrr": mrr, "per_query": per_query}


# ── Report ─────────────────────────────────────────────────────────────


def score_grade(pct):
    if pct >= 85: return "A"
    elif pct >= 70: return "B"
    elif pct >= 55: return "C"
    elif pct >= 40: return "D"
    else: return "F"


def build_report(all_results, query_count, db_size, timing, sample_queries):
    axes = ["vv", "mv", "fts", "triaxis"]
    levels = ["content", "folder", "complex"]

    lines = [
        "# Triaxis Real-World Search Benchmark",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries**: {query_count} (natural Korean, from MC data)",
        f"**DB Size**: {db_size} images",
        "",
        "## Query Examples (actual generated)",
        "",
        "| Level | Example |",
        "|---|---|",
    ]
    if sample_queries:
        for level in levels:
            # Find a good example
            for sq in sample_queries[:5]:
                q = sq["queries"][level]
                if len(q) > 10:
                    lines.append(f"| **{level}** | `{q[:65]}` |")
                    break

    for level in levels:
        results = all_results[level]
        sb5 = max(results[a]["recall"].get("R@5", 0) for a in ["vv", "mv", "fts"])
        t5 = results["triaxis"]["recall"].get("R@5", 0)
        lift5 = (t5 / sb5 - 1) * 100 if sb5 > 0 else 0

        lines.extend(["", "---", "", f"## {level.upper()}", ""])
        lines.append("| Axis | R@1 | R@3 | R@5 | R@10 | MRR |")
        lines.append("|---|:-:|:-:|:-:|:-:|:-:|")
        for axis in axes:
            r = results[axis]
            rc = r["recall"]
            p = "**" if axis == "triaxis" else ""
            lines.append(
                f"| {p}{axis.upper()}{p} | {p}{rc.get('R@1',0):.3f}{p} | "
                f"{p}{rc.get('R@3',0):.3f}{p} | {p}{rc.get('R@5',0):.3f}{p} | "
                f"{p}{rc.get('R@10',0):.3f}{p} | {p}{r['mrr']:.3f}{p} |"
            )
        lines.append(f"\n**Fusion Lift R@5**: {sb5:.3f} → {t5:.3f} (**{lift5:+.1f}%**)")

    # Summary
    lines.extend(["", "---", "", "## Summary", ""])
    lines.append("| Level | Triaxis R@1 | R@5 | R@10 | MRR | Best Single R@5 | Lift |")
    lines.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|")
    for level in levels:
        tri = all_results[level]["triaxis"]
        rc = tri["recall"]
        sb5 = max(all_results[level][a]["recall"]["R@5"] for a in ["vv", "mv", "fts"])
        t5 = rc["R@5"]
        lift = (t5 / sb5 - 1) * 100 if sb5 > 0 else 0
        lines.append(
            f"| **{level}** | {rc['R@1']:.3f} | {rc['R@5']:.3f} | "
            f"{rc['R@10']:.3f} | {tri['mrr']:.3f} | {sb5:.3f} | **{lift:+.1f}%** |"
        )

    # Per-query detail (folder level, top 20)
    if "folder" in all_results:
        results = all_results["folder"]
        lines.extend(["", "---", "", "## Per-Query Detail (folder, top 20)", ""])
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
    parser = argparse.ArgumentParser(description="Triaxis Real-World Search Benchmark")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    random.seed(42)
    timing = {}

    print("=" * 70)
    print("  Triaxis Real-World Benchmark")
    print("  (Natural Korean queries — actual user patterns)")
    print("=" * 70)

    # Load
    print("  Loading search engine...")
    t0 = time.perf_counter()
    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    db = SQLiteDB()
    searcher = SqliteVectorSearch(db=db)
    timing["init"] = round(time.perf_counter() - t0, 1)

    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT id, file_name, image_type, folder_path, art_style, mc_caption, ai_tags
        FROM files
        WHERE mc_caption IS NOT NULL AND mc_caption != ''
          AND ai_tags IS NOT NULL AND ai_tags != ''
        ORDER BY RANDOM() LIMIT ?
    """, (args.count * 2,))  # oversample to filter
    columns = [d[0] for d in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    cursor.execute("SELECT COUNT(*) FROM files")
    db_size = cursor.fetchone()[0]

    # Generate
    queries = generate_realworld_queries(rows)[:args.count]
    print(f"  Generated {len(queries)} queries (DB: {db_size})")

    if len(queries) < 5:
        print("  ERROR: Too few valid queries.")
        sys.exit(1)

    # Show samples
    print(f"\n  Sample queries:")
    for sq in queries[:3]:
        print(f"    {sq['file_name']} ({sq['folder']})")
        for level, q in sq["queries"].items():
            print(f"      {level:>8}: {q[:65]}")
        print()

    # Evaluate
    levels = ["content", "folder", "complex"]
    axes_config = {
        "vv": lambda q: search_vv(searcher, q, args.top_k),
        "mv": lambda q: search_mv(searcher, q, args.top_k),
        "fts": lambda q: search_fts(searcher, q, args.top_k),
        "triaxis": lambda q: search_triaxis(searcher, q, args.top_k),
    }
    all_results = {}

    for level in levels:
        print(f"\n{'=' * 70}")
        print(f"  Level: {level.upper()}")
        print(f"{'=' * 70}")
        all_results[level] = {}

        for axis_name, fn in axes_config.items():
            label = {"vv": "VV", "mv": "MV", "fts": "FTS", "triaxis": "TRIAXIS"}[axis_name]
            print(f"\n  {label}:")
            t0 = time.perf_counter()
            result = evaluate(axis_name, fn, queries, level)
            elapsed = round(time.perf_counter() - t0, 1)
            timing[f"{level}_{axis_name}"] = elapsed
            all_results[level][axis_name] = result
            rc = result["recall"]
            print(f"    R@1={rc['R@1']:.3f}  R@5={rc['R@5']:.3f}  "
                  f"R@10={rc['R@10']:.3f}  MRR={result['mrr']:.3f}  ({elapsed}s)")

    # Report
    report_md = build_report(all_results, len(queries), db_size, timing, queries)

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"triaxis_realworld_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "query_count": len(queries),
        "db_size": db_size,
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
    print(f"  RESULTS — Real-World Korean Queries")
    print(f"{'=' * 70}")
    print(f"  {'Level':<10} {'Tri R@1':>8} {'Tri R@5':>8} {'Tri R@10':>9} {'MRR':>8}  {'Best1 R@5':>10} {'Lift':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*9} {'─'*8}  {'─'*10} {'─'*8}")
    for level in levels:
        tri = all_results[level]["triaxis"]
        sb5 = max(all_results[level][a]["recall"]["R@5"] for a in ["vv", "mv", "fts"])
        t5 = tri["recall"]["R@5"]
        lift = (t5 / sb5 - 1) * 100 if sb5 > 0 else 0
        print(
            f"  {level:<10} {tri['recall']['R@1']:>7.3f} {tri['recall']['R@5']:>7.3f} "
            f"{tri['recall']['R@10']:>8.3f} {tri['mrr']:>7.3f}  "
            f"{sb5:>9.3f} {lift:>+7.1f}%"
        )
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
