#!/usr/bin/env python3
"""
Triaxis vs LLM — Search Quality Comparison
=============================================
Compares our Triaxis vector search against an LLM reading ALL MC data.

The LLM represents "perfect text understanding" — it reads every image's
caption and tags, then decides which are relevant. This is what you'd get
if you gave Claude/GPT all your image descriptions and asked it to find
matching images.

Triaxis is compared against this LLM baseline to measure how close
vector search gets to full language understanding.

Method:
  1. Generate natural Korean queries (same as realworld bench)
  2. LLM Judge: Feed ALL MC captions in batches to local LLM →
     "Is this image relevant to the query?" → LLM ground truth set
  3. Triaxis: Run same queries through full pipeline
  4. Compare: P@K of Triaxis results against LLM ground truth

Key metric: Agreement Rate — how often Triaxis and LLM agree on relevance

Usage:
    python tools/bench_vs_llm.py --count 20
    python tools/bench_vs_llm.py --count 20 --batch-size 50
"""

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"

# Reuse tag mapping
TAG_KO_MAP = {
    "sky": "하늘", "mountain": "산", "river": "강", "lake": "호수",
    "window": "창문", "door": "문", "room": "방", "classroom": "교실",
    "tree": "나무", "forest": "숲", "flower": "꽃", "garden": "정원",
    "building": "건물", "house": "집", "city": "도시", "street": "거리",
    "bridge": "다리", "stairs": "계단", "wall": "벽",
    "desk": "책상", "chair": "의자", "bed": "침대", "sofa": "소파",
    "night": "밤", "sunset": "일몰", "star": "별", "moon": "달",
    "park": "공원", "ocean": "바다", "kitchen": "주방",
    "curtain": "커튼", "painting": "그림", "mirror": "거울",
    "plant": "식물", "cloud": "구름", "fire": "불", "water": "물",
    "light": "빛", "rooftop": "옥상",
}
SKIP_TAGS = {
    "anime", "illustration", "sketch", "digital", "3d_render",
    "painterly", "realistic", "fantasy", "abstract", "cartoon",
    "cool_palette", "warm_palette", "neutral_color_palette",
    "atmospheric", "dynamic", "dramatic", "serene", "ethereal",
}


def _parse_tags(raw):
    if not raw: return []
    try:
        p = json.loads(raw)
        if isinstance(p, list): return [str(t).strip() for t in p if t]
    except Exception: pass
    return [t.strip() for t in raw.split(",") if t.strip()]


def _tag_ko(tag):
    t = tag.lower().replace("_", " ")
    return TAG_KO_MAP.get(t, TAG_KO_MAP.get(t.split()[0], tag) if " " in t else tag)


def _folder_short(fp):
    if not fp: return ""
    parts = [p for p in fp.split("/") if p and len(p) > 2 and p not in ("bg", "발송", "외주 회사")]
    return parts[-1] if parts else ""


# ── Query Generation ───────────────────────────────────────────────────


def generate_queries(db, count):
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT id, file_name, folder_path, mc_caption, ai_tags
        FROM files WHERE mc_caption IS NOT NULL AND ai_tags IS NOT NULL
        ORDER BY RANDOM() LIMIT ?
    """, (count * 3,))
    cols = [d[0] for d in cursor.description]
    rows = [dict(zip(cols, r)) for r in cursor.fetchall()]

    queries = []
    for row in rows:
        if len(queries) >= count: break
        tags = _parse_tags(row.get("ai_tags", ""))
        concrete = [t for t in tags if t.lower() not in SKIP_TAGS and len(t) > 2]
        if len(concrete) < 2: continue
        picks = random.sample(concrete, 2)
        el1, el2 = _tag_ko(picks[0]), _tag_ko(picks[1])
        folder = _folder_short(row.get("folder_path", ""))

        if folder:
            query = f"{folder}에서 {el1}과 {el2} 있는 이미지"
        else:
            query = f"{el1}이 있는 {el2} 이미지"

        queries.append({
            "source_id": row["id"],
            "file_name": row["file_name"],
            "query": query,
            "elements_en": picks,
            "elements_ko": [el1, el2],
        })
    return queries


# ── LLM Judge — Batch relevance scoring ────────────────────────────────


def _load_llm():
    """Load local LLM for judging (Qwen3.5-4B MLX)."""
    try:
        from mlx_lm import load, generate
        model_id = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
        print(f"    Loading LLM judge: {model_id}")
        model, tokenizer = load(model_id)
        return model, tokenizer, model_id
    except Exception as e:
        print(f"    LLM load failed: {e}")
        return None, None, None


def _llm_judge_batch(model, tokenizer, query: str, images_batch: List[dict]) -> Set[int]:
    """
    Ask LLM: "Which of these images are relevant to the query?"

    Input: query + list of {id, caption, tags}
    Output: set of relevant image IDs
    """
    from mlx_lm import generate as mlx_generate

    # Build image list text
    img_lines = []
    for img in images_batch:
        tags_str = ", ".join(img["tags"][:6]) if img["tags"] else ""
        img_lines.append(f'[{img["id"]}] {img["caption"][:80]} | tags: {tags_str}')

    images_text = "\n".join(img_lines)

    prompt_text = f"""You are an image search relevance judge.
Given a search query and a list of image descriptions, output ONLY the IDs of images that are relevant to the query.

Query: "{query}"

Images:
{images_text}

Output ONLY a JSON array of relevant image IDs. If none are relevant, output [].
Example: [123, 456, 789]

Relevant IDs:"""

    messages = [{"role": "user", "content": prompt_text}]
    tok = getattr(tokenizer, 'tokenizer', tokenizer)
    formatted = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    result = mlx_generate(
        model, tokenizer, prompt=formatted,
        max_tokens=200, verbose=False,
    )

    # Parse IDs from response
    text = result if isinstance(result, str) else str(result)
    try:
        # Find JSON array in response
        import re
        match = re.search(r'\[[\d,\s]*\]', text)
        if match:
            ids = json.loads(match.group())
            return {int(i) for i in ids}
    except Exception:
        pass
    return set()


def llm_judge_all(model, tokenizer, query: str, all_images: List[dict],
                  batch_size: int = 50) -> Set[int]:
    """Judge ALL images in batches, return set of relevant IDs."""
    relevant_ids = set()
    n = len(all_images)

    for start in range(0, n, batch_size):
        batch = all_images[start:start + batch_size]
        batch_relevant = _llm_judge_batch(model, tokenizer, query, batch)
        # Validate IDs exist in this batch
        batch_ids = {img["id"] for img in batch}
        valid = batch_relevant & batch_ids
        relevant_ids |= valid

    return relevant_ids


# ── Search Functions ───────────────────────────────────────────────────


def search_triaxis(searcher, query, top_k):
    try:
        return [r["id"] for r in searcher.triaxis_search(
            query, top_k=top_k, threshold=0.0, use_codex=False
        )]
    except Exception:
        return []


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


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Triaxis vs LLM Search Comparison")
    parser.add_argument("--count", type=int, default=15,
                        help="Number of queries (LLM judges all 11K images per query)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Images per LLM batch")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    random.seed(42)
    timing = {}

    print("=" * 70)
    print("  Triaxis vs LLM — Who Finds Better?")
    print("=" * 70)

    # Load DB + search
    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    db = SQLiteDB()
    searcher = SqliteVectorSearch(db=db)

    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files WHERE mc_caption IS NOT NULL")
    db_size = cursor.fetchone()[0]

    # Load ALL MC data for LLM judging
    print(f"  Loading all MC data ({db_size} images)...")
    t0 = time.perf_counter()
    cursor.execute("""
        SELECT id, file_name, mc_caption, ai_tags
        FROM files WHERE mc_caption IS NOT NULL AND mc_caption != ''
    """)
    cols = [d[0] for d in cursor.description]
    all_images = []
    for row in cursor.fetchall():
        r = dict(zip(cols, row))
        all_images.append({
            "id": r["id"],
            "file_name": r["file_name"],
            "caption": r.get("mc_caption", ""),
            "tags": _parse_tags(r.get("ai_tags", "")),
        })
    timing["load_mc"] = round(time.perf_counter() - t0, 1)
    print(f"  Loaded {len(all_images)} images in {timing['load_mc']}s")

    # Generate queries
    queries = generate_queries(db, args.count)
    print(f"  Generated {len(queries)} queries")
    for q in queries[:3]:
        print(f"    \"{q['query']}\"")

    # Load LLM judge
    print(f"\n{'─' * 70}")
    print("  Loading LLM Judge...")
    t0 = time.perf_counter()
    model, tokenizer, llm_id = _load_llm()
    timing["llm_load"] = round(time.perf_counter() - t0, 1)

    if model is None:
        print("  FATAL: Cannot load LLM judge")
        sys.exit(1)

    # Run comparison
    results = []
    total_batches = (len(all_images) + args.batch_size - 1) // args.batch_size

    for qi, q in enumerate(queries):
        print(f"\n{'─' * 70}")
        print(f"  Query [{qi+1}/{len(queries)}]: \"{q['query']}\"")

        # LLM judges all images
        print(f"    LLM judging {len(all_images)} images ({total_batches} batches)...")
        t0 = time.perf_counter()
        llm_relevant = llm_judge_all(model, tokenizer, q["query"], all_images, args.batch_size)
        llm_time = round(time.perf_counter() - t0, 1)
        print(f"    LLM found {len(llm_relevant)} relevant images ({llm_time}s)")

        # Triaxis search
        t0 = time.perf_counter()
        triaxis_ids = search_triaxis(searcher, q["query"], args.top_k)
        tri_time = round(time.perf_counter() - t0, 1)

        # VV / MV individual
        vv_ids = search_vv(searcher, q["query"], args.top_k)
        mv_ids = search_mv(searcher, q["query"], args.top_k)

        # Calculate P@K against LLM ground truth
        def calc_pk(ranked_ids, gt_set, k):
            top_k_set = set(ranked_ids[:k])
            hits = len(top_k_set & gt_set)
            return hits / k if k > 0 else 0

        k_values = [3, 5, 10]
        tri_pk = {k: calc_pk(triaxis_ids, llm_relevant, k) for k in k_values}
        vv_pk = {k: calc_pk(vv_ids, llm_relevant, k) for k in k_values}
        mv_pk = {k: calc_pk(mv_ids, llm_relevant, k) for k in k_values}

        # LLM's own P@K (= its top-N are by definition relevant)
        # LLM doesn't rank, so P@K = min(|relevant|, K) / K
        llm_pk = {k: min(len(llm_relevant), k) / k for k in k_values}

        print(f"    P@5: LLM={llm_pk[5]:.2f}  Triaxis={tri_pk[5]:.2f}  "
              f"VV={vv_pk[5]:.2f}  MV={mv_pk[5]:.2f}")

        # Agreement: how many of Triaxis top-K did LLM also find relevant?
        tri_top5 = set(triaxis_ids[:5])
        agreement = len(tri_top5 & llm_relevant) / min(5, len(tri_top5)) if tri_top5 else 0

        results.append({
            "query": q["query"],
            "file_name": q["file_name"],
            "llm_relevant_count": len(llm_relevant),
            "triaxis_pk": tri_pk,
            "vv_pk": vv_pk,
            "mv_pk": mv_pk,
            "llm_pk": llm_pk,
            "agreement_at_5": round(agreement, 3),
            "llm_time": llm_time,
            "tri_time": tri_time,
        })

    # Unload LLM
    del model, tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    # Aggregate
    n = len(results)
    avg = lambda key, k: round(np.mean([r[key][k] for r in results]), 4)
    avg_agreement = round(np.mean([r["agreement_at_5"] for r in results]), 4)

    # Report
    lines = [
        "# Triaxis vs LLM — Search Quality Comparison",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Queries**: {n}",
        f"**DB Size**: {db_size} images",
        f"**LLM Judge**: {llm_id}",
        f"**LLM judges ALL {db_size} images per query** (not sampling)",
        "",
        "## How to Read",
        "",
        "LLM reads every image's caption/tags and decides relevance.",
        "This is the \"give Claude all your images\" scenario.",
        "P@K measures how many of the search results the LLM agrees are relevant.",
        "",
        "## Results",
        "",
        "| System | P@3 | P@5 | P@10 |",
        "|---|:-:|:-:|:-:|",
        f"| **LLM (ceiling)** | **{avg('llm_pk', 3):.3f}** | **{avg('llm_pk', 5):.3f}** | **{avg('llm_pk', 10):.3f}** |",
        f"| **TRIAXIS** | **{avg('triaxis_pk', 3):.3f}** | **{avg('triaxis_pk', 5):.3f}** | **{avg('triaxis_pk', 10):.3f}** |",
        f"| MV only | {avg('mv_pk', 3):.3f} | {avg('mv_pk', 5):.3f} | {avg('mv_pk', 10):.3f} |",
        f"| VV only | {avg('vv_pk', 3):.3f} | {avg('vv_pk', 5):.3f} | {avg('vv_pk', 10):.3f} |",
        "",
        f"**Agreement@5**: {avg_agreement:.3f} — {avg_agreement*100:.0f}% of Triaxis top-5 results are also approved by LLM",
        "",
    ]

    # Ratio
    tri_p5 = avg('triaxis_pk', 5)
    llm_p5 = avg('llm_pk', 5)
    ratio = tri_p5 / llm_p5 if llm_p5 > 0 else 0
    lines.extend([
        "## Triaxis vs LLM Ratio",
        "",
        f"Triaxis P@5 / LLM P@5 = {tri_p5:.3f} / {llm_p5:.3f} = **{ratio:.1%}**",
        "",
        f"Our 1B vector search achieves **{ratio:.0%} of LLM-level search quality**.",
        "",
    ])

    # Per-query detail
    lines.extend(["---", "", "## Per-Query Detail", ""])
    lines.append("| Query | LLM Found | Tri P@5 | LLM P@5 | Agreement |")
    lines.append("|---|:-:|:-:|:-:|:-:|")
    for r in results:
        lines.append(
            f"| {r['query'][:45]} | {r['llm_relevant_count']} | "
            f"{r['triaxis_pk'][5]:.2f} | {r['llm_pk'][5]:.2f} | "
            f"{r['agreement_at_5']:.2f} |"
        )

    lines.append("")
    report_md = "\n".join(lines)

    # Save
    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"vs_llm_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "query_count": n, "db_size": db_size, "llm": llm_id,
        "avg_triaxis_p5": tri_p5, "avg_llm_p5": llm_p5,
        "ratio": round(ratio, 4), "avg_agreement": avg_agreement,
        "per_query": results, "timing": timing,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    # Console
    print(f"\n{'=' * 70}")
    print(f"  Triaxis vs LLM — Results")
    print(f"{'=' * 70}")
    print(f"  LLM P@5:      {llm_p5:.3f} (ceiling)")
    print(f"  Triaxis P@5:  {tri_p5:.3f}")
    print(f"  Ratio:        {ratio:.1%} of LLM quality")
    print(f"  Agreement@5:  {avg_agreement:.1%}")
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
