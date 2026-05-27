#!/usr/bin/env python3
"""LLM rejudge of a precision benchmark result.

The shipped bench uses substring keyword matching to build GT, which
misses semantic / multilingual / visual-but-not-tagged relevance. This
tool walks an existing precision_*.json, asks a local LLM to judge
whether each non-GT item in top-K is actually relevant to the query
(given its MC caption + tags), and re-computes P@K.

Usage:
    .venv/bin/python tools/bench_llm_rejudge.py \\
        benchmarks/results/precision_20260528_phaseCD.json \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── LLM judge backend ───────────────────────────────────────────────


_mlx_model = None
_mlx_tok = None


def _load_mlx() -> bool:
    """Load the MLX LLM lazily. Returns True on success."""
    global _mlx_model, _mlx_tok
    if _mlx_model is not None:
        return True
    try:
        from mlx_lm import load
        model_id = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
        _mlx_model, _mlx_tok = load(model_id)
        return True
    except Exception as e:
        print(f"[rejudge] MLX load failed: {e}", file=sys.stderr)
        return False


def _judge_batch(query: str, items: list[dict]) -> dict[int, str]:
    """Ask the LLM which items are relevant. Returns {file_id: 'yes'|'no'}."""
    if not items:
        return {}
    if not _load_mlx():
        return {it["id"]: "skip" for it in items}

    from mlx_lm import generate

    # Build a single-prompt batch (LLM returns JSON list of yes/no).
    item_lines = []
    for it in items:
        cap = (it.get("mc_caption") or "")[:200]
        tags = (it.get("ai_tags") or "")[:200]
        item_lines.append(f"  - id={it['id']}: caption='{cap}' tags='{tags}'")
    items_block = "\n".join(item_lines)

    prompt = (
        "You are judging image-search relevance. Each candidate has a MC caption "
        "and tag string. Korean queries may map to English caption/tag content.\n\n"
        f"USER QUERY (Korean): {query}\n\n"
        "CANDIDATES:\n"
        f"{items_block}\n\n"
        "For each id, decide if the image is RELEVANT to the user query. "
        "Be generous on visual/semantic match — if the caption/tags describe a scene that fits the query intent, answer yes. "
        "Reply with ONLY a JSON object mapping id (integer) to 'yes' or 'no'. No prose.\n"
        "Example: {\"123\": \"yes\", \"456\": \"no\"}\n"
        "Answer:"
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        text = _mlx_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        out = generate(
            _mlx_model, _mlx_tok, prompt=text,
            max_tokens=300, verbose=False,
        )
        # Find the first JSON object in the response
        match = re.search(r"\{[^{}]*\}", out, re.DOTALL)
        if not match:
            return {it["id"]: "skip" for it in items}
        parsed = json.loads(match.group(0))
        result: dict[int, str] = {}
        for k, v in parsed.items():
            try:
                fid = int(k)
            except (TypeError, ValueError):
                continue
            val = str(v).strip().lower()
            result[fid] = "yes" if val.startswith("y") else "no"
        # Default 'skip' for any items the LLM didn't answer
        for it in items:
            result.setdefault(it["id"], "skip")
        return result
    except Exception as e:
        print(f"[rejudge] judge_batch failed: {e}", file=sys.stderr)
        return {it["id"]: "skip" for it in items}


# ── DB lookup ───────────────────────────────────────────────────────


def _fetch_items(db: sqlite3.Connection, file_ids: list[int]) -> dict[int, dict]:
    if not file_ids:
        return {}
    placeholders = ",".join("?" * len(file_ids))
    cur = db.execute(
        f"SELECT id, mc_caption, ai_tags FROM files WHERE id IN ({placeholders})",
        file_ids,
    )
    return {row[0]: {"id": row[0], "mc_caption": row[1], "ai_tags": row[2]}
            for row in cur.fetchall()}


# ── Rejudge ─────────────────────────────────────────────────────────


def rejudge(report_path: Path, top_k: int) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    triaxis = report.get("axes", {}).get("triaxis", {})
    per_query = triaxis.get("per_query") or []
    if not per_query:
        raise SystemExit("report has no triaxis per_query data")

    db_path = REPO / "imageparser.db"
    db = sqlite3.connect(str(db_path))

    # Tighten ranked_ids: the bench may have stored full ranking but we
    # judge only the top-K slice.
    queries_processed = 0
    sum_orig_p_at_k = 0.0
    sum_llm_p_at_k = 0.0
    promotions = 0
    per_query_rejudge = []

    for q in per_query:
        ranked = list(q.get("ranked_ids") or [])[:top_k]
        if not ranked:
            continue

        items = _fetch_items(db, ranked)
        # Original GT — we'll recover it from the per_query "hits@K" field.
        # Better: use the precision pickle we don't have. So instead we run
        # rejudge on EVERY top-K item, comparing the LLM's verdict to the
        # original `relevant_in_top_k` count.
        judged = _judge_batch(q.get("query", ""), [items[fid] for fid in ranked if fid in items])

        # P@K (original) — recompute from hits@K stored in the report.
        orig_hits = q.get(f"hits@{top_k}")
        if orig_hits is None:
            orig_hits = round(q.get(f"P@{top_k}", 0.0) * top_k)
        orig_p = orig_hits / top_k

        # P@K (LLM) — count yes
        llm_hits = sum(1 for fid in ranked if judged.get(fid) == "yes")
        llm_p = llm_hits / top_k

        delta = llm_hits - orig_hits
        if delta > 0:
            promotions += delta

        sum_orig_p_at_k += orig_p
        sum_llm_p_at_k += llm_p
        queries_processed += 1
        per_query_rejudge.append({
            "query": q.get("query"),
            f"P@{top_k}_orig": round(orig_p, 3),
            f"P@{top_k}_llm": round(llm_p, 3),
            "delta_hits": delta,
            "judged": {str(fid): judged.get(fid) for fid in ranked},
        })

        print(
            f"  [{queries_processed}/{len(per_query)}] '{(q.get('query') or '')[:50]}': "
            f"hits={orig_hits}→{llm_hits} (Δ={delta:+d})",
            flush=True,
        )

    n = max(1, queries_processed)
    summary = {
        "queries_processed": queries_processed,
        "top_k": top_k,
        f"P@{top_k}_keyword": round(sum_orig_p_at_k / n, 4),
        f"P@{top_k}_llm": round(sum_llm_p_at_k / n, 4),
        "absolute_lift": round((sum_llm_p_at_k - sum_orig_p_at_k) / n, 4),
        "relative_lift_pct": round(
            100.0 * (sum_llm_p_at_k - sum_orig_p_at_k) / max(sum_orig_p_at_k, 1e-9),
            1,
        ),
        "promoted_items": promotions,
        "per_query": per_query_rejudge,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM rejudge of precision bench")
    parser.add_argument("report", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.report.exists():
        print(f"[rejudge] not found: {args.report}", file=sys.stderr)
        return 2

    summary = rejudge(args.report, args.top_k)

    print()
    print("=" * 70)
    print("  LLM REJUDGE — summary")
    print("=" * 70)
    print(f"  queries           : {summary['queries_processed']}")
    print(f"  P@{args.top_k} (keyword GT) : {summary[f'P@{args.top_k}_keyword']}")
    print(f"  P@{args.top_k} (LLM judge) : {summary[f'P@{args.top_k}_llm']}")
    print(f"  absolute lift     : {summary['absolute_lift']:+.4f}")
    print(f"  relative lift     : {summary['relative_lift_pct']:+.1f}%")
    print(f"  promoted items    : {summary['promoted_items']}")
    print("=" * 70)

    out = args.output or args.report.with_name(
        args.report.stem + f"_llm_rejudge_k{args.top_k}.json"
    )
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
