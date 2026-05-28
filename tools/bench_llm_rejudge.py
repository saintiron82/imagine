#!/usr/bin/env python3
"""LLM rejudge of a precision benchmark result.

The shipped bench uses substring keyword matching to build GT, which
misses semantic / multilingual / visual-but-not-tagged relevance. This
tool walks an existing precision_*.json, asks an LLM to judge whether
each top-K item is actually relevant to the query (given its MC caption
+ tags), and re-computes P@K.

Two backends in order of preference:
  1. agentcli (Codex CLI) — smarter, session-reused via alias.
  2. Local MLX — fallback when agentcli/codex isn't available.

Usage:
    .venv/bin/python tools/bench_llm_rejudge.py \\
        benchmarks/results/precision_20260528_phaseCD.json \\
        --top-k 5 \\
        [--backend agentcli|mlx]
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


# ── LLM judge backends ──────────────────────────────────────────────


_mlx_model = None
_mlx_tok = None
_agentcli_client = None
_AGENTCLI_TRIED = False


def _load_mlx() -> bool:
    """Load the MLX LLM lazily. Returns True on success."""
    global _mlx_model, _mlx_tok
    if _mlx_model is not None:
        return True
    try:
        from mlx_lm import load
        # Larger model is needed for reliable batch-JSON judgement;
        # Qwen3.5-9B 4-bit fits in ~6GB RAM and handles the format much
        # better than the 4B variant. Override via env if desired.
        import os
        model_id = os.environ.get(
            "IMAGINE_REJUDGE_MLX_MODEL",
            "mlx-community/Qwen3.5-9B-MLX-4bit",
        )
        _mlx_model, _mlx_tok = load(model_id)
        return True
    except Exception as e:
        print(f"[rejudge] MLX load failed: {e}", file=sys.stderr)
        return False


def _load_agentcli():
    """Lazy-init agentcli LLMClient for Codex provider. Returns client or None."""
    global _agentcli_client, _AGENTCLI_TRIED
    if _AGENTCLI_TRIED:
        return _agentcli_client
    _AGENTCLI_TRIED = True
    try:
        from agentcli import LLMClient, MemoryStore
        _agentcli_client = LLMClient(store=MemoryStore())
        return _agentcli_client
    except Exception as e:
        print(f"[rejudge] agentcli load failed: {e}", file=sys.stderr)
        return None


def _build_judge_prompt(query: str, items: list[dict]) -> str:
    item_lines = []
    for it in items:
        cap = (it.get("mc_caption") or "")[:200]
        tags = (it.get("ai_tags") or "")[:200]
        item_lines.append(f"  - id={it['id']}: caption='{cap}' tags='{tags}'")
    items_block = "\n".join(item_lines)

    return (
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


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced top-level {...} from arbitrary text.

    Handles nested braces, trailing prose, and ```json``` fences.
    Returns the JSON substring or None.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_judgement(out: str, items: list[dict]) -> dict[int, str]:
    blob = _extract_first_json_object(out)
    if blob is None:
        return {it["id"]: "skip" for it in items}
    try:
        parsed = json.loads(blob)
    except Exception:
        return {it["id"]: "skip" for it in items}
    result: dict[int, str] = {}
    if not isinstance(parsed, dict):
        return {it["id"]: "skip" for it in items}
    for k, v in parsed.items():
        try:
            fid = int(k)
        except (TypeError, ValueError):
            continue
        val = str(v).strip().lower()
        result[fid] = "yes" if val.startswith("y") else "no"
    for it in items:
        result.setdefault(it["id"], "skip")
    return result


def _judge_batch_agentcli(query: str, items: list[dict]) -> Optional[dict[int, str]]:
    client = _load_agentcli()
    if client is None:
        return None
    prompt = _build_judge_prompt(query, items)
    try:
        resp = client.chat(
            prompt=prompt,
            provider="codex",
            owner="imagine-bench",
            alias="imagine-rejudge",
            cwd=str(REPO),
            timeout=60,
        )
        content = (resp.content or "").strip()
        if not content:
            return None
        return _parse_judgement(content, items)
    except Exception as e:
        print(f"[rejudge] agentcli judge failed: {e}", file=sys.stderr)
        return None


def _judge_batch_mlx(query: str, items: list[dict]) -> dict[int, str]:
    if not _load_mlx():
        return {it["id"]: "skip" for it in items}
    from mlx_lm import generate

    prompt = _build_judge_prompt(query, items)
    try:
        messages = [{"role": "user", "content": prompt}]
        # Qwen3.x has a "thinking mode" that emits reasoning prose
        # before the final answer — disable it so we just get the JSON.
        try:
            text = _mlx_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizers don't accept enable_thinking; fall through.
            text = _mlx_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        out = generate(
            _mlx_model, _mlx_tok, prompt=text,
            max_tokens=400, verbose=False,
        )
        return _parse_judgement(out, items)
    except Exception as e:
        print(f"[rejudge] mlx judge failed: {e}", file=sys.stderr)
        return {it["id"]: "skip" for it in items}


def _judge_batch(query: str, items: list[dict], backend: str) -> dict[int, str]:
    """Dispatch to the requested backend with agentcli→mlx fallback."""
    if not items:
        return {}
    if backend in ("auto", "agentcli"):
        result = _judge_batch_agentcli(query, items)
        if result is not None:
            return result
        if backend == "agentcli":
            return {it["id"]: "skip" for it in items}
    return _judge_batch_mlx(query, items)


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


def rejudge(report_path: Path, top_k: int, backend: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    triaxis = report.get("axes", {}).get("triaxis", {})
    per_query = triaxis.get("per_query") or []
    if not per_query:
        raise SystemExit("report has no triaxis per_query data")

    db_path = REPO / "imageparser.db"
    db = sqlite3.connect(str(db_path))

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
        judged = _judge_batch(
            q.get("query", ""),
            [items[fid] for fid in ranked if fid in items],
            backend=backend,
        )

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
    parser.add_argument(
        "--backend", choices=("auto", "agentcli", "mlx"), default="auto",
        help="auto: try agentcli (Codex), fall back to MLX. agentcli: Codex only. mlx: local Qwen only.",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"[rejudge] not found: {args.report}", file=sys.stderr)
        return 2

    print(f"[rejudge] backend={args.backend} report={args.report.name}")
    summary = rejudge(args.report, args.top_k, args.backend)

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
