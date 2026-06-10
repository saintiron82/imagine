"""Run the multi-condition pair benchmark (frozen_multicondition_pairs queryset).

Reproduces the result format of multicondition_pairs_20260605_s19_* so
diagnose_multicondition_recall.py and prior summaries stay comparable.

Variants:
  current   — full pipeline as shipped
  no_guard  — IMAGINE_BENCH_DISABLE_EVIDENCE_GUARD=1 (object-evidence
              recall guard off; isolates the guard's contribution)

Usage:
    python tools/run_multicondition_bench.py \
        --queryset benchmarks/querysets/frozen_multicondition_pairs_s15_sample100.json \
        --output benchmarks/results/multicondition_pairs_<date>_<tag>.json \
        [--variants current,no_guard]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VARIANT_ENV = {
    "current": {},
    "no_guard": {"IMAGINE_BENCH_DISABLE_EVIDENCE_GUARD": "1"},
    "no_element_evidence": {"IMAGINE_BENCH_DISABLE_AND": "1"},
}


def run_variant(searcher, queries, variant: str) -> list[dict]:
    env_overrides = VARIANT_ENV[variant]
    saved = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    try:
        rows = []
        for q in queries:
            query = q["query"]
            gt = set(q["gt_ids"])
            t0 = time.perf_counter()
            results = searcher.triaxis_search(query, top_k=10, use_codex=False)
            elapsed = round(time.perf_counter() - t0, 2)
            ids10 = [r.get("id") for r in results[:10]]
            ids5 = ids10[:5]
            p5 = round(sum(1 for i in ids5 if i in gt) / 5, 4)
            p10 = round(sum(1 for i in ids10 if i in gt) / 10, 4)
            top = results[0] if results else {}
            rows.append({
                "query": query,
                "elements_ko": q.get("elements_ko"),
                "gt_count": len(gt),
                "gt_ids": sorted(gt),
                "ids5": ids5,
                "ids10": ids10,
                "p5": p5,
                "p10": p10,
                "elapsed_s": elapsed,
                "top_evidence_score": top.get("evidence_score"),
                "top_evidence_matrix": top.get("evidence_matrix"),
            })
            print(f"  [{variant}] p5={p5:.2f} p10={p10:.2f} ({elapsed}s) {query}")
        return rows
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "queries": n,
        "p5": round(sum(r["p5"] for r in rows) / n, 4),
        "p10": round(sum(r["p10"] for r in rows) / n, 4),
        "hit5": sum(1 for r in rows if any(i in set(r["gt_ids"]) for i in r["ids5"])),
        "hit10": sum(1 for r in rows if any(i in set(r["gt_ids"]) for i in r["ids10"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queryset", type=Path,
                    default=Path("benchmarks/querysets/frozen_multicondition_pairs_s15_sample100.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variants", default="current,no_guard")
    args = ap.parse_args()

    qs = json.loads(args.queryset.read_text())
    queries = qs["queries"]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANT_ENV:
            print(f"unknown variant: {v}", file=sys.stderr)
            return 2

    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    searcher = SqliteVectorSearch(db=SQLiteDB())

    runs = {}
    for variant in variants:
        print(f"=== variant: {variant} ({len(queries)} queries)")
        runs[variant] = run_variant(searcher, queries, variant)

    summary = {v: summarize(rows) for v, rows in runs.items()}
    comparison = {}
    if "current" in summary and "no_guard" in summary:
        comparison["current_vs_no_guard"] = {
            "delta_p5": round(summary["current"]["p5"] - summary["no_guard"]["p5"], 4),
            "delta_p10": round(summary["current"]["p10"] - summary["no_guard"]["p10"], 4),
        }

    out = {
        "run_id": args.output.stem,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "queryset": str(args.queryset),
        "sample_file_count": qs.get("sample_file_count"),
        "variants": variants,
        "summary": summary,
        "comparison": comparison,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(comparison, ensure_ascii=False))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
