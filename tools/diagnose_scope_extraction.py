"""Diagnose decomposer scope extraction against frozen_30_v1.

For each query, runs the same scope resolution chain as
SqliteVectorSearch.triaxis_search (decompose -> plan filter -> relax ->
query hint) and reports:
  - what scope the decomposer extracted
  - how many files each resolution stage matched
  - whether the GT files are inside the winning scope set

Usage:
    python tools/diagnose_scope_extraction.py [--queryset PATH] [--out PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_scope(searcher, decomposer, query: str) -> dict:
    """Mirror search_triaxis scope resolution, returning a trace dict."""
    from backend.search.sqlite_search import _relax_unmatched_scope

    t0 = time.perf_counter()
    unified = decomposer.decompose(query)
    decompose_ms = round((time.perf_counter() - t0) * 1000, 1)

    scope = unified.get("scope", {})
    backend = unified.get("_decomp_backend", "unknown")
    scope_requested = bool(
        scope.get("folder") or scope.get("image_type") or scope.get("format")
    )

    trace = {
        "query": query,
        "backend": backend,
        "decompose_ms": decompose_ms,
        "scope": dict(scope),
        "scope_requested": scope_requested,
        "stages": [],
        "final_source": None,
        "final_count": 0,
        "final_ids": set(),
    }

    def record(stage, ids):
        trace["stages"].append({"stage": stage, "count": len(ids or ())})

    scope_file_ids = set()
    if scope_requested:
        scope_file_ids, _ = searcher._apply_plan_filter_with_info(scope)
        record("decomposition", scope_file_ids)
        if not scope_file_ids:
            relaxed_scope, relaxed_keys = _relax_unmatched_scope(scope, query)
            if relaxed_keys:
                relaxed_ids, _ = searcher._apply_plan_filter_with_info(relaxed_scope)
                record(f"relaxed({','.join(sorted(relaxed_keys))})", relaxed_ids)
                if relaxed_ids:
                    trace["final_source"] = "decomposition_relaxed"
                    scope_file_ids = relaxed_ids
        else:
            trace["final_source"] = "decomposition"

        if not scope_file_ids:
            hinted_folder, hinted_ids = searcher._scope_ids_from_query_hint(
                query, base_scope=scope, skip_folder=scope.get("folder"),
            )
            record(f"query_hint({hinted_folder})", hinted_ids)
            if hinted_ids:
                trace["final_source"] = "query_hint"
                trace["hint_folder"] = hinted_folder
                scope_file_ids = hinted_ids
        elif scope.get("folder"):
            hinted_folder, hinted_ids = searcher._scope_ids_from_query_hint(
                query, base_scope=scope, skip_folder=scope.get("folder"),
            )
            if hinted_ids:
                record(f"query_hint_override({hinted_folder})", hinted_ids)
                trace["final_source"] = "query_hint_override"
                trace["hint_folder"] = hinted_folder
                scope_file_ids = hinted_ids
    else:
        hinted_folder, hinted_ids = searcher._scope_ids_from_query_hint(query)
        record(f"query_hint_no_scope({hinted_folder})", hinted_ids)
        if hinted_ids:
            trace["final_source"] = "query_hint_hard_or_soft"
            trace["hint_folder"] = hinted_folder
            scope_file_ids = hinted_ids

    trace["final_count"] = len(scope_file_ids or ())
    trace["final_ids"] = scope_file_ids or set()
    trace["zero_scope"] = scope_requested and not scope_file_ids
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queryset", default="benchmarks/querysets/frozen_30_v1.json")
    ap.add_argument("--out", default="benchmarks/results/scope_extraction_diagnosis.json")
    args = ap.parse_args()

    from backend.db.sqlite_client import SQLiteDB
    from backend.search.sqlite_search import SqliteVectorSearch
    from backend.search.query_decomposer import QueryDecomposer

    qs = json.loads(Path(args.queryset).read_text())
    queries = qs["queries"] if isinstance(qs, dict) else qs

    searcher = SqliteVectorSearch()
    decomposer = QueryDecomposer(use_codex=False)

    rows = []
    n_zero = 0
    n_gt_full = 0
    n_gt_partial = 0
    n_gt_none = 0
    for q in queries:
        query = q["query"]
        gt = set(q.get("gt_ids") or [])
        trace = resolve_scope(searcher, decomposer, query)
        ids = trace.pop("final_ids")
        covered = len(gt & ids) if ids else 0
        trace["gt_total"] = len(gt)
        trace["gt_in_scope"] = covered
        if trace["zero_scope"]:
            n_zero += 1
        if gt:
            if not ids:
                pass  # zero-scope already counted
            elif covered == len(gt):
                n_gt_full += 1
            elif covered > 0:
                n_gt_partial += 1
            else:
                n_gt_none += 1
        rows.append(trace)
        status = "ZERO" if trace["zero_scope"] else f"{covered}/{len(gt)} gt"
        print(f"[{status:>10}] {query}")
        print(f"             scope={trace['scope']} backend={trace['backend']} "
              f"source={trace['final_source']} files={trace['final_count']} "
              f"stages={trace['stages']}")

    summary = {
        "queries": len(rows),
        "zero_scope": n_zero,
        "gt_fully_in_scope": n_gt_full,
        "gt_partially_in_scope": n_gt_partial,
        "gt_outside_scope": n_gt_none,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    out = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
