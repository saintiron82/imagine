#!/usr/bin/env python3
"""
Benchmark CLI - Independent run & record system.

Usage:
    # Run a single engine and record scores
    python benchmarks/run.py --engine triaxis --tag "v3.7.0_balanced"
    python benchmarks/run.py --engine fts_only --tag "bm25_baseline"
    python benchmarks/run.py --engine random --tag "random_baseline"

    # Run with specific query set
    python benchmarks/run.py --engine triaxis --tag "smoke_test" --queries smoke

    # Compare two recorded runs
    python benchmarks/run.py --compare "v3.7.0_balanced,v3.7.0_visual"

    # Save a run as baseline
    python benchmarks/run.py --save-baseline "v3.7.0_balanced"

    # Gate check (exit code 0=pass, 1=fail)
    python benchmarks/run.py --gate --baseline latest --run-file benchmarks/runs/xxx.json

    # List recorded runs
    python benchmarks/run.py --list

    # Show score history for an engine
    python benchmarks/run.py --history triaxis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from benchmarks.lib.runner import (
    execute_run,
    get_engine,
    list_runs,
    load_labels,
    load_queries,
    load_run,
    save_run,
)
from benchmarks.lib.comparator import (
    compare_runs,
    load_baseline,
    save_baseline,
    score_run,
)
from benchmarks.lib.reporter import (
    generate_comparison_report,
    generate_history_report,
    generate_run_report,
    save_report,
    save_summary_json,
)


def load_config() -> dict:
    """Load benchmark config."""
    cfg_path = Path(__file__).parent / "config" / "benchmark.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def resolve_queries_path(queries_arg: str, config: dict) -> str:
    """Resolve query set path from arg or config."""
    base = Path(__file__).parent
    dataset = config.get("dataset", {})

    if queries_arg == "smoke":
        return str(base / dataset.get("smoke", "data/queries/smoke_50.jsonl"))
    elif queries_arg == "full":
        return str(base / dataset.get("full", "data/queries/full_300.jsonl"))
    elif Path(queries_arg).exists():
        return queries_arg
    else:
        # Try relative to benchmarks/
        return str(base / queries_arg)


def resolve_labels_path(config: dict) -> str:
    """Resolve labels path from config."""
    base = Path(__file__).parent
    dataset = config.get("dataset", {})
    return str(base / dataset.get("labels", "data/labels/relevance_v1.jsonl"))


def cmd_run(args, config):
    """Execute a benchmark run."""
    queries_path = resolve_queries_path(args.queries, config)

    if not Path(queries_path).exists():
        print(f"[ERROR] Query set not found: {queries_path}")
        print("Create a query set first: benchmarks/data/queries/")
        sys.exit(1)

    queries = load_queries(queries_path)
    print(f"[bench] Loaded {len(queries)} queries from {queries_path}")

    # Get engine config from yaml
    engine_cfg = config.get("engines", {}).get(args.engine, {})
    engine = get_engine(args.engine, engine_cfg)
    print(f"[bench] Engine: {engine.engine_id} ({engine.description})")

    top_k = config.get("run", {}).get("top_k", 50)
    tag = args.tag or f"{args.engine}_{datetime.now().strftime('%Y%m%d')}"

    print(f"[bench] Running {len(queries)} queries (top_k={top_k}, tag={tag})...")
    run_result = execute_run(engine, queries, top_k=top_k, tag=tag)

    # Save raw run
    run_path = save_run(run_result)
    print(f"[bench] Run saved: {run_path}")

    # Score against labels if available
    labels_path = resolve_labels_path(config)
    labels = load_labels(labels_path)

    if labels:
        scored = score_run(run_result, labels)
        print(f"\n[bench] === Scores ({scored.get('n_queries', 0)} scored queries) ===")
        for m in ["ndcg@10", "recall@50", "mrr@10"]:
            print(f"  {m}: {scored.get(m, 0):.4f}")
        if scored.get("p95_latency_ms"):
            print(f"  p95 latency: {scored['p95_latency_ms']}ms")

        # Save report
        report = generate_run_report(scored)
        report_name = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_path = save_report(report, report_name)
        save_summary_json(scored, report_name)
        print(f"[bench] Report: {report_path}")
    else:
        print(f"\n[bench] No labels found at {labels_path} - scores not computed")
        print("[bench] Latency stats:")
        latencies = [q["latency_ms"] for q in run_result["queries"]]
        if latencies:
            latencies.sort()
            mean_lat = sum(latencies) / len(latencies)
            p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
            print(f"  mean: {mean_lat:.0f}ms")
            print(f"  p95: {latencies[p95_idx]}ms")

    print(f"\n[bench] Done. Run ID: {run_result['run_id']}")


def cmd_compare(args, config):
    """Compare two recorded runs."""
    tags = [t.strip() for t in args.compare.split(",")]
    if len(tags) != 2:
        print("[ERROR] --compare requires exactly 2 tags separated by comma")
        sys.exit(1)

    labels_path = resolve_labels_path(config)
    labels = load_labels(labels_path)
    if not labels:
        print(f"[ERROR] Labels required for comparison: {labels_path}")
        sys.exit(1)

    # Find runs by tag
    all_runs = list_runs()
    runs_by_tag = {}
    for r in all_runs:
        if r["tag"] in tags:
            runs_by_tag[r["tag"]] = r["file"]

    for tag in tags:
        if tag not in runs_by_tag:
            print(f"[ERROR] Run not found for tag: {tag}")
            print(f"Available: {[r['tag'] for r in all_runs]}")
            sys.exit(1)

    run_a = load_run(runs_by_tag[tags[0]])
    run_b = load_run(runs_by_tag[tags[1]])

    scored_a = score_run(run_a, labels)
    scored_b = score_run(run_b, labels)

    gates = config.get("gates", {})
    n_bootstrap = config.get("statistics", {}).get("bootstrap_n", 10000)

    comparison = compare_runs(scored_a, scored_b, gates=gates, n_bootstrap=n_bootstrap)

    report = generate_comparison_report(comparison)
    report_name = f"compare_{tags[0]}_vs_{tags[1]}_{datetime.now().strftime('%Y%m%d')}"
    report_path = save_report(report, report_name)

    print(report)
    print(f"\n[bench] Report saved: {report_path}")


def cmd_save_baseline(args, config):
    """Save a run as baseline."""
    tag = args.save_baseline
    all_runs = list_runs()
    match = [r for r in all_runs if r["tag"] == tag]

    if not match:
        print(f"[ERROR] Run not found for tag: {tag}")
        sys.exit(1)

    labels_path = resolve_labels_path(config)
    labels = load_labels(labels_path)
    if not labels:
        print(f"[ERROR] Labels required: {labels_path}")
        sys.exit(1)

    run_data = load_run(match[0]["file"])
    scored = score_run(run_data, labels)
    path = save_baseline(scored, tag)
    print(f"[bench] Baseline saved: {path}")


def cmd_gate(args, config):
    """Gate check against baseline."""
    if not args.run_file:
        print("[ERROR] --run-file required for gate check")
        sys.exit(1)

    baseline_tag = args.baseline or "latest"
    try:
        baseline = load_baseline(baseline_tag)
    except FileNotFoundError:
        print(f"[ERROR] Baseline not found: {baseline_tag}")
        sys.exit(1)

    labels_path = resolve_labels_path(config)
    labels = load_labels(labels_path)
    if not labels:
        print(f"[ERROR] Labels required: {labels_path}")
        sys.exit(1)

    run_data = load_run(args.run_file)
    scored = score_run(run_data, labels)

    gates = config.get("gates", {})
    comparison = compare_runs(baseline, scored, gates=gates)

    gate_pass = comparison["gate_result"]["overall"]
    print(f"[bench] Gate: {'PASS' if gate_pass else 'FAIL'}")

    for metric, detail in comparison["gate_result"]["details"].items():
        status = "PASS" if detail["pass"] else "FAIL"
        reason = f" - {detail.get('reason', '')}" if not detail["pass"] else ""
        print(f"  {metric}: {status}{reason}")

    sys.exit(0 if gate_pass else 1)


def cmd_list(args, config):
    """List all recorded runs."""
    runs = list_runs()
    if not runs:
        print("[bench] No runs found in benchmarks/runs/")
        return

    print(f"[bench] {len(runs)} recorded runs:\n")
    print(f"{'Tag':<30} {'Engine':<15} {'Queries':<8} {'Date'}")
    print("-" * 75)
    for r in runs:
        tag = r.get("tag", "")[:29]
        engine = r.get("engine_id", "")[:14]
        nq = r.get("n_queries", 0)
        ts = r.get("timestamp", "")[:10]
        print(f"{tag:<30} {engine:<15} {nq:<8} {ts}")


def cmd_history(args, config):
    """Show score history for an engine."""
    engine_id = args.history
    labels_path = resolve_labels_path(config)
    labels = load_labels(labels_path)

    all_runs = list_runs()
    engine_runs = [r for r in all_runs if r["engine_id"] == engine_id]

    if not engine_runs:
        print(f"[bench] No runs found for engine: {engine_id}")
        return

    scored_list = []
    for r in engine_runs:
        run_data = load_run(r["file"])
        if labels:
            scored = score_run(run_data, labels)
            scored_list.append(scored)
        else:
            scored_list.append({
                "tag": r.get("tag", ""),
                "engine_id": engine_id,
                "timestamp": r.get("timestamp", ""),
            })

    report = generate_history_report(scored_list)
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CLI - Independent run & record",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Run mode
    parser.add_argument("--engine", help="Engine to benchmark (triaxis, vv_only, mv_only, fts_only, random)")
    parser.add_argument("--tag", help="Tag for this run (default: engine_YYYYMMDD)")
    parser.add_argument("--queries", default="smoke", help="Query set: smoke, full, or path (default: smoke)")

    # Compare mode
    parser.add_argument("--compare", help="Compare two tags: 'tag_a,tag_b'")

    # Baseline
    parser.add_argument("--save-baseline", help="Save a run (by tag) as baseline")
    parser.add_argument("--baseline", help="Baseline tag for gate check (default: latest)")

    # Gate
    parser.add_argument("--gate", action="store_true", help="Gate check mode")
    parser.add_argument("--run-file", help="Run JSON file for gate check")

    # Info
    parser.add_argument("--list", action="store_true", help="List all recorded runs")
    parser.add_argument("--history", help="Show score history for an engine")

    args = parser.parse_args()
    config = load_config()

    if args.list:
        cmd_list(args, config)
    elif args.history:
        cmd_history(args, config)
    elif args.compare:
        cmd_compare(args, config)
    elif args.save_baseline:
        cmd_save_baseline(args, config)
    elif args.gate:
        cmd_gate(args, config)
    elif args.engine:
        cmd_run(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
