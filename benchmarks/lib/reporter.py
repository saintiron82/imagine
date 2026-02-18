"""
Benchmark report generator.
Generates Markdown reports for individual runs and comparisons.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def format_score(value: float, precision: int = 4) -> str:
    """Format a metric score."""
    return f"{value:.{precision}f}"


def format_delta(delta: float, precision: int = 4) -> str:
    """Format delta with sign and color hint."""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{precision}f}"


def generate_run_report(scored: dict) -> str:
    """Generate Markdown report for a single scored run."""
    lines = [
        f"# Benchmark Report: {scored.get('tag', scored.get('run_id', 'unknown'))}",
        "",
        f"- **Engine**: {scored.get('engine_id', 'unknown')}",
        f"- **Timestamp**: {scored.get('timestamp', 'unknown')}",
        f"- **Queries**: {scored.get('n_queries', 0)}",
        "",
        "## Scores",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]

    for metric in ["ndcg@10", "recall@50", "mrr@10", "p@10"]:
        val = scored.get(metric, 0.0)
        lines.append(f"| {metric} | {format_score(val)} |")

    if scored.get("p95_latency_ms") is not None:
        lines.append(f"| p95 latency | {scored['p95_latency_ms']}ms |")
    if scored.get("mean_latency_ms") is not None:
        lines.append(f"| mean latency | {scored['mean_latency_ms']:.0f}ms |")
    if scored.get("error_rate") is not None:
        lines.append(f"| error rate | {scored['error_rate']:.3f} |")

    lines.append("")
    return "\n".join(lines)


def generate_comparison_report(comparison: dict) -> str:
    """Generate Markdown report comparing two runs."""
    base = comparison.get("baseline", {})
    cand = comparison.get("candidate", {})
    gate = comparison.get("gate_result", {})

    gate_status = "PASS" if gate.get("overall", True) else "FAIL"

    lines = [
        f"# Comparison Report",
        "",
        f"- **Baseline**: {base.get('tag', base.get('run_id', '?'))}",
        f"- **Candidate**: {cand.get('tag', cand.get('run_id', '?'))}",
        f"- **Gate**: **{gate_status}**",
        "",
        "## Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta | Significant | Gate |",
        "|--------|----------|-----------|-------|-------------|------|",
    ]

    metrics = comparison.get("metrics", {})
    gate_details = gate.get("details", {})

    for name, data in metrics.items():
        val_a = format_score(data.get("baseline", 0))
        val_b = format_score(data.get("candidate", 0))
        delta = format_delta(data.get("delta", 0))

        bs = data.get("bootstrap", {})
        sig = "Yes" if bs.get("significant", False) else "No"

        g = gate_details.get(name, {})
        gate_str = "PASS" if g.get("pass", True) else f"FAIL: {g.get('reason', '')}"

        lines.append(f"| {name} | {val_a} | {val_b} | {delta} | {sig} | {gate_str} |")

    # Latency
    lat = comparison.get("latency", {})
    if lat:
        lines.extend([
            "",
            "## Latency",
            "",
            f"| | Baseline | Candidate | Delta |",
            f"|--|----------|-----------|-------|",
            f"| p95 | {lat.get('baseline_p95', 0)}ms | {lat.get('candidate_p95', 0)}ms | {lat.get('delta_ms', 0):+d}ms ({lat.get('increase_pct', 0):+.1%}) |",
        ])

    # Bootstrap details
    lines.extend(["", "## Bootstrap Details (95% CI)", ""])
    for name, data in metrics.items():
        bs = data.get("bootstrap", {})
        if bs:
            lines.append(
                f"- **{name}**: mean diff = {bs.get('mean_diff', 0):+.4f}, "
                f"CI = [{bs.get('ci_lower', 0):+.4f}, {bs.get('ci_upper', 0):+.4f}]"
            )

    lines.append("")
    return "\n".join(lines)


def generate_history_report(runs_summary: List[dict]) -> str:
    """Generate Markdown report showing score history for an engine."""
    if not runs_summary:
        return "# Score History\n\nNo runs found.\n"

    engine = runs_summary[0].get("engine_id", "unknown")
    lines = [
        f"# Score History: {engine}",
        "",
        "| Tag | nDCG@10 | Recall@50 | MRR@10 | p95 Lat | Date |",
        "|-----|---------|-----------|--------|---------|------|",
    ]

    for r in runs_summary:
        tag = r.get("tag", r.get("run_id", "?"))
        ndcg = format_score(r.get("ndcg@10", 0))
        recall = format_score(r.get("recall@50", 0))
        mrr = format_score(r.get("mrr@10", 0))
        lat = f"{r.get('p95_latency_ms', '?')}ms"
        ts = r.get("timestamp", "")[:10]
        lines.append(f"| {tag} | {ndcg} | {recall} | {mrr} | {lat} | {ts} |")

    lines.append("")
    return "\n".join(lines)


def save_report(content: str, name: str, output_dir: str = "benchmarks/reports") -> str:
    """Save report to file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return str(path)


def save_summary_json(data: dict, name: str, output_dir: str = "benchmarks/reports") -> str:
    """Save JSON summary alongside report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    # Strip internal per-query data
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    return str(path)
