#!/usr/bin/env python3
"""Compare two Search Evaluation V1 summaries and gate regressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ("nDCG@10", "P@10", "Recall@10", "MRR@10")


def load_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if summary.get("schema_version") != "search_evaluation_v1":
        raise ValueError(f"{path}: expected schema_version search_evaluation_v1")
    if not isinstance(summary.get("runs"), list):
        raise ValueError(f"{path}: missing runs list")
    return summary


def index_runs(summary: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for run in summary["runs"]:
        engine_id = str(run.get("engine_id", ""))
        if not engine_id:
            raise ValueError(f"{label}: run missing engine_id")
        if engine_id in indexed:
            raise ValueError(f"{label}: duplicate engine_id: {engine_id}")
        indexed[engine_id] = run
    return indexed


def parse_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def compare_summaries(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metrics: list[str] | tuple[str, ...] = DEFAULT_METRICS,
    engines: list[str] | tuple[str, ...] | None = None,
    min_delta: float = -0.01,
    min_query_ratio: float = 1.0,
    max_latency_ratio: float | None = None,
) -> dict[str, Any]:
    baseline_runs = index_runs(baseline, "baseline")
    candidate_runs = index_runs(candidate, "candidate")
    selected_engines = list(engines) if engines else sorted(set(baseline_runs) & set(candidate_runs))

    comparisons = []
    checks = []
    failures = []

    if not selected_engines:
        failures.append("no common engines to compare")

    for engine_id in selected_engines:
        base_run = baseline_runs.get(engine_id)
        cand_run = candidate_runs.get(engine_id)
        if base_run is None:
            failures.append(f"baseline missing engine: {engine_id}")
            continue
        if cand_run is None:
            failures.append(f"candidate missing engine: {engine_id}")
            continue

        base_query_count = int(base_run.get("query_count") or 0)
        cand_query_count = int(cand_run.get("query_count") or 0)
        min_queries = base_query_count * min_query_ratio
        query_status = "pass" if cand_query_count >= min_queries else "fail"
        if query_status == "fail":
            failures.append(
                f"{engine_id}: query_count {cand_query_count} < required {min_queries:.2f}"
            )
        checks.append({
            "engine_id": engine_id,
            "check": "query_count",
            "baseline": base_query_count,
            "candidate": cand_query_count,
            "required_min": _round(min_queries),
            "status": query_status,
        })

        base_metrics = base_run.get("metrics") or {}
        cand_metrics = cand_run.get("metrics") or {}
        for metric in metrics:
            if metric not in base_metrics:
                failures.append(f"{engine_id}: baseline missing metric {metric}")
                status = "fail"
                base_value = None
                cand_value = cand_metrics.get(metric)
                delta = None
            elif metric not in cand_metrics:
                failures.append(f"{engine_id}: candidate missing metric {metric}")
                status = "fail"
                base_value = base_metrics.get(metric)
                cand_value = None
                delta = None
            else:
                base_value = float(base_metrics[metric])
                cand_value = float(cand_metrics[metric])
                delta = cand_value - base_value
                status = "pass" if delta >= min_delta else "fail"
                if status == "fail":
                    failures.append(
                        f"{engine_id}: {metric} delta {delta:.6f} < {min_delta:.6f}"
                    )

            comparisons.append({
                "engine_id": engine_id,
                "metric": metric,
                "baseline": _round(base_value),
                "candidate": _round(cand_value),
                "delta": _round(delta),
                "required_min_delta": min_delta,
                "status": status,
            })

        if max_latency_ratio is not None:
            base_latency = base_run.get("avg_latency_ms")
            cand_latency = cand_run.get("avg_latency_ms")
            if base_latency is None:
                latency_status = "skip"
                required_max = None
            elif cand_latency is None:
                latency_status = "fail"
                required_max = float(base_latency) * max_latency_ratio
                failures.append(f"{engine_id}: candidate missing avg_latency_ms")
            else:
                required_max = float(base_latency) * max_latency_ratio
                latency_status = "pass" if float(cand_latency) <= required_max else "fail"
                if latency_status == "fail":
                    failures.append(
                        f"{engine_id}: avg_latency_ms {cand_latency} > {required_max:.3f}"
                    )
            checks.append({
                "engine_id": engine_id,
                "check": "avg_latency_ms",
                "baseline": _round(base_latency),
                "candidate": _round(cand_latency),
                "required_max": _round(required_max),
                "status": latency_status,
            })

    return {
        "schema_version": "search_evaluation_compare_v1",
        "status": "fail" if failures else "pass",
        "metrics": list(metrics),
        "engines": selected_engines,
        "rules": {
            "min_delta": min_delta,
            "min_query_ratio": min_query_ratio,
            "max_latency_ratio": max_latency_ratio,
        },
        "baseline": {
            "label_query_count": baseline.get("label_query_count"),
            "run_count": baseline.get("run_count"),
        },
        "candidate": {
            "label_query_count": candidate.get("label_query_count"),
            "run_count": candidate.get("run_count"),
        },
        "checks": checks,
        "comparisons": comparisons,
        "failures": failures,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Search Evaluation Comparison",
        "",
        f"**Status**: `{report['status']}`",
        f"**Metrics**: {', '.join(report['metrics'])}",
        f"**Engines**: {', '.join(report['engines']) if report['engines'] else '-'}",
        "",
        "## Metric Gate",
        "",
        "| Engine | Metric | Baseline | Candidate | Delta | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for item in report["comparisons"]:
        lines.append(
            f"| `{item['engine_id']}` | `{item['metric']}` | "
            f"{item['baseline'] if item['baseline'] is not None else 'n/a'} | "
            f"{item['candidate'] if item['candidate'] is not None else 'n/a'} | "
            f"{item['delta'] if item['delta'] is not None else 'n/a'} | "
            f"`{item['status']}` |"
        )

    lines.extend([
        "",
        "## Run Checks",
        "",
        "| Engine | Check | Baseline | Candidate | Required | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for item in report["checks"]:
        required = item.get("required_min", item.get("required_max"))
        lines.append(
            f"| `{item['engine_id']}` | `{item['check']}` | "
            f"{item['baseline'] if item['baseline'] is not None else 'n/a'} | "
            f"{item['candidate'] if item['candidate'] is not None else 'n/a'} | "
            f"{required if required is not None else 'n/a'} | "
            f"`{item['status']}` |"
        )

    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- {failure}")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="baseline evaluation JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate evaluation JSON")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="comma-separated metric names")
    parser.add_argument("--engines", help="comma-separated engine filter; default = common engines")
    parser.add_argument("--min-delta", type=float, default=-0.01, help="minimum candidate-baseline delta")
    parser.add_argument("--min-query-ratio", type=float, default=1.0, help="candidate query_count / baseline query_count")
    parser.add_argument("--max-latency-ratio", type=float, help="optional candidate latency / baseline latency cap")
    parser.add_argument("--output-json", type=Path, help="write comparison JSON")
    parser.add_argument("--output-md", type=Path, help="write comparison Markdown")
    args = parser.parse_args(argv)

    try:
        report = compare_summaries(
            baseline=load_summary(args.baseline),
            candidate=load_summary(args.candidate),
            metrics=parse_csv(args.metrics),
            engines=parse_csv(args.engines) or None,
            min_delta=args.min_delta,
            min_query_ratio=args.min_query_ratio,
            max_latency_ratio=args.max_latency_ratio,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(report), encoding="utf-8")

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
