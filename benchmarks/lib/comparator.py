"""
Baseline comparison and bootstrap statistical testing.
Compares independently recorded runs.
"""

import json
import random as rng
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .metrics import aggregate_metrics, compute_query_metrics


def score_run(run: dict, labels: Dict[str, Dict[str, int]]) -> dict:
    """Score a run against labels. Returns aggregated metrics."""
    query_metrics = []

    for qr in run["queries"]:
        qid = qr["query_id"]
        relevance = labels.get(qid, {})

        if not relevance:
            continue

        m = compute_query_metrics(
            ranked_ids=qr["ranked_ids"],
            relevance=relevance,
            latency_ms=qr.get("latency_ms", 0),
        )
        m["query_id"] = qid
        m["error"] = qr.get("error")
        query_metrics.append(m)

    agg = aggregate_metrics(query_metrics)
    agg["run_id"] = run.get("run_id", "")
    agg["engine_id"] = run.get("engine_id", "")
    agg["tag"] = run.get("tag", "")
    agg["timestamp"] = run.get("timestamp", "")
    return agg


def bootstrap_ci(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for mean difference (B - A).

    Returns dict with mean_diff, ci_lower, ci_upper, significant.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have same length"
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}

    rng.seed(seed)
    diffs = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in indices) / n
        mean_b = sum(scores_b[i] for i in indices) / n
        diffs.append(mean_b - mean_a)

    diffs.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))

    mean_diff = sum(diffs) / len(diffs)
    ci_lower = diffs[lo_idx]
    ci_upper = diffs[min(hi_idx, len(diffs) - 1)]

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": ci_lower > 0 or ci_upper < 0,
    }


def compare_runs(
    scored_a: dict,
    scored_b: dict,
    gates: Optional[dict] = None,
    n_bootstrap: int = 10000,
) -> dict:
    """Compare two scored runs. A=baseline, B=candidate.

    Returns comparison with deltas, bootstrap CIs, and gate results.
    """
    metric_names = ["ndcg@10", "recall@50", "mrr@10", "p@10"]
    comparison = {
        "baseline": {
            "run_id": scored_a.get("run_id", ""),
            "tag": scored_a.get("tag", ""),
        },
        "candidate": {
            "run_id": scored_b.get("run_id", ""),
            "tag": scored_b.get("tag", ""),
        },
        "metrics": {},
        "latency": {},
        "gate_result": {"overall": True, "details": {}},
    }

    per_query_a = scored_a.get("_per_query", {})
    per_query_b = scored_b.get("_per_query", {})

    for name in metric_names:
        val_a = scored_a.get(name, 0.0)
        val_b = scored_b.get(name, 0.0)
        delta = val_b - val_a

        # Bootstrap if per-query data available
        ci = {}
        if name in per_query_a and name in per_query_b:
            qa = per_query_a[name]
            qb = per_query_b[name]
            if len(qa) == len(qb):
                ci = bootstrap_ci(qa, qb, n_bootstrap=n_bootstrap)

        comparison["metrics"][name] = {
            "baseline": val_a,
            "candidate": val_b,
            "delta": delta,
            "bootstrap": ci,
        }

    # Latency comparison
    lat_a = scored_a.get("p95_latency_ms", 0)
    lat_b = scored_b.get("p95_latency_ms", 0)
    comparison["latency"] = {
        "baseline_p95": lat_a,
        "candidate_p95": lat_b,
        "delta_ms": lat_b - lat_a,
        "increase_pct": (lat_b - lat_a) / lat_a if lat_a > 0 else 0.0,
    }

    # Gate evaluation
    if gates:
        overall = True
        details = {}

        for gate_metric, rule in gates.items():
            if gate_metric in comparison["metrics"]:
                delta = comparison["metrics"][gate_metric]["delta"]
                max_drop = rule.get("max_drop", None)
                if max_drop is not None and delta < -max_drop:
                    details[gate_metric] = {
                        "pass": False,
                        "reason": f"dropped {delta:.4f} (limit: {-max_drop})",
                    }
                    if rule.get("action") == "block":
                        overall = False
                else:
                    details[gate_metric] = {"pass": True}

            elif gate_metric == "p95_latency_ms":
                max_inc = rule.get("max_increase_pct", None)
                inc = comparison["latency"]["increase_pct"]
                if max_inc is not None and inc > max_inc:
                    details[gate_metric] = {
                        "pass": False,
                        "reason": f"increased {inc:.1%} (limit: {max_inc:.0%})",
                    }
                    if rule.get("action") == "block":
                        overall = False
                else:
                    details[gate_metric] = {"pass": True}

            elif gate_metric == "error_rate":
                er_a = scored_a.get("error_rate", 0)
                er_b = scored_b.get("error_rate", 0)
                max_inc = rule.get("max_increase", 0)
                if (er_b - er_a) > max_inc:
                    details[gate_metric] = {
                        "pass": False,
                        "reason": f"error rate +{er_b - er_a:.3f} (limit: {max_inc})",
                    }
                    if rule.get("action") == "block":
                        overall = False
                else:
                    details[gate_metric] = {"pass": True}

        comparison["gate_result"] = {"overall": overall, "details": details}

    return comparison


def save_baseline(scored: dict, tag: str, baselines_dir: str = "benchmarks/baselines") -> str:
    """Save scored run as a baseline."""
    out_dir = Path(baselines_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove per-query raw data for compact storage
    saved = {k: v for k, v in scored.items() if not k.startswith("_")}
    saved["baseline_tag"] = tag

    path = out_dir / f"{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)

    # Update latest symlink
    latest = out_dir / "latest.json"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(f"{tag}.json")

    return str(path)


def load_baseline(tag_or_path: str, baselines_dir: str = "benchmarks/baselines") -> dict:
    """Load a baseline by tag name or file path."""
    p = Path(tag_or_path)
    if not p.exists():
        p = Path(baselines_dir) / f"{tag_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(f"Baseline not found: {tag_or_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
