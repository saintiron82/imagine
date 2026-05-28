#!/usr/bin/env python3
"""Sprint 2 β2: calibrate ConfidenceThresholds from LLM-judge data.

Reads an LLM-judge precision report (output of bench_llm_rejudge.py)
plus the source precision report (which carries per-result raw axis
scores), aligns them, and emits low/mid/high cuts where the empirical
precision-at-confidence first crosses 0.5 / 0.7 / 0.85.

Usage:
    .venv/bin/python tools/calibrate_confidence.py \\
        --judged benchmarks/results/precision_20260528_frozen_run1_sprint1_slm_judge.json \\
        --raw    benchmarks/results/precision_20260528_frozen_run1_sprint1.json \\
        --output benchmarks/results/confidence_thresholds_20260528.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _precision_at(samples, threshold: float) -> float:
    above = [r for s, r in samples if s >= threshold]
    if not above:
        return 0.0
    return sum(1 for r in above if r) / len(above)


def calibrate(samples):
    """Fit low/mid/high to precision targets 0.5/0.7/0.85.

    Walks candidate thresholds at observed score quantiles (rounded to
    3 decimals) and picks the smallest threshold whose
    precision-at-threshold meets the target. Falls back to the
    documented defaults when no candidate satisfies a target.
    """
    if not samples:
        return {"low": 0.20, "mid": 0.35, "high": 0.55, "n_samples": 0}

    sorted_scores = sorted({round(s, 3) for s, _ in samples})
    targets = {"low": 0.5, "mid": 0.7, "high": 0.85}
    defaults = {"low": 0.20, "mid": 0.35, "high": 0.55}

    result = {"n_samples": len(samples)}
    for level, target in targets.items():
        chosen = defaults[level]
        for cand in sorted_scores:
            if _precision_at(samples, cand) >= target:
                chosen = cand
                break
        result[level] = round(float(chosen), 3)

    # Monotonicity guard.
    result["mid"] = max(result["mid"], result["low"])
    result["high"] = max(result["high"], result["mid"])
    return result


def _load_samples(judged_path: Path, raw_path: Path):
    """Align LLM judgements to per-rank score proxies."""
    judged_report = json.loads(judged_path.read_text(encoding="utf-8"))
    raw_report = json.loads(raw_path.read_text(encoding="utf-8"))

    per_query_raw = raw_report.get("axes", {}).get("triaxis", {}).get("per_query", [])
    judged_by_query = {}
    for q in judged_report.get("per_query", []):
        key = (q.get("query") or "").strip()
        judged_by_query[key] = {str(k): str(v) for k, v in (q.get("judged") or {}).items()}

    samples = []
    for q in per_query_raw:
        key = (q.get("query") or "").strip()
        judged = judged_by_query.get(key, {})
        ranked = q.get("ranked_ids") or []
        for rank, fid in enumerate(ranked[:5]):
            verdict = judged.get(str(fid))
            if verdict not in ("yes", "no"):
                continue
            # Rank-position proxy: rank 0 = score 1.0, rank 4 = 0.2.
            # We don't yet save per-result raw scores in the bench JSON;
            # this proxy is the best we can fit until the bench widens
            # its per_query schema.
            score = max(0.0, 1.0 - rank / 5.0)
            samples.append((score, verdict == "yes"))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    samples = _load_samples(args.judged, args.raw)
    thresholds = calibrate(samples)

    print(f"Samples aligned: {thresholds['n_samples']}")
    print(f"  low  = {thresholds['low']}  (target precision 0.5)")
    print(f"  mid  = {thresholds['mid']}  (target precision 0.7)")
    print(f"  high = {thresholds['high']}  (target precision 0.85)")

    if args.output:
        args.output.write_text(
            json.dumps(thresholds, indent=2), encoding="utf-8",
        )
        print(f"  written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
