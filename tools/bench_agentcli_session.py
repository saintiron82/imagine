#!/usr/bin/env python3
"""Measure agentcli session-reuse effect on Codex CLI calls.

Compares:
  - "fresh" mode: each chat() uses a unique alias → Codex starts a new
    session every time (cold path).
  - "reused" mode: chat() uses a fixed alias → agentcli resumes the
    Codex session after the first call (warm path).

We time N identical prompts and report mean/median latency for each mode.

Usage:
    .venv/bin/python tools/bench_agentcli_session.py --count 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


PROMPT = (
    "Return a single-line JSON object describing this Korean image search query. "
    "Use this exact schema: "
    '{"pre_filter":{"folder":"<korean folder name or empty>","image_type":null,"format":null},'
    '"search":{"query":"<english one-line description>","mode":"semantic"},'
    '"fallback_keywords":["<short keywords>"]}\n'
    "QUERY: #07에서 캐릭터과 방 있는 이미지\n"
    "JSON:"
)


def _make_client():
    from agentcli import LLMClient, MemoryStore
    return LLMClient(store=MemoryStore())


def _time_call(client, alias: str) -> tuple[float, bool, int]:
    t0 = time.perf_counter()
    try:
        resp = client.chat(
            prompt=PROMPT,
            provider="codex",
            owner="imagine-bench-session",
            alias=alias,
            cwd=str(REPO),
            timeout=60,
        )
        content = (resp.content or "").strip()
        ok = bool(content)
        size = len(content)
    except Exception as e:
        print(f"  call failed: {e}", file=sys.stderr)
        return time.perf_counter() - t0, False, 0
    elapsed = time.perf_counter() - t0
    return elapsed, ok, size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5,
                        help="number of calls per mode")
    args = parser.parse_args()

    client = _make_client()

    # ── Fresh: unique alias per call (no session reuse) ─────────────
    print(f"\n[fresh] {args.count} calls, unique alias each")
    fresh_times: list[float] = []
    for i in range(args.count):
        alias = f"fresh-{uuid.uuid4().hex[:8]}"
        t, ok, size = _time_call(client, alias)
        status = "OK" if ok else "FAIL"
        print(f"  [{i+1}/{args.count}] {t:6.2f}s {status} ({size} chars)")
        fresh_times.append(t)

    # ── Reused: fixed alias (agentcli resumes Codex session) ────────
    print(f"\n[reused] {args.count} calls, fixed alias 'session-reuse-test'")
    reused_times: list[float] = []
    fixed_alias = "session-reuse-test"
    for i in range(args.count):
        t, ok, size = _time_call(client, fixed_alias)
        status = "OK" if ok else "FAIL"
        print(f"  [{i+1}/{args.count}] {t:6.2f}s {status} ({size} chars)")
        reused_times.append(t)

    def _stats(times: list[float]) -> dict:
        return {
            "n": len(times),
            "mean": statistics.fmean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
        }

    fresh = _stats(fresh_times)
    reused = _stats(reused_times)

    print()
    print("=" * 60)
    print("  Codex session reuse — latency comparison")
    print("=" * 60)
    print(f"  {'mode':10s} {'n':>4s} {'mean':>8s} {'median':>8s} {'min':>8s} {'max':>8s}")
    print(
        f"  {'fresh':10s} {fresh['n']:>4d} {fresh['mean']:>8.2f} "
        f"{fresh['median']:>8.2f} {fresh['min']:>8.2f} {fresh['max']:>8.2f}"
    )
    print(
        f"  {'reused':10s} {reused['n']:>4d} {reused['mean']:>8.2f} "
        f"{reused['median']:>8.2f} {reused['min']:>8.2f} {reused['max']:>8.2f}"
    )
    if fresh["mean"] > 0:
        delta = reused["mean"] - fresh["mean"]
        pct = 100.0 * delta / fresh["mean"]
        print(f"\n  Δ mean: {delta:+.2f}s ({pct:+.1f}%)")
        if delta < 0:
            print(f"  → session reuse SAVES ~{-delta:.2f}s per call on average")
        else:
            print(f"  → no measurable session-reuse benefit observed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
