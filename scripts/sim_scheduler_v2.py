#!/usr/bin/env python3
"""
Worker Scheduling Simulator v2: Modeless (Job-Driven)

Old (v1): Server assigns mode → worker processes that mode only → mode mismatch possible
New (v2): Worker claims a task → checks phase status → processes needed phase

No "mode" concept. Workers just process whatever job they get.

Key difference from v1:
- v1: decide_mode() picks a phase → worker locked to that phase for batch
- v2: pick_best_job() picks a JOB from the queue → worker does what the job needs

Same Algorithm E' pressure for deciding WHICH phase's jobs to claim.
But the decision happens at claim time, not at heartbeat time.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Constants ──
PHASE_TIME = {"parse": 1, "mc": 50, "vv": 2, "mv": 0.5}
PHASE_ORDER = ["parse", "mc", "vv", "mv"]

# Pipeline flow: completing a phase produces work for downstream phases
PRODUCES = {
    "parse": ["mc"],
    "mc": ["vv", "mv"],
    "vv": [],
    "mv": [],
}


@dataclass
class Worker:
    name: str
    gpu_type: str  # "strong", "weak", "cpu", "embedded"
    capable: List[str] = field(default_factory=list)
    # No "assigned mode" - worker just processes whatever it claims
    current_phase: Optional[str] = None
    batch_remaining: int = 0
    processed: Dict[str, int] = field(default_factory=lambda: {p: 0 for p in PHASE_ORDER})
    switches: int = 0
    idle_steps: int = 0
    # Probe: try unknown phases once. True=success, False=fail
    probed: Dict[str, bool] = field(default_factory=dict)
    probe_penalty: float = 1.0  # mc time multiplier discovered via probe (1.0=normal, 3.0=slow)

    # Simulation param: can_mc_actually controls whether MC probe succeeds
    # (in real system, this is determined by whether VLM loads successfully)
    can_mc_actually: bool = True

    def __post_init__(self):
        if not self.capable:
            if self.gpu_type == "strong":
                self.capable = ["mc", "vv", "mv"]
            elif self.gpu_type == "weak":
                self.capable = ["vv", "mv"]
            elif self.gpu_type == "cpu":
                self.capable = ["mv"]
            elif self.gpu_type == "embedded":
                self.capable = ["parse", "mc", "vv", "mv"]

    @property
    def effective_capable(self) -> List[str]:
        """Capable phases + probed successes, minus probed failures."""
        result = list(self.capable)
        for phase, ok in self.probed.items():
            if ok and phase not in result:
                result.append(phase)
        return [p for p in result if self.probed.get(p) is not False]

    def can_probe(self, phase: str) -> bool:
        """Can try a phase not in default capable and not yet probed?"""
        return phase not in self.capable and phase not in self.probed


def decide_mode_v1(worker: Worker, buffers: Dict[str, int],
                   workers_on: Dict[str, int]) -> str:
    """v1: Algorithm E' mode-based decision (from sim_scheduler.py)."""
    if worker.batch_remaining > 0 and worker.current_phase:
        if buffers.get(worker.current_phase, 0) > 0:
            return worker.current_phase
        worker.batch_remaining = 0

    pressure = {}
    for phase in worker.capable:
        pending = buffers.get(phase, 0)
        if pending <= 0:
            continue
        n = workers_on.get(phase, 0)
        w = PHASE_TIME.get(phase, 1)
        p = (pending / (n + 1)) * w
        if n == 0 and pending > 0:
            p *= 1.5
        pressure[phase] = p

    if not pressure:
        return "idle"
    best = max(pressure, key=pressure.get)
    if (worker.current_phase and worker.current_phase in pressure
            and pressure[worker.current_phase] > 0):
        if pressure[best] <= pressure[worker.current_phase] * 2:
            return worker.current_phase
    return best


def claim_best_phase(worker: Worker, claimable: Dict[str, int],
                     workers_on: Dict[str, int]) -> str:
    """Pick the best phase to claim from, based on ACTUAL claimable counts.

    Uses effective_capable (includes probe successes).
    If idle and a probeable phase has high pressure, triggers probe.

    Key rules:
    1. Sticky: if last phase still has work, stay (avoid model switch)
    2. GPU-exclusive preference: GPU workers prefer MC/VV over MV
       (leave MV to CPU workers who can only do MV)
    3. Switch only if another phase is overwhelmingly more urgent (>2x)
    """
    def _calc_pressure(phase):
        pending = claimable.get(phase, 0)
        if pending <= 0:
            return 0
        n = workers_on.get(phase, 0)
        w = PHASE_TIME.get(phase, 1)
        p = (pending / (n + 1)) * w
        if n == 0 and pending > 0:
            p *= 1.5
        if phase == "mv":
            p += pending * 10
        return p

    capable = worker.effective_capable

    # GPU workers: prefer GPU-exclusive phases (mc, vv) over shared (mv)
    has_cpu_workers = any(n > 0 for phase, n in workers_on.items() if phase == "mv")
    gpu_exclusive = [p for p in ("mc", "vv") if p in capable and claimable.get(p, 0) > 0] if has_cpu_workers else []

    # Sticky: if last phase still has work, prefer staying
    if (worker.current_phase and worker.current_phase in capable
            and claimable.get(worker.current_phase, 0) > 0):
        cur = worker.current_phase
        cur_p = _calc_pressure(cur)

        # GPU worker on MV but GPU-exclusive work available? Switch back.
        if cur == "mv" and gpu_exclusive:
            best_gpu_p = max(_calc_pressure(p) for p in gpu_exclusive)
            if best_gpu_p > 0:
                return max(gpu_exclusive, key=_calc_pressure)

        # Check if any other phase is overwhelmingly more urgent
        best_other_p = max((_calc_pressure(p) for p in capable if p != cur), default=0)
        if best_other_p <= cur_p * 2:
            return cur

    # Current phase exhausted or overwhelmed - pick best
    if gpu_exclusive:
        return max(gpu_exclusive, key=_calc_pressure)

    # Normal selection from all effective capable phases
    pressure = {p: _calc_pressure(p) for p in capable if _calc_pressure(p) > 0}
    if not pressure:
        # Nothing in capable phases. Try probing an unknown phase?
        for phase in PHASE_ORDER:
            if worker.can_probe(phase) and claimable.get(phase, 0) > 0:
                return f"probe:{phase}"
        return "idle"
    return max(pressure, key=pressure.get)


def simulate_v1(name: str, workers: List[Worker], initial_buffers: Dict[str, int],
                max_steps: int = 500, batch_size: int = 5) -> Tuple[int, int, int]:
    """v1: Mode-based (original Algorithm E')."""
    from sim_scheduler import simulate as _sim_v1, Worker as _W1

    # Convert workers to v1 format
    v1_workers = [_W1(w.name, w.gpu_type) for w in workers]
    return _sim_v1(name, v1_workers, initial_buffers, max_steps, batch_size, verbose=False)


def simulate_v2(name: str, workers: List[Worker], initial_buffers: Dict[str, int],
                max_steps: int = 500, batch_size: int = 5,
                verbose: bool = False) -> Tuple[int, int, int]:
    """v2: Modeless (job-driven). Workers claim from claimable pool directly."""
    buffers = dict(initial_buffers)
    done_count = 0

    # Reset workers
    for w in workers:
        w.current_phase = None
        w.batch_remaining = 0
        w.processed = {p: 0 for p in PHASE_ORDER}
        w.switches = 0
        w.idle_steps = 0

    for step in range(1, max_steps + 1):
        if all(v == 0 for v in buffers.values()):
            break

        # In v2, "claimable" = buffers directly (simulation has no assigned/pending split)
        # In real code, claimable uses same WHERE as claim query
        claimable = {k: v for k, v in buffers.items() if v > 0}

        # Count workers per phase (for pressure calculation)
        workers_on = {}
        for w in workers:
            if w.current_phase and w.current_phase != "idle":
                workers_on[w.current_phase] = workers_on.get(w.current_phase, 0) + 1

        # Each worker claims
        for w in workers:
            # Batch commit: finish current batch first
            if w.batch_remaining > 0 and w.current_phase:
                if claimable.get(w.current_phase, 0) > 0:
                    continue  # stay on current phase
                w.batch_remaining = 0  # phase empty, release

            old_phase = w.current_phase
            phase = claim_best_phase(w, claimable, workers_on)

            # Handle probe result
            if isinstance(phase, str) and phase.startswith("probe:"):
                probe_phase = phase.split(":")[1]
                # Simulate probe: 1 step cost, check if worker can actually do it
                if w.can_mc_actually or probe_phase != "mc":
                    w.probed[probe_phase] = True
                    if verbose:
                        print(f"  PROBE {w.name}: {probe_phase} -> SUCCESS (added to capable)")
                else:
                    w.probed[probe_phase] = False
                    if verbose:
                        print(f"  PROBE {w.name}: {probe_phase} -> FAIL (excluded)")
                w.current_phase = None
                continue

            if phase == "idle":
                w.idle_steps += 1
                w.current_phase = None
                w.batch_remaining = 0
                continue

            if phase != old_phase:
                if old_phase and old_phase != "idle":
                    w.switches += 1
                w.current_phase = phase
                w.batch_remaining = min(claimable.get(phase, 0), batch_size)
                workers_on[phase] = workers_on.get(phase, 0) + 1
            else:
                if w.batch_remaining == 0:
                    w.batch_remaining = min(claimable.get(phase, 0), batch_size)

        # Process: each worker handles 1 item
        for w in workers:
            if not w.current_phase or w.current_phase == "idle":
                continue
            phase = w.current_phase
            if buffers.get(phase, 0) > 0:
                buffers[phase] -= 1
                w.processed[phase] += 1
                w.batch_remaining = max(0, w.batch_remaining - 1)

                for downstream in PRODUCES.get(phase, []):
                    buffers[downstream] = buffers.get(downstream, 0) + 1

                if phase == "mv":
                    done_count += 1

        if verbose and (step <= 5 or step % 10 == 0 or all(v == 0 for v in buffers.values())):
            buf_str = " ".join(f"{k}={v}" for k, v in buffers.items() if v > 0) or "(done)"
            assign_str = " ".join(f"{w.name}:{w.current_phase or 'idle'}" for w in workers)
            print(f"  {step:4d} | {buf_str:45s} | {assign_str}")

    total_switches = sum(w.switches for w in workers)
    return step, done_count, total_switches


# ══════════════════════════════════════════════════════════════
# COMPARISON: v1 (mode-based) vs v2 (modeless)
# ══════════════════════════════════════════════════════════════

def simulate_v1(name: str, workers: List[Worker], initial_buffers: Dict[str, int],
                max_steps: int = 500, batch_size: int = 5) -> Tuple[int, int, int]:
    """v1: Mode-based simulation (inline, no import needed)."""
    buffers = dict(initial_buffers)
    done_count = 0
    for w in workers:
        w.current_phase = None
        w.batch_remaining = 0
        w.processed = {p: 0 for p in PHASE_ORDER}
        w.switches = 0
        w.idle_steps = 0

    step = 0
    for step in range(1, max_steps + 1):
        if all(v == 0 for v in buffers.values()):
            break
        workers_on = {}
        for w in workers:
            if w.current_phase and w.current_phase != "idle":
                workers_on[w.current_phase] = workers_on.get(w.current_phase, 0) + 1
        for w in workers:
            old = w.current_phase
            mode = decide_mode_v1(w, buffers, workers_on)
            if mode == "idle":
                w.idle_steps += 1
                w.current_phase = None
                w.batch_remaining = 0
                continue
            if mode != old:
                if old and old != "idle":
                    w.switches += 1
                w.current_phase = mode
                w.batch_remaining = min(buffers.get(mode, 0), batch_size)
                workers_on[mode] = workers_on.get(mode, 0) + 1
            else:
                if w.batch_remaining == 0:
                    w.batch_remaining = min(buffers.get(mode, 0), batch_size)
        for w in workers:
            if not w.current_phase or w.current_phase == "idle":
                continue
            phase = w.current_phase
            if buffers.get(phase, 0) > 0:
                buffers[phase] -= 1
                w.processed[phase] += 1
                w.batch_remaining = max(0, w.batch_remaining - 1)
                for downstream in PRODUCES.get(phase, []):
                    buffers[downstream] = buffers.get(downstream, 0) + 1
                if phase == "mv":
                    done_count += 1

    return step, done_count, sum(w.switches for w in workers)


def compare(name, workers_spec, initial_buffers, max_steps=500, batch_size=5):
    """Run same scenario on v1 and v2, compare results."""
    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"  Workers: {', '.join(f'{s[0]}({s[1]})' for s in workers_spec)}")
    print(f"  Initial: {initial_buffers}")
    print(f"{'='*75}")

    v1_workers = [Worker(s[0], s[1]) for s in workers_spec]
    s1, d1, sw1 = simulate_v1(f"v1: {name}", v1_workers, dict(initial_buffers), max_steps, batch_size)

    v2_workers = [Worker(s[0], s[1]) for s in workers_spec]
    s2, d2, sw2 = simulate_v2(f"v2: {name}", v2_workers, dict(initial_buffers), max_steps, batch_size)

    print(f"\n  {'METRIC':<20s} {'v1 (mode)':>12s} {'v2 (modeless)':>14s} {'diff':>10s}")
    print(f"  {'-'*56}")
    print(f"  {'Steps':.<20s} {s1:>12d} {s2:>14d} {s2-s1:>+10d}")
    print(f"  {'Files done':.<20s} {d1:>12d} {d2:>14d} {d2-d1:>+10d}")
    print(f"  {'Mode switches':.<20s} {sw1:>12d} {sw2:>14d} {sw2-sw1:>+10d}")

    faster = "v2" if s2 < s1 else "v1" if s1 < s2 else "tie"
    if s2 != s1:
        print(f"  Winner: {faster} ({'%.1f' % ((1 - s2/s1)*100)}% faster)")
    else:
        print(f"  Winner: tie")
    return s1, s2


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("  COMPARISON: v1 (mode-based) vs v2 (modeless/job-driven)")
    print("#" * 75)

    scenarios = [
        ("S1: 5 files, server alone",
         [("Server", "embedded")],
         {"parse": 5}),

        ("S2: 20 files, server alone",
         [("Server", "embedded")],
         {"parse": 20}),

        ("S3: 100 files, server + GPU",
         [("Server", "embedded"), ("GPU-A", "strong")],
         {"parse": 100}),

        ("S4: 1000 parsed, server + GPU + CPU",
         [("Server", "embedded"), ("GPU-A", "strong"), ("CPU-B", "cpu")],
         {"mc": 1000}),

        ("S5: tail phase - mc=3, vv=200, mv=200",
         [("Server", "embedded")],
         {"mc": 3, "vv": 200, "mv": 200}),

        ("S6: mixed - parse=10, mc=50, vv=30, mv=80",
         [("Server", "embedded"), ("GPU-A", "strong"), ("CPU-B", "cpu")],
         {"parse": 10, "mc": 50, "vv": 30, "mv": 80}),

        ("S7: dregs - mc=1, vv=2",
         [("Server", "embedded"), ("GPU-A", "strong")],
         {"mc": 1, "vv": 2}),

        ("S8: CPU only, MC stuck, mv=50",
         [("CPU-1", "cpu"), ("CPU-2", "cpu"), ("CPU-3", "cpu")],
         {"mc": 100, "mv": 50}),

        ("S9: 5000 files, server + 3 GPU + CPU",
         [("Server", "embedded"), ("G1", "strong"), ("G2", "strong"),
          ("G3", "strong"), ("CPU", "cpu")],
         {"parse": 5000}, 8000),

        ("S10: server weak + external strong",
         [("Server", "embedded"), ("RTX4090", "strong")],
         {"parse": 100}),

        ("S11: dregs everywhere",
         [("Server", "embedded")],
         {"parse": 2, "mc": 3, "vv": 4, "mv": 1}),

        ("S12: mc=500, server alone",
         [("Server", "embedded")],
         {"mc": 500}, 2000),
    ]

    results = []
    for args in scenarios:
        name, workers_spec, buffers = args[0], args[1], args[2]
        max_steps = args[3] if len(args) > 3 else 500
        s1, s2 = compare(name, workers_spec, buffers, max_steps=max_steps)
        results.append((name, s1, s2))

    # Final summary
    print(f"\n\n{'#'*75}")
    print(f"  FINAL SUMMARY")
    print(f"{'#'*75}")
    print(f"\n  {'Scenario':<45s} {'v1':>6s} {'v2':>6s} {'diff':>8s}")
    print(f"  {'-'*65}")
    for name, s1, s2 in results:
        short = name[:43]
        diff_pct = f"{(1-s2/s1)*100:+.1f}%" if s1 > 0 else "n/a"
        print(f"  {short:<45s} {s1:>6d} {s2:>6d} {diff_pct:>8s}")
