#!/usr/bin/env python3
"""
Worker Scheduling Algorithm Simulator

Algorithm E' = Time-Weighted Pressure + Batch Commit + Minimum Guarantee

Rules:
1. Time-Weighted Pressure: pressure = (pending / (workers+1)) * time_weight
2. Batch Commit: finish all available items in current phase before re-evaluating
3. Minimum Guarantee: every phase with pending > 0 gets at least 1 batch per round
4. Stability: only switch if best pressure > current * 2x
5. GPU capability filter: MC/VV need GPU, MV is CPU-ok
6. Thermal: danger/critical → idle
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Constants ──
PHASE_TIME = {"parse": 1, "mc": 50, "vv": 2, "mv": 0.5}
PHASE_ORDER = ["parse", "mc", "vv", "mv"]  # pipeline order

# What each phase produces (downstream)
PRODUCES = {
    "parse": ["mc"],       # parse done → mc buffer
    "mc": ["vv", "mv"],    # mc done → vv + mv buffers (vv independent, mv needs mc)
    "vv": [],
    "mv": [],
}

@dataclass
class Worker:
    name: str
    gpu_type: str  # "strong", "weak", "cpu"
    capable: List[str] = field(default_factory=list)
    assigned: Optional[str] = None
    batch_remaining: int = 0  # items left in current batch commit
    processed: Dict[str, int] = field(default_factory=lambda: {p: 0 for p in PHASE_ORDER})
    switches: int = 0
    idle_steps: int = 0

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


def decide_mode_e_prime(worker: Worker, buffers: Dict[str, int],
                        workers_on: Dict[str, int]) -> str:
    """
    Algorithm E' = Time-Weighted + Batch Commit + Min Guarantee

    1. If batch_remaining > 0 and current phase has items → continue (batch commit)
    2. Otherwise: find phases with pending > 0 that this worker can do
    3. Time-weighted pressure = (pending / (workers_on + 1)) * time_weight
    4. Minimum guarantee: if a capable phase has pending but no workers → boost it
    5. Pick highest pressure; only switch if > current * 2
    """

    # 1. Batch commit: finish current batch
    if worker.batch_remaining > 0 and worker.assigned:
        if buffers.get(worker.assigned, 0) > 0:
            return worker.assigned
        # Current phase empty mid-batch → release
        worker.batch_remaining = 0

    # 2. Build pressure for capable phases
    pressure = {}
    for phase in worker.capable:
        pending = buffers.get(phase, 0)
        if pending <= 0:
            continue
        n = workers_on.get(phase, 0)
        w = PHASE_TIME.get(phase, 1)
        p = (pending / (n + 1)) * w

        # Minimum guarantee: if this phase has work but NO workers, boost pressure
        if n == 0 and pending > 0:
            p *= 1.5  # 50% boost for unserved phases

        pressure[phase] = p

    if not pressure:
        return "idle"

    best = max(pressure, key=pressure.get)

    # 3. Stability: only switch if best > current * 2
    if (worker.assigned and worker.assigned in pressure
            and pressure[worker.assigned] > 0):
        if pressure[best] <= pressure[worker.assigned] * 2:
            return worker.assigned

    return best


def simulate(name: str, workers: List[Worker], initial_buffers: Dict[str, int],
             max_steps: int = 500, batch_size: int = 5, verbose: bool = True):
    """Run full simulation."""
    buffers = dict(initial_buffers)
    done_count = 0
    total_files = sum(initial_buffers.values())

    # Track per-file completion (simplified: count files that passed all phases)
    # A file is "done" when it has been through mc + vv + mv
    files_needing = {"mc": 0, "vv": 0, "mv": 0}

    print(f"\n{'='*75}")
    print(f"  {name}")
    print(f"  Workers: {', '.join(f'{w.name}({w.gpu_type})' for w in workers)}")
    print(f"  Initial: {buffers}, batch_size={batch_size}")
    print(f"{'='*75}")

    for step in range(1, max_steps + 1):
        if all(v == 0 for v in buffers.values()):
            break

        # Count workers per phase
        workers_on = {}
        for w in workers:
            if w.assigned and w.assigned != "idle":
                workers_on[w.assigned] = workers_on.get(w.assigned, 0) + 1

        # Each worker decides
        for w in workers:
            old = w.assigned
            mode = decide_mode_e_prime(w, buffers, workers_on)

            if mode == "idle":
                w.idle_steps += 1
                w.assigned = None
                w.batch_remaining = 0
                continue

            if mode != old:
                if old and old != "idle":
                    w.switches += 1
                w.assigned = mode
                # Batch commit: lock in for min(pending, batch_size) items
                w.batch_remaining = min(buffers.get(mode, 0), batch_size)
                # Update workers_on for next worker's decision
                workers_on[mode] = workers_on.get(mode, 0) + 1
            else:
                if w.batch_remaining == 0:
                    w.batch_remaining = min(buffers.get(mode, 0), batch_size)

        # Process: each worker handles 1 item
        for w in workers:
            if not w.assigned or w.assigned == "idle":
                continue
            phase = w.assigned
            if buffers.get(phase, 0) > 0:
                buffers[phase] -= 1
                w.processed[phase] += 1
                w.batch_remaining = max(0, w.batch_remaining - 1)

                # Pipeline flow: produce downstream
                for downstream in PRODUCES.get(phase, []):
                    buffers[downstream] = buffers.get(downstream, 0) + 1

                # Track completion: mv done = file fully complete
                if phase == "mv":
                    done_count += 1

        # Print
        if step <= 5 or step % 10 == 0 or all(v == 0 for v in buffers.values()):
            buf_str = " ".join(f"{k}={v}" for k, v in buffers.items() if v > 0) or "(done)"
            assign_str = " ".join(f"{w.name}:{w.assigned or 'idle'}" for w in workers)
            if verbose:
                print(f"  {step:4d} | {buf_str:45s} | {assign_str:30s} | done={done_count}")

    print(f"\n  Completed in {step} steps. Files done: {done_count}")
    total_switches = 0
    for w in workers:
        total = sum(w.processed.values())
        detail = " ".join(f"{k}={v}" for k, v in w.processed.items() if v > 0)
        print(f"    {w.name:12s}: {total:4d} processed ({detail}), "
              f"{w.switches} switches, {w.idle_steps} idle")
        total_switches += w.switches
    print(f"  Total switches: {total_switches}")
    return step, done_count, total_switches


# ══════════════════════════════════════════════════════════════
# SCENARIOS
# ══════════════════════════════════════════════════════════════

print("\n" + "█" * 75)
print("  ALGORITHM E': Time-Weighted + Batch Commit + Min Guarantee")
print("█" * 75)

# S1: Tiny (5 files), server alone
simulate("S1: 5 files, server alone",
         [Worker("서버", "embedded")],
         {"parse": 5})

# S2: Small (20 files), server alone
simulate("S2: 20 files, server alone",
         [Worker("서버", "embedded")],
         {"parse": 20})

# S3: Medium (100), server + GPU
simulate("S3: 100 files, server + GPU(12GB)",
         [Worker("서버", "embedded"), Worker("외부A", "strong")],
         {"parse": 100})

# S4: Large (1000, pre-parsed), 2 GPU + CPU
simulate("S4: 1000 files (parsed), server + GPU + CPU",
         [Worker("서버", "embedded"), Worker("GPU-A", "strong"), Worker("CPU-B", "cpu")],
         {"mc": 1000})

# S5: Tail — mc almost done
simulate("S5: tail — mc=3, vv=200, mv=200",
         [Worker("서버", "embedded")],
         {"mc": 3, "vv": 200, "mv": 200})

# S6: Mixed
simulate("S6: mixed — parse=10, mc=50, vv=30, mv=80",
         [Worker("서버", "embedded"), Worker("GPU-A", "strong"), Worker("CPU-B", "cpu")],
         {"parse": 10, "mc": 50, "vv": 30, "mv": 80})

# S7: Dregs
simulate("S7: dregs — mc=1, vv=2, mv=0",
         [Worker("서버", "embedded"), Worker("GPU-A", "strong")],
         {"mc": 1, "vv": 2})

# S8: CPU only, MC stuck
simulate("S8: 3 CPU workers, mc=100 (stuck), mv=50",
         [Worker("CPU-1", "cpu"), Worker("CPU-2", "cpu"), Worker("CPU-3", "cpu")],
         {"mc": 100, "mv": 50})

# S9: Massive — 5000 files, 4 GPU workers
simulate("S9: 5000 files, server + 3 GPU + 1 CPU",
         [Worker("서버", "embedded"), Worker("G1", "strong"), Worker("G2", "strong"),
          Worker("G3", "strong"), Worker("CPU", "cpu")],
         {"parse": 5000}, max_steps=8000, verbose=True)

# S10: Server weak(8GB) + external strong(24GB)
simulate("S10: server weak + external strong, 100 files",
         [Worker("서버", "embedded"), Worker("RTX4090", "strong")],
         {"parse": 100})

# S11: All phases have dregs
simulate("S11: dregs everywhere — parse=2, mc=3, vv=4, mv=1",
         [Worker("서버", "embedded")],
         {"parse": 2, "mc": 3, "vv": 4, "mv": 1})

# S12: One phase dominates
simulate("S12: mc=500, everything else 0, server alone",
         [Worker("서버", "embedded")],
         {"mc": 500}, max_steps=2000)
