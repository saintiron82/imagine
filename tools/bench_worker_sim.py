#!/usr/bin/env python3
"""
Worker Simulation Benchmark
============================
Simulates the real worker environment by exercising the full model lifecycle
(load -> inference -> unload -> GC) for each phase using the production
PhaseRunner code.

Unlike test_qwen35_eval.py which measures pure inference only, this benchmark
includes all overhead: model cold-load, PSD preprocessing, Phase switching,
GC/cache cleanup, and memory tracking.

Results serve as the "design speed" baseline for comparing against actual
worker throughput.

Usage:
    python tools/bench_worker_sim.py                     # default: 10 images, all phases
    python tools/bench_worker_sim.py --count 20          # 20 images
    python tools/bench_worker_sim.py --phases mc         # MC phase only
    python tools/bench_worker_sim.py --phases mc,vv      # MC + VV
    python tools/bench_worker_sim.py --no-full           # skip full pipeline
    python tools/bench_worker_sim.py --compare prev.json # compare with previous
    python tools/bench_worker_sim.py --output result.json
"""

import argparse
import gc
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE_DIR = Path("/Users/saintiron/imageDB/마캬베리즈무/실내소품")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks"

# ── Memory Helpers ──────────────────────────────────────────────────────


def get_rss_mb() -> float:
    """Process RSS in MB via psutil."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return -1.0


def get_metal_mb() -> float:
    """MLX Metal active memory in MB. Returns -1 if unavailable."""
    try:
        import mlx.core as mx
        return mx.get_active_memory() / 1024**2
    except Exception:
        return -1.0


def get_mps_mb() -> float:
    """torch MPS current allocated memory in MB. Returns -1 if unavailable."""
    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return torch.mps.current_allocated_memory() / 1024**2
    except Exception:
        pass
    return -1.0


@dataclass
class MemSnapshot:
    rss_mb: float = -1.0
    metal_mb: float = -1.0
    mps_mb: float = -1.0


def snapshot_memory() -> MemSnapshot:
    return MemSnapshot(
        rss_mb=round(get_rss_mb(), 1),
        metal_mb=round(get_metal_mb(), 1),
        mps_mb=round(get_mps_mb(), 1),
    )


def mem_max(a: MemSnapshot, b: MemSnapshot) -> MemSnapshot:
    """Element-wise max of two snapshots (for peak tracking)."""
    return MemSnapshot(
        rss_mb=max(a.rss_mb, b.rss_mb),
        metal_mb=max(a.metal_mb, b.metal_mb),
        mps_mb=max(a.mps_mb, b.mps_mb),
    )


# ── Data Classes ────────────────────────────────────────────────────────


@dataclass
class PhaseResult:
    phase: str
    model_id: str
    file_count: int
    batch_size: int
    # Timing
    model_load_s: float = 0.0
    processing_s: float = 0.0
    model_unload_s: float = 0.0
    gc_s: float = 0.0
    total_s: float = 0.0
    # Derived
    per_file_s: float = 0.0
    files_per_min: float = 0.0
    # Memory
    mem_before_load: MemSnapshot = field(default_factory=MemSnapshot)
    mem_after_load: MemSnapshot = field(default_factory=MemSnapshot)
    mem_peak: MemSnapshot = field(default_factory=MemSnapshot)
    mem_after_unload: MemSnapshot = field(default_factory=MemSnapshot)
    # Status
    success_count: int = 0
    fail_count: int = 0


@dataclass
class PipelineResult:
    phases: Dict[str, PhaseResult] = field(default_factory=dict)
    total_s: float = 0.0
    files_per_min: float = 0.0
    file_count: int = 0


# ── Stub Implementations ───────────────────────────────────────────────


class NullStorageBackend:
    """No-op storage backend for benchmarking."""

    def __init__(self):
        self.call_count = 0

    def save_vision(self, item, vision_result):
        self.call_count += 1

    def save_vv(self, item, vv_vec, structure_vec=None):
        self.call_count += 1

    def save_mv(self, item, mv_vec, text):
        self.call_count += 1

    def flush(self):
        pass


class TimingProgressReporter:
    """Progress reporter that tracks per-file timing and peak memory."""

    def __init__(self):
        self.events: List[dict] = []
        self.peak_mem = MemSnapshot()

    def phase_start(self, phase: str, count: int) -> None:
        print(f"    [{phase}] {count} items...")

    def file_done(self, phase: str, index: int, count: int,
                  file_name: str, success: bool) -> None:
        self.events.append({
            "phase": phase, "index": index, "file": file_name,
            "success": success, "t": time.perf_counter(),
        })
        # Sample memory for peak tracking
        self.peak_mem = mem_max(self.peak_mem, snapshot_memory())

    def phase_complete(self, phase: str, elapsed_s: float) -> None:
        print(f"    [{phase}] done in {elapsed_s:.1f}s")


# ── Image Loading ───────────────────────────────────────────────────────


def load_psd_images(image_dir: Path, n: int, max_edge: int = 768):
    """
    Load N PSD files, convert to RGB PIL Images, save as temp PNGs.
    Returns (list[(PIL.Image, filename, temp_png_path)], elapsed_seconds).
    """
    from PIL import Image
    from psd_tools import PSDImage

    psd_files = sorted(image_dir.glob("*.psd"))[:n]
    if not psd_files:
        # Fallback to PNG/JPG
        psd_files = sorted(image_dir.glob("*.png"))[:n]
        psd_files += sorted(image_dir.glob("*.jpg"))[:n]
        psd_files = psd_files[:n]

    if not psd_files:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    t0 = time.perf_counter()
    results = []
    tmp_dir = tempfile.mkdtemp(prefix="bench_sim_")

    for pf in psd_files:
        try:
            if pf.suffix.lower() == ".psd":
                psd = PSDImage.open(str(pf))
                img = psd.composite()
            else:
                img = Image.open(str(pf))

            # Convert to RGB
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize
            w, h = img.size
            scale = max_edge / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # Save as temp PNG (PhaseRunner needs file path)
            tmp_path = os.path.join(tmp_dir, pf.stem + ".png")
            img.save(tmp_path)

            results.append((img, pf.name, tmp_path))
        except Exception as e:
            print(f"    SKIP {pf.name}: {e}")

    elapsed = time.perf_counter() - t0
    return results, elapsed, tmp_dir


# ── Phase Benchmarks (Standalone) ───────────────────────────────────────


def bench_mc_phase(images: list, config: dict) -> PhaseResult:
    """Benchmark MC (VLM) phase standalone with full model lifecycle."""
    from backend.pipeline.model_manager import ModelManager

    n = len(images)
    result = PhaseResult(phase="mc", model_id="", file_count=n, batch_size=1)

    models = ModelManager()

    # 1. Before load
    result.mem_before_load = snapshot_memory()

    # 2. Load VLM
    t_load = time.perf_counter()
    try:
        analyzer = models.get_vlm()
    except Exception as e:
        print(f"    VLM load failed: {e}")
        result.model_id = "FAILED"
        return result
    result.model_load_s = time.perf_counter() - t_load
    result.model_id = getattr(analyzer, 'model_id', None) or type(analyzer).__name__
    result.mem_after_load = snapshot_memory()
    peak = result.mem_after_load

    print(f"    VLM loaded: {result.model_id} ({result.model_load_s:.1f}s)")

    # 3. Get domain
    domain = None
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        domain_id = cfg.get("classification.active_domain")
        if domain_id:
            from backend.vision.domain_loader import load_domain
            domain = load_domain(domain_id)
    except Exception:
        pass

    # 4. Process images
    t_proc = time.perf_counter()
    ok = 0
    for i, (img, fname, _) in enumerate(images):
        try:
            r = analyzer.classify_and_analyze(img, domain=domain)
            caption = r.get("mc_caption", "") or r.get("caption", "")
            if caption:
                ok += 1
            else:
                print(f"    [{i+1}/{n}] {fname}: empty caption")
        except Exception as e:
            print(f"    [{i+1}/{n}] {fname}: FAIL {e}")
        peak = mem_max(peak, snapshot_memory())

    result.processing_s = time.perf_counter() - t_proc
    result.success_count = ok
    result.fail_count = n - ok

    # 5. Unload
    t_unload = time.perf_counter()
    models.unload_vlm()
    result.model_unload_s = time.perf_counter() - t_unload

    # 6. GC
    t_gc = time.perf_counter()
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    time.sleep(0.3)  # Let OS reclaim
    result.gc_s = time.perf_counter() - t_gc

    result.mem_peak = peak
    result.mem_after_unload = snapshot_memory()
    result.total_s = result.model_load_s + result.processing_s + result.model_unload_s + result.gc_s
    result.per_file_s = result.total_s / n if n > 0 else 0
    result.files_per_min = 60.0 / result.per_file_s if result.per_file_s > 0 else 0

    return result


def bench_vv_phase(images: list, config: dict) -> PhaseResult:
    """Benchmark VV (SigLIP2) phase standalone with full model lifecycle."""
    from backend.pipeline.model_manager import ModelManager

    n = len(images)
    batch_size = config.get("vv_batch", 8)
    result = PhaseResult(phase="vv", model_id="", file_count=n, batch_size=batch_size)

    models = ModelManager()

    result.mem_before_load = snapshot_memory()

    # Load
    t_load = time.perf_counter()
    encoder = models.get_vv_encoder()
    result.model_load_s = time.perf_counter() - t_load
    result.model_id = getattr(encoder, 'model_name', type(encoder).__name__)
    result.mem_after_load = snapshot_memory()
    peak = result.mem_after_load

    print(f"    SigLIP2 loaded: {result.model_id} ({result.model_load_s:.1f}s)")

    # Process in batches
    t_proc = time.perf_counter()
    ok = 0
    pil_images = [img for img, _, _ in images]
    for start in range(0, n, batch_size):
        chunk = pil_images[start:start + batch_size]
        try:
            if hasattr(encoder, "encode_image_batch") and len(chunk) > 1:
                vecs = encoder.encode_image_batch(chunk)
            else:
                vecs = [encoder.encode_image(img) for img in chunk]
            ok += len(vecs)
        except Exception as e:
            print(f"    VV batch [{start}:{start+len(chunk)}] FAIL: {e}")
        peak = mem_max(peak, snapshot_memory())

    result.processing_s = time.perf_counter() - t_proc
    result.success_count = ok
    result.fail_count = n - ok

    # Unload
    t_unload = time.perf_counter()
    models.unload_vv()
    result.model_unload_s = time.perf_counter() - t_unload

    # GC
    t_gc = time.perf_counter()
    gc.collect()
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    time.sleep(0.3)
    result.gc_s = time.perf_counter() - t_gc

    result.mem_peak = peak
    result.mem_after_unload = snapshot_memory()
    result.total_s = result.model_load_s + result.processing_s + result.model_unload_s + result.gc_s
    result.per_file_s = result.total_s / n if n > 0 else 0
    result.files_per_min = 60.0 / result.per_file_s if result.per_file_s > 0 else 0

    return result


def bench_mv_phase(images: list, config: dict) -> PhaseResult:
    """Benchmark MV (Qwen3-Embedding) phase standalone.

    Uses synthetic MC texts since MV needs captions as input.
    """
    from backend.pipeline.model_manager import ModelManager

    n = len(images)
    batch_size = config.get("mv_batch", 16)
    result = PhaseResult(phase="mv", model_id="", file_count=n, batch_size=batch_size)

    # Generate synthetic MC texts (realistic lengths)
    texts = []
    for _, fname, _ in images:
        texts.append(
            f"A detailed illustration of {fname}. "
            "Interior scene with furniture, decorative objects, warm lighting. "
            "Tags: interior, furniture, props, game_asset, illustration"
        )

    models = ModelManager()
    result.mem_before_load = snapshot_memory()

    # Load
    t_load = time.perf_counter()
    provider = models.get_mv_encoder()
    result.model_load_s = time.perf_counter() - t_load
    result.model_id = type(provider).__name__
    result.mem_after_load = snapshot_memory()
    peak = result.mem_after_load

    print(f"    MV loaded: {result.model_id} ({result.model_load_s:.1f}s)")

    # Process in batches
    t_proc = time.perf_counter()
    ok = 0
    for start in range(0, n, batch_size):
        chunk = texts[start:start + batch_size]
        try:
            if hasattr(provider, "encode_batch") and len(chunk) > 1:
                vecs = provider.encode_batch(chunk)
            else:
                vecs = [provider.encode(t) for t in chunk]
            ok += len(vecs)
        except Exception as e:
            print(f"    MV batch [{start}:{start+len(chunk)}] FAIL: {e}")
        peak = mem_max(peak, snapshot_memory())

    result.processing_s = time.perf_counter() - t_proc
    result.success_count = ok
    result.fail_count = n - ok

    # Unload
    t_unload = time.perf_counter()
    models.unload_mv()
    result.model_unload_s = time.perf_counter() - t_unload

    # GC
    t_gc = time.perf_counter()
    gc.collect()
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    time.sleep(0.3)
    result.gc_s = time.perf_counter() - t_gc

    result.mem_peak = peak
    result.mem_after_unload = snapshot_memory()
    result.total_s = result.model_load_s + result.processing_s + result.model_unload_s + result.gc_s
    result.per_file_s = result.total_s / n if n > 0 else 0
    result.files_per_min = 60.0 / result.per_file_s if result.per_file_s > 0 else 0

    return result


# ── Full Pipeline Benchmark ─────────────────────────────────────────────


def bench_full_pipeline(images: list, config: dict) -> PipelineResult:
    """
    Benchmark full V -> VV -> MV pipeline using PhaseRunner.

    This exercises the EXACT production code path including model
    load/unload/GC between phases.
    """
    from backend.pipeline.model_manager import ModelManager
    from backend.pipeline.phase_runner import PhaseRunner
    from backend.pipeline.protocols import FixedBatchStrategy, PhaseItem

    n = len(images)
    pipeline = PipelineResult(file_count=n)

    models = ModelManager()
    storage = NullStorageBackend()
    progress = TimingProgressReporter()
    batch_strategy = FixedBatchStrategy(
        vision=1,
        vv=config.get("vv_batch", 8),
        mv=config.get("mv_batch", 16),
    )
    runner = PhaseRunner(models, storage, progress, batch_strategy)

    # Create PhaseItems from loaded images
    items = []
    for img, fname, tmp_path in images:
        item = PhaseItem(
            file_path=tmp_path,
            thumb_path=tmp_path,
            folder_path=str(config.get("image_dir", "")),
        )
        items.append(item)

    mem_before = snapshot_memory()

    t0 = time.perf_counter()
    success, failed = runner.run_all(items)
    total = time.perf_counter() - t0

    mem_after = snapshot_memory()

    pipeline.total_s = total
    pipeline.files_per_min = (n / total * 60) if total > 0 else 0

    # Extract per-phase timing from progress events
    phase_times = {}
    for event in progress.events:
        phase = event["phase"]
        if phase not in phase_times:
            phase_times[phase] = {"first": event["t"], "last": event["t"], "count": 0}
        phase_times[phase]["last"] = event["t"]
        phase_times[phase]["count"] += 1

    for phase_name, pt in phase_times.items():
        pr = PhaseResult(
            phase=phase_name,
            model_id="(via PhaseRunner)",
            file_count=pt["count"],
            batch_size=batch_strategy.get_batch_size(phase_name),
            processing_s=round(pt["last"] - pt["first"], 2),
            success_count=pt["count"],
        )
        pipeline.phases[phase_name] = pr

    # Store memory info on first phase
    if pipeline.phases:
        first_phase = list(pipeline.phases.values())[0]
        first_phase.mem_before_load = mem_before
        first_phase.mem_peak = progress.peak_mem
        first_phase.mem_after_unload = mem_after

    print(f"    Pipeline: {success} ok, {failed} failed, {total:.1f}s total")

    return pipeline


# ── Output Formatting ───────────────────────────────────────────────────


def format_console_table(phase_results: Dict[str, PhaseResult],
                         pipeline: Optional[PipelineResult],
                         preprocess_s: float, file_count: int):
    """Print formatted console summary."""
    sep = "─" * 78
    print()
    print("=" * 78)
    print("  Worker Simulation Benchmark")
    print("=" * 78)
    print(f"  Images: {file_count}")
    print(f"  PSD preprocess: {preprocess_s:.1f}s ({preprocess_s/file_count:.2f}s/file)")
    print(sep)

    # Phase table
    header = f"  {'Phase':<10} {'Model':<28} {'Load':>6} {'Proc':>7} {'Unload':>7} {'Total':>7} {'f/min':>7}"
    print(header)
    print(f"  {'─'*10} {'─'*28} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for name, pr in phase_results.items():
        model_short = pr.model_id[:28] if pr.model_id else "?"
        print(
            f"  {pr.phase:<10} {model_short:<28} "
            f"{pr.model_load_s:>5.1f}s {pr.processing_s:>6.1f}s "
            f"{pr.model_unload_s:>6.1f}s {pr.total_s:>6.1f}s "
            f"{pr.files_per_min:>6.1f}"
        )

    if pipeline:
        print(f"  {'─'*10}")
        print(
            f"  {'Full':.<10} {'(V->VV->MV via PhaseRunner)':<28} "
            f"{'':>6} {'':>7} {'':>7} "
            f"{pipeline.total_s:>6.1f}s {pipeline.files_per_min:>6.1f}"
        )

    # Memory table
    print()
    print(f"  {'Memory (MB)':<16} {'RSS':>8} {'Metal':>8} {'MPS':>8}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}")

    for name, pr in phase_results.items():
        _print_mem_row(f"  {pr.phase} before", pr.mem_before_load)
        _print_mem_row(f"  {pr.phase} peak", pr.mem_peak)
        _print_mem_row(f"  {pr.phase} after", pr.mem_after_unload)

    print(sep)
    print()


def _print_mem_row(label: str, mem: MemSnapshot):
    def _fmt(v):
        return f"{v:>7.0f}" if v >= 0 else "    n/a"
    print(f"  {label:<16} {_fmt(mem.rss_mb)} {_fmt(mem.metal_mb)} {_fmt(mem.mps_mb)}")


def build_json_report(phase_results: Dict[str, PhaseResult],
                      pipeline: Optional[PipelineResult],
                      preprocess_s: float, config: dict) -> dict:
    """Build JSON-serializable report dict."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": _get_system_info(),
        "config": {
            "image_count": config.get("count", 0),
            "image_dir": str(config.get("image_dir", "")),
            "vv_batch_size": config.get("vv_batch", 8),
            "mv_batch_size": config.get("mv_batch", 16),
            "max_edge": config.get("max_edge", 768),
        },
        "psd_preprocess": {
            "total_s": round(preprocess_s, 2),
            "per_file_s": round(preprocess_s / max(config.get("count", 1), 1), 3),
        },
        "phases": {},
        "reference_speeds": {},
    }

    for name, pr in phase_results.items():
        report["phases"][name] = {
            "model_id": pr.model_id,
            "file_count": pr.file_count,
            "batch_size": pr.batch_size,
            "model_load_s": round(pr.model_load_s, 2),
            "processing_s": round(pr.processing_s, 2),
            "model_unload_s": round(pr.model_unload_s, 2),
            "gc_s": round(pr.gc_s, 2),
            "total_s": round(pr.total_s, 2),
            "per_file_s": round(pr.per_file_s, 3),
            "files_per_min": round(pr.files_per_min, 1),
            "success": pr.success_count,
            "failed": pr.fail_count,
            "memory": {
                "before_load": asdict(pr.mem_before_load),
                "after_load": asdict(pr.mem_after_load),
                "peak": asdict(pr.mem_peak),
                "after_unload": asdict(pr.mem_after_unload),
            },
        }
        report["reference_speeds"][f"{name}_files_per_min"] = round(pr.files_per_min, 1)

    if pipeline:
        report["full_pipeline"] = {
            "total_s": round(pipeline.total_s, 2),
            "files_per_min": round(pipeline.files_per_min, 1),
            "file_count": pipeline.file_count,
            "phases": {
                k: {
                    "processing_s": round(v.processing_s, 2),
                    "file_count": v.file_count,
                }
                for k, v in pipeline.phases.items()
            },
        }
        report["reference_speeds"]["full_files_per_min"] = round(pipeline.files_per_min, 1)

    return report


def _get_system_info() -> dict:
    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import mlx.core as mx
        info["mlx_version"] = mx.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    # System memory
    try:
        import psutil
        info["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    return info


# ── Compare Mode ────────────────────────────────────────────────────────


def compare_results(current: dict, previous: dict):
    """Print comparison between two benchmark results."""
    print()
    print("=" * 78)
    print("  Comparison with previous result")
    print("=" * 78)
    print(f"  Previous: {previous.get('timestamp', '?')}")
    print(f"  Current:  {current.get('timestamp', '?')}")
    print()

    header = f"  {'Phase':<10} {'Prev f/min':>12} {'Curr f/min':>12} {'Delta':>10} {'Change':>10}"
    print(header)
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

    prev_speeds = previous.get("reference_speeds", {})
    curr_speeds = current.get("reference_speeds", {})

    all_keys = set(list(prev_speeds.keys()) + list(curr_speeds.keys()))
    for key in sorted(all_keys):
        prev_val = prev_speeds.get(key, 0)
        curr_val = curr_speeds.get(key, 0)
        delta = curr_val - prev_val
        pct = (delta / prev_val * 100) if prev_val > 0 else 0
        sign = "+" if delta >= 0 else ""
        label = key.replace("_files_per_min", "")
        print(
            f"  {label:<10} {prev_val:>11.1f} {curr_val:>11.1f} "
            f"{sign}{delta:>9.1f} {sign}{pct:>8.1f}%"
        )
    print()


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Worker Simulation Benchmark — measures real pipeline speed"
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="Image directory")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of images to process")
    parser.add_argument("--max-edge", type=int, default=768,
                        help="Max image edge for resize")
    parser.add_argument("--vv-batch", type=int, default=8,
                        help="VV sub-batch size")
    parser.add_argument("--mv-batch", type=int, default=16,
                        help="MV sub-batch size")
    parser.add_argument("--phases", type=str, default="mc,vv,mv",
                        help="Comma-separated phases to benchmark: mc,vv,mv")
    parser.add_argument("--no-full", action="store_true",
                        help="Skip full pipeline benchmark")
    parser.add_argument("--output", type=Path, default=None,
                        help="JSON output file path")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Path to previous JSON result for comparison")
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",")]
    config = {
        "image_dir": args.images,
        "count": args.count,
        "max_edge": args.max_edge,
        "vv_batch": args.vv_batch,
        "mv_batch": args.mv_batch,
    }

    print()
    print("=" * 78)
    print("  Worker Simulation Benchmark — Loading images...")
    print("=" * 78)

    # Load images
    images, preprocess_s, tmp_dir = load_psd_images(
        args.images, args.count, args.max_edge
    )
    actual_count = len(images)
    config["count"] = actual_count
    print(f"  Loaded {actual_count} images in {preprocess_s:.1f}s "
          f"({preprocess_s/actual_count:.2f}s/file)")
    print()

    phase_results: Dict[str, PhaseResult] = {}

    # Phase benchmarks
    if "mc" in phases:
        print("  Phase MC (VLM):")
        phase_results["mc"] = bench_mc_phase(images, config)
        print(f"    => {phase_results['mc'].files_per_min:.1f} files/min, "
              f"{phase_results['mc'].success_count}/{actual_count} ok")
        print()

    if "vv" in phases:
        print("  Phase VV (SigLIP2):")
        phase_results["vv"] = bench_vv_phase(images, config)
        print(f"    => {phase_results['vv'].files_per_min:.1f} files/min, "
              f"{phase_results['vv'].success_count}/{actual_count} ok")
        print()

    if "mv" in phases:
        print("  Phase MV (Qwen3-Embedding):")
        phase_results["mv"] = bench_mv_phase(images, config)
        print(f"    => {phase_results['mv'].files_per_min:.1f} files/min, "
              f"{phase_results['mv'].success_count}/{actual_count} ok")
        print()

    # Full pipeline
    pipeline = None
    if not args.no_full and len(phases) >= 2:
        print("  Full Pipeline (V -> VV -> MV via PhaseRunner):")
        pipeline = bench_full_pipeline(images, config)
        print(f"    => {pipeline.files_per_min:.1f} files/min")
        print()

    # Close images
    for img, _, _ in images:
        try:
            img.close()
        except Exception:
            pass

    # Cleanup temp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # Output
    format_console_table(phase_results, pipeline, preprocess_s, actual_count)

    # Save JSON
    report = build_json_report(phase_results, pipeline, preprocess_s, config)

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"worker_sim_{ts}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON saved: {output_path}")

    # Compare
    if args.compare and args.compare.exists():
        with open(args.compare, "r", encoding="utf-8") as f:
            prev = json.load(f)
        compare_results(report, prev)

    print()


if __name__ == "__main__":
    main()
