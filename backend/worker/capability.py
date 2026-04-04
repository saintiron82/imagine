"""
Worker capability spec — hardware profile + measured speed.

Two parts:
1. Hardware spec (GPU, VRAM) — collected instantly
2. Speed profile (mc/vv/mv files/min) — measured once with test image, cached

Sent to server on connect so scheduler can:
- Classify GPU class (strong/weak/cpu)
- Use measured speeds for optimal batch sizing (skip cold start trial)

Speed profile is saved to ~/.imagine/worker_profile.json and reused
across restarts. Re-benchmark only when GPU changes.
"""

import json
import logging
import os
import platform
import socket
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Profile cache location
PROFILE_DIR = Path.home() / ".imagine"
PROFILE_FILE = PROFILE_DIR / "worker_profile.json"
BENCH_SAMPLE_FILE = PROFILE_DIR / "sample.png"


def collect_capability(include_speed: bool = True) -> Dict[str, Any]:
    """Collect worker hardware capability spec + cached speed profile.

    Args:
        include_speed: If True, include speed profile from cache.
            Does NOT run benchmark — call run_benchmark() separately first.

    Returns:
        {
            "hostname": "my-pc",
            "os": "Darwin" | "Windows" | "Linux",
            "cpu_count": 10,
            "memory_total_gb": 32.0,
            "gpu_name": "Apple M5" | "NVIDIA RTX 4090" | "",
            "gpu_type": "mps" | "cuda" | "cpu",
            "vram_gb": 32.0,
            "is_metal": True | False,
            "compute_capability": "8.9" | None,
            "mc_speed": 7.8 | None,   # files/min (from cached profile)
            "vv_speed": 81.0 | None,
            "mv_speed": 119.0 | None,
        }
    """
    spec = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "cpu_name": _get_cpu_name(),
        "cpu_count": os.cpu_count() or 1,
        "memory_total_gb": _get_system_memory_gb(),
        "gpu_name": "",
        "gpu_type": "cpu",
        "vram_gb": 0.0,
        "is_metal": False,
        "compute_capability": None,
    }

    # Try GPU detection
    try:
        import torch
    except ImportError:
        return spec

    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            spec["gpu_name"] = props.name
            spec["gpu_type"] = "cuda"
            spec["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
            spec["compute_capability"] = f"{props.major}.{props.minor}"
        except Exception as e:
            logger.warning(f"CUDA detection failed: {e}")
        return spec

    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        spec["gpu_type"] = "mps"
        spec["is_metal"] = True
        spec["gpu_name"] = _get_apple_chip_name()
        spec["vram_gb"] = _get_system_memory_gb()  # unified memory

    # Attach cached speed profile + scores if available
    if include_speed:
        profile = load_speed_profile()
        if profile:
            spec["mc_speed"] = profile.get("mc_speed")
            spec["vv_speed"] = profile.get("vv_speed")
            spec["mv_speed"] = profile.get("mv_speed")
            scores = profile.get("scores")
            if not scores:
                scores = calculate_scores(profile)
            spec["scores"] = scores

    return spec


def _get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    # Fallback: read from /proc or sysctl
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _get_cpu_name() -> str:
    """Get CPU brand string."""
    try:
        if platform.system() == "Darwin":
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if r.stdout.strip():
                return r.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Windows":
            return platform.processor() or "Unknown"
    except Exception:
        pass
    return platform.processor() or "Unknown"


def _get_apple_chip_name() -> str:
    """Get Apple chip name (M1/M2/M3/M4/M5)."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        brand = result.stdout.strip()
        if brand:
            return f"Apple Silicon ({brand})"
    except Exception:
        pass
    return "Apple Silicon (MPS)"


# ── Speed Profile ───────────────────────────────────────────

def load_speed_profile() -> Optional[Dict[str, float]]:
    """Load cached speed profile from disk.

    Returns:
        {"mc_speed": 7.8, "vv_speed": 81.0, "mv_speed": 119.0, "gpu_name": "..."}
        or None if not available or GPU changed.
    """
    if not PROFILE_FILE.exists():
        return None

    try:
        data = json.loads(PROFILE_FILE.read_text())
        # Validate: must have speed values
        if not data.get("mc_speed") and not data.get("vv_speed"):
            return None
        # Check if GPU changed since last benchmark
        current_gpu = collect_capability(include_speed=False).get("gpu_name", "")
        if data.get("gpu_name") and data["gpu_name"] != current_gpu:
            logger.info(
                f"GPU changed ({data['gpu_name']} → {current_gpu}), "
                f"speed profile invalidated"
            )
            return None
        return data
    except Exception as e:
        logger.warning(f"Failed to load speed profile: {e}")
        return None


def save_speed_profile(profile: Dict[str, Any]):
    """Save speed profile to disk."""
    try:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        PROFILE_FILE.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False)
        )
        logger.info(f"Speed profile saved: {PROFILE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save speed profile: {e}")


def ensure_sample_image() -> Optional[str]:
    """Ensure a sample image exists for benchmarking.

    Looks for sample.png in ~/.imagine/. If not present, copies from:
    1. Project thumbnails directory (real content, best for VLM benchmark)
    2. Fallback: generates a simple test image

    Returns path to the image, or None on failure.
    """
    if BENCH_SAMPLE_FILE.exists():
        return str(BENCH_SAMPLE_FILE)

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Try 1: Copy an existing thumbnail from project
    try:
        import shutil
        project_root = Path(__file__).parent.parent.parent
        thumb_dir = project_root / "output" / "thumbnails"
        if thumb_dir.exists():
            # Pick the first real thumbnail
            for f in sorted(thumb_dir.glob("*_thumb.png")):
                if f.stat().st_size > 10000:  # skip tiny files
                    shutil.copy2(str(f), str(BENCH_SAMPLE_FILE))
                    logger.info(f"Sample image: copied {f.name} → {BENCH_SAMPLE_FILE}")
                    return str(BENCH_SAMPLE_FILE)
    except Exception:
        pass

    # Try 2: Generate a simple test image
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (1024, 1024), (30, 30, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 400, 400], fill=(200, 50, 50))
        draw.ellipse([500, 200, 800, 500], fill=(50, 150, 200))
        draw.rectangle([300, 600, 700, 900], fill=(50, 200, 100))
        img.save(str(BENCH_SAMPLE_FILE), "PNG")
        logger.info(f"Sample image: generated → {BENCH_SAMPLE_FILE}")
        return str(BENCH_SAMPLE_FILE)
    except Exception as e:
        logger.warning(f"Failed to create sample image: {e}")
        return None


def run_benchmark() -> Dict[str, Any]:
    """Run local benchmark: measure MC/VV/MV speed with test image.

    Returns speed profile dict with measured files/min per phase.
    Results are saved to disk for reuse.
    """
    bench_image = ensure_sample_image()
    if not bench_image:
        logger.warning("No bench image available, skipping benchmark")
        return {}

    cap = collect_capability()
    profile = {
        "gpu_name": cap.get("gpu_name", ""),
        "gpu_type": cap.get("gpu_type", "cpu"),
        "vram_gb": cap.get("vram_gb", 0),
        "benchmarked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info("Running worker benchmark...")

    # VV benchmark (SigLIP2)
    try:
        vv_speed = _bench_vv(bench_image)
        profile["vv_speed"] = round(vv_speed, 1)
        logger.info(f"  VV: {vv_speed:.1f} files/min")
    except Exception as e:
        logger.warning(f"  VV benchmark failed: {e}")
        profile["vv_speed"] = 0  # 0 = failed (not None = unmeasured)

    # MV benchmark (Qwen3-Embedding)
    try:
        mv_speed = _bench_mv()
        profile["mv_speed"] = round(mv_speed, 1)
        logger.info(f"  MV: {mv_speed:.1f} files/min")
    except Exception as e:
        logger.warning(f"  MV benchmark failed: {e}")
        profile["mv_speed"] = 0

    # MC benchmark (VLM) — always try (CPU/Ollama can also run VLM)
    try:
        mc_speed = _bench_mc(bench_image)
        profile["mc_speed"] = round(mc_speed, 1)
        logger.info(f"  MC: {mc_speed:.1f} files/min")
    except Exception as e:
        logger.warning(f"  MC benchmark failed: {e}")
        profile["mc_speed"] = 0  # 0 = tried but failed → incapable

    # Calculate scores
    profile["scores"] = calculate_scores(profile)

    save_speed_profile(profile)
    return profile


def calculate_scores(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate performance scores with difficulty multiplier.

    MC is ~400× harder than MV per file. Score reflects this:
      weighted_score = speed × difficulty_multiplier

    Difficulty multiplier (MV 기준):
      MC: ×397 (MC 1파일 처리 = MV 397파일 가치)
      VV: ×10  (VV 1파일 처리 = MV 10파일 가치)
      MV: ×1   (기준)

    Reference M5 32GB: MC=8.6/m, VV=347/m, MV=3412/m
    Multiplier = MV_ref / phase_ref = { MC: 3412/8.6=397, VV: 3412/347=10 }
    """
    # Difficulty multipliers (MV-equivalent, from M5 reference)
    DIFFICULTY = {"mc": 397, "vv": 10, "mv": 1}

    scores = {
        "difficulty": dict(DIFFICULTY),
        "phases": {},        # 개별 phase 점수
        "incapable": [],     # 불능 phase 목록
    }

    total_weighted = 0
    for phase in ("mc", "vv", "mv"):
        speed = profile.get(f"{phase}_speed")

        if speed is not None and speed == 0:
            # 0 = benchmark 시도했지만 실패 → 불능
            scores["phases"][phase] = {
                "speed": 0,
                "weighted": 0,
                "status": "incapable",
            }
            scores["incapable"].append(phase)
        elif speed and speed > 0:
            raw = round(speed, 1)
            weighted = round(raw * DIFFICULTY[phase])
            scores["phases"][phase] = {
                "speed": raw,
                "weighted": weighted,
                "status": "ok",
            }
            total_weighted += weighted
        else:
            # None = 미측정
            scores["phases"][phase] = {
                "speed": None,
                "weighted": None,
                "status": "unmeasured",
            }

    # 종합 점수 = 가중 합계
    scores["total"] = total_weighted

    # 등급 = MC 속도 기준 (파이프라인 병목)
    mc_speed = scores["phases"].get("mc", {}).get("speed") or 0
    if mc_speed >= 12:
        scores["grade"] = "S"
    elif mc_speed >= 7:
        scores["grade"] = "A"
    elif mc_speed >= 3:
        scores["grade"] = "B"
    elif mc_speed > 0:
        scores["grade"] = "C"
    else:
        scores["grade"] = "F"

    return scores


def _bench_vv(image_path: str) -> float:
    """Benchmark VV (SigLIP2) — encode 1 image, return files/min."""
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    from PIL import Image as PILImage

    encoder = SigLIP2Encoder()
    img = PILImage.open(image_path).convert("RGB")

    # Warm up
    encoder.encode_image(img)

    # Measure
    t = time.perf_counter()
    for _ in range(3):
        encoder.encode_image(img)
    elapsed = (time.perf_counter() - t) / 3

    return 60.0 / elapsed if elapsed > 0 else 0


def _bench_mv() -> float:
    """Benchmark MV (Qwen3-Embedding) — encode 1 text, return files/min."""
    from backend.vector.text_embedding import get_text_embedding_provider

    provider = get_text_embedding_provider()
    test_text = "A fantasy knight holding a sword in full armor, anime style character illustration"

    # Warm up
    provider.encode(test_text)

    # Measure
    t = time.perf_counter()
    for _ in range(3):
        provider.encode(test_text)
    elapsed = (time.perf_counter() - t) / 3

    return 60.0 / elapsed if elapsed > 0 else 0


def _bench_mc(image_path: str) -> float:
    """Benchmark MC (VLM) — generate caption for 1 image, return files/min."""
    from backend.vision.vision_factory import get_vision_analyzer
    from PIL import Image as PILImage

    analyzer = get_vision_analyzer()
    img = PILImage.open(image_path).convert("RGB")

    # Warm up (first call loads model)
    analyzer.analyze(img)

    # Measure
    t = time.perf_counter()
    analyzer.analyze(img)
    elapsed = time.perf_counter() - t

    return 60.0 / elapsed if elapsed > 0 else 0
