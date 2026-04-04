#!/usr/bin/env python3
"""
VV (Visual Vector) Quality Benchmark
======================================
Evaluates VV encoder quality using two metrics:

1. Self-Consistency (0-50):
   Apply transforms (crop, flip, brightness, blur, resize) to each image,
   measure cosine similarity between original and transformed vectors.
   A good encoder should be robust to these augmentations.

2. Retrieval@K (0-50):
   For each image, find top-K nearest neighbors in the corpus.
   Images from the same group (filename prefix or folder) should rank higher.
   Measures how well the encoder captures visual similarity for retrieval.

Total score: 0-100.

Usage:
    python tools/bench_vv_quality.py                          # default: 30 images
    python tools/bench_vv_quality.py --count 50               # more images
    python tools/bench_vv_quality.py --profile benchmarks/profiles/vv_siglip2.yaml
    python tools/bench_vv_quality.py --encoder-model google/siglip2-so400m-patch16-naflex
"""

import argparse
import gc
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE_DIR = Path("/Users/saintiron/imageDB/마캬베리즈무/실내소품")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ── Image Transforms for Self-Consistency ──────────────────────────────


def augment_center_crop(img: Image.Image, ratio: float = 0.7) -> Image.Image:
    """Center crop to ratio of original size."""
    w, h = img.size
    nw, nh = int(w * ratio), int(h * ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def augment_horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def augment_brightness(img: Image.Image, factor: float = 1.4) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def augment_darken(img: Image.Image, factor: float = 0.6) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def augment_blur(img: Image.Image, radius: float = 3.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def augment_small_resize(img: Image.Image, scale: float = 0.5) -> Image.Image:
    """Downscale then upscale back — simulates quality loss."""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return small.resize((w, h), Image.LANCZOS)


def augment_color_jitter(img: Image.Image) -> Image.Image:
    """Shift color saturation."""
    return ImageEnhance.Color(img).enhance(1.5)


AUGMENTATIONS = {
    "center_crop_70": lambda img: augment_center_crop(img, 0.7),
    "center_crop_50": lambda img: augment_center_crop(img, 0.5),
    "h_flip": augment_horizontal_flip,
    "brighten_1.4x": lambda img: augment_brightness(img, 1.4),
    "darken_0.6x": lambda img: augment_darken(img, 0.6),
    "blur_r3": lambda img: augment_blur(img, 3.0),
    "resize_50pct": lambda img: augment_small_resize(img, 0.5),
    "color_jitter": augment_color_jitter,
}


# ── Group Extraction (for Retrieval@K) ─────────────────────────────────


def extract_group(filename: str) -> str:
    """
    Extract group ID from filename for Retrieval@K.

    Strategy: use the prefix before the first underscore or number change.
    e.g. "#02_01.psd" → "#02", "00-002-v08R.psd" → "00-002",
         "001_EP14.psd" → "001"
    """
    stem = Path(filename).stem

    # Pattern 1: #NN_ prefix
    m = re.match(r'(#\d+)', stem)
    if m:
        return m.group(1)

    # Pattern 2: NNN_ or NNN- prefix (3+ digits)
    m = re.match(r'(\d{2,4})', stem)
    if m:
        return m.group(1)

    # Pattern 3: alphabetic prefix before first digit
    m = re.match(r'([a-zA-Z]+)', stem)
    if m:
        return m.group(1)

    return stem[:6]


# ── Image Loading ──────────────────────────────────────────────────────


def load_images(image_dir: Path, n: int, max_edge: int = 512):
    """Load N images, return list of (PIL.Image, filename)."""
    from psd_tools import PSDImage

    files = sorted(image_dir.glob("*.psd"))[:n]
    if not files:
        files = sorted(image_dir.glob("*.png"))[:n]
        files += sorted(image_dir.glob("*.jpg"))[:n]
        files = files[:n]

    if not files:
        raise FileNotFoundError(f"No image files in {image_dir}")

    results = []
    for pf in files:
        try:
            if pf.suffix.lower() == ".psd":
                psd = PSDImage.open(str(pf))
                img = psd.composite()
            else:
                img = Image.open(str(pf))

            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            scale = max_edge / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            results.append((img, pf.name))
        except Exception as e:
            print(f"  SKIP {pf.name}: {e}")

    return results


# ── Scoring: Self-Consistency ──────────────────────────────────────────


def score_self_consistency(
    images: List[Tuple[Image.Image, str]], encoder
) -> Dict:
    """
    Score 0-50: How robust is the encoder to image transforms?

    For each image, encode original + N augmented versions.
    Measure mean cosine similarity between original and each augmented.

    Expected cosine ranges for a good encoder:
      - h_flip: 0.85-0.95 (mirror is semantically similar)
      - center_crop_70: 0.80-0.95 (most content preserved)
      - center_crop_50: 0.60-0.85 (less content)
      - brightness: 0.90-0.99 (minor change)
      - blur: 0.85-0.95 (shape preserved)
      - resize: 0.90-0.99 (same content, quality loss)

    Score mapping: mean cosine [0.5, 0.95] → [0, 50]
    """
    n = len(images)
    aug_names = list(AUGMENTATIONS.keys())
    all_sims = {aug: [] for aug in aug_names}
    per_image = []

    print(f"    Self-consistency: {n} images × {len(aug_names)} transforms")

    for i, (img, fname) in enumerate(images):
        orig_vec = encoder.encode_image(img)
        img_sims = {}

        for aug_name, aug_fn in AUGMENTATIONS.items():
            try:
                aug_img = aug_fn(img)
                aug_vec = encoder.encode_image(aug_img)
                cos_sim = float(np.dot(orig_vec, aug_vec))
                all_sims[aug_name].append(cos_sim)
                img_sims[aug_name] = round(cos_sim, 4)
            except Exception as e:
                img_sims[aug_name] = -1.0

        per_image.append({"file": fname, "sims": img_sims})

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"      [{i+1}/{n}] done")

    # Aggregate
    aug_means = {}
    for aug_name, sims in all_sims.items():
        valid = [s for s in sims if s > 0]
        aug_means[aug_name] = round(np.mean(valid), 4) if valid else 0.0

    overall_mean = np.mean(list(aug_means.values()))

    # Score: [0.5, 0.95] → [0, 50]
    normalized = max(0, min(1, (overall_mean - 0.5) / 0.45))
    score = round(normalized * 50, 1)

    return {
        "score": score,
        "overall_mean_cosine": round(overall_mean, 4),
        "per_augmentation": aug_means,
        "per_image": per_image,
    }


# ── Scoring: Retrieval@K ──────────────────────────────────────────────


def score_retrieval_at_k(
    images: List[Tuple[Image.Image, str]], encoder,
    k_values: List[int] = [3, 5, 10],
) -> Dict:
    """
    Score 0-50: How well does the encoder retrieve similar images?

    For each image, find top-K nearest neighbors (excluding self).
    Check if neighbors belong to the same group (filename prefix).
    Measures Precision@K for each K.

    Score: weighted average of P@3, P@5, P@10 mapped to [0, 50].
    """
    n = len(images)

    # Encode all images
    print(f"    Retrieval@K: encoding {n} images...")
    vectors = []
    groups = []
    for img, fname in images:
        vec = encoder.encode_image(img)
        vectors.append(vec)
        groups.append(extract_group(fname))

    vectors = np.array(vectors)  # (N, dim)

    # Check group distribution
    from collections import Counter
    group_counts = Counter(groups)
    groups_with_pairs = sum(1 for c in group_counts.values() if c >= 2)
    print(f"    Groups: {len(group_counts)} unique, {groups_with_pairs} with 2+ members")

    if groups_with_pairs < 2:
        print("    WARNING: Not enough groups with multiple members for Retrieval@K")
        return {
            "score": 0,
            "warning": "Insufficient groups for retrieval evaluation",
            "group_distribution": dict(group_counts),
        }

    # Compute all-pairs cosine similarity
    # vectors are L2-normalized (SigLIP2 output), so dot product = cosine
    sim_matrix = vectors @ vectors.T  # (N, N)

    # Evaluate Retrieval@K
    precision_at_k = {k: [] for k in k_values}
    per_image_results = []

    for i in range(n):
        my_group = groups[i]
        # Same-group count (excluding self)
        same_group_total = group_counts[my_group] - 1
        if same_group_total == 0:
            continue  # Skip singletons

        # Get similarities, exclude self
        sims = sim_matrix[i].copy()
        sims[i] = -1.0  # exclude self
        ranked = np.argsort(sims)[::-1]  # descending

        img_result = {"file": images[i][1], "group": my_group, "precision": {}}

        for k in k_values:
            top_k = ranked[:k]
            same_in_k = sum(1 for j in top_k if groups[j] == my_group)
            # Precision = hits / min(k, total_possible)
            max_possible = min(k, same_group_total)
            p_at_k = same_in_k / max_possible if max_possible > 0 else 0
            precision_at_k[k].append(p_at_k)
            img_result["precision"][f"P@{k}"] = round(p_at_k, 3)

        per_image_results.append(img_result)

    # Aggregate
    mean_precision = {}
    for k in k_values:
        vals = precision_at_k[k]
        mean_precision[f"P@{k}"] = round(np.mean(vals), 4) if vals else 0.0

    # Score: weighted average P@3(0.5) + P@5(0.3) + P@10(0.2) → [0, 50]
    weights = {3: 0.5, 5: 0.3, 10: 0.2}
    weighted_avg = sum(
        mean_precision.get(f"P@{k}", 0) * w
        for k, w in weights.items()
    )
    score = round(weighted_avg * 50, 1)

    return {
        "score": score,
        "mean_precision": mean_precision,
        "group_distribution": dict(group_counts.most_common(20)),
        "images_evaluated": len(per_image_results),
        "images_skipped_singleton": n - len(per_image_results),
        "per_image": per_image_results[:20],  # Limit output size
    }


# ── Report Builder ─────────────────────────────────────────────────────


def score_grade(total: float) -> str:
    if total >= 85:
        return "A"
    elif total >= 70:
        return "B"
    elif total >= 55:
        return "C"
    elif total >= 40:
        return "D"
    else:
        return "F"


def build_markdown_report(
    encoder_id: str,
    sc_result: Dict,
    ret_result: Dict,
    timing: Dict,
    image_count: int,
) -> str:
    total = sc_result["score"] + ret_result["score"]
    lines = [
        f"# VV Quality Benchmark",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Encoder**: `{encoder_id}`",
        f"**Images**: {image_count}",
        f"",
        f"## Overall Score: {total:.1f}/100 ({score_grade(total)})",
        f"",
        f"| Component | Score | Details |",
        f"|---|:-:|---|",
        f"| Self-Consistency | {sc_result['score']:.1f}/50 | Mean cosine: {sc_result['overall_mean_cosine']:.4f} |",
        f"| Retrieval@K | {ret_result['score']:.1f}/50 | {', '.join(f'{k}={v:.3f}' for k, v in ret_result.get('mean_precision', {}).items())} |",
        f"",
        f"## Timing",
        f"",
        f"| Phase | Time |",
        f"|---|---|",
        f"| Image loading | {timing.get('load_s', 0):.1f}s |",
        f"| Model loading | {timing.get('model_load_s', 0):.1f}s |",
        f"| Self-consistency | {timing.get('sc_s', 0):.1f}s |",
        f"| Retrieval@K | {timing.get('ret_s', 0):.1f}s |",
        f"",
        f"---",
        f"",
        f"## Self-Consistency Detail",
        f"",
        f"Per-augmentation mean cosine similarity (higher = more robust):",
        f"",
        f"| Augmentation | Mean Cosine |",
        f"|---|:-:|",
    ]
    for aug, val in sc_result.get("per_augmentation", {}).items():
        bar = "█" * int(val * 20) if val > 0 else ""
        lines.append(f"| {aug} | {val:.4f} {bar} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Retrieval@K Detail",
        f"",
        f"Group distribution (top 15):",
        f"",
        f"| Group | Count |",
        f"|---|:-:|",
    ])
    for grp, cnt in list(ret_result.get("group_distribution", {}).items())[:15]:
        lines.append(f"| {grp} | {cnt} |")

    lines.extend([
        f"",
        f"Images evaluated: {ret_result.get('images_evaluated', 0)} "
        f"(skipped {ret_result.get('images_skipped_singleton', 0)} singletons)",
        f"",
    ])

    return "\n".join(lines)


# ── Memory Helpers ─────────────────────────────────────────────────────


def get_mem_info() -> Dict:
    info = {}
    try:
        import psutil
        info["rss_mb"] = round(psutil.Process().memory_info().rss / 1024**2, 1)
    except Exception:
        pass
    try:
        import torch
        if hasattr(torch.mps, "current_allocated_memory"):
            info["mps_mb"] = round(torch.mps.current_allocated_memory() / 1024**2, 1)
    except Exception:
        pass
    return info


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="VV (Visual Vector) Quality Benchmark")
    parser.add_argument("--profile", type=Path, default=None,
                        help="YAML profile file")
    parser.add_argument("--encoder-model", type=str, default=None,
                        help="Override VV encoder model ID")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load profile
    if args.profile:
        try:
            import yaml
            with open(args.profile, "r", encoding="utf-8") as f:
                profile = yaml.safe_load(f) or {}
        except ImportError:
            from ruamel.yaml import YAML
            with open(args.profile, "r", encoding="utf-8") as f:
                profile = YAML().load(f) or {}

        if "encoder_model" in profile and not args.encoder_model:
            args.encoder_model = profile["encoder_model"]
        if "count" in profile and args.count == 30:
            args.count = profile["count"]
        if "max_edge" in profile:
            args.max_edge = profile["max_edge"]

    timing = {}

    print("=" * 70)
    print("  VV Quality Benchmark")
    print("=" * 70)

    # Load images
    t0 = time.perf_counter()
    images = load_images(args.images, args.count, args.max_edge)
    timing["load_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Loaded {len(images)} images in {timing['load_s']}s")

    # Load encoder
    print(f"\n  Loading VV encoder...")
    t0 = time.perf_counter()
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    encoder = SigLIP2Encoder(model_name=args.encoder_model)
    # Warm up with first image
    _ = encoder.encode_image(images[0][0])
    timing["model_load_s"] = round(time.perf_counter() - t0, 1)
    encoder_id = encoder.model_name
    mem = get_mem_info()
    print(f"  Encoder: {encoder_id} ({timing['model_load_s']}s)")
    if mem:
        print(f"  Memory: {mem}")

    # 1. Self-Consistency
    print(f"\n{'─' * 70}")
    print("  1. Self-Consistency")
    t0 = time.perf_counter()
    sc_result = score_self_consistency(images, encoder)
    timing["sc_s"] = round(time.perf_counter() - t0, 1)
    print(f"    Score: {sc_result['score']:.1f}/50 "
          f"(mean cosine: {sc_result['overall_mean_cosine']:.4f})")

    # 2. Retrieval@K
    print(f"\n{'─' * 70}")
    print("  2. Retrieval@K")
    t0 = time.perf_counter()
    ret_result = score_retrieval_at_k(images, encoder)
    timing["ret_s"] = round(time.perf_counter() - t0, 1)
    print(f"    Score: {ret_result['score']:.1f}/50")
    for k, v in ret_result.get("mean_precision", {}).items():
        print(f"      {k}: {v:.3f}")

    # Total
    total = sc_result["score"] + ret_result["score"]

    # Cleanup
    del encoder
    gc.collect()
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass

    # Report
    report_md = build_markdown_report(
        encoder_id, sc_result, ret_result, timing, len(images)
    )

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"vv_quality_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # JSON
    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "encoder": encoder_id,
        "image_count": len(images),
        "total_score": total,
        "self_consistency": sc_result,
        "retrieval_at_k": ret_result,
        "timing": timing,
        "system_memory": mem,
    }
    # Trim per_image for JSON size
    raw["self_consistency"] = {
        k: v for k, v in sc_result.items() if k != "per_image"
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {total:.1f}/100 ({score_grade(total)})")
    print(f"    Self-Consistency: {sc_result['score']:.1f}/50")
    print(f"    Retrieval@K:     {ret_result['score']:.1f}/50")
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"  JSON:   {json_path}")
    print(f"{'=' * 70}")

    # Cleanup images
    for img, _ in images:
        try:
            img.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
