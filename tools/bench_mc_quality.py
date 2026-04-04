#!/usr/bin/env python3
"""
MC Quality Comparison with Auto-Scoring
=========================================
Compares VLM caption/tag quality between two models on identical images.
Outputs a side-by-side markdown report with automated quality scores.

Scoring System (0-100):
  - Visual Alignment (0-35): SigLIP2 image↔caption cosine similarity
  - Caption Quality  (0-25): length, specificity, concrete object mentions
  - Tag Quality      (0-25): count, diversity, specificity
  - Structure        (0-15): field completeness, valid classification

Usage:
    python tools/bench_mc_quality.py --profile benchmarks/profiles/ab_gemma4_vs_qwen35.yaml
    python tools/bench_mc_quality.py --ab MODEL_A MODEL_B --count 5
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
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_IMAGE_DIR = Path("/Users/saintiron/imageDB/마캬베리즈무/실내소품")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "results"


# ── Scoring Constants ──────────────────────────────────────────────────

# Vague/abstract words that indicate the model isn't really seeing the image
VAGUE_WORDS = {
    "passage", "threshold", "transition", "journey", "metaphor",
    "conceptual", "abstract", "representation", "symbol", "notion",
    "concept", "idea", "essence", "undefined", "ambiguous",
    "disparate", "miscellaneous",
}

# Concrete object words that indicate real visual understanding
CONCRETE_INDICATORS = {
    # Furniture
    "desk", "chair", "table", "bed", "sofa", "couch", "shelf", "bookshelf",
    "dresser", "cabinet", "drawer", "lamp", "chandelier", "curtain", "rug",
    # Room types
    "bedroom", "kitchen", "living room", "bathroom", "office", "library",
    "hallway", "dining", "study",
    # Objects
    "window", "door", "wall", "floor", "ceiling", "stairs",
    "plant", "flower", "vase", "painting", "mirror", "clock",
    "book", "bottle", "cup", "basket", "pillow", "blanket",
    # Outdoor
    "tree", "mountain", "lake", "river", "sky", "cloud", "sun",
    "house", "building", "road", "bridge", "boat", "dock",
    # Style
    "anime", "cartoon", "realistic", "pixel", "watercolor",
    # Colors
    "red", "blue", "green", "yellow", "pink", "purple", "orange",
    "wooden", "colorful", "bright", "dark", "warm", "cozy",
}

VALID_IMAGE_TYPES = {
    "character", "background", "ui", "icon", "effect", "item",
    "illustration", "concept_art", "texture", "other",
}


# ── Scoring Functions ──────────────────────────────────────────────────


def score_visual_alignment(
    image: Image.Image, caption: str, encoder
) -> float:
    """
    Score 0-35: SigLIP2 image↔caption cosine similarity.

    SigLIP2 cosine similarity typically ranges 0.05-0.35 for relevant pairs.
    We map this to 0-35 with a tuned curve.
    """
    if not caption or len(caption) < 5:
        return 0.0

    try:
        img_vec = encoder.encode_image(image)
        txt_vec = encoder.encode_text(caption)
        cos_sim = float(np.dot(img_vec, txt_vec))

        # Map SigLIP2 cosine range [0.05, 0.35] → [0, 35]
        # Below 0.05 = irrelevant, above 0.35 = exceptional
        normalized = max(0, min(1, (cos_sim - 0.05) / 0.30))
        return round(normalized * 35, 1)
    except Exception:
        return 0.0


def score_caption_quality(caption: str) -> float:
    """
    Score 0-25: Caption quality based on length, specificity, concreteness.

    Breakdown:
      - Length (0-8): Optimal 40-120 chars
      - Specificity (0-9): Penalize vague/abstract words
      - Concreteness (0-8): Reward concrete object mentions
    """
    if not caption:
        return 0.0

    caption_lower = caption.lower()
    words = set(re.findall(r'[a-z]+', caption_lower))

    # Length score (0-8)
    n = len(caption)
    if n < 10:
        len_score = 0
    elif n < 20:
        len_score = 2
    elif n < 40:
        len_score = 5
    elif n <= 120:
        len_score = 8  # sweet spot
    elif n <= 200:
        len_score = 6  # verbose
    else:
        len_score = 3  # too verbose

    # Specificity score (0-9): penalize vague words
    vague_count = len(words & VAGUE_WORDS)
    if vague_count == 0:
        spec_score = 9
    elif vague_count == 1:
        spec_score = 6
    elif vague_count == 2:
        spec_score = 3
    else:
        spec_score = 0  # mostly vague

    # Concreteness score (0-8): reward concrete objects
    # Check multi-word phrases too
    concrete_count = len(words & CONCRETE_INDICATORS)
    for phrase in ["living room", "dining table", "full body"]:
        if phrase in caption_lower:
            concrete_count += 1
    concrete_score = min(8, concrete_count * 2)

    return round(len_score + spec_score + concrete_score, 1)


def score_tag_quality(tags: list) -> float:
    """
    Score 0-25: Tag quality based on count, diversity, specificity.

    Breakdown:
      - Count (0-8): 5-8 tags optimal
      - Diversity (0-9): Unique, no near-duplicates
      - Specificity (0-8): Concrete vs abstract tags
    """
    if not tags or not isinstance(tags, list):
        return 0.0

    # Count score (0-8)
    n = len(tags)
    if n == 0:
        count_score = 0
    elif n < 3:
        count_score = 3
    elif n < 5:
        count_score = 6
    elif n <= 8:
        count_score = 8  # sweet spot
    elif n <= 12:
        count_score = 6
    else:
        count_score = 4  # too many

    # Diversity score (0-9): penalize duplicates / near-duplicates
    tag_set = set(t.lower().strip() for t in tags)
    unique_ratio = len(tag_set) / max(n, 1)
    # Check for near-duplicates (e.g., "threshold" and "threshold_crossing")
    stems = set()
    near_dup = 0
    for t in tag_set:
        stem = t.split("_")[0].split("-")[0][:6]
        if stem in stems:
            near_dup += 1
        stems.add(stem)
    dup_penalty = near_dup * 2
    diversity_score = max(0, round(unique_ratio * 9 - dup_penalty))

    # Specificity score (0-8): concrete vs vague tags
    concrete_tags = sum(
        1 for t in tag_set
        if any(c in t for c in CONCRETE_INDICATORS) or len(t) > 3
    )
    vague_tags = sum(1 for t in tag_set if t in VAGUE_WORDS)
    spec_ratio = (concrete_tags - vague_tags) / max(n, 1)
    spec_score = max(0, min(8, round((spec_ratio + 0.5) * 8)))

    return round(count_score + diversity_score + spec_score, 1)


def score_structure(result: Dict[str, Any]) -> float:
    """
    Score 0-15: Structural completeness.

    Breakdown:
      - JSON parse / no error (0-5)
      - Required fields present & non-empty (0-5)
      - Valid image_type (0-5)
    """
    if "_error" in result:
        return 0.0

    # Parse success (5)
    parse_score = 5

    # Field completeness (0-5)
    caption = result.get("mc_caption", "") or result.get("caption", "")
    tags = result.get("tags", []) or result.get("ai_tags", [])
    image_type = result.get("image_type", "")
    field_count = 0
    if caption:
        field_count += 1
    if tags:
        field_count += 1
    if image_type:
        field_count += 1
    field_score = round(field_count / 3 * 5, 1)

    # Valid image_type (0-5)
    type_score = 5 if image_type in VALID_IMAGE_TYPES else 2

    return round(parse_score + field_score + type_score, 1)


def score_mc(
    image: Image.Image, result: Dict[str, Any], encoder
) -> Dict[str, float]:
    """Compute all scores for a single MC result. Returns breakdown + total."""
    caption = result.get("mc_caption", "") or result.get("caption", "")
    tags = result.get("tags", []) or result.get("ai_tags", [])

    visual = score_visual_alignment(image, caption, encoder)
    caption_q = score_caption_quality(caption)
    tag_q = score_tag_quality(tags)
    structure = score_structure(result)
    total = visual + caption_q + tag_q + structure

    return {
        "visual_alignment": visual,
        "caption_quality": caption_q,
        "tag_quality": tag_q,
        "structure": structure,
        "total": round(total, 1),
    }


# ── Image Loading ──────────────────────────────────────────────────────


def load_images(image_dir: Path, n: int, max_edge: int = 768):
    """Load N images, return list of (PIL.Image, filename, tmp_path)."""
    from psd_tools import PSDImage

    files = sorted(image_dir.glob("*.psd"))[:n]
    if not files:
        files = sorted(image_dir.glob("*.png"))[:n]
        files += sorted(image_dir.glob("*.jpg"))[:n]
        files = files[:n]

    if not files:
        raise FileNotFoundError(f"No image files in {image_dir}")

    results = []
    tmp_dir = tempfile.mkdtemp(prefix="bench_quality_")

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

            tmp_path = os.path.join(tmp_dir, pf.stem + ".png")
            img.save(tmp_path)
            results.append((img, pf.name, tmp_path))
        except Exception as e:
            print(f"  SKIP {pf.name}: {e}")

    return results, tmp_dir


# ── Model Runner ───────────────────────────────────────────────────────


def run_model(model_id: str, images: list, domain,
              model_opts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run a VLM on all images, return list of MC results.

    Args:
        model_opts: Per-model settings from profile (concise, max_tokens,
                    temperature, system_prompt, stage2_prompt)
    """
    from backend.vision.mlx_adapter import MLXVisionAdapter

    opts = model_opts or {}
    print(f"\n  Loading {model_id}...")
    if opts:
        print(f"  Options: {opts}")
    t0 = time.perf_counter()
    adapter = MLXVisionAdapter(
        model=model_id,
        max_tokens=opts.get("max_tokens", 512),
        temperature=opts.get("temperature", 0.1),
        concise=opts.get("concise"),
        system_prompt=opts.get("system_prompt"),
        stage2_prompt=opts.get("stage2_prompt"),
    )
    adapter._load_model()
    load_time = time.perf_counter() - t0

    try:
        import mlx.core as mx
        metal_mb = mx.get_active_memory() / 1024 ** 2
        print(f"  Loaded in {load_time:.1f}s — Metal: {metal_mb:.0f}MB")
    except Exception:
        print(f"  Loaded in {load_time:.1f}s")

    results = []
    for i, (img, fname, _) in enumerate(images):
        t1 = time.perf_counter()
        try:
            r = adapter.classify_and_analyze(img, domain=domain)
            elapsed = time.perf_counter() - t1
            r["_file"] = fname
            r["_time_s"] = round(elapsed, 2)
            results.append(r)
            caption = r.get("mc_caption", "") or r.get("caption", "")
            tags = r.get("tags", []) or r.get("ai_tags", [])
            img_type = r.get("image_type", "?")
            print(f"    [{i+1}/{len(images)}] {fname} → {img_type} ({elapsed:.1f}s) "
                  f"caption={len(caption)}chars tags={len(tags)}")
        except Exception as e:
            elapsed = time.perf_counter() - t1
            print(f"    [{i+1}/{len(images)}] {fname} → FAIL ({e})")
            results.append({"_file": fname, "_time_s": round(elapsed, 2), "_error": str(e)})

    # Unload
    if hasattr(adapter, 'unload_model'):
        adapter.unload_model()
    del adapter
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass

    return results


# ── Report Builder ─────────────────────────────────────────────────────


def score_grade(total: float) -> str:
    """Map total score to letter grade."""
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
    model_a_id: str, model_b_id: str,
    results_a: list, results_b: list,
    scores_a: list, scores_b: list,
    images: list,
) -> str:
    """Build side-by-side markdown comparison with scores."""
    avg_score_a = sum(s["total"] for s in scores_a) / max(len(scores_a), 1)
    avg_score_b = sum(s["total"] for s in scores_b) / max(len(scores_b), 1)

    lines = [
        f"# MC Quality Comparison (Scored)",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model A**: `{model_a_id}`",
        f"**Model B**: `{model_b_id}`",
        f"**Images**: {len(images)}",
        f"",
        f"## Overall Scores",
        f"",
        f"| Metric | Model A | Model B |",
        f"|---|:-:|:-:|",
        f"| **Total (avg)** | **{avg_score_a:.1f}/100 ({score_grade(avg_score_a)})** | **{avg_score_b:.1f}/100 ({score_grade(avg_score_b)})** |",
        f"| Visual Alignment (/35) | {sum(s['visual_alignment'] for s in scores_a)/len(scores_a):.1f} | {sum(s['visual_alignment'] for s in scores_b)/len(scores_b):.1f} |",
        f"| Caption Quality (/25) | {sum(s['caption_quality'] for s in scores_a)/len(scores_a):.1f} | {sum(s['caption_quality'] for s in scores_b)/len(scores_b):.1f} |",
        f"| Tag Quality (/25) | {sum(s['tag_quality'] for s in scores_a)/len(scores_a):.1f} | {sum(s['tag_quality'] for s in scores_b)/len(scores_b):.1f} |",
        f"| Structure (/15) | {sum(s['structure'] for s in scores_a)/len(scores_a):.1f} | {sum(s['structure'] for s in scores_b)/len(scores_b):.1f} |",
        f"",
        f"### Score Breakdown",
        f"",
        f"| Component | Max | Measures |",
        f"|---|:-:|---|",
        f"| Visual Alignment | 35 | SigLIP2 image↔caption cosine similarity |",
        f"| Caption Quality | 25 | Length (0-8) + Specificity (0-9) + Concreteness (0-8) |",
        f"| Tag Quality | 25 | Count (0-8) + Diversity (0-9) + Specificity (0-8) |",
        f"| Structure | 15 | Parse success (0-5) + Fields (0-5) + Valid type (0-5) |",
        f"",
        f"---",
        f"",
    ]

    # Per-image details
    for i, ((img, fname, _), ra, rb, sa, sb) in enumerate(
        zip(images, results_a, results_b, scores_a, scores_b)
    ):
        caption_a = ra.get("mc_caption", "") or ra.get("caption", "")
        caption_b = rb.get("mc_caption", "") or rb.get("caption", "")
        tags_a = ra.get("tags", []) or ra.get("ai_tags", [])
        tags_b = rb.get("tags", []) or rb.get("ai_tags", [])
        type_a = ra.get("image_type", "?")
        type_b = rb.get("image_type", "?")
        time_a = ra.get("_time_s", 0)
        time_b = rb.get("_time_s", 0)

        winner = "A" if sa["total"] > sb["total"] else ("B" if sb["total"] > sa["total"] else "=")

        lines.extend([
            f"## {i+1}. {fname} (Winner: **{winner}**)",
            f"",
            f"| | Model A | Model B |",
            f"|---|---|---|",
            f"| **Score** | **{sa['total']:.1f}** ({score_grade(sa['total'])}) | **{sb['total']:.1f}** ({score_grade(sb['total'])}) |",
            f"| Visual Align | {sa['visual_alignment']:.1f}/35 | {sb['visual_alignment']:.1f}/35 |",
            f"| Caption | {sa['caption_quality']:.1f}/25 | {sb['caption_quality']:.1f}/25 |",
            f"| Tags | {sa['tag_quality']:.1f}/25 | {sb['tag_quality']:.1f}/25 |",
            f"| Structure | {sa['structure']:.1f}/15 | {sb['structure']:.1f}/15 |",
            f"| image_type | {type_a} | {type_b} |",
            f"| time | {time_a}s | {time_b}s |",
            f"| tags | {', '.join(tags_a) if isinstance(tags_a, list) else tags_a} | {', '.join(tags_b) if isinstance(tags_b, list) else tags_b} |",
            f"",
            f"**Caption A** ({len(caption_a)} chars):",
            f"> {caption_a}",
            f"",
            f"**Caption B** ({len(caption_b)} chars):",
            f"> {caption_b}",
            f"",
            f"---",
            f"",
        ])

    # Summary stats
    avg_time_a = sum(r.get("_time_s", 0) for r in results_a) / max(len(results_a), 1)
    avg_time_b = sum(r.get("_time_s", 0) for r in results_b) / max(len(results_b), 1)

    wins_a = sum(1 for sa, sb in zip(scores_a, scores_b) if sa["total"] > sb["total"])
    wins_b = sum(1 for sa, sb in zip(scores_a, scores_b) if sb["total"] > sa["total"])
    ties = len(scores_a) - wins_a - wins_b

    lines.extend([
        f"## Final Summary",
        f"",
        f"| | Model A | Model B |",
        f"|---|:-:|:-:|",
        f"| **Avg Score** | **{avg_score_a:.1f}** ({score_grade(avg_score_a)}) | **{avg_score_b:.1f}** ({score_grade(avg_score_b)}) |",
        f"| Wins | {wins_a} | {wins_b} |",
        f"| Ties | {ties} | {ties} |",
        f"| Avg time/file | {avg_time_a:.1f}s | {avg_time_b:.1f}s |",
        f"",
    ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="MC Quality Comparison with Scoring")
    parser.add_argument("--profile", type=Path, default=None,
                        help="YAML profile (reuses ab field)")
    parser.add_argument("--ab", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                        help="Two model IDs to compare")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load profile
    profile = {}
    if args.profile:
        try:
            import yaml
            with open(args.profile, "r", encoding="utf-8") as f:
                profile = yaml.safe_load(f) or {}
        except ImportError:
            from ruamel.yaml import YAML
            with open(args.profile, "r", encoding="utf-8") as f:
                profile = YAML().load(f) or {}

        if not args.ab and "ab" in profile:
            args.ab = profile["ab"]
        if "count" in profile and args.count == 5:
            args.count = profile["count"]

    if not args.ab:
        print("Error: --ab MODEL_A MODEL_B or --profile with ab field required")
        sys.exit(1)

    model_a, model_b = args.ab

    # Per-model options from profile
    model_a_opts = profile.get("model_a_opts", {})
    model_b_opts = profile.get("model_b_opts", {})

    print("=" * 70)
    print("  MC Quality Comparison (Scored)")
    print("=" * 70)
    print(f"  Model A: {model_a}")
    print(f"  Model B: {model_b}")
    print(f"  Images:  {args.count}")

    # Load images
    images, tmp_dir = load_images(args.images, args.count, args.max_edge)
    print(f"  Loaded {len(images)} images")

    # Load domain
    domain = None
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        domain_id = cfg.get("classification.active_domain")
        if domain_id:
            from backend.vision.domain_loader import load_domain
            domain = load_domain(domain_id)
            print(f"  Domain: {domain_id}")
    except Exception:
        pass

    # Run models with per-model options
    print(f"\n{'─' * 70}")
    print(f"  Model A: {model_a}")
    results_a = run_model(model_a, images, domain, model_opts=model_a_opts)

    print(f"\n{'─' * 70}")
    print(f"  Model B: {model_b}")
    results_b = run_model(model_b, images, domain, model_opts=model_b_opts)

    # Load SigLIP2 for visual alignment scoring
    print(f"\n{'─' * 70}")
    print("  Loading SigLIP2 for visual alignment scoring...")
    from backend.vector.siglip2_encoder import SigLIP2Encoder
    encoder = SigLIP2Encoder()

    # Score all results
    print("  Scoring...")
    scores_a = []
    scores_b = []
    for (img, fname, _), ra, rb in zip(images, results_a, results_b):
        sa = score_mc(img, ra, encoder)
        sb = score_mc(img, rb, encoder)
        scores_a.append(sa)
        scores_b.append(sb)
        print(f"    {fname}: A={sa['total']:.1f} B={sb['total']:.1f}")

    # Unload SigLIP2
    del encoder
    gc.collect()
    try:
        import torch
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass

    # Build report
    report_md = build_markdown_report(
        model_a, model_b, results_a, results_b, scores_a, scores_b, images
    )

    # Save markdown
    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"mc_quality_{ts}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Save JSON
    json_path = output_path.with_suffix(".json")
    raw = {
        "timestamp": datetime.now().isoformat(),
        "model_a": {"id": model_a, "results": results_a, "scores": scores_a},
        "model_b": {"id": model_b, "results": results_b, "scores": scores_b},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)

    # Print summary
    avg_a = sum(s["total"] for s in scores_a) / len(scores_a)
    avg_b = sum(s["total"] for s in scores_b) / len(scores_b)
    wins_a = sum(1 for sa, sb in zip(scores_a, scores_b) if sa["total"] > sb["total"])
    wins_b = sum(1 for sa, sb in zip(scores_a, scores_b) if sb["total"] > sa["total"])

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Model A ({model_a.split('/')[-1]}): {avg_a:.1f}/100 ({score_grade(avg_a)})")
    print(f"  Model B ({model_b.split('/')[-1]}): {avg_b:.1f}/100 ({score_grade(avg_b)})")
    print(f"  Wins: A={wins_a}  B={wins_b}  Tie={len(scores_a)-wins_a-wins_b}")
    print(f"{'─' * 70}")
    print(f"  Report: {output_path}")
    print(f"  JSON:   {json_path}")
    print(f"{'=' * 70}")

    # Cleanup
    for img, _, _ in images:
        try:
            img.close()
        except Exception:
            pass
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
