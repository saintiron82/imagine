#!/usr/bin/env python3
"""
Qwen3.5 VLM Evaluation Test Script
===================================
Qwen3-VL → Qwen3.5 전환 가능성 평가를 위한 독립 테스트 스크립트.
기존 코드를 수정하지 않고, 직접 모델을 로드하여 테스트합니다.

Usage:
    python tools/test_qwen35_eval.py --all
    python tools/test_qwen35_eval.py --phase 1
    python tools/test_qwen35_eval.py --phase 3 --phase 5
    python tools/test_qwen35_eval.py --all --output results.json
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_IMAGE_DIR = Path("/Users/saintiron/imageDB/마캬베리즈무/실내소품")

# ── Helpers ─────────────────────────────────────────────────────────────

def _load_test_image(max_edge=768):
    """Load first available PSD as RGB PIL Image, resized to max_edge."""
    from PIL import Image
    from psd_tools import PSDImage

    psd_files = sorted(TEST_IMAGE_DIR.glob("*.psd"))
    if not psd_files:
        raise FileNotFoundError(f"No PSD files in {TEST_IMAGE_DIR}")

    psd = PSDImage.open(str(psd_files[0]))
    img = psd.composite()
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize keeping aspect ratio
    w, h = img.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    print(f"  Image: {psd_files[0].name} ({img.size[0]}x{img.size[1]})")
    return img, psd_files[0].name


def _load_test_images(n=5, max_edge=768):
    """Load up to n PSD files as RGB PIL Images."""
    from PIL import Image
    from psd_tools import PSDImage

    psd_files = sorted(TEST_IMAGE_DIR.glob("*.psd"))[:n]
    images = []
    for pf in psd_files:
        try:
            psd = PSDImage.open(str(pf))
            img = psd.composite()
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
            images.append((img, pf.name))
        except Exception as e:
            print(f"  SKIP {pf.name}: {e}")
    return images


def _get_rss_mb():
    """Current process RSS in MB."""
    import psutil
    return psutil.Process().memory_info().rss / 1024 ** 2


def _try_json_parse(text):
    """Try to parse JSON from text, return (parsed_dict, tier)."""
    # Strip thinking tags if present
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Tier 1: direct
    try:
        d = json.loads(cleaned)
        if isinstance(d, dict):
            return d, 1
    except json.JSONDecodeError:
        pass

    # Tier 2: strip markdown fences
    fenced = re.sub(r'```(?:json)?\s*', '', cleaned)
    fenced = re.sub(r'```\s*$', '', fenced).strip()
    # Fix trailing commas
    fenced = re.sub(r',\s*([}\]])', r'\1', fenced)
    try:
        d = json.loads(fenced)
        if isinstance(d, dict):
            return d, 2
    except json.JSONDecodeError:
        pass

    return None, 3


STAGE1_SYSTEM = "You are a strict JSON generator. Output valid JSON only. No explanation, no markdown fences."
STAGE1_USER = """Classify this image into exactly ONE type.

{
  "image_type": "ONE OF: character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other",
  "confidence": "ONE OF: high, medium, low"
}"""

STAGE2_SYSTEM = "You are a strict JSON generator. Output valid JSON only matching the provided schema. No explanation, no markdown fences."
STAGE2_USER = """Analyze this image in detail.

Return JSON with these fields:
{
  "caption": "A detailed description of the image content",
  "tags": ["tag1", "tag2", "tag3", "..."],
  "art_style": "The art style (e.g., anime, realistic, pixel, watercolor, other)",
  "color_palette": "Dominant color palette description"
}"""


def _mlx_generate(model, processor, config, image, system_prompt, user_prompt,
                  max_tokens=192, temperature=0.1, enable_thinking=False):
    """MLX generation with proper chat template + thinking mode control."""
    from mlx_vlm import generate

    # Build messages with proper multimodal content
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    # Use tokenizer directly to control enable_thinking
    tok = getattr(processor, 'tokenizer', processor)
    formatted = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    result = generate(
        model, processor, formatted, [image],
        max_tokens=max_tokens, temperature=temperature, verbose=False,
    )
    return result.text if hasattr(result, 'text') else str(result)


# ── Phase 1: Environment Check ──────────────────────────────────────────

def phase1_env_check():
    print("\n" + "=" * 60)
    print("Phase 1: Environment Check")
    print("=" * 60)

    checks = {}

    # Python
    v = sys.version.split()[0]
    ok = tuple(int(x) for x in v.split(".")[:2]) >= (3, 10)
    checks["python"] = (v, ok)
    print(f"  Python: {v} {'PASS' if ok else 'FAIL (need >=3.10)'}")

    # transformers
    try:
        import transformers
        v = transformers.__version__
        ok = tuple(int(x) for x in v.split(".")[:2]) >= (5, 4)
        checks["transformers"] = (v, ok)
        print(f"  transformers: {v} {'PASS' if ok else 'FAIL (need >=5.4.0)'}")
    except ImportError:
        checks["transformers"] = (None, False)
        print("  transformers: NOT INSTALLED")

    # torch + MPS
    try:
        import torch
        mps = torch.backends.mps.is_available()
        checks["torch"] = (torch.__version__, True)
        checks["mps"] = (str(mps), mps)
        print(f"  torch: {torch.__version__} PASS")
        print(f"  MPS: {'PASS' if mps else 'FAIL'}")
    except ImportError:
        checks["torch"] = (None, False)
        checks["mps"] = (None, False)
        print("  torch: NOT INSTALLED")

    # mlx-vlm
    try:
        import mlx_vlm
        v = getattr(mlx_vlm, '__version__', 'unknown')
        checks["mlx_vlm"] = (v, True)
        print(f"  mlx-vlm: {v} PASS")
    except ImportError:
        checks["mlx_vlm"] = (None, False)
        print("  mlx-vlm: NOT INSTALLED (CRITICAL for Mac)")

    # mlx
    try:
        import mlx.core as mx
        checks["mlx"] = ("available", True)
        print(f"  mlx: available PASS")
    except ImportError:
        checks["mlx"] = (None, False)
        print("  mlx: NOT INSTALLED")

    # System memory
    import psutil
    mem_gb = psutil.virtual_memory().total / 1024 ** 3
    ok = mem_gb >= 30
    checks["memory"] = (f"{mem_gb:.0f}GB", ok)
    print(f"  Memory: {mem_gb:.0f}GB {'PASS' if ok else 'FAIL (need >=32GB)'}")

    # Test images
    psd_count = len(list(TEST_IMAGE_DIR.glob("*.psd"))) if TEST_IMAGE_DIR.exists() else 0
    ok = psd_count >= 5
    checks["test_images"] = (str(psd_count), ok)
    print(f"  Test PSD files: {psd_count} {'PASS' if ok else 'WARN (need >=5)'}")

    all_pass = all(v[1] for v in checks.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    return checks


# ── Phase 2: Transformers Backend Test ───────────────────────────────────

def phase2_transformers_test():
    print("\n" + "=" * 60)
    print("Phase 2: Transformers Backend — Qwen3.5-4B (float16, MPS)")
    print("=" * 60)

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "Qwen/Qwen3.5-4B"
    results = {"model": model_id, "backend": "transformers"}

    # Load image
    img, fname = _load_test_image()
    results["test_image"] = fname

    # Load model
    print(f"  Loading {model_id}...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": "mps"},
        attn_implementation="sdpa",
    )
    model.eval()
    load_time = time.perf_counter() - t0
    results["load_time_s"] = round(load_time, 1)
    print(f"  Model loaded in {load_time:.1f}s")

    rss_after_load = _get_rss_mb()
    results["rss_after_load_mb"] = round(rss_after_load)
    print(f"  RSS after load: {rss_after_load:.0f} MB")

    # Build messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": STAGE1_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": STAGE1_USER},
        ]},
    ]

    # Generate (thinking OFF)
    print("  Generating (thinking OFF)...")
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        chat_template_kwargs={"enable_thinking": False},
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=192)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
    output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    gen_time = time.perf_counter() - t0

    results["gen_time_s"] = round(gen_time, 2)
    results["raw_output"] = output
    print(f"  Generation: {gen_time:.2f}s")
    print(f"  Raw output: {output[:200]}")

    # Parse JSON
    parsed, tier = _try_json_parse(output)
    results["json_tier"] = tier
    results["parsed"] = parsed
    print(f"  JSON parse tier: {tier} ({'direct' if tier == 1 else 'repaired' if tier == 2 else 'FAILED'})")
    if parsed:
        print(f"  Parsed: {json.dumps(parsed, ensure_ascii=False, indent=2)}")

    rss_peak = _get_rss_mb()
    results["rss_peak_mb"] = round(rss_peak)
    print(f"  RSS peak: {rss_peak:.0f} MB")

    # Cleanup
    del model, processor, inputs, gen_ids
    gc.collect()
    torch.mps.empty_cache()

    return results


# ── Phase 3: MLX Backend Test (MANDATORY) ─────────────────────────────

def phase3_mlx_test():
    print("\n" + "=" * 60)
    print("Phase 3: MLX Backend — Qwen3.5-4B-MLX-4bit (MANDATORY)")
    print("=" * 60)

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model_id = "mlx-community/Qwen3.5-4B-MLX-4bit"
    results = {"model": model_id, "backend": "mlx"}

    # Load image
    img, fname = _load_test_image()
    results["test_image"] = fname

    # Load model
    print(f"  Loading {model_id}...")
    t0 = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    load_time = time.perf_counter() - t0
    results["load_time_s"] = round(load_time, 1)
    print(f"  Model loaded in {load_time:.1f}s")

    metal_after = mx.get_active_memory() / 1024 ** 2
    results["metal_after_load_mb"] = round(metal_after)
    print(f"  Metal memory after load: {metal_after:.0f} MB")

    # Generate with proper system/user role separation
    print("  Generating...")
    t0 = time.perf_counter()
    output = _mlx_generate(model, processor, config, img, STAGE1_SYSTEM, STAGE1_USER)
    gen_time = time.perf_counter() - t0

    results["gen_time_s"] = round(gen_time, 2)
    results["raw_output"] = output
    print(f"  Generation: {gen_time:.2f}s")
    print(f"  Raw output: {output[:200]}")

    # Check for thinking tags
    has_thinking = '<think>' in output
    results["has_thinking_tags"] = has_thinking
    if has_thinking:
        print("  WARNING: Thinking tags detected in output!")

    # Parse JSON
    parsed, tier = _try_json_parse(output)
    results["json_tier"] = tier
    results["parsed"] = parsed
    print(f"  JSON parse tier: {tier}")
    if parsed:
        print(f"  Parsed: {json.dumps(parsed, ensure_ascii=False, indent=2)}")

    metal_peak = mx.get_peak_memory() / 1024 ** 2
    results["metal_peak_mb"] = round(metal_peak)
    print(f"  Metal peak memory: {metal_peak:.0f} MB")

    # Cleanup
    del model, processor
    gc.collect()
    mx.clear_cache()

    return results


# ── Phase 4: Thinking Mode Control ────────────────────────────────────

def phase4_thinking_mode():
    print("\n" + "=" * 60)
    print("Phase 4: Thinking Mode Control (MLX)")
    print("=" * 60)

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model_id = "mlx-community/Qwen3.5-4B-MLX-4bit"
    results = {"model": model_id}

    img, fname = _load_test_image()

    print(f"  Loading {model_id}...")
    model, processor = load(model_id)
    config = load_config(model_id)

    # --- Sub-test A: Thinking ON (enable_thinking=True) ---
    print("\n  [A] Thinking ON (enable_thinking=True)...")
    t0 = time.perf_counter()
    out_a = _mlx_generate(model, processor, config, img,
                          STAGE1_SYSTEM, STAGE1_USER, max_tokens=512,
                          enable_thinking=True)
    time_a = time.perf_counter() - t0

    has_think_a = '<think>' in out_a
    parsed_a, tier_a = _try_json_parse(out_a)
    results["thinking_on"] = {
        "time_s": round(time_a, 2),
        "has_thinking": has_think_a,
        "output_len": len(out_a),
        "json_tier": tier_a,
        "parsed": parsed_a,
    }
    print(f"    Time: {time_a:.2f}s | Thinking tags: {has_think_a} | Len: {len(out_a)} | JSON tier: {tier_a}")
    print(f"    Output preview: {out_a[:300]}")

    # --- Sub-test B: Thinking OFF (enable_thinking=False) ---
    print("\n  [B] Thinking OFF (enable_thinking=False)...")
    t0 = time.perf_counter()
    out_b = _mlx_generate(model, processor, config, img,
                          STAGE1_SYSTEM, STAGE1_USER, max_tokens=192,
                          enable_thinking=False)
    time_b = time.perf_counter() - t0

    has_think_b = '<think>' in out_b
    parsed_b, tier_b = _try_json_parse(out_b)
    results["thinking_off_prompt"] = {
        "time_s": round(time_b, 2),
        "has_thinking": has_think_b,
        "output_len": len(out_b),
        "json_tier": tier_b,
        "parsed": parsed_b,
    }
    print(f"    Time: {time_b:.2f}s | Thinking tags: {has_think_b} | Len: {len(out_b)} | JSON tier: {tier_b}")
    print(f"    Output preview: {out_b[:300]}")

    # Summary
    speedup = time_a / time_b if time_b > 0 else 0
    print(f"\n  Summary: Thinking ON={time_a:.1f}s vs OFF={time_b:.1f}s (speedup: {speedup:.1f}x)")

    del model, processor
    gc.collect()
    mx.clear_cache()

    return results


# ── Phase 5: 2-Stage Pipeline Compatibility ────────────────────────────

def phase5_pipeline_compat():
    print("\n" + "=" * 60)
    print("Phase 5: 2-Stage Pipeline Compatibility (MLX)")
    print("=" * 60)

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model_id = "mlx-community/Qwen3.5-4B-MLX-4bit"
    results = {"model": model_id}

    img, fname = _load_test_image()

    print(f"  Loading {model_id}...")
    model, processor = load(model_id)
    config = load_config(model_id)

    sys_nothink = STAGE1_SYSTEM + "\nDo NOT think. Output JSON only."

    # Stage 1: Classification
    print("\n  [Stage 1] Classification...")
    t0 = time.perf_counter()
    s1_out = _mlx_generate(model, processor, config, img, sys_nothink, STAGE1_USER)
    s1_time = time.perf_counter() - t0

    s1_parsed, s1_tier = _try_json_parse(s1_out)
    valid_types = {"character", "background", "ui_element", "item", "icon", "texture", "effect", "logo", "photo", "illustration", "other"}
    s1_type = s1_parsed.get("image_type", "unknown") if s1_parsed else "unknown"
    s1_valid = s1_type in valid_types

    results["stage1"] = {
        "time_s": round(s1_time, 2),
        "json_tier": s1_tier,
        "image_type": s1_type,
        "type_valid": s1_valid,
        "parsed": s1_parsed,
    }
    print(f"    Time: {s1_time:.2f}s | Type: {s1_type} | Valid: {s1_valid} | JSON tier: {s1_tier}")

    # Stage 2: Structured Analysis
    print(f"\n  [Stage 2] Structured Analysis (type={s1_type})...")
    sys2_nothink = STAGE2_SYSTEM + "\nDo NOT think. Output JSON only."
    t0 = time.perf_counter()
    s2_out = _mlx_generate(model, processor, config, img, sys2_nothink, STAGE2_USER, max_tokens=384)
    s2_time = time.perf_counter() - t0

    s2_parsed, s2_tier = _try_json_parse(s2_out)
    has_caption = bool(s2_parsed and s2_parsed.get("caption"))
    has_tags = bool(s2_parsed and s2_parsed.get("tags"))

    results["stage2"] = {
        "time_s": round(s2_time, 2),
        "json_tier": s2_tier,
        "has_caption": has_caption,
        "caption_preview": (s2_parsed.get("caption", "")[:100] if s2_parsed else ""),
        "tag_count": len(s2_parsed.get("tags", [])) if s2_parsed else 0,
        "parsed": s2_parsed,
    }
    print(f"    Time: {s2_time:.2f}s | JSON tier: {s2_tier}")
    if s2_parsed:
        print(f"    Caption: {s2_parsed.get('caption', '')[:100]}")
        print(f"    Tags: {s2_parsed.get('tags', [])}")
        print(f"    Art style: {s2_parsed.get('art_style', 'N/A')}")

    total = s1_time + s2_time
    success = s1_valid and has_caption and has_tags and s1_tier <= 2 and s2_tier <= 2
    results["total_time_s"] = round(total, 2)
    results["pipeline_success"] = success
    print(f"\n  Pipeline: {'SUCCESS' if success else 'ISSUES FOUND'} | Total: {total:.2f}s")

    del model, processor
    gc.collect()
    mx.clear_cache()

    return results


# ── Phase 6: Performance Benchmark ───────────────────────────────────

def phase6_benchmark():
    print("\n" + "=" * 60)
    print("Phase 6: Performance Benchmark")
    print("=" * 60)

    images = _load_test_images(n=3)
    if len(images) < 1:
        print("  ERROR: No test images available")
        return {}

    configs = [
        ("A: Qwen3-VL-4B MLX (baseline)", "mlx", "mlx-community/Qwen3-VL-4B-Instruct-4bit", False),
        ("B: Qwen3.5-4B MLX (think OFF)", "mlx", "mlx-community/Qwen3.5-4B-MLX-4bit", False),
        ("C: Qwen3.5-4B MLX (think ON)", "mlx", "mlx-community/Qwen3.5-4B-MLX-4bit", True),
        ("D: Qwen3.5-9B MLX (think OFF)", "mlx", "mlx-community/Qwen3.5-9B-MLX-4bit", False),
    ]

    all_results = {}

    for cfg_name, backend, model_id, thinking_on in configs:
        print(f"\n  --- {cfg_name} ---")
        cfg_results = {"model": model_id, "backend": backend, "thinking": thinking_on}

        try:
            import mlx.core as mx
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            # Cold load
            t0 = time.perf_counter()
            model, processor = load(model_id)
            config = load_config(model_id)
            load_time = time.perf_counter() - t0
            cfg_results["load_time_s"] = round(load_time, 1)
            print(f"    Load: {load_time:.1f}s")

            metal_load = mx.get_peak_memory() / 1024 ** 2
            cfg_results["metal_after_load_mb"] = round(metal_load)
            print(f"    Metal after load: {metal_load:.0f} MB")

            # Process images
            img_results = []
            max_tok = 512 if thinking_on else 192

            for img, fname in images:
                t0 = time.perf_counter()
                out = _mlx_generate(model, processor, config, img,
                                    STAGE1_SYSTEM, STAGE1_USER, max_tokens=max_tok,
                                    enable_thinking=thinking_on)
                gen_time = time.perf_counter() - t0

                parsed, tier = _try_json_parse(out)
                has_think = '<think>' in out

                img_results.append({
                    "file": fname,
                    "gen_time_s": round(gen_time, 2),
                    "json_tier": tier,
                    "has_thinking": has_think,
                    "image_type": parsed.get("image_type", "?") if parsed else "?",
                    "output_len": len(out),
                })
                print(f"    {fname}: {gen_time:.2f}s | tier={tier} | type={parsed.get('image_type', '?') if parsed else '?'} | think={has_think}")

            cfg_results["images"] = img_results
            avg_time = sum(r["gen_time_s"] for r in img_results) / len(img_results)
            cfg_results["avg_gen_time_s"] = round(avg_time, 2)

            metal_peak = mx.get_peak_memory() / 1024 ** 2
            cfg_results["metal_peak_mb"] = round(metal_peak)
            print(f"    Avg gen: {avg_time:.2f}s | Metal peak: {metal_peak:.0f} MB")

            del model, processor
            gc.collect()
            mx.clear_cache()

        except Exception as e:
            cfg_results["error"] = str(e)
            print(f"    ERROR: {e}")

        all_results[cfg_name] = cfg_results

    # Summary table
    print("\n  === Benchmark Summary ===")
    print(f"  {'Config':<35} {'Load(s)':<8} {'Avg Gen(s)':<11} {'Metal(MB)':<10} {'JSON OK'}")
    print("  " + "-" * 80)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<35} ERROR: {r['error'][:40]}")
        else:
            json_ok = sum(1 for ir in r.get("images", []) if ir["json_tier"] <= 2)
            total = len(r.get("images", []))
            print(f"  {name:<35} {r.get('load_time_s', '?'):<8} {r.get('avg_gen_time_s', '?'):<11} {r.get('metal_peak_mb', '?'):<10} {json_ok}/{total}")

    return all_results


# ── Phase 7: Batch Processing Test ───────────────────────────────────

def phase7_batch_test():
    print("\n" + "=" * 60)
    print("Phase 7: Batch Processing (5 images, memory tracking)")
    print("=" * 60)

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model_id = "mlx-community/Qwen3.5-4B-MLX-4bit"
    results = {"model": model_id}

    images = _load_test_images(n=5)
    print(f"  Loaded {len(images)} images")

    print(f"  Loading {model_id}...")
    model, processor = load(model_id)
    config = load_config(model_id)

    rss_readings = []
    metal_readings = []
    successes = 0

    sys_nothink = STAGE2_SYSTEM + "\nDo NOT think. Output JSON only."

    for i, (img, fname) in enumerate(images):
        t0 = time.perf_counter()
        out = _mlx_generate(model, processor, config, img,
                            sys_nothink, STAGE2_USER, max_tokens=384)
        gen_time = time.perf_counter() - t0

        parsed, tier = _try_json_parse(out)
        if parsed and parsed.get("caption"):
            successes += 1

        rss = _get_rss_mb()
        metal = mx.get_active_memory() / 1024 ** 2
        rss_readings.append(rss)
        metal_readings.append(metal)

        print(f"  [{i+1}/5] {fname}: {gen_time:.2f}s | tier={tier} | RSS={rss:.0f}MB | Metal={metal:.0f}MB")

    # Memory growth analysis
    rss_growth = rss_readings[-1] - rss_readings[0] if len(rss_readings) >= 2 else 0
    metal_growth = metal_readings[-1] - metal_readings[0] if len(metal_readings) >= 2 else 0

    results["image_count"] = len(images)
    results["successes"] = successes
    results["rss_start_mb"] = round(rss_readings[0]) if rss_readings else 0
    results["rss_end_mb"] = round(rss_readings[-1]) if rss_readings else 0
    results["rss_growth_mb"] = round(rss_growth)
    results["metal_growth_mb"] = round(metal_growth)
    results["memory_stable"] = abs(rss_growth) < 500

    print(f"\n  Results: {successes}/{len(images)} success")
    print(f"  RSS growth: {rss_growth:.0f} MB ({'STABLE' if abs(rss_growth) < 500 else 'LEAK?'})")
    print(f"  Metal growth: {metal_growth:.0f} MB")

    # Unload test
    print("  Testing model unload...")
    rss_before = _get_rss_mb()
    del model, processor
    gc.collect()
    mx.clear_cache()
    time.sleep(1)
    rss_after = _get_rss_mb()
    freed = rss_before - rss_after
    results["unload_freed_mb"] = round(freed)
    print(f"  Unload freed: {freed:.0f} MB")

    return results


# ── Phase 8: 9B Model Viability ──────────────────────────────────────

def phase8_9b_test():
    print("\n" + "=" * 60)
    print("Phase 8: Qwen3.5-9B-MLX-4bit Viability (32GB M5)")
    print("=" * 60)

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    model_id = "mlx-community/Qwen3.5-9B-MLX-4bit"
    results = {"model": model_id}

    img, fname = _load_test_image()

    print(f"  Loading {model_id} (this may take a while)...")
    t0 = time.perf_counter()
    try:
        model, processor = load(model_id)
        config = load_config(model_id)
    except Exception as e:
        results["error"] = f"Load failed: {e}"
        results["verdict"] = "FAIL"
        print(f"  LOAD FAILED: {e}")
        return results

    load_time = time.perf_counter() - t0
    results["load_time_s"] = round(load_time, 1)
    print(f"  Load: {load_time:.1f}s")

    metal_load = mx.get_peak_memory() / 1024 ** 2
    results["metal_after_load_mb"] = round(metal_load)
    print(f"  Metal after load: {metal_load:.0f} MB")

    sys_nothink = STAGE1_SYSTEM + "\nDo NOT think. Output JSON only."

    # Stage 1
    print("  Stage 1 (classification)...")
    t0 = time.perf_counter()
    try:
        out = _mlx_generate(model, processor, config, img,
                            sys_nothink, STAGE1_USER)
        s1_time = time.perf_counter() - t0
        parsed, tier = _try_json_parse(out)
        results["stage1_time_s"] = round(s1_time, 2)
        results["stage1_tier"] = tier
        results["stage1_parsed"] = parsed
        print(f"  Stage 1: {s1_time:.2f}s | tier={tier}")
        if parsed:
            print(f"    {json.dumps(parsed, ensure_ascii=False)}")
    except Exception as e:
        results["stage1_error"] = str(e)
        print(f"  Stage 1 FAILED: {e}")

    # Stage 2
    print("  Stage 2 (structured analysis)...")
    sys2_nothink = STAGE2_SYSTEM + "\nDo NOT think. Output JSON only."
    t0 = time.perf_counter()
    try:
        out2 = _mlx_generate(model, processor, config, img,
                             sys2_nothink, STAGE2_USER, max_tokens=384)
        s2_time = time.perf_counter() - t0
        parsed2, tier2 = _try_json_parse(out2)
        results["stage2_time_s"] = round(s2_time, 2)
        results["stage2_tier"] = tier2
        results["stage2_parsed"] = parsed2
        print(f"  Stage 2: {s2_time:.2f}s | tier={tier2}")
        if parsed2:
            cap = parsed2.get("caption", "")[:100]
            print(f"    Caption: {cap}")
            print(f"    Tags: {parsed2.get('tags', [])}")
    except Exception as e:
        results["stage2_error"] = str(e)
        print(f"  Stage 2 FAILED: {e}")

    # Peak memory
    metal_peak = mx.get_peak_memory() / 1024 ** 2
    results["metal_peak_mb"] = round(metal_peak)
    print(f"\n  Metal peak: {metal_peak:.0f} MB")

    # Verdict
    s1_ok = results.get("stage1_time_s", 999) < 60 and results.get("stage1_tier", 3) <= 2
    s2_ok = results.get("stage2_time_s", 999) < 60 and results.get("stage2_tier", 3) <= 2
    mem_ok = metal_peak < 24000

    if s1_ok and s2_ok and mem_ok:
        verdict = "PASS"
    elif metal_peak > 28000:
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"

    results["verdict"] = verdict
    print(f"\n  Verdict: {verdict}")
    print(f"    Memory: {metal_peak:.0f} MB ({'OK' if mem_ok else 'HIGH'})")
    if "stage1_time_s" in results:
        print(f"    Stage 1: {results['stage1_time_s']:.1f}s ({'OK' if s1_ok else 'SLOW/FAIL'})")
    if "stage2_time_s" in results:
        print(f"    Stage 2: {results['stage2_time_s']:.1f}s ({'OK' if s2_ok else 'SLOW/FAIL'})")

    del model, processor
    gc.collect()
    mx.clear_cache()

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 VLM Evaluation")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--phase", type=int, action="append", help="Run specific phase(s)")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not args.all and not args.phase:
        args.all = True

    phases_to_run = set(range(1, 9)) if args.all else set(args.phase or [])

    all_results = {}
    start = time.perf_counter()

    if 1 in phases_to_run:
        all_results["phase1"] = phase1_env_check()

    if 2 in phases_to_run:
        all_results["phase2"] = phase2_transformers_test()

    if 3 in phases_to_run:
        all_results["phase3"] = phase3_mlx_test()

    if 4 in phases_to_run:
        all_results["phase4"] = phase4_thinking_mode()

    if 5 in phases_to_run:
        all_results["phase5"] = phase5_pipeline_compat()

    if 6 in phases_to_run:
        all_results["phase6"] = phase6_benchmark()

    if 7 in phases_to_run:
        all_results["phase7"] = phase7_batch_test()

    if 8 in phases_to_run:
        all_results["phase8"] = phase8_9b_test()

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print(f"COMPLETE — Total: {elapsed:.0f}s")
    print("=" * 60)

    if args.output:
        # Serialize, skipping non-serializable items
        def default_ser(o):
            return str(o)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=default_ser)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
