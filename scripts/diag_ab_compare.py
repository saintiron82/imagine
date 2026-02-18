"""
A/B Comparison: Old 2-step vs New 1-step apply_chat_template.

Tests:
  A (old): 2-step, no system role, "Respond with JSON only" in user prompt
  B (new): 1-step, system role, clean user prompt

Compares:
  1. Tokenization equivalence (A-tokenize vs B-tokenize)
  2. JSON output quality (Stage 1 + Stage 2)
"""
import sys, gc, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from backend.utils.tier_config import get_active_tier

_, tier_cfg = get_active_tier()
model_id = tier_cfg.get("vlm", {}).get("model", "Qwen/Qwen3-VL-4B-Instruct")
print(f"Model: {model_id}\n")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("mps")

# Load test images (pick 3 diverse ones)
import glob
thumbs = sorted(glob.glob(str(PROJECT_ROOT / "output/thumbnails/agm_12_*_thumb.png")))
test_thumbs = [thumbs[0], thumbs[5], thumbs[10]] if len(thumbs) > 10 else thumbs[:3]
test_images = [Image.open(t).convert("RGB") for t in test_thumbs]
print(f"Test images: {[Path(t).name for t in test_thumbs]}")
print(f"Sizes: {[img.size for img in test_images]}\n")


def cleanup():
    gc.collect()
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


def generate_old_2step(image, prompt):
    """OLD: 2-step apply_chat_template, no system role."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to("mps")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    del inputs, generated_ids, trimmed
    cleanup()
    return decoded


def generate_new_1step(image, user_prompt, system_prompt=None):
    """NEW: 1-step apply_chat_template, with system role."""
    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]
    })

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    del inputs, generated_ids, trimmed
    cleanup()
    return decoded


# ── Test 1: Tokenization Equivalence ──────────────────────────
print("=" * 70)
print("TEST 1: TOKENIZATION EQUIVALENCE (same prompt, 2-step vs 1-step)")
print("=" * 70)

test_prompt = "Describe this image briefly."
img = test_images[0]

# 2-step tokenization
msgs_a = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": test_prompt}
]}]
text_a = processor.apply_chat_template(msgs_a, tokenize=False, add_generation_prompt=True)
inputs_a = processor(text=[text_a], images=[img], return_tensors="pt")
tokens_a = inputs_a.input_ids[0].tolist()

# 1-step tokenization
inputs_b = processor.apply_chat_template(
    msgs_a, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
tokens_b = inputs_b["input_ids"][0].tolist()

print(f"  2-step tokens: {len(tokens_a)}")
print(f"  1-step tokens: {len(tokens_b)}")
print(f"  IDENTICAL: {tokens_a == tokens_b}")
if tokens_a != tokens_b:
    # Find first difference
    for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
        if a != b:
            print(f"  First diff at position {i}: 2-step={a}, 1-step={b}")
            break
    if len(tokens_a) != len(tokens_b):
        print(f"  Length diff: 2-step has {len(tokens_a) - len(tokens_b)} more tokens")

del inputs_a, inputs_b
cleanup()

# ── Test 2: Stage 1 Classification ────────────────────────────
print("\n" + "=" * 70)
print("TEST 2: STAGE 1 CLASSIFICATION (A=old prompt, B=new system+user)")
print("=" * 70)

# OLD Stage 1 prompt
OLD_S1 = """Classify this image into exactly ONE type. Respond with JSON only, no explanation.

{
  "image_type": "ONE OF: character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other",
  "confidence": "ONE OF: high, medium, low"
}"""

# NEW Stage 1
NEW_S1_SYSTEM = "You are a strict JSON generator. Output valid JSON only. No explanation, no markdown fences."
NEW_S1_USER = """Classify this image into exactly ONE type.

{
  "image_type": "ONE OF: character, background, ui_element, item, icon, texture, effect, logo, photo, illustration, other",
  "confidence": "ONE OF: high, medium, low"
}"""

from backend.vision.repair import parse_structured_output
from backend.vision.schemas import STAGE1_SCHEMA

for i, img in enumerate(test_images):
    name = Path(test_thumbs[i]).stem
    print(f"\n  [{name}]")

    t0 = time.perf_counter()
    raw_a = generate_old_2step(img, OLD_S1)
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw_b = generate_new_1step(img, NEW_S1_USER, NEW_S1_SYSTEM)
    t_b = time.perf_counter() - t0

    parsed_a = parse_structured_output(raw_a, STAGE1_SCHEMA, image_type="other")
    parsed_b = parse_structured_output(raw_b, STAGE1_SCHEMA, image_type="other")

    print(f"    A (old 2-step): {raw_a.strip()[:120]}")
    print(f"      → parsed: {parsed_a}  ({t_a:.1f}s)")
    print(f"    B (new 1-step): {raw_b.strip()[:120]}")
    print(f"      → parsed: {parsed_b}  ({t_b:.1f}s)")
    print(f"    MATCH: {parsed_a.get('image_type') == parsed_b.get('image_type')}")


# ── Test 3: Stage 2 Structured Analysis ───────────────────────
print("\n" + "=" * 70)
print("TEST 3: STAGE 2 ANALYSIS (A=old prompt, B=new system+user)")
print("=" * 70)

from backend.vision.prompts import get_stage2_prompt
from backend.vision.schemas import get_schema

# Use first image, classify it first
img = test_images[0]
name = Path(test_thumbs[0]).stem

# Get image type from Stage 1
s1_result = parse_structured_output(
    generate_new_1step(img, NEW_S1_USER, NEW_S1_SYSTEM),
    STAGE1_SCHEMA, image_type="other"
)
image_type = s1_result.get("image_type", "other")
print(f"\n  [{name}] image_type={image_type}")

# Build Stage 2 prompt (includes schema)
s2_user = get_stage2_prompt(image_type)

# OLD: same prompt with "Output JSON only." still present (simulate)
OLD_S2_SYSTEM = None
old_s2_user = s2_user  # Note: we already removed "Output JSON only." from templates
# Re-add it to simulate old behavior
old_s2_user_with_json = s2_user + "\nOutput JSON only."

# NEW: system role + clean user prompt
NEW_S2_SYSTEM = "You are a strict JSON generator. Output valid JSON only matching the provided schema. No explanation, no markdown fences."

schema = get_schema(image_type)

print(f"\n  A (old: 'Output JSON only.' in user prompt, no system role):")
t0 = time.perf_counter()
raw_a = generate_old_2step(img, old_s2_user_with_json)
t_a = time.perf_counter() - t0
parsed_a = parse_structured_output(raw_a, schema, image_type=image_type)
print(f"    Raw: {raw_a.strip()[:200]}")
print(f"    Parsed keys: {list(parsed_a.keys())}")
print(f"    caption: {parsed_a.get('caption', '')[:80]}")
print(f"    tags: {parsed_a.get('tags', [])[:5]}")
print(f"    Time: {t_a:.1f}s")

print(f"\n  B (new: system role + clean user prompt):")
t0 = time.perf_counter()
raw_b = generate_new_1step(img, s2_user, NEW_S2_SYSTEM)
t_b = time.perf_counter() - t0
parsed_b = parse_structured_output(raw_b, schema, image_type=image_type)
print(f"    Raw: {raw_b.strip()[:200]}")
print(f"    Parsed keys: {list(parsed_b.keys())}")
print(f"    caption: {parsed_b.get('caption', '')[:80]}")
print(f"    tags: {parsed_b.get('tags', [])[:5]}")
print(f"    Time: {t_b:.1f}s")

# ── Test 4: JSON compliance rate ──────────────────────────────
print("\n" + "=" * 70)
print("TEST 4: JSON COMPLIANCE (3 images × Stage 1, raw output inspection)")
print("=" * 70)

def is_clean_json(raw):
    """Check if raw output is valid JSON without markdown fences."""
    s = raw.strip()
    if s.startswith("```"):
        return False, "markdown fence"
    try:
        json.loads(s)
        return True, "clean JSON"
    except json.JSONDecodeError:
        # Try to extract JSON from mixed output
        try:
            start = s.index('{')
            end = s.rindex('}') + 1
            json.loads(s[start:end])
            return False, f"JSON with prefix/suffix: '{s[:20]}...'"
        except:
            return False, f"invalid: '{s[:40]}...'"

a_clean = 0
b_clean = 0
for i, img in enumerate(test_images):
    raw_a = generate_old_2step(img, OLD_S1)
    raw_b = generate_new_1step(img, NEW_S1_USER, NEW_S1_SYSTEM)

    ok_a, reason_a = is_clean_json(raw_a)
    ok_b, reason_b = is_clean_json(raw_b)
    if ok_a: a_clean += 1
    if ok_b: b_clean += 1

    name = Path(test_thumbs[i]).stem
    print(f"  [{name}]  A: {reason_a:30s}  B: {reason_b}")

print(f"\n  JSON compliance: A(old)={a_clean}/{len(test_images)}  B(new)={b_clean}/{len(test_images)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

# Cleanup
del model, processor
cleanup()
