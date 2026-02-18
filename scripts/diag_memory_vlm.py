"""
VLM Memory Diagnostic: Qwen3-VL vs Pipeline memory attribution.

Measures MPS driver allocation at each step to determine whether
memory growth comes from model inference or pipeline accumulation.
"""
import sys, gc, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

def mps_driver_gb():
    """MPS driver allocated memory in GB."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / (1024**3)
    return 0.0

def mps_current_gb():
    """MPS current allocated memory in GB."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / (1024**3)
    return 0.0

def rss_gb():
    """Process RSS in GB."""
    import psutil, os
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def report(label):
    print(f"[{label:30s}]  Driver={mps_driver_gb():6.2f}GB  Current={mps_current_gb():6.2f}GB  RSS={rss_gb():5.2f}GB")

def cleanup():
    gc.collect(); gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()

# ── 0. Baseline ────────────────────────────
report("0. baseline")

# ── 1. Load Model ──────────────────────────
from transformers import AutoProcessor, AutoModelForImageTextToText
from backend.utils.tier_config import get_active_tier

_, tier_cfg = get_active_tier()
model_id = tier_cfg.get("vlm", {}).get("model", "Qwen/Qwen3-VL-4B-Instruct")
print(f"\nModel: {model_id}")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("mps")

report("1. model loaded")

# ── 2. Load test image ────────────────────
from PIL import Image
import glob

thumbs = sorted(glob.glob(str(PROJECT_ROOT / "output/thumbnails/agm_12_24*_thumb.png")))
if not thumbs:
    thumbs = sorted(glob.glob(str(PROJECT_ROOT / "output/thumbnails/*_thumb.png")))
test_images = [Image.open(t).convert("RGB") for t in thumbs[:8]]
print(f"\nLoaded {len(test_images)} test images: {[img.size for img in test_images]}")
report("2. images loaded (CPU)")

# ── 3. Single inference ───────────────────
def run_inference(images, label):
    """Run VLM inference on N images and measure memory."""
    system_prompt = "You are a strict JSON generator. Output valid JSON only."
    user_prompt = 'Classify this image. {"image_type": "one of: character, background, other"}'

    cleanup()
    report(f"{label} BEFORE")

    messages_list = []
    for img in images:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        messages_list.append(messages)

    if len(images) == 1:
        # Single: 1-step apply_chat_template
        inputs = processor.apply_chat_template(
            messages_list[0], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to("mps") if hasattr(v, "to") else v for k, v in inputs.items()}
    else:
        # Batch: 2-step (tokenize=False + processor)
        texts = []
        for msgs in messages_list:
            texts.append(processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))
        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to("mps")

    report(f"{label} inputs on MPS")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    report(f"{label} after generate")

    # Trim + decode
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)

    # Cleanup
    del inputs, generated_ids, trimmed
    if hasattr(model, 'rope_deltas'):
        model.rope_deltas = None
    cleanup()

    report(f"{label} after cleanup")
    print(f"  Output[0]: {decoded[0][:80]}")
    print()

# ── Run tests ──────────────────────────────
print("\n" + "="*70)
print("SINGLE IMAGE INFERENCE")
print("="*70)
run_inference(test_images[:1], "3a. batch=1")

print("="*70)
print("BATCH=2 INFERENCE")
print("="*70)
run_inference(test_images[:2], "3b. batch=2")

print("="*70)
print("BATCH=4 INFERENCE")
print("="*70)
run_inference(test_images[:4], "3c. batch=4")

print("="*70)
print("BATCH=8 INFERENCE")
print("="*70)
if len(test_images) >= 8:
    run_inference(test_images[:8], "3d. batch=8")
else:
    print(f"  Only {len(test_images)} images available, skipping batch=8")

# ── 4. Unload model ───────────────────────
print("="*70)
print("MODEL UNLOAD")
print("="*70)
report("4a. before unload")
del model, processor
cleanup()
time.sleep(1)
cleanup()
report("4b. after unload + cleanup")
