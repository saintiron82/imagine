#!/usr/bin/env python3
"""
Install required Hugging Face models for local/offline runtime.

Usage:
  .venv/bin/python scripts/install_hf_models.py --all
  .venv/bin/python scripts/install_hf_models.py --vlm
  .venv/bin/python scripts/install_hf_models.py --dinov2
  .venv/bin/python scripts/install_hf_models.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils.tier_config import get_active_tier


def _is_hf_model_id(model_id: str) -> bool:
    if not model_id:
        return False
    # Ollama tags are usually like "qwen3-vl:8b"
    return "/" in model_id and ":" not in model_id


def install_vlm_model() -> None:
    from transformers import AutoProcessor, AutoModelForImageTextToText

    tier_name, tier_cfg = get_active_tier()
    vlm_cfg = (tier_cfg or {}).get("vlm", {})
    model_id = vlm_cfg.get("model", "")
    backend = (vlm_cfg.get("backend") or "").lower()

    if not _is_hf_model_id(model_id):
        print(f"[SKIP] VLM model is not HuggingFace id: {model_id}")
        return

    print(f"[INSTALL] VLM ({tier_name}/{backend}): {model_id}")
    AutoProcessor.from_pretrained(model_id)
    AutoModelForImageTextToText.from_pretrained(model_id)
    print("[OK] VLM model installed")


def install_dinov2_model() -> None:
    from transformers import AutoImageProcessor, AutoModel

    model_id = "facebook/dinov2-base"
    print(f"[INSTALL] DINOv2: {model_id}")
    AutoImageProcessor.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)
    print("[OK] DINOv2 model installed")


def check_local_models() -> int:
    from transformers import AutoImageProcessor, AutoModel

    code = 0
    tier_name, tier_cfg = get_active_tier()
    vlm_cfg = (tier_cfg or {}).get("vlm", {})
    model_id = vlm_cfg.get("model", "")

    if _is_hf_model_id(model_id):
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            AutoProcessor.from_pretrained(model_id, local_files_only=True)
            AutoModelForImageTextToText.from_pretrained(model_id, local_files_only=True)
            print(f"[OK] Local VLM ready: {model_id}")
        except Exception:
            print(f"[MISSING] Local VLM missing: {model_id}")
            code = 1
    else:
        print(f"[SKIP] VLM local-check skipped (non-HF model): {model_id}")

    try:
        AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
        AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True)
        print("[OK] Local DINOv2 ready: facebook/dinov2-base")
    except Exception:
        print("[MISSING] Local DINOv2 missing: facebook/dinov2-base")
        code = 1

    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Install/check local HF models for ImageParser")
    parser.add_argument("--vlm", action="store_true", help="Install active-tier VLM model")
    parser.add_argument("--dinov2", action="store_true", help="Install DINOv2 model")
    parser.add_argument("--all", action="store_true", help="Install both VLM and DINOv2")
    parser.add_argument("--check", action="store_true", help="Check local-only availability")
    args = parser.parse_args()

    if args.check:
        return check_local_models()

    do_vlm = args.all or args.vlm
    do_dino = args.all or args.dinov2

    if not do_vlm and not do_dino:
        parser.print_help()
        return 1

    if do_vlm:
        install_vlm_model()
    if do_dino:
        install_dinov2_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
