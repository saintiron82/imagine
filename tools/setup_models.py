#!/usr/bin/env python3
"""
Ollama 모델 자동 설치 스크립트 - ImageParser v3.1

티어별 AI 모델을 자동으로 다운로드하고 설치합니다.

Usage:
    python tools/setup_models.py --tier=standard
    python tools/setup_models.py --tier=pro
    python tools/setup_models.py --tier=ultra
    python tools/setup_models.py --all
"""

import argparse
import subprocess
import sys
import json
from typing import List, Dict


# Fallback if config.yaml is unavailable. Single source of truth is now
# config.yaml's ai_mode.tiers.{tier}.vlm.backends.{plat}.ollama.model and
# ai_mode.tiers.{tier}.text_embed.model. See _load_tier_models_from_config().
_FALLBACK_TIER_MODELS: Dict[str, List[str]] = {
    "standard": ["qwen3.5:4b", "qwen3-embedding:0.6b"],
    "pro": ["qwen3.5:9b", "qwen3-embedding:0.6b"],
    "ultra": ["qwen3.5:9b", "qwen3-embedding:8b"],
}


def _load_tier_models_from_config() -> Dict[str, List[str]]:
    """v3 P16: derive Ollama install list from config.yaml (single source).

    Returns the same shape as _FALLBACK_TIER_MODELS, sourced from
    ai_mode.tiers.{tier}.vlm.backends.{plat}.ollama.model + text_embed.model.
    Falls back to constants if config can't be loaded.
    """
    try:
        import platform as _plat
        from pathlib import Path as _Path
        # Avoid heavy imports — tools/setup_models.py runs before any backend init.
        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
        from backend.utils.config import get_config  # type: ignore
        cfg = get_config()
        plat_key = {"Darwin": "darwin", "Windows": "windows", "Linux": "linux"}.get(
            _plat.system(), "linux"
        )
        tiers = (cfg.get("ai_mode.tiers") or {})
        out: Dict[str, List[str]] = {}
        for tier_name, tier_cfg in tiers.items():
            if not isinstance(tier_cfg, dict):
                continue
            vlm = tier_cfg.get("vlm", {}) or {}
            ollama_cfg = (vlm.get("backends", {}) or {}).get(plat_key, {}).get("ollama", {})
            vlm_model = (ollama_cfg or {}).get("model") or vlm.get("model")
            embed = tier_cfg.get("text_embed", {}) or {}
            embed_model = embed.get("model")
            models = []
            if vlm_model and ":" in str(vlm_model):  # only Ollama-format names
                models.append(vlm_model)
            if embed_model and ":" in str(embed_model):
                models.append(embed_model)
            if models:
                out[tier_name] = models
        return out or _FALLBACK_TIER_MODELS
    except Exception as e:
        print(f"⚠️  config.yaml load failed ({e}); using fallback model list")
        return _FALLBACK_TIER_MODELS


TIER_MODELS: Dict[str, List[str]] = _load_tier_models_from_config()


def check_ollama() -> bool:
    """
    Ollama 설치 여부 확인.

    Returns:
        bool: Ollama가 설치되어 있으면 True
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        version = result.stdout.strip()
        print(f"✅ Ollama installed: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_ollama_server() -> bool:
    """
    Ollama 서버 실행 여부 확인.

    Returns:
        bool: 서버가 실행 중이면 True
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def list_installed_models() -> List[str]:
    """
    현재 설치된 Ollama 모델 목록 조회.

    Returns:
        List[str]: 설치된 모델 이름 목록
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = []
        for line in lines:
            if line.strip():
                # Extract model name (first column)
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def pull_model(model: str, dry_run: bool = False) -> bool:
    """
    Ollama 모델 다운로드.

    Args:
        model: 모델 이름 (예: qwen3.5:9b)
        dry_run: True면 실제 다운로드 없이 시뮬레이션

    Returns:
        bool: 성공 시 True
    """
    print(f"\n📥 Pulling {model}...")

    if dry_run:
        print(f"   [DRY RUN] Would execute: ollama pull {model}")
        return True

    try:
        # Use subprocess.run with real-time output
        process = subprocess.Popen(
            ["ollama", "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Print output line by line
        for line in process.stdout:
            print(f"   {line.rstrip()}")

        process.wait()

        if process.returncode == 0:
            print(f"   ✅ {model} installed successfully")
            return True
        else:
            print(f"   ❌ {model} installation failed (exit code: {process.returncode})")
            return False

    except FileNotFoundError:
        print(f"   ❌ Ollama command not found")
        return False
    except Exception as e:
        print(f"   ❌ {model} installation error: {e}")
        return False


def setup_tier(tier: str, dry_run: bool = False, skip_installed: bool = True):
    """
    특정 티어의 모델 설치.

    Args:
        tier: "standard" | "pro" | "ultra"
        dry_run: True면 실제 다운로드 없이 시뮬레이션
        skip_installed: True면 이미 설치된 모델 건너뛰기
    """
    models = TIER_MODELS.get(tier, [])

    if not models:
        print(f"⚠️  No Ollama models defined for tier '{tier}'")
        print(f"   Note: {tier.upper()} tier may use Transformers models (Hugging Face)")
        return

    print(f"\n{'='*60}")
    print(f"📦 Setting up {tier.upper()} tier ({len(models)} models)")
    print(f"{'='*60}")

    # Check already installed models
    installed = []
    if skip_installed:
        installed = list_installed_models()
        print(f"📋 Currently installed models: {len(installed)}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for model in models:
        # Check if already installed
        if skip_installed and model in installed:
            print(f"\n⏭️  Skipping {model} (already installed)")
            skip_count += 1
            continue

        # Pull model
        if pull_model(model, dry_run):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"📊 Summary for {tier.upper()} tier:")
    print(f"   ✅ Installed: {success_count}")
    print(f"   ⏭️  Skipped: {skip_count}")
    print(f"   ❌ Failed: {fail_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Ollama models for ImageParser v3.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/setup_models.py --tier=pro
  python tools/setup_models.py --all
  python tools/setup_models.py --all --dry-run
  python tools/setup_models.py --tier=ultra --force

Tiers:
  standard: Lightweight models (≤6GB VRAM)
  pro:      Balanced models (8-16GB VRAM)
  ultra:    High-end models (≥20GB VRAM)
        """
    )
    parser.add_argument(
        "--tier",
        choices=["standard", "pro", "ultra"],
        help="Setup specific tier"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Setup all tiers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate installation without downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models are installed"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 ImageParser v3.1 Model Setup")
    print("=" * 60)

    # Check Ollama installation
    if not check_ollama():
        print("\n❌ Ollama not found!")
        print("\n💡 Please install Ollama first:")
        print("   - Windows/Mac: https://ollama.com/download")
        print("   - Linux: curl https://ollama.com/install.sh | sh")
        sys.exit(1)

    # Check Ollama server
    if not args.dry_run and not check_ollama_server():
        print("\n⚠️  Ollama server is not running!")
        print("   Starting Ollama server in background...")
        print("   Please run: ollama serve")
        print("   Or just continue - ollama pull will start the server automatically")

    # Determine tiers to setup
    tiers_to_setup = []
    if args.all:
        tiers_to_setup = ["standard", "pro", "ultra"]
    elif args.tier:
        tiers_to_setup = [args.tier]
    else:
        print("\n❌ Please specify --tier or --all")
        print("   Example: python tools/setup_models.py --tier=pro")
        sys.exit(1)

    # Setup each tier
    for tier in tiers_to_setup:
        setup_tier(tier, dry_run=args.dry_run, skip_installed=not args.force)

    print("\n" + "=" * 60)
    print("✅ Model setup complete!")
    print("=" * 60)

    if not args.dry_run:
        print("\n💡 Next steps:")
        print("   1. Verify installation: ollama list")
        print("   2. Start processing files:")
        print("      python backend/pipeline/ingest_engine.py --file \"path/to/image.psd\"")
        print("   3. Or auto-detect tier:")
        print("      python -c \"from backend.utils.tier_config import get_active_tier; print(get_active_tier())\"")
    print()


if __name__ == "__main__":
    main()
