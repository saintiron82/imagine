"""
Sprint 2.1 (P14/P15/P17): VLM Factory platform-aware resolver tests.

Tests `VisionAnalyzerFactory.resolve_platform_model` and
`resolve_batch_size` — pure-Python static methods, no model loads.

We isolate import by reading the module file directly so torch/PIL/etc.
need not be installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_resolver():
    """Import only resolve_platform_model + resolve_batch_size."""
    spec_path = PROJECT_ROOT / "backend" / "vision" / "vision_factory.py"
    src = spec_path.read_text()
    # Extract just the two staticmethods + class shell so we can exec without
    # triggering top-level imports (torch, etc.).
    ns: dict = {}
    # Grab the resolve_platform_model and resolve_batch_size source
    import re
    m = re.search(
        r"(    @staticmethod\s+def resolve_platform_model.*?)(?=    # ── Fallback)",
        src,
        re.DOTALL,
    )
    assert m, "Could not locate resolver source"
    body = m.group(1)
    # Body is already 4-space indented (was inside the class). Wrap in fresh class.
    code = "class VisionAnalyzerFactory:\n" + body
    exec(code, ns)
    return ns["VisionAnalyzerFactory"]


VAF = _load_resolver()


_PRO_VLM = {
    "backend": "auto",
    "device": "auto",
    "model": "Qwen/Qwen3.5-9B",
    "mlx_model": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "backends": {
        "darwin": {
            "mlx": {"model": "mlx-community/Qwen3.5-9B-MLX-4bit"},
            "ollama": {"model": "qwen3.5:9b"},
            "transformers": {"model": "Qwen/Qwen3.5-9B"},
        },
        "windows": {
            "backend": "ollama",
            "model": "qwen3.5:9b",
            "ollama": {"model": "qwen3.5:9b"},
            "transformers": {"model": "Qwen/Qwen3.5-9B"},
        },
        "linux": {
            "backend": "ollama",
            "model": "qwen3.5:9b",
            "vllm": {"model": "Qwen/Qwen3.5-9B"},
            "ollama": {"model": "qwen3.5:9b"},
            "transformers": {"model": "Qwen/Qwen3.5-9B"},
        },
    },
}


def test_darwin_mlx_uses_per_backend_model():
    assert VAF.resolve_platform_model(_PRO_VLM, "darwin", "mlx") \
        == "mlx-community/Qwen3.5-9B-MLX-4bit"


def test_darwin_ollama_uses_per_backend_model():
    assert VAF.resolve_platform_model(_PRO_VLM, "darwin", "ollama") == "qwen3.5:9b"


def test_windows_ollama_uses_per_backend_model():
    """The whole point of P14: Windows + Ollama must NOT get HF format."""
    got = VAF.resolve_platform_model(_PRO_VLM, "windows", "ollama")
    assert got == "qwen3.5:9b", \
        f"Windows + Ollama should resolve to Ollama-format model, got {got}"


def test_linux_vllm_uses_per_backend_model():
    assert VAF.resolve_platform_model(_PRO_VLM, "linux", "vllm") == "Qwen/Qwen3.5-9B"


def test_linux_transformers_uses_per_backend_model():
    assert VAF.resolve_platform_model(_PRO_VLM, "linux", "transformers") == "Qwen/Qwen3.5-9B"


def test_legacy_fallback_when_per_backend_missing():
    """If backends.{plat}.{backend}.model is absent, fall back to legacy backends.{plat}.model."""
    legacy_cfg = {
        "backend": "auto",
        "model": "Qwen/Qwen3.5-9B",
        "backends": {
            "windows": {
                "backend": "ollama",
                "model": "qwen3.5:9b",
                # no nested 'ollama' key
            },
        },
    }
    assert VAF.resolve_platform_model(legacy_cfg, "windows", "ollama") == "qwen3.5:9b"


def test_top_level_fallback_when_no_platform_config():
    """When backends.{plat} is empty entirely, fall back to top-level model."""
    minimal_cfg = {"backend": "auto", "model": "qwen3.5:9b", "backends": {}}
    assert VAF.resolve_platform_model(minimal_cfg, "linux", "ollama") == "qwen3.5:9b"


def test_mlx_model_legacy_shortcut_on_darwin():
    """Old config without backends.darwin.mlx still resolves via mlx_model key."""
    legacy_cfg = {
        "backend": "auto",
        "model": "Qwen/Qwen3.5-9B",
        "mlx_model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "backends": {},
    }
    got = VAF.resolve_platform_model(legacy_cfg, "darwin", "mlx")
    assert got == "mlx-community/Qwen3.5-9B-MLX-4bit"


def test_batch_size_per_backend_override():
    cfg = {
        "model": "x",
        "batch_size": 4,
        "backends": {
            "linux": {"vllm": {"model": "y", "batch_size": 16}},
        },
    }
    assert VAF.resolve_batch_size(cfg, "linux", "vllm") == 16
    # ollama not configured — falls back to top-level
    assert VAF.resolve_batch_size(cfg, "linux", "ollama") == 4
