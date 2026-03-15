# Imagine - AI Model Setup Guide

## Overview

Imagine uses three AI models in its **Triaxis** search architecture:

| Model | Role | Input | Output |
|-------|------|-------|--------|
| **SigLIP2** (VV) | Visual similarity | Image pixels | 1152-dim vector |
| **Qwen3-VL** (VLM) | Caption & tag generation | Image + prompt | MC text (caption + tags) |
| **Qwen3-Embedding** (MV) | Semantic search | MC text | 1024-dim vector |

```
Image ──→ [SigLIP2] ──→ VV (visual search)
  │
  └──→ [Qwen3-VL] ──→ MC caption/tags ──→ [Qwen3-Embedding] ──→ MV (semantic search)
```

---

## Tier System

Hardware varies — so we provide three tiers:

| Tier | Target | VRAM | VLM | MV Model |
|------|--------|------|-----|----------|
| **standard** | Integrated GPU laptop | ~6GB | Qwen3-VL-**2B** | qwen3-embedding:0.6b (1024d) |
| **pro** | Mac M-series, RTX 3060+ | 8-16GB | Qwen3-VL-**4B** | qwen3-embedding:0.6b (1024d) |
| **ultra** | RTX 4090, A100 | 20GB+ | Qwen3-VL-**8B** | qwen3-embedding:8b (4096d) |

**Key design**: VV (SigLIP2) is **identical across all tiers** — switching tiers never requires VV regeneration. standard↔pro switching is fully seamless (same MV model too).

---

## Why MLX?

### The Problem

On macOS with Apple Silicon, PyTorch's MPS backend works but is **not optimized** for vision-language model inference. Token generation is slow, and memory usage is high with full-precision models.

### MLX Advantage

MLX is Apple's **native ML framework** designed specifically for Apple Silicon's unified memory architecture.

| Metric | PyTorch MPS | MLX | Improvement |
|--------|------------|-----|-------------|
| Time to first token (TTFT) | ~2000ms | ~500ms | **4x faster** |
| Token generation | ~1 token/sec | ~30 tokens/sec | **30x faster** |
| Memory (4B model) | ~8GB (fp16) | ~3GB (4-bit) | **2.5x less** |

### Why It Matters

The VLM (Qwen3-VL) generates captions for every image in the database. With thousands of images, the difference between 1 token/sec and 30 tokens/sec is the difference between hours and minutes.

### When MLX Is Used

MLX is **only used for VLM inference on macOS**:

```
macOS + pro tier:   MLX (4-bit quantized) → transformers (fallback)
macOS + ultra tier: MLX (4-bit quantized) → transformers (fallback)
Windows:            ollama → transformers (MLX not available)
Linux:              vllm → ollama → transformers (MLX not available)
```

VV (SigLIP2) and MV (Qwen3-Embedding) always use **transformers** directly — they're fast enough without MLX.

### MLX Models

MLX uses **quantized community models** from HuggingFace:

| Tier | MLX Model | Quantization |
|------|-----------|-------------|
| standard | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | 4-bit |
| pro | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 4-bit |
| ultra | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | 4-bit |

---

## Backend Fallback Chain

The VLM factory (`vision_factory.py`) uses an explicit fallback chain to ensure the app always works, regardless of what's installed:

```
1. Check platform (macOS / Windows / Linux)
2. Check tier (standard / pro / ultra)
3. Build backend chain based on config.yaml
4. Try each backend in order:
   - Check availability (is package installed? is service running?)
   - Try to instantiate
   - If success → use it
   - If fail → try next
5. transformers is always the last resort (guaranteed available)
```

### Platform × Tier Matrix

| Platform | standard | pro | ultra |
|----------|----------|-----|-------|
| **macOS** | transformers | mlx → transformers | mlx → transformers |
| **Windows** | transformers | transformers | ollama → transformers |
| **Linux** | transformers | transformers | vllm → ollama → transformers |

### Backend Characteristics

| Backend | Pros | Cons | Platform |
|---------|------|------|----------|
| **MLX** | Native Apple Silicon, 4-bit quant, fastest single-image | macOS only, batch_size=1 | macOS |
| **vLLM** | Batch processing (8-16x throughput) | CUDA only, no Windows | Linux GPU |
| **Ollama** | Easy install, auto model management | Slower, batch_size=1 | All |
| **Transformers** | Always available, MPS/CUDA/CPU auto | Slower than MLX/vLLM | All (fallback) |

---

## Installation

### Quick Start

```bash
# Full setup (recommended)
python backend/setup/installer.py --full-setup

# Or step by step:
python backend/setup/installer.py --install           # pip packages
python backend/setup/installer.py --download-model    # SigLIP2 (HuggingFace)
python backend/setup/installer.py --setup-ollama      # Ollama models (if needed)
python backend/setup/installer.py --init-db           # SQLite schema

# Verify
python backend/setup/installer.py --check
```

### What Gets Downloaded

| Model | Size | Cache Location | Required? |
|-------|------|---------------|-----------|
| SigLIP2 so400m-naflex | ~1.6GB | `~/.cache/huggingface/hub/` | Yes (all tiers) |
| Qwen3-VL-4B (transformers) | ~8GB | `~/.cache/huggingface/hub/` | Yes (pro tier) |
| Qwen3-VL-4B-4bit (MLX) | ~3GB | `~/.cache/huggingface/hub/` | macOS only |
| qwen3-embedding:0.6b (Ollama) | ~1.2GB | `~/.ollama/models/` | If using Ollama |

### First Run

On first run, the Electron app will:
1. Check environment (`checkEnv`)
2. Show missing dependencies
3. Offer to install them automatically
4. Download models in background

---

## Model Selection Rationale

### SigLIP2 (not Meta PE-Core)

| Criteria | SigLIP2 so400m | Meta PE-Core |
|----------|---------------|-------------|
| License | **Apache 2.0** | CC-BY-NC (non-commercial) |
| Ecosystem | HuggingFace transformers native | Custom library / OpenCLIP |
| macOS MPS | **Verified on M5** | Unverified, xformers dependency |
| Performance | ~83% ImageNet ZS | ~83.5% ImageNet ZS |
| NaFlex | **Yes** (variable aspect ratio) | No |

Decision: 0.5% performance gap is not worth the license restriction and ecosystem incompatibility.

### Qwen3-VL (not LLaVA, InternVL, Phi-4)

| Criteria | Qwen3-VL | Others |
|----------|---------|--------|
| Size lineup | **2B / 4B / 8B** (maps 1:1 to tiers) | Large gaps (7B/13B) |
| Korean+English | **Excellent** | Mostly English-focused |
| Cross-platform | transformers + MLX + Ollama | transformers only |
| License | **Apache 2.0** | Mixed |

### Qwen3-Embedding (not BGE-M3, Jina, E5)

| Criteria | Qwen3-Embedding | Others |
|----------|----------------|--------|
| MRL support | **Yes** (dimension truncation) | BGE: No, Jina: Yes |
| Size lineup | **0.6B + 8B** | Single size |
| Korean quality | **Excellent** | Varies |
| Same family as VLM | **Yes** (tokenizer compatible) | No |

---

## Pipeline: Phase-based Processing

```
Phase P (Parse)   → Extract metadata, thumbnails         [No model]
  ↓
Phase V (Vision)  → Generate MC captions/tags             [VLM loaded]
  ↓ unload VLM
Phase E (Embed)   → Generate VV + MV vectors             [SigLIP2 + Embedding loaded]
  ↓ unload encoders
Phase S (Summary) → Confirm completion                    [No model]
```

Only **one model family** in memory at a time → GPU memory efficient.

---

## Configuration

All model settings are in `config.yaml` (single source of truth):

```yaml
ai_mode:
  override: pro          # manual tier selection
  auto_detect: false     # or true for automatic VRAM detection

  tiers:
    pro:
      visual:
        model: google/siglip2-so400m-patch16-naflex
        dimensions: 1152
      vlm:
        backend: auto    # auto selects best available
        model: Qwen/Qwen3-VL-4B-Instruct
        backends:
          darwin:
            backend: mlx
            model: mlx-community/Qwen3-VL-4B-Instruct-4bit
            fallback: transformers
          windows:
            backend: transformers
          linux:
            backend: transformers
      text_embed:
        model: qwen3-embedding:0.6b
        dimensions: 1024
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| MLX model load fails | mlx-vlm not installed | `pip install mlx-vlm` or fallback to transformers |
| Ollama connection refused | Ollama not running | `ollama serve` or install from ollama.com |
| CUDA out of memory | Model too large for GPU | Lower tier in config.yaml |
| Slow VLM on Mac | Using transformers instead of MLX | Install mlx-vlm: `pip install mlx-vlm` |
| MV results empty | Vision phase not run | Re-process with `--no-skip` |
