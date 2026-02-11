# ImageParser Installation Guide

Cross-platform setup guide for Windows, macOS, and Linux.

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for Electron frontend)
- **Git**

## Quick Start (All Platforms)

### 1. Clone & Virtual Environment

```bash
git clone <repository-url>
cd ImageParser
python -m venv .venv
```

Activate the virtual environment:

| Platform | Command |
|----------|---------|
| **Windows** (PowerShell) | `.venv\Scripts\activate` |
| **Windows** (CMD) | `.venv\Scripts\activate.bat` |
| **macOS / Linux** | `source .venv/bin/activate` |

### 2. Install Backend (Python)

```bash
pip install -r requirements.txt
python backend/setup/installer.py --full-setup
```

This will:
- Install all Python packages
- Download SigLIP2 VV model (~600MB-3GB depending on tier)
- Initialize SQLite database
- Check and pull Ollama models (if Ollama is running)

### 3. Install Ollama (AI Models)

Ollama provides the MV and VLM (Vision Language Model) backends.

| Platform | Install Method |
|----------|---------------|
| **Windows** | Download from [ollama.com/download](https://ollama.com/download) |
| **macOS** | `brew install ollama` or download from [ollama.com/download](https://ollama.com/download) |
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |

After installing, pull the required models:

```bash
# Standard tier (default, ~6GB VRAM)
ollama pull qwen3-embedding:0.6b

# Or let the installer do it:
python backend/setup/installer.py --setup-ollama
```

### 4. Install Frontend (Electron)

```bash
cd frontend
npm install
```

### 5. Run

```bash
# From frontend/ directory
npm run electron:dev
```

## Verify Installation

```bash
python backend/setup/installer.py --check
```

Expected output:
```
✅ Python Dependencies: OK
✅ Visual Model Cached: google/siglip2-base-patch16-224
✅ SQLite: 3.xx.x
✅ sqlite-vec Extension: 0.x.x
✅ Ollama Running: Yes
✅ Ollama Models: Ready
✅ GPU: NVIDIA RTX xxxx (xxxx MB VRAM)  # or Apple Silicon MPS / CPU mode
```

## Platform-Specific Notes

### Windows

- **GPU**: NVIDIA GPU with CUDA recommended. Check with `nvidia-smi`.
- **PyTorch**: Installed automatically via `pip install -r requirements.txt`. For CUDA-specific versions, see [pytorch.org](https://pytorch.org/get-started/locally/).
- **Encoding**: All file I/O uses UTF-8. No cp949 issues.

### macOS

- **Apple Silicon (M1/M2/M3/M4)**: MPS acceleration is automatically detected for SigLIP2 encoding. No extra setup needed.
- **PyTorch MPS**: Installed automatically. Verify with:
  ```bash
  python -c "import torch; print(torch.backends.mps.is_available())"
  ```
- **Intel Mac**: CPU mode only for embeddings. Ollama handles VLM acceleration.

### Linux

- **GPU**: NVIDIA GPU with CUDA recommended.
- **SQLite**: Ensure system SQLite is 3.41+. Check with:
  ```bash
  python -c "import sqlite3; print(sqlite3.sqlite_version)"
  ```

## Tier System

ImageParser supports 3 tiers configured in `config.yaml` (`ai_mode.override`):

| Tier | VRAM | VV Model | VLM | MV Model |
|------|------|-------------|-----|----------------|
| **standard** (default) | ~6GB | SigLIP2-base (768d) | Qwen3-VL-2B | qwen3-embedding:0.6b |
| **pro** | 8-16GB | SigLIP2-so400m (1152d) | Qwen3-VL-4B | qwen3-embedding:0.6b |
| **ultra** | 20GB+ | SigLIP2-giant (1664d) | Qwen3-VL-8B | qwen3-embedding:8b |

To change tier:
```bash
# Edit config.yaml
ai_mode:
  override: standard  # Change to: pro or ultra
```

Then re-run setup to download the appropriate models:
```bash
python backend/setup/installer.py --download-model
python backend/setup/installer.py --setup-ollama
```

## Individual Setup Commands

```bash
# Install Python packages only
python backend/setup/installer.py --install

# Download VV model only
python backend/setup/installer.py --download-model

# Pull Ollama models only
python backend/setup/installer.py --setup-ollama

# Initialize database only
python backend/setup/installer.py --init-db

# Show SQLite setup guide
python backend/setup/installer.py --setup-sqlite
```

## Troubleshooting

### Ollama connection refused

Ollama is not running. Start it:
- **Windows**: Ollama runs as a system service after installation. Check system tray.
- **macOS/Linux**: Run `ollama serve` in a terminal.

### SigLIP2 model download stuck

Clear HuggingFace cache and retry:
```bash
# Remove cached model
rm -rf ~/.cache/huggingface/hub/models--google--siglip2*

# Re-download
python backend/setup/installer.py --download-model
```

### sqlite-vec extension load failed

```bash
pip install --force-reinstall sqlite-vec
```

Requires Python 3.11+ with SQLite 3.41+.

### Search returns no results

Files need to be processed first. Use the Electron app to select a folder and run "Process", or:
```bash
python backend/pipeline/ingest_engine.py --discover "path/to/images"
```

### macOS: "python3" not found in Electron

Ensure `.venv` is created with `python3 -m venv .venv` and the virtual environment is active before running `npm run electron:dev`.

## System Requirements

| | Minimum | Recommended |
|--|---------|------------|
| **RAM** | 8 GB | 16 GB |
| **Disk** | 10 GB | 20 GB |
| **GPU** | None (CPU mode) | NVIDIA 6GB+ VRAM or Apple Silicon |
| **Python** | 3.11 | 3.12+ |
| **Node.js** | 18 | 20+ |
