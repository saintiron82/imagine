# Ollama Setup Guide

Complete guide for setting up Ollama as the AI vision backend for ImageParser.

## Why Ollama?

**Deployment Benefits**:
- ✅ **Memory Efficient**: Models load only during use (`keep_alive=0`)
- ✅ **Easy Installation**: Single binary, no Python dependencies
- ✅ **Windows Compatible**: Native Windows support
- ✅ **Standalone**: No internet connection needed after model download
- ✅ **Fast**: Optimized inference with llama.cpp backend

**Best for**:
- Production deployments
- Windows workstations
- Memory-constrained environments
- Standalone installations

## Installation

### Windows

```powershell
# 1. Download Ollama installer
# Visit: https://ollama.com/download/windows

# 2. Run installer (OllamaSetup.exe)
# Default installation: C:\Users\<username>\AppData\Local\Programs\Ollama

# 3. Verify installation
ollama --version

# 4. Start Ollama service (runs automatically after install)
# Check system tray for Ollama icon
```

### macOS

```bash
# Download and install
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

### Linux

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version
```

## Download Vision Model

After installing Ollama, download the vision model:

```powershell
# Qwen2.5-VL 7B (recommended, ~4.7GB)
ollama pull qwen2.5-vl:7b

# Alternative models:
# ollama pull llava:7b         # LLaVA 7B (~4.5GB)
# ollama pull bakllava:7b      # BakLLaVA 7B (~4.5GB)
# ollama pull llava:13b        # LLaVA 13B (higher quality, ~8GB)
```

**Model Download Time**:
- ~10-20 minutes on good internet connection
- Models are cached locally after first download

## Configure ImageParser

### 1. Create .env file

```powershell
# Copy template
copy .env.example .env
```

### 2. Edit .env

```ini
# Set Ollama as vision backend
VISION_BACKEND=ollama

# Specify model (must match pulled model)
VISION_MODEL=qwen2.5-vl:7b

# Ollama server URL (default: local)
OLLAMA_HOST=http://localhost:11434
```

### 3. Verify Connection

```powershell
# Test Ollama API
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

## Usage

Once configured, ImageParser will automatically use Ollama:

```powershell
# Process images with Ollama vision analysis
python backend/pipeline/ingest_engine.py --file "test.psd"

# Ollama will:
# 1. Load model when needed
# 2. Analyze image
# 3. Immediately unload model (keep_alive=0)
```

**Memory Usage**:
- Idle: ~100MB (Ollama service)
- During analysis: ~5-8GB (model + inference)
- After analysis: Returns to ~100MB

## Troubleshooting

### Ollama not responding

```powershell
# Check if Ollama service is running
Get-Process ollama

# Restart Ollama service
Stop-Process -Name "ollama" -Force
# Restart from Start Menu or double-click Ollama icon
```

### Model not found

```powershell
# List installed models
ollama list

# If model missing, pull it
ollama pull qwen2.5-vl:7b
```

### Connection refused

```powershell
# Check Ollama server
curl http://localhost:11434/api/tags

# If fails, ensure Ollama is running
# Check system tray for Ollama icon
# Or start manually: ollama serve
```

### Slow inference

**Possible causes**:
- CPU-only inference (no GPU)
- Large model on limited RAM
- Other applications using resources

**Solutions**:
- Use smaller model: `llava:7b` instead of `llava:13b`
- Close other applications
- Check GPU availability: `nvidia-smi` (NVIDIA GPUs)

## Comparison: Transformers vs Ollama

| Feature | Transformers | Ollama |
|---------|-------------|--------|
| **Installation** | Python packages (~5GB) | Single binary (~200MB) |
| **Model Download** | Automatic (Hugging Face) | Manual (ollama pull) |
| **Memory (Idle)** | ~2GB (models stay loaded) | ~100MB (models unload) |
| **Memory (Active)** | ~6GB | ~5-8GB |
| **Inference Speed** | Fast (GPU) / Slow (CPU) | Optimized (llama.cpp) |
| **Quality** | High (latest models) | High (same models) |
| **Deployment** | Complex (Python env) | Simple (standalone) |
| **Best For** | Development, GPU servers | Production, Windows PCs |

## Advanced Configuration

### Custom Ollama Host (Server Mode)

For centralized server deployment:

```ini
# .env on client machines
VISION_BACKEND=ollama
OLLAMA_HOST=http://192.168.1.100:11434  # Server IP
```

```powershell
# On server machine: expose Ollama to network
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

### GPU Acceleration

Ollama automatically uses available GPUs:
- **NVIDIA**: CUDA (automatic)
- **AMD**: ROCm (Linux only)
- **Apple**: Metal (macOS M-series)

Check GPU usage:
```powershell
# NVIDIA
nvidia-smi

# During inference, you should see:
# - GPU Memory usage increase
# - GPU utilization spike
```

### Model Customization

Create custom vision prompts via Modelfile:

```dockerfile
# custom-vision.Modelfile
FROM qwen2.5-vl:7b

PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM You are an expert image analyst specializing in PSD layer analysis.
```

```powershell
# Create custom model
ollama create custom-vision -f custom-vision.Modelfile

# Use in .env
VISION_MODEL=custom-vision
```

## Next Steps

- ✅ [Installation Guide](../INSTALLATION.md) - Complete setup
- ✅ [PostgreSQL Setup](postgresql_setup.md) - Database configuration
- ✅ [Phase Roadmap](phase_roadmap.md) - Development timeline
