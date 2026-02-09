# Platform-Specific Optimization Guide

**v3.1.1** - Cross-platform vision backend optimization

---

## Overview

ImageParser v3.1.1 introduces **platform-specific optimization** for vision processing. The system automatically selects the best backend based on your operating system and available software.

**Key Benefits:**
- **Windows**: Stable processing with Ollama
- **Mac**: 3-5x faster with vLLM batch processing
- **Linux**: 3-5x faster with vLLM batch processing

---

## Platform Comparison

| Platform | Recommended Backend | Batch Processing | Performance (10 images) |
|----------|---------------------|------------------|------------------------|
| **Windows** | Ollama | Sequential (1) | ~510s (51s × 10) |
| **Mac** | vLLM | Parallel (16) | ~60s (8.5x faster) |
| **Linux** | vLLM | Parallel (16) | ~60s (8.5x faster) |

---

## Backend Selection

### AUTO Mode (Recommended)

The simplest way to use platform optimization is **AUTO mode**:

```yaml
# config.yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: auto  # Automatically selects best backend
```

**AUTO mode behavior:**
1. **Detects your platform** (Windows/Mac/Linux)
2. **Checks available backends** (vLLM, Ollama, Transformers)
3. **Selects optimal configuration** based on platform
4. **Sets optimal batch size** for the backend

### Manual Selection

You can explicitly specify a backend:

```yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: vllm     # or 'ollama', 'transformers'
        model: Qwen/Qwen3-VL-8B-Instruct
```

**Warning:** Manual selection may not work on all platforms (e.g., vLLM on Windows).

---

## Installation Guides

### Windows

**Step 1: Install Ollama**
```powershell
# Download from https://ollama.com
winget install Ollama.Ollama

# Start Ollama
ollama serve

# Pull Qwen3-VL model
ollama pull qwen3-vl:8b
```

**Step 2: Configure ImageParser**
```yaml
# config.yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: auto  # Will automatically use Ollama on Windows
```

**Expected Performance:**
- Batch size: 1 (sequential)
- Processing time: ~51s per image
- Stable and reliable

---

### Mac (macOS)

**Option 1: vLLM (Recommended for Speed)**

```bash
# Install vLLM
pip install vllm

# vLLM will automatically download models on first use
```

**Option 2: Ollama (Fallback)**

```bash
# Install Ollama
brew install ollama

# Start Ollama
ollama serve

# Pull model
ollama pull qwen3-vl:8b
```

**Configure ImageParser:**
```yaml
# config.yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: auto  # Will prefer vLLM if installed, fallback to Ollama
```

**Expected Performance:**
- **vLLM**: Batch size 16, ~6s per image (8.5x faster)
- **Ollama**: Batch size 1, ~51s per image

---

### Linux

**Option 1: vLLM (Recommended for Speed)**

```bash
# Install vLLM
pip install vllm

# For CUDA support (NVIDIA GPU required)
pip install vllm[cuda]
```

**Option 2: Ollama (Fallback)**

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# Pull model
ollama pull qwen3-vl:8b
```

**Configure ImageParser:**
```yaml
# config.yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: auto  # Will prefer vLLM if installed
```

**Expected Performance:**
- **vLLM**: Batch size 16, ~6s per image (8.5x faster)
- **Ollama**: Batch size 1, ~51s per image

---

## Platform Detection

To check your platform and recommended configuration:

```bash
# Run platform detector
python -m backend.utils.platform_detector
```

**Example Output:**
```
======================================================================
PLATFORM INFORMATION
======================================================================
OS: Windows 11
Architecture: AMD64
Python: 3.11.5

======================================================================
BACKEND RECOMMENDATIONS
======================================================================
Optimal Backend: ollama
Optimal Batch Size: 1
Available Backends: ollama, transformers

======================================================================
WARNINGS
======================================================================
⚠ Windows: Using Ollama (sequential processing).
  For faster batch processing, consider Mac/Linux with vLLM.
======================================================================
```

---

## Batch Processing

### Windows (Ollama)

```bash
# Process images sequentially (optimal for Windows)
python backend/pipeline/ingest_engine.py \
  --discover "C:\path\to\images" \
  --batch-size 1
```

**Why batch_size=1?**
- Ollama Vision API doesn't benefit from batching
- Actually becomes **slower** with multiple requests (0.6x performance)
- Sequential processing is optimal

### Mac/Linux (vLLM)

```bash
# Process images in parallel (optimal for Mac/Linux)
python backend/pipeline/ingest_engine.py \
  --discover "/path/to/images" \
  --batch-size 16
```

**Why batch_size=16?**
- vLLM excels at batch processing
- PagedAttention enables efficient memory management
- Continuous Batching dynamically schedules requests

### AUTO Batch Size

Let the system decide:

```bash
python backend/pipeline/ingest_engine.py \
  --discover "path/to/images" \
  --batch-size auto
```

**AUTO mode will:**
1. Detect your platform
2. Select optimal backend
3. Choose optimal batch size
4. Cache calibration for future runs

---

## Performance Benchmarks

### Standard Tier (Transformers)

| Files | Windows | Mac | Linux |
|-------|---------|-----|-------|
| 1 | 12.2s | 12.2s | 12.2s |
| 10 | 17.0s | 17.0s | 17.0s |
| 20 | 17.4s | 17.4s | 17.4s |

**Speedup**: 14x (batch processing)

---

### Pro Tier (Transformers)

| Files | Windows | Mac | Linux |
|-------|---------|-----|-------|
| 1 | 17.5s | 17.5s | 17.5s |
| 10 | 37.6s | 37.6s | 37.6s |
| 20 | FAIL | FAIL | FAIL |

**Speedup**: 4.6x (VRAM limited to 10 files)

---

### Ultra Tier (Platform-Dependent)

| Files | Windows (Ollama) | Mac/Linux (vLLM) | Mac/Linux (Ollama) |
|-------|------------------|------------------|--------------------|
| 1 | 51.0s | ~6.0s | 51.0s |
| 10 | 510s | ~60s | 510s |
| 20 | 1020s | ~120s | 1020s |

**Windows**: Sequential processing (batch_size=1)
**Mac/Linux with vLLM**: 8.5x faster batch processing (batch_size=16)
**Mac/Linux with Ollama**: Same as Windows (sequential)

---

## Troubleshooting

### vLLM Not Available on Mac/Linux

**Symptoms:**
```
⚠ vLLM not available: package not installed
✓ Ollama available (vLLM not installed)
Optimal Backend: ollama
```

**Solution:**
```bash
# Install vLLM
pip install vllm

# For CUDA support (NVIDIA GPU)
pip install vllm[cuda]
```

---

### Windows: "vLLM not supported"

**Symptoms:**
```
ERROR: vLLM is not supported on Windows. Falling back to Ollama.
```

**Explanation:**
- vLLM requires Unix-like systems (Mac/Linux)
- Windows is not supported upstream

**Solutions:**
1. **Use Ollama** (recommended):
   - Stable and reliable
   - Sequential processing only
   - ~51s per image

2. **Use WSL2** (advanced):
   - Install Windows Subsystem for Linux 2
   - Install vLLM inside WSL2
   - Requires GPU passthrough setup

3. **Use Transformers** (fallback):
   - CPU-based processing
   - Slower than Ollama
   - Works on any platform

---

### Ollama Not Running

**Symptoms:**
```
ERROR: Ollama server is not running!
```

**Solution:**
```bash
# Start Ollama server
ollama serve

# In another terminal, test connection
ollama list
```

---

### Model Not Found

**Symptoms:**
```
ERROR: Model 'qwen3-vl:8b' not found
```

**Solution:**
```bash
# Pull the model
ollama pull qwen3-vl:8b

# Verify
ollama list
```

---

## FAQ

### Q: Why is Windows slower?

**A:** Windows uses Ollama with sequential processing because:
1. vLLM is not available on Windows (Linux/Mac only)
2. Ollama's Vision API doesn't benefit from batching
3. Text API supports batching (3x speedup), but Vision API shows 0.6x slowdown

**Solution:** For faster processing, use Mac or Linux with vLLM.

---

### Q: Can I use vLLM on Windows?

**A:** Not natively. vLLM requires Unix-like systems.

**Options:**
- **WSL2**: Install Linux subsystem and vLLM inside WSL2
- **Dual Boot**: Install Linux alongside Windows
- **Switch Platform**: Use Mac or Linux for production

---

### Q: Should I always use AUTO mode?

**A:** Yes, AUTO mode is recommended because:
- Automatically detects optimal backend
- Sets optimal batch size
- Handles platform differences
- Fallback to available backends

**When to use manual selection:**
- Testing different backends
- Debugging performance issues
- Specific deployment requirements

---

### Q: How do I switch between backends?

**Option 1: Change config.yaml**
```yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: vllm  # or 'ollama', 'transformers', 'auto'
```

**Option 2: Environment variable**
```bash
export VISION_BACKEND=vllm
python backend/pipeline/ingest_engine.py ...
```

**Option 3: Reset factory cache**
```python
from backend.vision.vision_factory import VisionAnalyzerFactory
VisionAnalyzerFactory.reset()  # Forces re-detection
```

---

### Q: What is the difference between vLLM and Ollama?

| Feature | vLLM | Ollama |
|---------|------|--------|
| **Platform** | Mac/Linux only | Windows/Mac/Linux |
| **Batch Processing** | Excellent (16+) | Poor (1 only) |
| **Performance** | 3-5x faster | Baseline |
| **Setup** | `pip install vllm` | Install app + pull models |
| **Memory** | PagedAttention | Standard |
| **Best For** | Production (Mac/Linux) | Development (all platforms) |

---

## Best Practices

### Development (Local Testing)

```yaml
ai_mode:
  override: standard  # Fast, lightweight model
  tiers:
    standard:
      vlm:
        backend: auto  # Use whatever is available
```

**Why:**
- Fast iteration (12s for 10 images)
- Low VRAM requirements (~4GB)
- Works on all platforms

---

### Production (Mac/Linux Server)

```yaml
ai_mode:
  override: ultra  # High quality
  tiers:
    ultra:
      vlm:
        backend: vllm  # Explicit vLLM for speed
        model: Qwen/Qwen3-VL-8B-Instruct
```

**Why:**
- Maximum quality (Qwen3-VL 8B)
- 8.5x faster with vLLM batch processing
- 16 images in ~60s

---

### Production (Windows Server)

```yaml
ai_mode:
  override: ultra
  tiers:
    ultra:
      vlm:
        backend: ollama  # Only option for Windows
        model: qwen3-vl:8b
```

**Why:**
- Stable and reliable
- No vLLM alternative on Windows
- Sequential processing only (~51s per image)

**Recommendation:** Consider migrating to Linux server for 8.5x speedup.

---

## Migration Guide

### From Ollama (Windows) to vLLM (Linux)

**Step 1: Prepare Linux Environment**
```bash
# Install Python and dependencies
sudo apt-get install python3-pip

# Install vLLM
pip install vllm
```

**Step 2: Update config.yaml**
```yaml
ai_mode:
  tiers:
    ultra:
      vlm:
        backend: vllm  # Changed from ollama
        model: Qwen/Qwen3-VL-8B-Instruct  # HuggingFace format
```

**Step 3: Test Performance**
```bash
# Run benchmark
python run_benchmark.py

# Expected: 8.5x speedup
```

---

## Summary

| Scenario | Platform | Backend | Batch Size | Performance |
|----------|----------|---------|------------|-------------|
| **Development** | Any | auto | auto | Good enough |
| **Production (Fast)** | Mac/Linux | vLLM | 16 | 8.5x faster |
| **Production (Windows)** | Windows | Ollama | 1 | Baseline |
| **Legacy** | Any | Transformers | varies | Fallback |

**Key Takeaway:**
- **Use AUTO mode** for automatic optimization
- **Mac/Linux + vLLM** for maximum speed (8.5x)
- **Windows + Ollama** for stability (no alternatives)

---

## Related Documents

- [Ollama Batch Processing Analysis](ollama_batch_processing_analysis.md) - Detailed benchmark results
- [V3.1 Release Notes](V3.1.md) - 3-Tier AI system documentation
- [Installation Guide](../INSTALLATION.md) - General setup instructions

---

**Version**: v3.1.1
**Last Updated**: 2026-02-09
**Author**: ImageParser Team
