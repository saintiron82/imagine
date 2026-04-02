# Benchmarks

Performance benchmark results for the Imagine pipeline.

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── baselines/                         # Reference baselines (model decisions)
│   ├── YYYYMMDD_<description>.json    # Frozen baseline snapshots
│   └── ...
└── results/                           # Ad-hoc benchmark runs
    └── worker_sim_YYYYMMDD_HHMMSS.json
```

### baselines/

Model selection decisions and reference speeds. These are **frozen snapshots** — 
not overwritten, only new baselines are added when models change.

| File | Description |
|------|-------------|
| `20260401_qwen35_eval_10img.json` | Qwen3.5 migration eval: 4 model configs (pure inference), tok/s, thinking mode, memory stability |
| `20260401_windows_ollama_eval.json` | Windows RTX 3060Ti Ollama eval: 4 models compared, qwen3.5:4b selected |
| `20260402_mc_only_3img.json` | Worker sim: MC phase only, 3 images (first valid MLX 4bit run) |
| `20260402_full_pipeline_5img.json` | Worker sim: Full V→VV→MV pipeline, 5 images — **current design speed baseline** |

### results/

`bench_worker_sim.py` output. Automatically saved with timestamps. Use `--compare` to 
diff against baselines.

## Current Design Speed (2026-04-02, M5 32GB)

| Phase | Model | files/min | per-file | Memory Peak |
|-------|-------|:---------:|:--------:|:-----------:|
| MC (VLM) | Qwen3.5-9B MLX 4bit | 7.8 | 7.7s | Metal 5.7GB |
| VV (SigLIP2) | SigLIP2-NaFlex | 81.1 | 0.74s | MPS 2.2GB |
| MV (Embed) | Qwen3-Embed 0.6B | 119.3 | 0.5s | MPS 1.1GB |
| **Full Pipeline** | V→VV→MV | **7.1** | **8.4s** | — |

MC is the bottleneck. VV/MV are negligible.

## Running Benchmarks

```bash
# Full benchmark (10 images, all phases + pipeline)
python tools/bench_worker_sim.py

# MC phase only (quick check)
python tools/bench_worker_sim.py --phases mc --no-full

# Compare with baseline
python tools/bench_worker_sim.py --compare benchmarks/baselines/20260402_full_pipeline_5img.json

# Custom image count
python tools/bench_worker_sim.py --count 20 --output benchmarks/results/my_test.json
```

## When to Run

- **Model change**: Before and after switching VLM/VV/MV models
- **Config tuning**: Batch size, prompt changes, quantization changes
- **Performance regression**: When actual worker throughput drops below design speed
- **New hardware**: First setup on a new machine

## Interpreting Results

**Design speed** (this benchmark) vs **actual worker speed** (server throughput):

| Gap | Likely Cause |
|-----|-------------|
| < 20% | Normal — network, DB, prefetch overhead |
| 20-50% | Check: download bottleneck, DB contention, prefetch pool config |
| > 50% | Bug — wrong model loaded, GPU contention, memory pressure |

The factory bug discovered on 2026-04-02 (using fp16 instead of 4bit) caused a **10x** gap.
