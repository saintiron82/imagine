# Benchmarks

Performance benchmark results for the Imagine pipeline.

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── baselines/                         # Reference baselines (model decisions)
│   ├── YYYYMMDD_<description>.json    # Frozen baseline snapshots
│   └── ...
├── profiles/                          # Benchmark profiles for bench_*.py
├── runs/                              # Standard Search Evaluation V1 runs
│   └── search_YYYYMMDD_HHMMSS/
│       ├── run_results.jsonl
│       ├── query_events.jsonl
│       ├── evaluation.json
│       ├── evaluation.md
│       └── manifest.json
└── results/                           # Ad-hoc benchmark runs
    └── worker_sim_YYYYMMDD_HHMMSS.json
```

### baselines/

Model selection decisions and reference speeds. These are **frozen snapshots** — 
not overwritten, only new baselines are added when models change.

| File | Description |
|------|-------------|
| `20260401_qwen35_eval_10img.json` | Qwen3.5 migration eval: 4 model configs (pure inference), tok/s, thinking mode |
| `20260401_windows_ollama_eval.json` | Windows RTX 3060Ti Ollama eval: 4 models compared |
| `20260402_mc_4b_10img.json` | MC phase: Qwen3.5-**4B** MLX 4bit (actual + theoretical speed) |
| `20260402_mc_9b_10img.json` | MC phase: Qwen3.5-**9B** MLX 4bit (actual + theoretical speed) |
| `20260402_ab_4b_vs_9b_10img.json` | A/B comparison: 4B vs 9B MC phase (same 10 images) |
| `20260402_embed_vv_mv_10img.json` | VV (SigLIP2) + MV (Qwen3-Embed) — VLM-independent |
| `20260402_full_pipeline_5img.json` | Full V→VV→MV pipeline (early baseline, 5 images) |
| `20260402_mc_only_3img.json` | MC phase first valid MLX 4bit run (3 images) |

### results/

`bench_worker_sim.py` output. Automatically saved with timestamps. Use `--compare` to 
diff against baselines.

### runs/

`tools/run_search_benchmark.py` output. This is the standard objective search
benchmark path:

- input: fixed QuerySet JSONL and optional LabelSet JSONL
- output: Search Evaluation V1 `RunResult` JSONL
- optional: metric evaluation and baseline/candidate gate

Use this path for before/after search quality comparison. Keep `results/` for
older or ad-hoc benchmark scripts.

## Current Design Speed (2026-04-02, M5 32GB)

### MC (VLM) — Bottleneck Phase

| Model | Actual f/min | Theoretical f/min | per-file | Metal Peak |
|-------|:-----------:|:-----------------:|:--------:|:----------:|
| Qwen3.5-**4B** MLX 4bit | 13.4 | **14.8** | 4.5s / 4.1s | 2.9GB |
| Qwen3.5-**9B** MLX 4bit | 7.9 | **8.3** | 7.6s / 7.4s | 5.7GB |

- Actual = model load + inference + unload + GC
- Theoretical = pure inference only (model stays resident)

### VV / MV — VLM-Independent

| Phase | Model | Actual f/min | Theoretical f/min | Memory Peak |
|-------|-------|:-----------:|:-----------------:|:-----------:|
| VV | SigLIP2-NaFlex | 112.9 | 124.6 | MPS 2.2GB |
| MV | Qwen3-Embed 0.6B | 224.0 | 277.0 | MPS 1.1GB |

MC is the bottleneck. VV/MV are 15-35x faster — negligible in pipeline total.

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

# Standard search quality run (fixed QuerySet + LabelSet)
python tools/run_search_benchmark.py \
  --queries benchmarks/data/queries/queryset_v1.jsonl \
  --labels benchmarks/data/labels/labels_v1.jsonl \
  --engines vv,mv,fts,triaxis \
  --top-k 50

# Compare candidate against frozen search baseline
python tools/run_search_benchmark.py \
  --queries benchmarks/data/queries/queryset_v1.jsonl \
  --labels benchmarks/data/labels/labels_v1.jsonl \
  --baseline benchmarks/baselines/search_eval_triaxis.json \
  --engines triaxis \
  --top-k 50
```

## When to Run

- **Model change**: Before and after switching VLM/VV/MV models
- **Config tuning**: Batch size, prompt changes, quantization changes
- **Performance regression**: When actual worker throughput drops below design speed
- **New hardware**: First setup on a new machine
- **Search changes**: Run `tools/run_search_benchmark.py` before/after ranking,
  fusion, query decomposition, or scoring changes

## Interpreting Results

**Design speed** (this benchmark) vs **actual worker speed** (server throughput):

| Gap | Likely Cause |
|-----|-------------|
| < 20% | Normal — network, DB, prefetch overhead |
| 20-50% | Check: download bottleneck, DB contention, prefetch pool config |
| > 50% | Bug — wrong model loaded, GPU contention, memory pressure |

The factory bug discovered on 2026-04-02 (using fp16 instead of 4bit) caused a **10x** gap.

## Search Quality Rules

- Do not compare runs generated from different QuerySets or LabelSets.
- Do not overwrite frozen baselines; add a new baseline file with date/context.
- Weak labels are acceptable for smoke checks, but quality claims require
  human or adjudicated labels.
- A candidate run should produce at least the same evaluated query count as
  the baseline.
- Default gate metrics are `nDCG@10`, `P@10`, `Recall@10`, and `MRR@10`.
