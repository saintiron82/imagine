# Spatial backfill — baseline (2026-06-02)

## Coverage before P10

| Table | Files w/ data | Coverage |
|-------|---:|---:|
| file_objects | 12 / 17,726 (16,470 with caption) | 0.07% |
| file_spatial_relations | 0 / 17,726 | 0.00% |
| file_depth_layers | 0 / 17,726 | 0.00% |

## Coverage after 1-file warm-up (file_id=28695)

| Table | Files w/ data | Coverage |
|-------|---:|---:|
| file_objects | 13 / 17,726 | 0.07% (+1 file, +5 object rows) |
| file_spatial_relations | 0 / 17,726 | 0.00% |
| file_depth_layers | 0 / 17,726 | 0.00% |

Object samples extracted for `agm03_190.psd`:
- wireframe figure (center)
- concrete pillar (center / left / right)
- pipe (top-left / top-right)
- stair (bottom-left)
- window (left)

Sample looks plausible — the analyzer is producing structured Korean+English names with per-location lists.

**Note**: VLM did **not** populate `file_spatial_relations` or `file_depth_layers` for this file. P10 gate is on `file_objects` only; relations/depth_layers handling is investigated separately if needed.

## VLM cost — 1-file end-to-end

| Phase | Time |
|-------|-----:|
| Parse + MC (VLM) | included in total |
| VV (SigLIP2 load + encode) | 24.7s |
| MV (Qwen3-Embedding 0.6B load + encode) | 39.3s |
| **Wall-time total (cold start)** | **108.9 s/file** |

Hardware: macOS, MPS backend, RSS peak ~1.4 GB.

**This is a cold-start number** — includes model loading time. Steady-state per-file cost during a batch should be substantially lower since the analyzer reuses loaded models.

## Cost projections (cold-start basis — conservative)

| Sample | Wall-time (cold-start basis) |
|--------|---:|
| 50 files (Stage 2) | ~90 min |
| 500 files (Stage 3) | ~15 hours |
| 5,000 files | ~150 hours |
| 16,000 files (full backfill) | ~480 hours (~20 days) |

**These numbers will halve or better in steady state** as models stay loaded across batch.

## Implication for sprint

- Stage 2 (50 samples) feasible in one session (~1.5 hr).
- Stage 3 (500 samples) requires overnight or split-day runs.
- Stage 7 full backfill needs a dedicated multi-day operation — not a session task.

## Tooling fix recorded

`backend/pipeline/ingest_engine.py` uses `start_parent_watchdog()` (`backend/utils/parent_watchdog.py`) which exits the process when stdin reaches EOF (Layer 2 stdin pipe watchdog). Background invocation via `Bash run_in_background=true` closes stdin → instant exit (0.2s). Workaround for backfill runs: pipe a never-EOF source into stdin, e.g. `< /dev/zero`. The wrapper at `tools/run_spatial_backfill.py` does not yet plumb this — callers must add the redirection themselves.
