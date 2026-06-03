# Spatial backfill — Stage 7 follow-up (2026-06-03)

## Why this stage

The Stage 6 decision said **STOP** on the basis that the spatial axis was a no-op at the v1 data scale (411 object rows over 112 files). The user then requested a precise re-measurement at higher coverage: backfill ~1000 files from the local `마키아벨리즘` folder (effectively all 634 not-yet-populated files) and rerun the A/B with a fresh queryset.

## Backfill run

| Metric | Value |
|---|---:|
| Candidate count | 634 (`--path-prefix /Users/saintiron/imageDB/마캬베리즈무/`) |
| Wall time | 16,002.4 s ≈ **4h 27m** |
| Avg per file | **25.24 s** (steady-state, no model reload) |
| **Parse errors** | **0** (100% local files → 100% parseable) |
| Object yield | 249 of 634 (39.3%) — lower than Stage 2/3's 73–80% |
| Relation yield | 26 of 634 (4.1%) |
| Depth-layer yield | 104 of 634 (16.4%) |

The 39.3% object yield on 마키아벨리즘 is notably below the ~75% seen on the prior smaller samples. Probable cause: this folder is dominated by `실내소품` (interior props) — 621 of 745 files. Many are isolated single-object reference cuts where the VLM cannot identify multiple discrete objects with locations.

## Database state delta

| Table | Before follow-up | After follow-up | Change |
|-------|---:|---:|---:|
| `file_objects` files | 112 | **361** | +249 |
| `file_objects` rows | 411 | **1,408** | +997 |
| `file_spatial_relations` files | 12 | **38** | +26 |
| `file_depth_layers` files | 44 | **148** | +104 |

The 마키아벨리즘 folder spatial-object coverage is now 360 of 745 captioned files = **48.3 %** (was 14.9 %).

## A/B measurement — `frozen_spatial_30_v2`

Regenerated `frozen_spatial_30_v2.json` from the larger pool. The new queryset has 30 queries, GT mean **14.6** (was 5.7 in v1), drawing from 182 (object, location) pairs with ≥2 supporting files (was 64 in v1).

| Metric | **v2 ON** | **v2 OFF** | Δ |
|---|---:|---:|---:|
| P@5 keyword | 0.400 | 0.393 | +0.007 |
| P@5 SLM-judge | **0.8933** | **0.8867** | **+0.007** |
| Fusion lift over best single axis | +27.7 % | +25.5 % | — |
| found / 30 | 30 | 30 | 0 |

### Gate verdict

Plan §Stage 5 gate: "spatial-on **must beat** spatial-off on SLM-judge by ≥+0.05p". Actual Δ at the larger data scale: **+0.007 p**. **FAIL again.** The spatial-axis code path remains a no-op even with 3.4 × more object rows.

## Cross-check — `frozen_30_v1` (Sprint 3 queryset) post-follow-up

| Run | P@5 keyword | P@5 SLM-judge | found / 30 |
|---|---:|---:|---:|
| Sprint 3 final | 0.380 | 0.673 | 25 |
| v1 P10 post-backfill (112 files) | 0.380 | 0.673 | 25 |
| **v2 P10 post-backfill (361 files)** | **0.367** | **0.6667** | **24** |

The Sprint 3 ceiling of 0.673 on general queries is **unchanged within noise** (0.673 → 0.667). The bigger backfill does not spill over to non-spatial query workloads.

## Interpretation

1. **The spatial axis code path is conclusively a no-op.** Confirmed at two data scales (411 and 1,408 object rows). Disabling `apply_spatial_intent_boost` via `IMAGINE_BENCH_DISABLE_SPATIAL=1` changes ranking by less than 0.01 P@5 SLM at either scale.
2. **FTS does the work for spatial queries.** `file_objects.spatial_text` is denormalized into the FTS5 index. A query like "오른쪽에 벽이 있는 이미지" naturally matches the row that has `"벽 right 오른쪽 ..."` in `spatial_text`. The spatial-axis ranking code adds nothing on top of that.
3. **Spatial queryset SLM-judge scales with data volume.** v1 → v2: 0.787 → 0.893. But this is queryset-internal: every query is built from a populated (object, location) pair, so the more populated rows there are, the more obvious FTS retrievals get. It is *not* evidence that the broader system improved.
4. **General workload (Sprint 3 ceiling) unchanged.** The added spatial data does not help non-spatial queries.

## Cost spent vs. payoff

- VLM wall time: 4h 27m on Apple Silicon (MLX, Qwen3.5-9B 4-bit)
- New data: +1,000 object rows, +30 relations, +240 depth layers
- Generalisable lift: **0 p** (within noise)
- Specific lift: spatial queryset SLM-judge 0.787 → 0.893 (queryset-internal)

## Final verdict (re-affirmed)

> **STOP.** The Stage 6 verdict holds at the larger data scale. Do not run further VLM backfill in pursuit of a spatial-axis lift. The next genuine lever (if needed) is to **redesign the spatial axis itself** so that `apply_spatial_intent_boost` actually shifts rankings, or to **expand the FTS spatial_text expansion** so that more spatial-position queries naturally match — neither requires a model run.

## What this experiment did add

- **Stronger negative evidence**: the no-op finding is now backed by 3.4 × more data.
- **A larger v2 queryset** (182 candidate pairs → 30 selected with mean 14.6 GT) that can serve as a stress test for any future spatial-axis design.
- **Calibration on PSD parse rate**: with local paths, parse is **100% reliable**. The 84% parse-failure rate from Stage 3 was entirely a WebDAV reachability issue, not a parser limitation. This unblocks future targeted local-folder backfills if a real spatial use case emerges.
- **Tooling**: `tools/run_spatial_backfill.py` now accepts `--path-prefix` for focused folder backfills.

## Cross-link

- `benchmarks/spatial_backfill/baseline.md`, `s2-sanity.md`, `s3-bench.md`, `s5-ab.md` — earlier stages
- `docs/state_report_spatial_backfill_2026-06-03.md` — Stage 6 scale decision (verdict re-affirmed by this follow-up)
- `benchmarks/querysets/frozen_spatial_30_v2.json` — new queryset
- `tools/run_spatial_backfill.py` — updated wrapper with `--path-prefix`
