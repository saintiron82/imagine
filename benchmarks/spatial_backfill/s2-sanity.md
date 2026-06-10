# Spatial backfill — Stage 2 sanity (2026-06-02)

## Run

- Started: ~18:12 KST (2026-06-02)
- Finished: ~18:36 KST
- Wall time: 1444.4 s (24 min)
- Files processed: 50 / 50 (0 parse errors)
- **Avg per file (steady state): 28.92 s** ← down from 108.9 s cold-start

## Coverage after

| Table | Files w/ data | New (this batch) |
|-------|---:|---:|
| file_objects | 53 / 17,726 | +40 |
| file_spatial_relations | 0 / 17,726 | 0 |
| file_depth_layers | 0 / 17,726 | 0 |

10 of 50 batch files yielded **0 object rows** — likely files where the VLM stage-2 output was empty / unparseable for that schema. Investigated separately if it becomes a blocker.

## Quality gate

**PASS** — 40/50 = 80% yield meets the gate (≥80%).

## Objects-per-file distribution (across all 53 files)

| Objects | # files |
|---:|---:|
| 1 | 6 |
| 2 | 6 |
| 3 | 9 |
| 4 | 15 |
| 5 | 13 |
| 6 | 4 |

Mean ≈ 3.8 objects/file. Total rows: 194 across 53 files.

## 5 random sample inspections

```
-- file_id=28725 --
  pillar         기둥       left
  staircase      계단       center
  ceiling        천장       top
  floor          바닥       bottom
  window         창문       right

-- file_id=28702 --
  anime character    애니메이션 캐릭터  center
  ruler              자                 right
  grid line          격자선             top-left
  pencil mark        연필 자국           center

-- file_id=28709 --
  character      캐릭터     center
  lamp           등         top-right
  doorframe      문틀       left
  wall           벽         left
  overlay        오버레이   center

-- file_id=28718 --
  metallic beam        금속 빔        top-left
  character silhouette 캐릭터 실루엣  bottom-right

-- file_id=28699 --
  shadow   그림자   right
  floor    바닥     center
  sword    검       right
  tile     타일     center
```

Judgment: all 5 samples look plausible — Korean+English names paired correctly, locations diverse, no obvious hallucinations.

## Implications

- Steady-state ~30 s/file is 4× faster than cold-start projection. New projections:
  - 500 files (Stage 3): **~4 hours**
  - 16,000 files (full backfill): **~133 hours** (~5.5 days)
- `file_spatial_relations` and `file_depth_layers` remain 0 across all 53 files. VLM analyzer is not emitting these for this image-DB. Whether that's a VLM limitation or a schema-extraction gap needs investigation, but it does not block P10 — the spatial axis search uses `file_objects.primary_location` primarily.

## Decision

Proceed to Stage 3 (500-sample) in background. Expected ~4 hours.
