# Spatial backfill — Stage 3 bench-sized (2026-06-02 → 2026-06-03)

## Run

- Wall time: 2118.3 s (~35 min)
- Candidates selected: 500
- Successful: **81** (16.2%)
- Parse-errors: **419** (83.8%)
- Avg per **successful** file: **26.1 s** ← matches Stage 2's 28.9 s
- Avg across all 500 attempts: 4.24 s (parse-failures are fast)

## Coverage after Stage 3

| Table | Files w/ data | New (this batch) |
|-------|---:|---:|
| file_objects | 112 / 17,726 | +59 |
| file_spatial_relations | 0 / 17,726 | 0 |
| file_depth_layers | 0 / 17,726 | 0 |

Yield on **processed** files: 59 / 81 = **73 %** (slightly below Stage 2's 80%).

## Parse-error nature

```
psd_tools.api.layers: "No bounding box could be extracted from the given layers."
```

This INFO-level message from the `psd_tools` library causes the upstream parse step to bail. Specific PSD files in this sample range have layer structures that the parser cannot extract a canvas-level bounding box from. This is a pre-existing issue surfaced by the spatial backfill — not caused by it.

Stage 2's 50-file sample was lucky (0 parse-errors). The wider sample exposed the real rate.

## Objects-per-file distribution (across all 112 files)

| Objects | # files |
|---:|---:|
| 1 | 12 |
| 2 | 7 |
| 3 | 27 |
| 4 | 29 |
| 5 | 28 |
| 6 | 9 |

Mean ≈ 3.9 obj/file. Total ~411 object rows across 112 files.

## Implications for full backfill

If the 83.8 % parse-failure rate is representative of the broader corpus, full backfill of 16,000 remaining files would yield:

- Successful processes: ~2,600
- New populated files: ~1,900 (at 73 % yield on success)
- Wall time: ~17,000 × 4.24 s = ~20 hours (parse failures are fast)
- **OR** if we filter to parseable files first: ~2,600 × 26 s = ~19 hours

Either path: a single overnight run can complete the entire backfill of *parseable* files.

The 83 % parse-failure rate is a separate problem (PSD parser limitation) and is **not** a P10 blocker — what gets populated is enough to evaluate the spatial axis.

## Decision

Proceed to Stage 4. 112 populated files yields ~411 (object × location) rows, ample for a 30-query spatial queryset (Stage 4 requires ≥30 distinct supported pairs).
