# Spatial axis A/B + cross-check (Stage 5, 2026-06-03)

## Primary A/B — `frozen_spatial_30_v1` queryset

|  | spatial ON | spatial OFF | Δ |
|---|---:|---:|---:|
| P@3 keyword | 0.222 | 0.222 | 0 |
| P@5 keyword | 0.260 | 0.253 | +0.007 |
| P@10 keyword | 0.250 | 0.250 | 0 |
| **P@5 SLM-judge** | **0.7867** | **0.7867** | **0** |
| promoted items | 80 | 81 | −1 |
| found / 30 | 30 | 30 | 0 |

Per-axis (both ON and OFF, identical):

| Axis | P@3 | P@5 | P@10 |
|------|---:|---:|---:|
| VV | 0.011 | 0.007 | 0.013 |
| MV | 0.000 | 0.000 | 0.003 |
| FTS | 0.333 | 0.293 | 0.287 ★ |
| TRIAXIS | 0.222 | 0.260 | 0.250 |

Fusion lift is **negative** (−11 %) — Triaxis underperforms FTS-only on this queryset.

### Gate verdict

Plan §Stage 5 gate: "spatial-on **must beat** spatial-off on SLM-judge by ≥+0.05p". Actual Δ = **0 p**. **FAIL.**

### Why the spatial axis does nothing here

`IMAGINE_BENCH_DISABLE_SPATIAL=1` toggles the S3.2 `apply_spatial_intent_boost` call. The toggle's null result means:

1. The boost adds no rank-shifting weight at this scale of populated data, OR
2. The boost runs but the rows it could promote are already at the top via FTS.

The second is the dominant cause: `file_objects.spatial_text` is denormalized into the FTS index as Korean + English token expansion. When the user types "오른쪽에 벽이 있는 이미지", FTS already matches "오른쪽" + "벽" + "right" + "wall" in the same row. The boost has nothing left to promote.

## Cross-check — Sprint 3 queryset (`frozen_30_v1`) post-backfill

Re-ran the original Sprint 3 folder + element queryset against the now-populated database (112 files with spatial objects).

| | Sprint 3 final | Post-backfill |
|---|---:|---:|
| P@5 keyword | 0.380 | 0.380 |
| P@5 SLM-judge | 0.673 | 0.673 |
| found / 30 | 25 | 25 |

**No change.** The 112 newly-populated files do **not** lift quality on non-spatial queries. The Sprint 3 ceiling of 0.673 holds.

## Two readings of P@5 SLM = 0.787 on spatial queryset

`frozen_spatial_30_v1` SLM-judge = 0.787 (higher than Sprint 3's 0.673), but this is **not** a true quality breakthrough:

1. The queryset is generated from `file_objects` data. Each query targets an (object, location) pair that exists in the populated 112 files. FTS naturally retrieves these.
2. The SLM judge sees the populated files' MC captions describing the queried scene → marks them relevant.

So `0.787` reflects "FTS does a great job retrieving the data we just inserted" rather than "the spatial axis improves ranking". A more honest comparison would use a queryset with spatial-position intent **not** derived from `file_objects` — but no such queryset exists in the corpus.

## Conclusions

1. The spatial axis code path (S3.2 boost) is a **no-op at this data scale**. Disabling it changes nothing.
2. Spatial data population helps **only** queries whose terms are present in the populated `spatial_text` of `file_objects`. It does not lift quality on other query types.
3. The Sprint 3 P@5 SLM ceiling of **0.673** holds for the general-purpose query workload.
4. The 30-sample queryset is small; results could shift ±0.03 p with a larger queryset, but the **direction** (no axis-toggle effect, no cross-queryset spillover) is unlikely to flip.

These three statements drive Stage 6's go / partial / stop decision.
