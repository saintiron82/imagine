# Spatial Backfill (P10) — Scale Decision

> Date: 2026-06-03
> Sprint: P10 Spatial Backfill
> Parent plan: `docs/superpowers/plans/2026-06-02-spatial-backfill-sprint.md`
> Detailed measurement: `benchmarks/spatial_backfill/s5-ab.md`

## TL;DR

> **Stop full backfill. Keep what we have. The spatial-axis code path is a no-op at any plausible data scale; the apparent quality lift on spatial queries comes entirely from FTS retrieving denormalized `spatial_text` tokens.**

## Sprint outcome

| | Start | End |
|---|---:|---:|
| `file_objects` files populated | 12 | 112 (+100) |
| `file_objects` rows | 23 | ~411 |
| `file_spatial_relations` | 0 | 0 |
| `file_depth_layers` | 0 | 0 |
| Sprint 3 ceiling (frozen_30_v1) | 0.673 P@5 SLM | 0.673 P@5 SLM (no change) |
| Spatial queryset baseline | n/a | 0.787 P@5 SLM |
| Spatial axis A/B Δ | n/a | **0 p (FAIL gate)** |

## What `s5-ab.md` proves

1. **Spatial axis toggle is null.** `IMAGINE_BENCH_DISABLE_SPATIAL=1` changes nothing — the S3.2 `apply_spatial_intent_boost` call has no rank-shifting effect on this queryset.
2. **The 0.787 lift is queryset-internal.** The frozen_spatial_30_v1 queries are derived from `file_objects` data. FTS naturally retrieves the same rows via the denormalized `spatial_text` index. The spatial code path is bypassed entirely.
3. **No spillover to other queries.** Re-running Sprint 3's `frozen_30_v1` queryset against the now-populated DB returns exactly 0.673 P@5 SLM — identical to Sprint 3's final number.

## Cost economics for full backfill

From Stage 3 observed numbers:

| Metric | Value |
|---|---:|
| Successful processes / 500 candidates | 81 (16.2 %) |
| Parse failures (psd_tools "No bounding box") | 419 (83.8 %) |
| Successful per-file wall time | 26.1 s |
| Parse-failure wall time | ~1 s (fast bail) |
| New populated files / 500 candidates | 59 |

Extrapolated to remaining 16,000 candidates:

| Path | Wall time | New populated files |
|---|---:|---:|
| Naive full | ~17 hours (16,000 × ~4 s avg incl. failures) | ~1,900 (at 12 % yield) |
| Parseable-only (if pre-filtered) | ~19 hours (~2,600 × 26 s) | ~1,900 |

Either path: roughly **20 hours** of overnight VLM work for ~1,900 new populated files. **Plus** a separate effort to fix the 84 % parse-failure rate before the work is worth doing on the broader corpus.

## ROI analysis

The lift this would provide:

1. **On spatial-position queries** — would scale roughly linearly with coverage. P@5 SLM might rise from 0.787 → ~0.85 at full coverage. But this metric is queryset-internally biased; the absolute lift is hard to translate to user experience.
2. **On all other queries** — **zero**. Sprint 3 ceiling holds. Backfill does not help folder + element queries, semantic queries, or visual-similarity queries.
3. **On axis-toggle behaviour** — **zero**. The spatial code path is a no-op regardless of data scale.

The lift is real **only** for spatial-positioning queries. The user has not yet told us this query type is a primary use case. Without that signal, building toward 1.9k more populated files is speculative.

## Hidden cost: parse-failure debt

The 84 % parse-failure rate on this random sample is a **pre-existing** issue, not introduced by P10. But it means:

- The image corpus has a large class of PSDs that `psd_tools` cannot extract layer bounding boxes from.
- These files presumably **already exist** in the DB without spatial data — and would have always failed during the original analysis pass too.
- Fixing this is a separate workstream: investigate the failing PSDs (sample 10–20), determine if they share a structural property (versioning, embedded smart objects, missing layer attrs), then update the parser.

Recommend treating this as a separate bug-track issue rather than blocking P10 completion.

## Decision

> **STOP** full backfill. Mark P10 complete at 112 populated files / 411 rows.

Rationale:

1. The spatial axis as wired does not pay rent — confirmed by Stage 5 toggle Δ = 0.
2. The data-side lift is queryset-internal and does not generalise to other workloads.
3. Sprint 3 ceiling 0.673 is unchanged by backfill.
4. The cost (20 h VLM + parse-failure debt) is real; the benefit on user-facing search is hard to defend without a stated spatial-query use case.

## What stays

- `tools/run_spatial_backfill.py` — wrapper that handles non-ASCII paths and watchdog gotcha.
- `tools/generate_spatial_queryset.py` — Korean spatial-intent queryset generator.
- `benchmarks/querysets/frozen_spatial_30_v1.json` — frozen 30-query spatial queryset.
- `benchmarks/spatial_backfill/baseline.md`, `s2-sanity.md`, `s3-bench.md`, `s5-ab.md` — full measurement trail.
- The 112 newly-populated `file_objects` rows in `imageparser.db` — small benefit on FTS-routed spatial queries.

## What does NOT happen now

- No full 16k-file backfill.
- No code change to the spatial axis (S3.2 boost remains toggleable; no API change).
- No investment in `psd_tools` parse-failure fix (separate workstream, low priority absent a user request).

## Conditions under which this decision would flip

Re-open the decision if **any** of the following becomes true:

1. A user use case appears that needs spatial-position queries on the broader corpus (not 112 files).
2. The `psd_tools` parse-failure rate is solved separately, dropping the per-file effective cost.
3. A future spatial-axis design lifts SLM-judge by ≥+0.05p with `IMAGINE_BENCH_DISABLE_SPATIAL` toggle on the same queryset — i.e. the axis becomes a real ranking signal, not a no-op.

## Cross-link

- `docs/state_report_2026-05-31.md` — pre-P10 system snapshot.
- `docs/imagine_operations_control_plane_2026-05-31.md` — P10 sits at the end of the roadmap precisely because it's the last analysis-side lever; this verdict removes that lever from the immediate roadmap.
- `docs/superpowers/plans/2026-06-02-spatial-backfill-sprint.md` — the 7-stage plan executed.
