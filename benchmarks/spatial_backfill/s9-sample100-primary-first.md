# Spatial sample100 primary-first rerun — Stage 9 (2026-06-03)

## What changed

This run strengthened the spatial object signal before reprocessing:

- extraction prompt now asks for exactly one `primary_location` first,
- secondary `locations` are only for objects that clearly span multiple cells,
- search scoring now separates primary, secondary, and text-only location hits,
- the benchmark/queryset is limited to the 100 reprocessed sample files.

## Reprocessing run

Command shape:

```bash
.venv/bin/python backend/pipeline/ingest_engine.py \
  --files '<100 sample paths>' \
  --no-skip
```

Result:

| Metric | Value |
|---|---:|
| Files | 100 |
| Parse errors | 0 |
| VLM MC | 100 / 100 |
| VV embedding | 100 / 100 |
| MV embedding | 100 / 100 |
| Total time | 3511.9s |
| Avg time | 35.12s / file |

## Sample coverage

Limited to `benchmarks/spatial_backfill/s9_sample100_files.json`:

| Table | Files | Rows |
|---|---:|---:|
| `file_objects` | 57 | 231 |
| `file_spatial_relations` | 8 | 10 |
| `file_depth_layers` | 22 | 52 |

Object location shape:

| Metric | Value |
|---|---:|
| Multi-location object rows | 179 / 231 |
| Avg locations per object row | 2.98 |
| Eligible `(object, primary_location)` pairs with >=2 files | 26 |

Stage 8 full-DB avg locations was 3.39, so the primary-first prompt did reduce
location breadth in this sample. It did not remove it: broad objects are still
common.

## Queryset

Generated:

```bash
.venv/bin/python tools/generate_spatial_queryset.py \
  --output benchmarks/querysets/frozen_spatial_30_s9_sample100.json \
  --count 30 \
  --min-files 2 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Output was 26 queries, not 30, because only 26 sample-local pairs had at least
two supporting files.

## Ablation

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s9_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260603_s9_sample100.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

| Variant | P@5 | P@10 |
|---|---:|---:|
| `current` | 0.2692 | 0.1885 |
| `no_spatial_axis` | 0.1538 | 0.1192 |
| `strict_primary` | 0.2615 | 0.1846 |

Comparisons:

| Comparison | Delta P@5 | Delta P@10 | Wins / Losses / Ties | Same top-5 |
|---|---:|---:|---:|---:|
| `current` vs `no_spatial_axis` | +0.1154 | +0.0693 | 11 / 1 / 14 | 1 / 26 |
| `current` vs `strict_primary` | +0.0077 | +0.0039 | 2 / 2 / 22 | 14 / 26 |

## Interpretation

This is the useful result:

> Strengthening primary-location extraction and grading primary/secondary/text
> matches made the production path behave almost like the strict-primary
> diagnostic, while still beating true no-axis.

So the earlier “no effect” conclusion was too broad. What failed was not the
entire spatial signal. The weak part was broad location matching and measuring
only the late boost switch.

The relation table is still too sparse to carry the feature. In this sample it
has only 10 rows over 8 files. The actual lift is still coming from
`file_objects.primary_location`, not object-to-object relations such as “castle
to the right of wall”.

Absolute P@5 is lower than Stage 8 full-DB current P@5 because this is a
sample-only queryset with just 26 weak-label queries and many queries have only
2 ground-truth files. Also, the decomposer still classifies several center
queries as `semantic`, which weakens explicit spatial intent.

## Verdict

For “처음에 정보를 어떻게 뽑는 게 가장 이득인가?” the answer is:

1. Extract one strong representative `primary_location` per visible object.
2. Keep secondary cells, but score them much lower.
3. Do not rely on relation rows yet; coverage is too low.
4. Next improvement should freeze or bypass query decomposition for spatial
   benchmarks, because decomposition drift still hides some spatial intent.
