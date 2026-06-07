# Spatial sample100 processing tightening — Stage 10 (2026-06-03)

## What changed

This run tested whether stronger processing-side extraction improves the same
100-file spatial sample.

Processing changes:

- spatial prompts now explicitly ask for searchable structural objects,
- objects may include `salience`,
- secondary `locations` are capped to at most 3 cells total,
- relation extraction is limited to high-confidence visible relations,
- SQLite normalization preserves valid `salience` and canonicalizes more
  structural object names.

## Reprocessing run

Same file set as Stage 9:

```bash
.venv/bin/python backend/pipeline/ingest_engine.py \
  --files '<100 sample paths from benchmarks/spatial_backfill/s9_sample100_files.json>' \
  --no-skip
```

Result:

| Metric | Stage 9 | Stage 10 |
|---|---:|---:|
| Files | 100 | 100 |
| Parse errors | 0 | 0 |
| VLM MC | 100 / 100 | 100 / 100 |
| VV embedding | 100 / 100 | 100 / 100 |
| MV embedding | 100 / 100 | 100 / 100 |
| Total time | 3511.9s | 2729.9s |
| Avg time | 35.12s / file | 27.30s / file |

## Sample coverage

Limited to the same 100 sample files.

| Metric | Stage 9 | Stage 10 | Change |
|---|---:|---:|---:|
| `file_objects` files | 57 | 41 | -16 |
| `file_objects` rows | 231 | 157 | -74 |
| `file_spatial_relations` files | 8 | 6 | -2 |
| `file_spatial_relations` rows | 10 | 6 | -4 |
| `file_depth_layers` files | 22 | 14 | -8 |
| `file_depth_layers` rows | 52 | 32 | -20 |
| Multi-location rows | 179 | 120 | -59 |
| Avg locations per object row | 2.98 | 2.34 | -0.64 |
| Max locations per object row | not capped | 3 | capped |
| Eligible `(object, primary_location)` pairs | 26 | 18 | -8 |
| Files with `salience` in structured metadata | 0 | 41 | +41 |

The tightening did reduce noisy broad locations, but it also removed too many
usable object labels from the sample.

## Queryset

Generated from Stage 10 sample-local labels:

```bash
.venv/bin/python tools/generate_spatial_queryset.py \
  --output benchmarks/querysets/frozen_spatial_30_s10_sample100.json \
  --count 30 \
  --min-files 2 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Only 18 queries were generated because only 18 sample-local
`(object, primary_location)` pairs had at least two supporting files.

## Ablation

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s10_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260603_s10_sample100.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

| Metric | Stage 9 current | Stage 10 current |
|---|---:|---:|
| Queries | 26 | 18 |
| P@5 | 0.2692 | 0.2222 |
| P@10 | 0.1885 | 0.1222 |
| Hit@5 | 22 / 26 | 13 / 18 |

Position-axis contribution:

| Comparison | Stage 9 | Stage 10 |
|---|---:|---:|
| Current P@5 | 0.2692 | 0.2222 |
| No-axis P@5 | 0.1538 | 0.1111 |
| Delta P@5 | +0.1154 | +0.1111 |
| Current P@10 | 0.1885 | 0.1222 |
| No-axis P@10 | 0.1192 | 0.0944 |
| Delta P@10 | +0.0693 | +0.0278 |
| Wins / Losses / Ties | 11 / 1 / 14 | 7 / 0 / 11 |
| Hit@5 current | 22 / 26 | 13 / 18 |
| Hit@5 no-axis | 13 / 26 | 8 / 18 |

Strict-primary comparison:

| Metric | Stage 9 | Stage 10 |
|---|---:|---:|
| Current P@5 | 0.2692 | 0.2222 |
| Strict-primary P@5 | 0.2615 | 0.2222 |
| Delta P@5 | +0.0077 | +0.0000 |
| Wins / Losses / Ties | 2 / 2 / 22 | 0 / 0 / 18 |

## Interpretation

The processing-side tightening did not improve the sample100 result.

It kept the useful spatial-axis effect: Stage 10 current still beats no-axis by
+0.1111 P@5. But the absolute score fell because fewer files and fewer repeated
labels were available for query generation and retrieval.

The strict-primary gap became exactly zero. That means the current search path
is now effectively behaving as primary-location search. The extra secondary
location information is not adding measurable lift in this sample.

## Verdict

Do not use this exact tighter prompt as the full-data reprocessing recipe.

Keep:

- primary-location-first normalization,
- `locations` cap at 3,
- `salience` preservation,
- structural object canonicalization in SQLite.

Relax before full reprocessing:

- object extraction should recover Stage 9-level coverage,
- relation extraction should not be so narrow that it drops below the already
  sparse Stage 9 relation table,
- structural objects should be included, but the prompt should not make the
  model omit ordinary objects that create repeated queryable labels.

The best next processing experiment is a hybrid S11 configuration:

1. keep the database-side cap and `salience`,
2. keep structural-object guidance,
3. relax object coverage wording back toward Stage 9,
4. allow up to 5 relations again, but require visible subject/object names.
