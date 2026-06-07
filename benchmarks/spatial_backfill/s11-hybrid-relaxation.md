# Spatial sample100 hybrid relaxation — Stage 11 (2026-06-04)

## Goal

Stage 10 reduced noisy broad locations, but it also reduced object coverage.
Stage 11 tested a hybrid prompt:

- keep `primary_location` first,
- keep max 3 `locations`,
- keep `salience`,
- keep structural-object guidance,
- relax object coverage wording so ordinary objects are not omitted,
- allow up to 5 visible relations again.

The acceptance gate was simple: recover coverage and improve over Stage 10
current P@5.

## Reprocessing run

Same 100 files as Stage 9 and Stage 10:

```bash
.venv/bin/python backend/pipeline/ingest_engine.py \
  --files '<100 sample paths from benchmarks/spatial_backfill/s9_sample100_files.json>' \
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
| Total time | 3420.2s |
| Avg time | 34.20s / file |
| VLM backend | MLXVisionAdapter |
| VLM model | mlx-community/Qwen3.5-9B-MLX-4bit |

The pipeline did not switch VLM models per file. It loaded one MLX VLM for the
vision phase, then unloaded it before VV/MV embedding phases.

## Sample coverage

Limited to the same 100 sample files.

| Metric | Stage 9 | Stage 10 | Stage 11 |
|---|---:|---:|---:|
| `file_objects` files | 57 | 41 | 40 |
| `file_objects` rows | 231 | 157 | 151 |
| `file_spatial_relations` files | 8 | 6 | 3 |
| `file_spatial_relations` rows | 10 | 6 | 4 |
| `file_depth_layers` files | 22 | 14 | 12 |
| `file_depth_layers` rows | 52 | 32 | 24 |
| Multi-location rows | 179 | 120 | 121 |
| Avg locations per object row | 2.98 | 2.34 | 2.46 |
| Max locations per object row | not capped | 3 | 3 |
| Eligible sample queries | 26 | 18 | 21 |
| Files with `salience` in structured metadata | 0 | 41 | 40 |

Stage 11 did not recover object coverage. It produced slightly more eligible
queries than Stage 10, but fewer object rows and fewer relation/depth rows.

## Queryset

Generated:

```bash
.venv/bin/python tools/generate_spatial_queryset.py \
  --output benchmarks/querysets/frozen_spatial_30_s11_sample100.json \
  --count 30 \
  --min-files 2 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Output was 21 queries, not 30, because only 21 sample-local pairs had enough
support.

## Ablation

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s11_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260604_s11_sample100.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

| Metric | Stage 9 | Stage 10 | Stage 11 |
|---|---:|---:|---:|
| Queries | 26 | 18 | 21 |
| Current P@5 | 0.2692 | 0.2222 | 0.2095 |
| No-axis P@5 | 0.1538 | 0.1111 | 0.1143 |
| Delta P@5 | +0.1154 | +0.1111 | +0.0952 |
| Current P@10 | 0.1885 | 0.1222 | 0.1333 |
| No-axis P@10 | 0.1192 | 0.0944 | 0.0952 |
| Delta P@10 | +0.0693 | +0.0278 | +0.0381 |
| Current Hit@5 | 22 / 26 | 13 / 18 | 15 / 21 |
| No-axis Hit@5 | 13 / 26 | 8 / 18 | 10 / 21 |
| Wins / Losses / Ties vs no-axis | 11 / 1 / 14 | 7 / 0 / 11 | 8 / 0 / 13 |
| Strict-primary P@5 | 0.2615 | 0.2222 | 0.2095 |
| Current minus strict P@5 | +0.0077 | +0.0000 | +0.0000 |

## Interpretation

Stage 11 failed the gate.

It did not recover object coverage, and current P@5 fell below Stage 10:

- Stage 10 current P@5: 0.2222
- Stage 11 current P@5: 0.2095

The position-axis signal still helps compared with no-axis, but the lift is
weaker than Stage 9 and Stage 10. Current and strict-primary are identical at
P@5, so secondary locations still add no measurable benefit.

The prompt wording alone is not enough to recover coverage. The likely problem
is not just "ask for ordinary objects too"; the model is still returning fewer
structured objects under the concise Qwen3.5 prompt shape.

## Decision

Do not use the Stage 11 prompt for full reprocessing.

The Stage 11 code patch was reverted after recording the benchmark. The S11
benchmark artifacts remain for comparison:

- `benchmarks/querysets/frozen_spatial_30_s11_sample100.json`
- `benchmarks/results/spatial_axis_ablation_20260604_s11_sample100.json`

The next useful processing-side unit should not be another wording-only prompt
tweak. It should test extraction shape:

1. keep DB normalization: primary first, max 3 locations, salience,
2. split structural object extraction into a separate optional field or pass,
3. keep ordinary object extraction close to the older Stage 9 prompt,
4. compare whether a two-list or two-pass extraction recovers object coverage.
