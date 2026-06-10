# Spatial sample100 structural split — Stage 12 (2026-06-04)

## Goal

Stage 11 showed that prompt wording alone did not recover object coverage.
Stage 12 tested a structural split:

- keep ordinary visible objects in `objects`,
- put architecture/room/large scene anchors in `structural_objects`,
- merge both fields into `file_objects` for search,
- keep primary-first locations, max 3 cells, and salience.

The acceptance signal was whether coverage and current P@5 recovered over
Stage 10/11 without losing the position-axis lift.

## Code change

Implemented:

- `backend/vision/schemas.py`: added `structural_objects`.
- `backend/vision/prompts.py`: separated ordinary objects from structural
  objects in the extraction prompt and concise JSON format.
- `backend/vision/repair.py`: repairs and fallback parsing now preserve
  `structural_objects`.
- `backend/db/sqlite_client.py`: normalizes `objects + structural_objects`
  into `file_objects`, with max 3 locations, salience preservation, and
  duplicate suppression.
- Tests added for schema, prompt, repair path, and DB normalization.

## Verification before reprocessing

```bash
.venv/bin/python -m pytest \
  tests/test_search_spatial_rerank.py \
  tests/test_vision_spatial_objects.py \
  tests/test_sqlite_spatial_objects.py \
  tests/test_spatial_axis_ablation.py \
  tests/test_spatial_intent_boost.py \
  tests/test_generate_spatial_queryset.py -q
```

Result: `37 passed, 2 warnings`.

## Reprocessing run

Same 100 files as Stage 9/10/11:

```bash
.venv/bin/python backend/pipeline/ingest_engine.py \
  --files '<100 sample paths from benchmarks/spatial_backfill/s9_sample100_files.json>' \
  --no-skip
```

Result:

| Metric | Value |
|---|---:|
| Files | 100 |
| Process return code | 0 |
| Parse phase | 100 / 100 |
| VLM MC | 100 / 100 |
| VV embedding | 100 / 100 |
| MV embedding | 100 / 100 |
| Total time | 3105.8s |
| Avg time | 31.06s / file |
| VLM backend | MLXVisionAdapter |
| VLM model | mlx-community/Qwen3.5-9B-MLX-4bit |

The pipeline did not switch VLM models per file. It used one MLX VLM phase,
then SigLIP/VV, then Qwen MV embedding.

The MV phase initially hit DNS/HuggingFace lookup warnings, then loaded
`Qwen/Qwen3-Embedding-0.6B` and completed 100 / 100.

The final ingest log reported 44 parse-errors while still completing all 100
files with return code 0. Treat this as a non-blocking parse diagnostic that
needs separate parser-level audit if it appears in full backfill.

## Sample coverage

Limited to the same 100 sample files.

| Metric | Stage 9 | Stage 10 | Stage 11 | Stage 12 |
|---|---:|---:|---:|---:|
| `file_objects` files | 57 | 41 | 40 | 76 |
| `file_objects` rows | 231 | 157 | 151 | 272 |
| `file_spatial_relations` files | 8 | 6 | 3 | 4 |
| `file_spatial_relations` rows | 10 | 6 | 4 | 4 |
| `file_depth_layers` files | 22 | 14 | 12 | 13 |
| `file_depth_layers` rows | 52 | 32 | 24 | 28 |
| Multi-location rows | 179 | 120 | 121 | 181 |
| Avg locations per object row | 2.98 | 2.34 | 2.46 | 2.08 |
| Max locations per object row | not capped | 3 | 3 | 3 |
| Eligible sample queries | 26 | 18 | 21 | 36 |
| Generated queryset size | 26 | 18 | 21 | 30 |
| Files with `salience` in structured metadata | 0 | 41 | 40 | 76 |
| Files with `structural_objects` field | 0 | 0 | 0 | 100 |

Stage 12 recovered and exceeded object coverage. It produced the largest
number of eligible sample-local position-object query pairs.

## Queryset

Generated:

```bash
.venv/bin/python tools/generate_spatial_queryset.py \
  --output benchmarks/querysets/frozen_spatial_30_s12_sample100.json \
  --count 30 \
  --min-files 2 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Output:

- pairs with at least 2 supporting files: 36
- generated queries: 30 / 30
- spatial-language queries: 30 / 30

## Ablation

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s12_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260604_s12_sample100.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

| Metric | Stage 9 | Stage 10 | Stage 11 | Stage 12 |
|---|---:|---:|---:|---:|
| Queries | 26 | 18 | 21 | 30 |
| Current P@5 | 0.2692 | 0.2222 | 0.2095 | 0.2333 |
| No-axis P@5 | 0.1538 | 0.1111 | 0.1143 | 0.1200 |
| Delta P@5 | +0.1154 | +0.1111 | +0.0952 | +0.1133 |
| Current P@10 | 0.1885 | 0.1222 | 0.1333 | 0.1500 |
| No-axis P@10 | 0.1192 | 0.0944 | 0.0952 | 0.0900 |
| Delta P@10 | +0.0693 | +0.0278 | +0.0381 | +0.0600 |
| Current Hit@5 | 22 / 26 | 13 / 18 | 15 / 21 | 22 / 30 |
| No-axis Hit@5 | 13 / 26 | 8 / 18 | 10 / 21 | 14 / 30 |
| Strict-primary P@5 | 0.2615 | 0.2222 | 0.2095 | 0.2267 |
| Strict-primary Hit@5 | 20 / 26 | 13 / 18 | 14 / 21 | 21 / 30 |
| Current minus strict P@5 | +0.0077 | +0.0000 | +0.0000 | +0.0066 |

## Interpretation

Stage 12 is a partial success.

Confirmed:

- The new extraction shape works better than Stage 10/11 for coverage.
- Current P@5 recovered from Stage 11 `0.2095` and Stage 10 `0.2222` to
  `0.2333`.
- Position information still matters: current beats no-axis by `+0.1133` P@5.
- Hit@5 is 22 / 30, so the system finds at least one correct result for about
  73% of these sample-local position queries.

Not confirmed:

- Stage 12 did not beat Stage 9 current P@5 `0.2692`.
- Current and strict-primary are still almost identical. Secondary location
  cells add almost no extra measurable value.
- Relations and depth did not recover; the main gain came from object coverage.

## Search-layer finding

During the run, the decomposer produced at least one wrong location keyword:

- query: `왼쪽 위에 텍스트 오버레이가 있는 이미지`
- decomposed `fts_keywords` included `오른쪽`

That is not an extraction-data problem. It is a query decomposition/search-layer
problem and can directly hurt top-5 ranking.

## Decision

Keep the Stage 12 structural split code. It improves the processing side over
Stage 10/11 and gives the best coverage so far.

Do not treat it as enough for full-corpus backfill by itself. The next unit
should be search-layer hardening on the fixed S12 sample:

1. deterministic Korean location parser before LLM decomposition,
2. remove contradictory location keywords from decomposer output,
3. force location/object constraints into the spatial axis before vector/FTS,
4. rerun the same S12 queryset and require P@5 to beat Stage 9 `0.2692`.

Artifacts:

- `benchmarks/querysets/frozen_spatial_30_s12_sample100.json`
- `benchmarks/results/spatial_axis_ablation_20260604_s12_sample100.json`
