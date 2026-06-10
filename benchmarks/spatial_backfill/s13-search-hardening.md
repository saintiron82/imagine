# Spatial sample100 search hardening — Stage 13 (2026-06-04)

## Goal

Stage 12 improved processing coverage but did not beat Stage 9 P@5.
During S12 ablation, the query decomposer produced a contradictory location
keyword:

- query: `왼쪽 위에 텍스트 오버레이가 있는 이미지`
- LLM `fts_keywords`: included `오른쪽`

Stage 13 keeps the S12 processed data unchanged and hardens the search layer.

## Code change

Implemented in `backend/search/sqlite_search.py`:

- extract spatial locations from the original query separately,
- if the original query contains a location, treat it as authoritative,
- remove decomposer-generated FTS keywords that contradict that original
  location,
- avoid joining separate Korean keywords before compact matching, because
  `오른쪽` + `위` can accidentally become `오른쪽위`.

Added regression tests in `tests/test_search_spatial_rerank.py`:

- contradictory decomposer location keywords do not enter spatial intent,
- contradictory location keywords are removed from spatial FTS keywords.

## RED / GREEN

RED:

```bash
.venv/bin/python -m pytest \
  tests/test_search_spatial_rerank.py::test_extract_spatial_intent_ignores_contradictory_decomposer_location_keywords \
  tests/test_search_spatial_rerank.py::test_sanitize_spatial_fts_keywords_removes_contradictory_locations -q
```

Result before implementation:

- `right` incorrectly appeared in locations for a `왼쪽 위` query,
- `_sanitize_spatial_fts_keywords` did not exist.

GREEN:

```bash
.venv/bin/python -m pytest \
  tests/test_search_spatial_rerank.py::test_extract_spatial_intent_ignores_contradictory_decomposer_location_keywords \
  tests/test_search_spatial_rerank.py::test_sanitize_spatial_fts_keywords_removes_contradictory_locations -q
```

Result: `2 passed`.

Focused search tests:

```bash
.venv/bin/python -m pytest tests/test_search_spatial_rerank.py -q
```

Result: `11 passed, 2 warnings`.

## Benchmark

No reprocessing. Reused S12 sample data and S12 frozen queryset:

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s12_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260604_s13_search_hardening_sample100.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

## Result

| Metric | S9 | S12 | S13 |
|---|---:|---:|---:|
| Queries | 26 | 30 | 30 |
| Current P@5 | 0.2692 | 0.2333 | 0.2733 |
| No-axis P@5 | 0.1538 | 0.1200 | 0.1200 |
| Delta P@5 | +0.1154 | +0.1133 | +0.1533 |
| Current P@10 | 0.1885 | 0.1500 | 0.1633 |
| No-axis P@10 | 0.1192 | 0.0900 | 0.0933 |
| Delta P@10 | +0.0693 | +0.0600 | +0.0700 |
| Current Hit@5 | 22 / 26 | 22 / 30 | 24 / 30 |
| No-axis Hit@5 | 13 / 26 | 14 / 30 | 14 / 30 |
| Strict-primary P@5 | 0.2615 | 0.2267 | 0.2600 |
| Strict-primary Hit@5 | 20 / 26 | 21 / 30 | 22 / 30 |
| Wins / Losses / Ties vs no-axis | 11 / 1 / 14 | 12 / 0 / 18 | 17 / 0 / 13 |

S13 passed the gate:

- It improved S12 current P@5 from `0.2333` to `0.2733`.
- It slightly beat the Stage 9 P@5 reference `0.2692`.
- It increased current Hit@5 from `22 / 30` to `24 / 30`.
- It widened the position-axis lift from `+0.1133` to `+0.1533`.

## Improved queries vs S12

| Query | S12 P@5 | S13 P@5 | Delta |
|---|---:|---:|---:|
| `왼쪽 위에 조명이 있는 이미지` | 0.2 | 0.6 | +0.4 |
| `왼쪽 위에 텍스트 오버레이가 있는 이미지` | 0.2 | 0.4 | +0.2 |
| `왼쪽 위에 천장이 있는 이미지` | 0.2 | 0.4 | +0.2 |
| `왼쪽 위에 창문이 있는 이미지` | 0.0 | 0.2 | +0.2 |
| `왼쪽 위에 병이 있는 이미지` | 0.0 | 0.2 | +0.2 |

All measured improvements were in top-left queries. This confirms that the
location-conflict bug was a real ranking problem, not just a diagnostic issue.

## Interpretation

The processing side and search side both mattered:

- S12 recovered structured object coverage.
- S13 made the search layer stop trusting contradictory decomposer locations.

The system is still not broadly "solved": P@5 is `0.2733`, so precision remains
low in absolute terms. But within this fixed 100-sample experiment, this is the
first variant that both has full 30-query coverage and beats the earlier S9
P@5 reference.

## Decision

Keep the S12 processing split and S13 search hardening.

Next useful unit before full-corpus backfill:

1. inspect remaining S13 zero-hit/low-hit queries,
2. decide whether failures are bad labels, missing extraction, or ranking,
3. only then run the full reprocess/backfill with the accepted S12+S13 code.

Artifacts:

- `benchmarks/querysets/frozen_spatial_30_s12_sample100.json`
- `benchmarks/results/spatial_axis_ablation_20260604_s13_search_hardening_sample100.json`
