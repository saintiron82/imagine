# Spatial sample100 scoped evaluation fix — Stage 14 (2026-06-05)

## Goal

The S13 benchmark used a sample-local queryset and sample-local `gt_ids`, but
the search runner queried the full database. This meant valid results outside
the 100-file sample were counted as false positives.

Stage 14 fixes the evaluation population:

- queryset ground truth: 100-file sample,
- search target: same 100-file sample,
- diagnostics: same 100-file sample.

## Code change

`tools/spatial_axis_ablation.py` now passes `file_ids` into
`SqliteVectorSearch.triaxis_search(...)` when `--file-id-json` is provided.

This was already supported by the search layer. The benchmark runner simply was
not forwarding the filter.

Added a regression test:

```bash
.venv/bin/python -m pytest \
  tests/test_spatial_axis_ablation.py::test_run_variant_limits_search_to_sample_file_ids -q
```

RED: `run_variant()` did not accept `file_ids`.

GREEN: `run_variant()` forwards `{1, 2}` into `triaxis_search(..., file_ids={1, 2})`.

## Corrected benchmark

Command:

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s12_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260605_s13_sample100_scoped.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Output confirms the search filter:

```text
file_id_filter_count: 100
```

## Result

| Metric | S13 full-DB search | S14 sample-scoped search |
|---|---:|---:|
| Search target | full DB | same 100 sample files |
| Queries | 30 | 30 |
| Current P@5 | 0.2733 | 0.5200 |
| No-axis P@5 | 0.1200 | 0.3467 |
| Delta P@5 | +0.1533 | +0.1733 |
| Current P@10 | 0.1633 | 0.2733 |
| No-axis P@10 | 0.0933 | 0.2233 |
| Delta P@10 | +0.0700 | +0.0500 |
| Current Hit@5 | 24 / 30 | 30 / 30 |
| Strict-primary P@5 | 0.2600 | 0.5267 |
| Current minus strict P@5 | +0.0133 | -0.0067 |

## Interpretation

The earlier low S13 P@5 was partly an evaluation artifact.

When the benchmark searches the same 100 files used to generate the ground
truth, every query returns at least one ground-truth hit in the top 5:

- current Hit@5: `30 / 30`
- current P@5: `0.5200`

This answers the question:

> Did we ask for positions that had no data?

No. The questions had sample-local data, and when the search target is limited
to the same sample, the system retrieves ground-truth results for every query.

There are still two real findings:

1. Position axis still helps: current beats no-axis by `+0.1733` P@5.
2. Secondary broad locations still add little: strict-primary is slightly better
   than current on this scoped run (`0.5267` vs `0.5200`).

## Decision

Use the scoped S14 number when judging the 100-sample experiment.

For full-corpus evaluation, either:

1. search the same sample used by the queryset, or
2. build ground truth from the full corpus.

Do not mix sample-local labels with full-DB search results.

Artifacts:

- `benchmarks/results/spatial_axis_ablation_20260605_s13_sample100_scoped.json`
