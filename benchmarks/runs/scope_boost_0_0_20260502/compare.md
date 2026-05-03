# Search Evaluation Comparison

**Status**: `fail`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.53484 | 0.528647 | -0.006193 | `pass` |
| `triaxis` | `P@10` | 0.31 | 0.31 | 0.0 | `pass` |
| `triaxis` | `Recall@10` | 0.450774 | 0.450774 | 0.0 | `pass` |
| `triaxis` | `MRR@10` | 0.659206 | 0.643704 | -0.015502 | `fail` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |

## Failures

- triaxis: MRR@10 delta -0.015502 < -0.010000
