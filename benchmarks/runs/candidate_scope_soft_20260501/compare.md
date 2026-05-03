# Search Evaluation Comparison

**Status**: `pass`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.448008 | 0.445769 | -0.002239 | `pass` |
| `triaxis` | `P@10` | 0.263333 | 0.26 | -0.003333 | `pass` |
| `triaxis` | `Recall@10` | 0.29494 | 0.294557 | -0.000383 | `pass` |
| `triaxis` | `MRR@10` | 0.79537 | 0.795833 | 0.000463 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |
