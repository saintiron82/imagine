# Search Evaluation Comparison

**Status**: `pass`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.577753 | 0.591037 | 0.013284 | `pass` |
| `triaxis` | `P@10` | 0.463333 | 0.456667 | -0.006666 | `pass` |
| `triaxis` | `Recall@10` | 0.297402 | 0.328129 | 0.030727 | `pass` |
| `triaxis` | `MRR@10` | 0.738889 | 0.762222 | 0.023333 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |
