# Search Evaluation Comparison

**Status**: `pass`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.577753 | 0.577753 | 0.0 | `pass` |
| `triaxis` | `P@10` | 0.463333 | 0.463333 | 0.0 | `pass` |
| `triaxis` | `Recall@10` | 0.297402 | 0.297402 | 0.0 | `pass` |
| `triaxis` | `MRR@10` | 0.738889 | 0.738889 | 0.0 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |
