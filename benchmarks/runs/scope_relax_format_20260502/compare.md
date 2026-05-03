# Search Evaluation Comparison

**Status**: `pass`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.53484 | 0.561475 | 0.026635 | `pass` |
| `triaxis` | `P@10` | 0.31 | 0.326667 | 0.016667 | `pass` |
| `triaxis` | `Recall@10` | 0.450774 | 0.477573 | 0.026799 | `pass` |
| `triaxis` | `MRR@10` | 0.659206 | 0.725556 | 0.06635 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |
