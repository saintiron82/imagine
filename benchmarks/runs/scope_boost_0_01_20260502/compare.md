# Search Evaluation Comparison

**Status**: `pass`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.53484 | 0.532655 | -0.002185 | `pass` |
| `triaxis` | `P@10` | 0.31 | 0.31 | 0.0 | `pass` |
| `triaxis` | `Recall@10` | 0.450774 | 0.450774 | 0.0 | `pass` |
| `triaxis` | `MRR@10` | 0.659206 | 0.651151 | -0.008055 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |
