# Search Evaluation Comparison

**Status**: `pass`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: fts, mv, triaxis, vv

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `fts` | `nDCG@10` | 0.134755 | 0.134755 | 0.0 | `pass` |
| `fts` | `P@10` | 0.073333 | 0.073333 | 0.0 | `pass` |
| `fts` | `Recall@10` | 0.10932 | 0.10932 | 0.0 | `pass` |
| `fts` | `MRR@10` | 0.275873 | 0.275873 | 0.0 | `pass` |
| `mv` | `nDCG@10` | 0.208415 | 0.208415 | 0.0 | `pass` |
| `mv` | `P@10` | 0.12 | 0.12 | 0.0 | `pass` |
| `mv` | `Recall@10` | 0.160752 | 0.160752 | 0.0 | `pass` |
| `mv` | `MRR@10` | 0.417778 | 0.417778 | 0.0 | `pass` |
| `triaxis` | `nDCG@10` | 0.448008 | 0.448008 | 0.0 | `pass` |
| `triaxis` | `P@10` | 0.263333 | 0.263333 | 0.0 | `pass` |
| `triaxis` | `Recall@10` | 0.29494 | 0.29494 | 0.0 | `pass` |
| `triaxis` | `MRR@10` | 0.79537 | 0.79537 | 0.0 | `pass` |
| `vv` | `nDCG@10` | 0.09549 | 0.09549 | 0.0 | `pass` |
| `vv` | `P@10` | 0.086667 | 0.086667 | 0.0 | `pass` |
| `vv` | `Recall@10` | 0.054917 | 0.054917 | 0.0 | `pass` |
| `vv` | `MRR@10` | 0.168413 | 0.168413 | 0.0 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `fts` | `query_count` | 30 | 30 | 30.0 | `pass` |
| `mv` | `query_count` | 30 | 30 | 30.0 | `pass` |
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |
| `vv` | `query_count` | 30 | 30 | 30.0 | `pass` |
