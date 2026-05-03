# Search Evaluation Comparison

**Status**: `fail`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.448008 | 0.394878 | -0.05313 | `fail` |
| `triaxis` | `P@10` | 0.263333 | 0.203333 | -0.06 | `fail` |
| `triaxis` | `Recall@10` | 0.29494 | 0.271256 | -0.023684 | `fail` |
| `triaxis` | `MRR@10` | 0.79537 | 0.796111 | 0.000741 | `pass` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |

## Failures

- triaxis: nDCG@10 delta -0.053130 < -0.010000
- triaxis: P@10 delta -0.060000 < -0.010000
- triaxis: Recall@10 delta -0.023684 < -0.010000
