# Search Evaluation Comparison

**Status**: `fail`
**Metrics**: nDCG@10, P@10, Recall@10, MRR@10
**Engines**: triaxis

## Metric Gate

| Engine | Metric | Baseline | Candidate | Delta | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `nDCG@10` | 0.577753 | 0.558685 | -0.019068 | `fail` |
| `triaxis` | `P@10` | 0.463333 | 0.436667 | -0.026666 | `fail` |
| `triaxis` | `Recall@10` | 0.297402 | 0.294795 | -0.002607 | `pass` |
| `triaxis` | `MRR@10` | 0.738889 | 0.728889 | -0.01 | `fail` |

## Run Checks

| Engine | Check | Baseline | Candidate | Required | Status |
|---|---|---:|---:|---:|---|
| `triaxis` | `query_count` | 30 | 30 | 30.0 | `pass` |

## Failures

- triaxis: nDCG@10 delta -0.019068 < -0.010000
- triaxis: P@10 delta -0.026666 < -0.010000
- triaxis: MRR@10 delta -0.010000 < -0.010000
