# Search Evaluation Report

**Schema**: `search_evaluation_v1`
**Label queries**: 30
**Runs**: 1

## Overall

### triaxis / scope_exact_segments_v2_20260503

- Queries evaluated: 30
- Avg latency: 9715.0 ms

| Metric | Score |
|---|---:|
| `MRR@10` | 0.762222 |
| `MRR@3` | 0.755556 |
| `MRR@5` | 0.762222 |
| `MRR@50` | 0.766443 |
| `P@10` | 0.456667 |
| `P@3` | 0.644444 |
| `P@5` | 0.566667 |
| `P@50` | 0.252667 |
| `Recall@10` | 0.328129 |
| `Recall@3` | 0.185610 |
| `Recall@5` | 0.237479 |
| `Recall@50` | 0.609602 |
| `nDCG@10` | 0.591037 |
| `nDCG@3` | 0.669064 |
| `nDCG@5` | 0.630572 |
| `nDCG@50` | 0.589446 |

Query type breakdown:

| Type | Queries | nDCG@10 | P@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `scoped` | 30 | 0.591037 | 0.456667 | 0.609602 |
