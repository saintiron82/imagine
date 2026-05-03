# Search Evaluation Report

**Schema**: `search_evaluation_v1`
**Label queries**: 30
**Runs**: 1

## Overall

### triaxis / scope_boost_0_025_20260502

- Queries evaluated: 30
- Avg latency: 1171.793 ms

| Metric | Score |
|---|---:|
| `MRR@10` | 0.659206 |
| `MRR@3` | 0.644444 |
| `MRR@5` | 0.651111 |
| `MRR@50` | 0.666975 |
| `P@10` | 0.310000 |
| `P@3` | 0.500000 |
| `P@5` | 0.420000 |
| `P@50` | 0.148000 |
| `Recall@10` | 0.450774 |
| `Recall@3` | 0.287885 |
| `Recall@5` | 0.371437 |
| `Recall@50` | 0.750105 |
| `nDCG@10` | 0.534840 |
| `nDCG@3` | 0.541576 |
| `nDCG@5` | 0.532692 |
| `nDCG@50` | 0.615440 |

Query type breakdown:

| Type | Queries | nDCG@10 | P@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `scoped` | 30 | 0.534840 | 0.310000 | 0.750105 |
