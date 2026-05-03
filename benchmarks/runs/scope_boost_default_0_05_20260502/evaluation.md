# Search Evaluation Report

**Schema**: `search_evaluation_v1`
**Label queries**: 30
**Runs**: 1

## Overall

### triaxis / scope_boost_default_0_05_20260502

- Queries evaluated: 30
- Avg latency: 5028.517 ms

| Metric | Score |
|---|---:|
| `MRR@10` | 0.692222 |
| `MRR@3` | 0.666667 |
| `MRR@5` | 0.673333 |
| `MRR@50` | 0.692222 |
| `P@10` | 0.323333 |
| `P@3` | 0.500000 |
| `P@5` | 0.426667 |
| `P@50` | 0.150000 |
| `Recall@10` | 0.473869 |
| `Recall@3` | 0.287885 |
| `Recall@5` | 0.388104 |
| `Recall@50` | 0.762486 |
| `nDCG@10` | 0.553640 |
| `nDCG@3` | 0.549397 |
| `nDCG@5` | 0.547715 |
| `nDCG@50` | 0.628841 |

Query type breakdown:

| Type | Queries | nDCG@10 | P@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `scoped` | 30 | 0.553640 | 0.323333 | 0.762486 |
