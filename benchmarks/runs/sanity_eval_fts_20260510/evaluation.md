# Search Evaluation Report

**Schema**: `search_evaluation_v1`
**Label queries**: 30
**Runs**: 1

## Overall

### fts / sanity_eval_fts_20260510

- Queries evaluated: 30
- Avg latency: 2.833 ms

| Metric | Score |
|---|---:|
| `MRR@10` | 0.076667 |
| `MRR@5` | 0.067778 |
| `MRR@50` | 0.076667 |
| `P@10` | 0.060000 |
| `P@5` | 0.060000 |
| `P@50` | 0.012000 |
| `Recall@10` | 0.107708 |
| `Recall@5` | 0.053333 |
| `Recall@50` | 0.107708 |
| `nDCG@10` | 0.084495 |
| `nDCG@5` | 0.060042 |
| `nDCG@50` | 0.074783 |

Query type breakdown:

| Type | Queries | nDCG@10 | P@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `scoped` | 30 | 0.084495 | 0.060000 | 0.107708 |
