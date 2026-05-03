# Search Evaluation Report

**Schema**: `search_evaluation_v1`
**Label queries**: 30
**Runs**: 1

## Overall

### triaxis / scope_boost_0_0_20260502

- Queries evaluated: 30
- Avg latency: 8584.103 ms

| Metric | Score |
|---|---:|
| `MRR@10` | 0.643704 |
| `MRR@3` | 0.627778 |
| `MRR@5` | 0.627778 |
| `MRR@50` | 0.650684 |
| `P@10` | 0.310000 |
| `P@3` | 0.488889 |
| `P@5` | 0.406667 |
| `P@50` | 0.145333 |
| `Recall@10` | 0.450774 |
| `Recall@3` | 0.276774 |
| `Recall@5` | 0.343659 |
| `Recall@50` | 0.747075 |
| `nDCG@10` | 0.528647 |
| `nDCG@3` | 0.531707 |
| `nDCG@5` | 0.515484 |
| `nDCG@50` | 0.605141 |

Query type breakdown:

| Type | Queries | nDCG@10 | P@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `scoped` | 30 | 0.528647 | 0.310000 | 0.747075 |
