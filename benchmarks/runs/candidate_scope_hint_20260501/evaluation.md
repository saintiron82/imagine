# Search Evaluation Report

**Schema**: `search_evaluation_v1`
**Label queries**: 30
**Runs**: 1

## Overall

### triaxis / candidate_scope_hint_20260501

- Queries evaluated: 30
- Avg latency: 5815.433 ms

| Metric | Score |
|---|---:|
| `MRR@10` | 0.796111 |
| `MRR@3` | 0.777778 |
| `MRR@5` | 0.792778 |
| `MRR@50` | 0.799141 |
| `P@10` | 0.203333 |
| `P@3` | 0.488889 |
| `P@5` | 0.346667 |
| `P@50` | 0.054000 |
| `Recall@10` | 0.271256 |
| `Recall@3` | 0.217669 |
| `Recall@5` | 0.250885 |
| `Recall@50` | 0.331275 |
| `nDCG@10` | 0.394878 |
| `nDCG@3` | 0.560594 |
| `nDCG@5` | 0.471853 |
| `nDCG@50` | 0.373615 |

Query type breakdown:

| Type | Queries | nDCG@10 | P@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `scoped` | 30 | 0.394878 | 0.203333 | 0.331275 |
