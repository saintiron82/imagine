# Scoped Gold v1 Seed Review

Purpose: human-review the high-impact scoped queries affected by `U-I110`.

Inputs:

- QuerySet: `benchmarks/runs/scoped_weak_v3_clean_tags3_20260502/queryset.jsonl`
- Weak LabelSet reference: `benchmarks/runs/scoped_weak_v3_clean_tags3_20260502/labels.jsonl`
- Baseline run: `benchmarks/runs/current_on_v3_clean_20260502/run_results.jsonl`
- Candidate run: `benchmarks/runs/scope_hint_hard_small_20260502/run_results.jsonl`

Review files:

- `focus_query_ids.txt`: 5 scoped queries whose top-10 results changed after U-I110.
- `review_tasks.csv`: editable human review sheet.
- `review_tasks.jsonl`: same tasks in JSONL form.
- `review_tasks_prefilled.csv`: assisted review sheet with conservative prefilled `reviewer_relevance`.
- `review_tasks_prefilled.jsonl`: same prefilled tasks in JSONL form.

Relevance scale:

- `0`: unrelated to the query.
- `1`: partially relevant, but misses one important query condition.
- `2`: clearly relevant to the requested scope and visual/content terms.

After filling `reviewer_relevance`, convert the completed review into a gold LabelSet:

```bash
python3 tools/finalize_search_label_review.py \
  --review benchmarks/reviews/scoped_gold_v1_seed_20260502/review_tasks_prefilled.csv \
  --output-labels benchmarks/reviews/scoped_gold_v1_seed_20260502/labels.scoped_gold_v1.jsonl \
  --label-version scoped_gold_v1 \
  --reviewer-id reviewer_name
```

Use the prefilled sheet as a draft only. Check `review_notes`; `scope=substring` means
the folder name appeared only as a substring, not an exact path segment.

Then run the guardrail benchmark:

```bash
.venv/bin/python tools/run_search_benchmark.py \
  --queries benchmarks/runs/scoped_weak_v3_clean_tags3_20260502/queryset.jsonl \
  --labels benchmarks/reviews/scoped_gold_v1_seed_20260502/labels.scoped_gold_v1.jsonl \
  --engines triaxis \
  --top-k 50 \
  --k 3,5,10,50 \
  --run-id scoped_gold_v1_guardrail_20260502 \
  --output-dir benchmarks/runs/scoped_gold_v1_guardrail_20260502
```
