# Scoped Gold v1 Broad Review

Purpose: build a wider human-reviewed Golden seed from existing v3 search result lists.

Inputs:

- QuerySet: `benchmarks/runs/scoped_weak_v3_clean_tags3_20260502/queryset.jsonl`
- Weak labels: `benchmarks/runs/scoped_weak_v3_clean_tags3_20260502/labels.jsonl`
- Combined runs: `combined_v3_run_results.jsonl`
- Source runs:
  - `benchmarks/runs/scoped_weak_v3_clean_tags3_20260502/run_results.jsonl`
  - `benchmarks/runs/current_on_v3_clean_20260502/run_results.jsonl`
  - `benchmarks/runs/scope_exact_segments_20260503/run_results.jsonl`
  - `benchmarks/runs/scope_exact_segments_v2_20260503/run_results.jsonl`
  - `benchmarks/runs/scope_exact_segments_v3_20260503/run_results.jsonl`
  - `benchmarks/runs/scope_hint_hard_small_20260502/run_results.jsonl`

Review files:

- `review_tasks_raw.csv`: broad raw candidate pool.
- `review_tasks_prefilled.csv`: broad pool with assisted notes.
- `review_tasks_diverse.csv`: active human review sheet, source/visual-diversified.
- `review_tasks_diverse.jsonl`: JSONL copy of the active human review sheet.

Sampling rule:

- 30 queries.
- 10 candidates per query.
- 300 total candidates.
- Visual near-duplicates are removed per query.
- Source folders are capped before relaxing to maintain the target count.

Relevance scale:

- `0`: unrelated or wrong scope/source.
- `1`: partially relevant, but misses an important condition.
- `2`: clearly relevant to the requested scope and visual/content terms.

Run review server:

```bash
python3 tools/serve_search_label_review.py \
  --csv benchmarks/reviews/scoped_gold_v1_broad_20260505/review_tasks_diverse.csv
```

Open:

```text
http://127.0.0.1:8765/benchmarks/reviews/scoped_gold_v1_seed_20260502/review_gallery.html
```

Finalize after human review:

```bash
python3 tools/finalize_search_label_review.py \
  --review benchmarks/reviews/scoped_gold_v1_broad_20260505/review_tasks_diverse.csv \
  --output-labels benchmarks/reviews/scoped_gold_v1_broad_20260505/labels.scoped_gold_v1_broad.jsonl \
  --label-version scoped_gold_v1_broad \
  --reviewer-id reviewer_name \
  --require-all
```
