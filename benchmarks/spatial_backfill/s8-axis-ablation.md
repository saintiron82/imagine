# Spatial axis ablation — Stage 8 (2026-06-03)

## Question

Stage 7 looked like "no measurable lift", but that A/B only disabled the final
`apply_spatial_intent_boost` step. It did **not** disable the spatial evidence
axis itself. This stage separates three possible causes:

1. weak extracted spatial data,
2. weak query/label construction,
3. search-layer matching/ranking not using the data sharply enough.

## Experiment

Tool:

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_v2.json \
  --output benchmarks/results/spatial_axis_ablation_20260603.json \
  --top-k 10
```

Variants:

| Variant | Meaning |
|---|---|
| `current` | Production search path on this branch. |
| `no_spatial_axis` | Monkeypatch `_spatial_evidence_search` to return `[]`; this is the true spatial-axis-off test. |
| `strict_primary` | Monkeypatch spatial search to require `file_objects.primary_location == query spatial_location`. This is a diagnostic variant, not a production proposal. |

## Result

| Comparison | P@5 | P@10 | Wins / Losses / Ties | Same top-5 |
|---|---:|---:|---:|---:|
| `current` | 0.4267 | 0.4500 | — | — |
| `no_spatial_axis` | 0.3400 | 0.3500 | current wins 11 / loses 4 / ties 15 | 2 / 30 |
| `strict_primary` | **0.7667** | **0.7700** | current wins 0 / loses 21 / ties 9 | 5 / 30 |

The true spatial-axis-off comparison changes the conclusion:

- Spatial evidence is **not** a no-op. Removing it drops P@5 by **0.0867** and P@10 by **0.10**.
- The previous ON/OFF result only proves that the final `apply_spatial_intent_boost` has almost no added value.
- `strict_primary` is much stronger than current, which points to search-layer matching dilution.

## Data diagnostics

| Metric | Value |
|---|---:|
| `file_objects` rows / files | 1,408 / 361 |
| Multi-location object rows | 1,155 / 1,408 |
| Avg locations per object row | 3.39 |
| `file_spatial_relations` rows / files | 47 / 38 |

The object data is usable, but most rows are broad. A wall can have
`primary_location='top'` while `locations` also includes `left`, `center`,
`right`, `top-left`, and `top-right`. That is useful for recall but poor for
ranking exact position queries.

## Interpretation

### Data extraction

The initial extracted data is not useless. There are enough object-location rows
for the spatial axis to improve current over no-axis. However, the VLM often
emits multi-location objects. This makes the data closer to "object occupies
these regions" than "object is primarily at this position".

The relation table is too small to carry relation queries yet: only 47 relation
rows over 38 files. Most measured lift comes from object-location evidence, not
object-to-object spatial relations.

### Queryset / labels

The v2 queryset is built from `(object, primary_location)` pairs, so it rewards
primary-location exact matches. `strict_primary` therefore partially fits the
weak-label construction. Its 0.7667 P@5 should not be read as real user quality
without a human/SLM label pass.

Still, the direction is meaningful: when the search layer is forced to use the
same exact spatial signal that generated the labels, ranking improves sharply.

### Search layer

The main current failure is in matching/ranking:

1. Production spatial search accepts broad `locations` and `spatial_text`
   matches almost as strongly as `primary_location`.
2. Many candidates get the same spatial score, so spatial ranking has weak
   separation.
3. The final `apply_spatial_intent_boost` adds little because the spatial axis
   has already participated in RRF/rerank; it is not the real control point.
4. Query decomposition has occasional semantic drift. One observed example:
   `중앙에 선화가 있는 이미지` was decomposed as `chrysanthemum flower in the center`.

## Verdict

The failure was not "bad data only" and not "spatial axis no-op".

The current best explanation is:

> Spatial extraction produced a broad but usable signal. The search layer then
> uses that signal too broadly, so exact position intent is diluted. The previous
> A/B also measured the wrong switch by disabling only a late boost.

## Next unit

Do not run more VLM backfill yet.

Next experiment should be search-layer only:

1. Add a real benchmark flag to disable `_spatial_evidence_search` itself.
2. Split spatial object scoring into:
   - `primary_location` exact match: strong score,
   - secondary `locations` match: weaker score,
   - `spatial_text` text hit only: weakest score.
3. Freeze decomposition for spatial queryset runs, or bypass the LLM decomposer
   by deriving intent from queryset fields.
4. Re-run `current`, `true_no_axis`, and `graded_location_score`.
5. Accept only if graded scoring beats current on the same frozen queryset and
   does not regress a small general queryset.
