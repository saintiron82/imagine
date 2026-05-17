# Spatial Processing V1 Review

Purpose: label search results for relation/depth questions after spatial backfill.

Inputs:

- QuerySet: `benchmarks/querysets/spatial_processing_v1_20260517/queryset.jsonl`
- Target evidence: `file_spatial_relations`, `file_depth_layers`, `files_fts.spatial`

Relevance scale:

- `2`: visible relation/depth condition clearly matches the query.
- `1`: partially matches, or related object is visible but relation/depth is uncertain.
- `0`: does not match.

Operational note:

- Run the QuerySet after a scoped backfill batch creates relation/depth evidence.
- Store human or adjudicated labels as LabelSet JSONL with `query_id`, `item_id`, `relevance`, `label_source`, and `label_version`.
