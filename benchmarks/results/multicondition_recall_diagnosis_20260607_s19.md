# Multicondition Recall Diagnosis

- Source: `benchmarks/results/multicondition_pairs_20260605_s19_post_ce_evidence_sample100.json`
- Variant: `current`
- Queries: 24
- GT total: 60
- Found@10: 55
- Missed@10: 5
- Micro Recall@10: 0.916667
- Condition group issues: 18

## Miss Causes

- `object_evidence_present_but_not_top10`: 5

## Recommendations

- `validate_condition_group_pairing`
- `add_object_evidence_recall_guard`
- `run_top50_miss_trace`

## Rows

### 벽, 커튼가 함께 있는 이미지
- found_at10: 4/5
- condition_group_issues: 2
- miss `31639`: `object_evidence_present_but_not_top10` matched=벽,커튼

### 창문, 커튼가 함께 있는 이미지
- found_at10: 4/5
- condition_group_issues: 2
- miss `31346`: `object_evidence_present_but_not_top10` matched=창문,커튼

### 문, 벽가 함께 있는 이미지
- found_at10: 2/4
- condition_group_issues: 2
- miss `31639`: `object_evidence_present_but_not_top10` matched=문,벽
- miss `31654`: `object_evidence_present_but_not_top10` matched=문,벽

### 등, 커튼가 함께 있는 이미지
- found_at10: 3/3
- condition_group_issues: 2

### 바닥, 벽가 함께 있는 이미지
- found_at10: 2/3
- condition_group_issues: 2
- miss `31654`: `object_evidence_present_but_not_top10` matched=바닥,벽

### 등, 창문가 함께 있는 이미지
- found_at10: 2/2
- condition_group_issues: 1

### 수납장, 창문가 함께 있는 이미지
- found_at10: 2/2
- condition_group_issues: 2

### 벽, 책장가 함께 있는 이미지
- found_at10: 2/2
- condition_group_issues: 2

### 식물, 의자가 함께 있는 이미지
- found_at10: 2/2
- condition_group_issues: 2

### 조명, 커튼가 함께 있는 이미지
- found_at10: 2/2
- condition_group_issues: 1
