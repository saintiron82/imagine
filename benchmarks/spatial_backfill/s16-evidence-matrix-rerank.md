# S16 Evidence Matrix Rerank

## Question

기존 데이터와 검색축을 다시 만들지 않고, 이미 검색 결과 row에 실려 있는 근거를 더 잘 조합하면 성과가 나는지 확인했다.

이번 변경은 새 파이프라인이 아니다. 기존 `triaxis_search`의 VV/MV/FTS/spatial/metadata 결과를 유지하고, 후보별로 다음을 명시적으로 붙여 리랭크한다.

- 조건별 만족 여부: 예를 들어 `설산|snowy mountain`, `새|bird`, `날고|flying`
- 검색축별 근거 존재 여부: visual, text_vec, fts, spatial, metadata

## Code Change

- `backend/search/sqlite_search.py`
  - `apply_evidence_matrix_rerank()` 추가
  - 후보 row에 `evidence_score`, `evidence_matrix`를 부착
  - element verification 직후, final `top_k` trim 이전에 evidence matrix 리랭크 적용
  - 기본 boost는 `search.rerank.evidence_matrix_boost`, 미설정 시 `0.20`
- `tests/test_search_spatial_rerank.py`
  - metadata/spatial evidence가 있는 후보가 partial 후보보다 올라오는지 검증
  - `triaxis_search(..., return_diagnostic=True)` 결과에 pre-trim evidence matrix가 붙는지 검증

## Multi-Condition Pair Check

QuerySet:

- `benchmarks/querysets/frozen_multicondition_pairs_s15_sample100.json`
- 100개 샘플 안의 object co-occurrence 기반 24개 weak-label 질의

Result artifact:

- `benchmarks/results/multicondition_pairs_20260605_s16_evidence_matrix_sample100.json`

| Variant | P@5 | P@10 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| current | 0.3000 | 0.2375 | 22/24 | 24/24 |
| no_element_evidence | 0.2750 | 0.2042 | 21/24 | 23/24 |

Comparison:

- current vs no_element_evidence: `delta P@5 +0.0250`
- wins 3, losses 0, ties 21

S15 대비:

- S15 current P@5: `0.2917`
- S16 current P@5: `0.3000`
- 순증: `+0.0083`

## Spatial Regression Check

Command:

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s12_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260605_s16_evidence_matrix_sample100_scoped.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Result artifact:

- `benchmarks/results/spatial_axis_ablation_20260605_s16_evidence_matrix_sample100_scoped.json`

| Variant | P@5 | P@10 |
| --- | ---: | ---: |
| current | 0.5200 | 0.2733 |
| no_spatial_axis | 0.3533 | 0.2467 |
| strict_primary | 0.5200 | 0.2733 |

Comparison:

- current vs no_spatial_axis: `delta P@5 +0.1667`, wins 15, losses 0, ties 15
- current vs strict_primary: `delta P@5 0.0000`, wins 0, losses 0, ties 30

## Verification

```bash
.venv/bin/python -m pytest tests/test_search_spatial_rerank.py::test_evidence_matrix_rerank_uses_metadata_and_spatial_evidence -q
.venv/bin/python -m pytest tests/test_search_spatial_rerank.py::test_triaxis_search_attaches_evidence_matrix_before_trim -q
.venv/bin/python -m pytest tests/test_search_spatial_rerank.py tests/test_element_and_verification.py tests/test_spatial_axis_ablation.py -q
.venv/bin/python -m py_compile backend/search/sqlite_search.py tools/spatial_axis_ablation.py
jq empty benchmarks/results/multicondition_pairs_20260605_s16_evidence_matrix_sample100.json benchmarks/results/spatial_axis_ablation_20260605_s16_evidence_matrix_sample100_scoped.json
git diff --check
```

Observed:

- targeted evidence tests passed
- regression suite: 29 passed, 2 warnings
- py_compile passed
- JSON validity check passed
- git diff whitespace check passed

## Interpretation

효과는 있다. 하지만 크지는 않다.

이번 결과는 "검색축이 없어서 실패했다"가 아니라 "후보 안의 근거를 더 명시적으로 조합하면 top5 정렬이 조금 좋아진다"는 쪽이다. 다중 조건 검색은 `P@5 0.2750 -> 0.3000`까지 올라왔고, 공간축 검색은 `P@5 0.5200`을 유지했다.

다음 개선 단위는 실패 질의의 evidence matrix를 직접 감사하는 것이다. 특히 `병 + 창문`, `상자 + 창문`처럼 아직 top5가 약한 질의에서 정답 후보가 후보 풀에 있는데 리랭크가 못 올리는지, 아니면 row 자체에 필요한 label/evidence가 없는지 분리해야 한다.
