# S19 Post Cross-Encoder Evidence Rerank

## Question

S17 감사에서 저득점 질의의 정답이 대부분 top10 안에 있지만 top5로 못 올라오는 것을 확인했다. 특히 `병 + 창문`, `상자 + 창문`은 정답이 top10 안에 있었고, structured object evidence를 강하게 주면 오프라인 top10 재정렬에서 P@5가 올라갔다.

원인은 evidence matrix가 final trim 전에 적용된 뒤, cross-encoder가 top10을 다시 섞는 구조였다. 따라서 조건 근거를 cross-encoder 이후에도 한 번 더 적용해야 했다.

## Code Change

- `backend/search/sqlite_search.py`
  - decomposer phrase noise 제거 강화
    - 예: `병과 창문`, `병, 창문가 함께 있는 이미지`, `window and bottle`을 condition group에서 제거
  - `evidence_matrix.conditions`에 structured object 근거 추가
    - `object_matched`
    - `object_missing`
    - `object_matches`
  - `spatial_objects`가 모든 조건을 만족하면 evidence score에 object full-match bonus 반영
  - cross-encoder 이후에도 evidence matrix를 재적용
- `tests/test_search_spatial_rerank.py`
  - phrase noise 제거 테스트 추가
  - structured object co-occurrence 우선 테스트 추가
  - cross-encoder 이후 evidence 재적용 테스트 추가

## Label Audit Finding

Artifact:

- `benchmarks/results/multicondition_pairs_20260605_s17_failure_evidence_audit.json`

저득점 13개 질의 감사 결과:

- 13/13 질의에서 weak-label 정답이 top10 안에 있었다.
- 11/13 질의에서 top5 안에 weak-label에는 없지만 evidence 조건을 모두 만족하는 후보가 있었다.
- top5 안의 full-evidence non-GT 후보는 총 26개였다.

따라서 기존 P@5는 순수 검색 실패가 아니라 label 누락과 rerank 실패가 섞인 값이다.

## Multi-Condition Pair Check

Result artifact:

- `benchmarks/results/multicondition_pairs_20260605_s19_post_ce_evidence_sample100.json`

| Variant | P@5 | P@10 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| current | 0.4083 | 0.2292 | 24/24 | 24/24 |
| no_element_evidence | 0.2750 | 0.2042 | 21/24 | 23/24 |

Comparison:

- current vs no_element_evidence: `delta P@5 +0.1333`
- wins 13, losses 1, ties 10

S16/S17 대비:

- S16 evidence matrix current P@5: `0.3000`
- S17 phrase-noise current P@5: `0.3000`
- S19 post-cross-encoder evidence current P@5: `0.4083`

Representative fixes:

- `병, 창문가 함께 있는 이미지`: `P@5 0.0 -> 0.4`
  - top5: `31569`, `31179`, `31185`, `31326`, `31577`
- `상자, 창문가 함께 있는 이미지`: `P@5 0.0 -> 0.4`
  - top5: `31569`, `31179`, `31378`, `31183`, `31144`

Known loss:

- `벽, 책장가 함께 있는 이미지`: current `0.2`, no_element_evidence `0.4`

## Spatial Regression Check

Result artifact:

- `benchmarks/results/spatial_axis_ablation_20260605_s19_post_ce_evidence_sample100_scoped.json`

| Variant | P@5 | P@10 |
| --- | ---: | ---: |
| current | 0.5200 | 0.2733 |
| no_spatial_axis | 0.3600 | 0.2367 |
| strict_primary | 0.5200 | 0.2700 |

Comparison:

- current vs no_spatial_axis: `delta P@5 +0.1600`, wins 14, losses 0, ties 16
- current vs strict_primary: `delta P@5 0.0000`, wins 0, losses 0, ties 30

## Interpretation

이번 변경은 실제 성과가 있다. 다중조건 weak-label P@5가 `0.3000 -> 0.4083`으로 올랐고, 공간 검색은 `0.5200`을 유지했다.

핵심 병목은 검색 후보 생성이 아니라 최종 리랭크 순서였다. 후보는 top10 안에 있었고, structured object evidence를 cross-encoder 이후에 다시 반영하자 top5로 올라왔다.

다만 weak-label 자체에도 누락이 있다. 예를 들어 object table에는 한 조건만 있지만 caption/tag에는 나머지 조건이 있는 후보가 실제 사용자 검색에서는 맞을 수 있다. 다음 단계는 weak-label을 `file_objects only`와 `evidence-expanded`로 분리해서 gold review 후보를 만드는 것이다.
