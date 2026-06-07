# S15 Multi-Condition Pre-Trim Rerank

## Question

기존 검색이 이미 여러 검색축을 조합하고 있다면, 효과가 없던 이유가 무엇인지 확인했다.

핵심 가설은 기존 element verification이 `top_k` 이후에 적용되어 넓은 후보 풀에서 정답을 끌어올리지 못한다는 것이다.

## Code Change

- `backend/search/sqlite_search.py`
  - element verification을 final `top_k` trim 이전으로 이동했다.
  - `IMAGINE_BENCH_DISABLE_AND=1`로 해당 단계만 끄는 벤치 비교 스위치를 추가했다.
  - `함께`, `모두`, `보이는`, `와`, `가`, `together` 같은 관계어/조사 노이즈를 element 조건에서 제외했다.
- `tests/test_search_spatial_rerank.py`
  - 다중 조건 후보가 full-condition evidence를 가진 결과를 올리는지 확인했다.
  - 관계어/조사 노이즈가 element group에 들어가지 않는지 확인했다.

## Spatial Regression Check

Command:

```bash
.venv/bin/python tools/spatial_axis_ablation.py \
  --queryset benchmarks/querysets/frozen_spatial_30_s12_sample100.json \
  --output benchmarks/results/spatial_axis_ablation_20260605_s15_stopword_sample100_scoped.json \
  --top-k 10 \
  --file-id-json benchmarks/spatial_backfill/s9_sample100_files.json
```

Result:

| Variant | P@5 | P@10 | Hit@5 |
| --- | ---: | ---: | ---: |
| current | 0.5200 | 0.2733 | 30/30 |
| no_spatial_axis | 0.3533 | 0.2467 | 28/30 |
| strict_primary | 0.5200 | 0.2700 | 30/30 |

Comparison:

- current vs no_spatial_axis: `delta P@5 +0.1667`, wins 15, losses 0, ties 15
- current vs strict_primary: `delta P@5 0.0000`, wins 0, losses 0, ties 30

## Multi-Condition Pair Check

QuerySet:

- `benchmarks/querysets/frozen_multicondition_pairs_s15_sample100.json`
- 100 sample 안의 `file_objects` co-occurrence에서 `gt_count >= 2`인 상위 24개 물체쌍
- 예: `벽, 커튼가 함께 있는 이미지`, `창문, 커튼가 함께 있는 이미지`

Command:

```bash
IMAGINE_BENCH_DISABLE_AND=1  # only for comparison variant
```

Result artifact:

- `benchmarks/results/multicondition_pairs_20260605_s15_sample100.json`

| Variant | P@5 | P@10 | Hit@5 | Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| current | 0.2917 | 0.2375 | 22/24 | 24/24 |
| no_element_verification | 0.2750 | 0.2042 | 21/24 | 23/24 |

Comparison:

- current vs no_element_verification: `delta P@5 +0.0167`
- wins 2, losses 0, ties 22

## Interpretation

기존 검색은 이미 다중 검색축을 조합하고 있었다. 이번 변경은 새 축을 추가한 것이 아니라, 조건 검증을 더 이른 단계로 옮긴 것이다.

효과는 있다. 다만 크지는 않다. 위치 검색은 회귀 없이 기존 수준을 유지했고, 다중 조건 weak-label 쿼리에서는 P@5가 `0.2750 -> 0.2917`로 올랐다.

즉 현재 병목은 "조합을 안 해서"가 아니다. 병목은 다음 쪽이다.

- LLM decomposer가 조건 키워드를 가끔 이상하게 만든다.
- element verification은 아직 substring 기반이라 의미적으로 같은 물체를 강하게 묶지 못한다.
- 후보 풀 안에는 정답이 들어오지만, top5 정렬 품질이 아직 낮다.

다음 개선 단위는 처리 파이프라인보다 검색/리랭커 쪽이다. 특히 `candidate evidence matrix`를 만들고, object/text/spatial 조건별 만족 개수를 explicit feature로 리랭킹하는 쪽이 맞다.
