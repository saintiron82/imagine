# Metadata Quality Profile

이 문서는 `metadata_quality_v1_20260510` 수동 리뷰 결과를 검색/검출 시스템에 연결하는 운영 기준이다.

## 의미

- 이 리뷰셋은 개별 이미지 수동 보정표가 아니라, AI 캡션/태그 신뢰도를 추정하는 샘플 골든이다.
- 현재 리뷰 완료 행은 186/500개이며, 캡션/태그 품질 신호를 검색 리랭커에 shadow 값으로 붙인다.
- 기본 설정의 `metadata_quality_weight`는 `0.0`이다. 즉 현재 순위는 바꾸지 않고 진단값만 노출한다.

## 산출물

- `benchmarks/reviews/metadata_quality_v1_20260510/metadata_quality_profile.json`
  - 전체/상태/태그별 신뢰도 프로파일
  - `caption_reliability`, `tag_reliability`, `metadata_reliability` 포함
- `benchmarks/reviews/metadata_quality_v1_20260510/metadata_quality_signals.jsonl`
  - 리뷰된 `item_id`별 강한 신호
  - 개별 리뷰 신호는 프로파일 추정보다 우선한다.
- `benchmarks/reviews/metadata_quality_v1_20260510/summary.md`
  - 사람이 보기 위한 현재 집계 요약

## 검색 연결 방식

검색 결과에는 다음 shadow 필드가 붙는다.

- `metadata_reliability_score`
- `caption_reliability`
- `tag_reliability`
- `metadata_quality_source`
- `metadata_quality_confidence`
- `metadata_quality_basis`
- `metadata_quality_adjustment`

현재 기본값에서는 `metadata_quality_adjustment`가 `0.0`이라서 순위 영향이 없다. 검증량이 늘고 benchmark에서 개선이 확인되면 `search.rerank.metadata_quality_weight`를 작게 올린다. 초기 운영 권장 범위는 `0.02`에서 `0.05`다.

## 재생성 명령

```bash
python3 tools/build_metadata_quality_profile.py
python3 tools/summarize_metadata_review.py \
  --csv benchmarks/reviews/metadata_quality_v1_20260510/metadata_review_sample.csv \
  --output-json benchmarks/reviews/metadata_quality_v1_20260510/summary.json \
  --output-md benchmarks/reviews/metadata_quality_v1_20260510/summary.md
```

## 주의

- 현재 샘플은 작품군 편향이 있으므로 전체 30만 장에 강한 보정으로 바로 쓰지 않는다.
- 태그별 보정은 `runtime_tag_min_reviewed` 기준을 만족하는 태그만 사용한다.
- profile 기반 추정은 약한 신호이고, `item_id` 단위 수동 리뷰 신호는 강한 신호다.
- 품질 개선을 주장하려면 `tools/run_search_benchmark.py` 기반 before/after 비교가 필요하다.
