# Search Evaluation V1

Imagine 검색 품질 평가는 QuerySet, LabelSet, RunResult 세 가지 JSONL 계약을 기준으로 한다.

## 목적

검색 개선은 정답셋 기반 metric으로만 판단한다. 단일 실행의 체감 결과나 일부 예시만으로 품질 개선을 선언하지 않는다.

## QuerySet

파일 예시: `benchmarks/data/queries/queryset_v1.jsonl`

```json
{"query_id":"q001","query_text":"창문과 책상이 있는 교실 이미지","query_type":"semantic","locale":"ko-KR","created_at":"2026-05-01T00:00:00+09:00"}
```

필수 필드:

- `query_id`: query 고유 ID
- `query_text`: 사용자 질의
- `query_type`: `exact`, `semantic`, `scoped`, `complex`, `ambiguous` 중 하나
- `locale`: 예: `ko-KR`
- `created_at`: ISO8601 timestamp

## LabelSet

파일 예시: `benchmarks/data/labels/labels_v1.jsonl`

```json
{"query_id":"q001","item_id":"123","relevance":2,"label_source":"human","label_version":"v1"}
```

필수 필드:

- `query_id`: QuerySet의 ID
- `item_id`: 검색 결과 item/file ID
- `relevance`: `0`, `1`, `2`
- `label_source`: `weak`, `human`, `adjudicated`
- `label_version`: 라벨셋 버전

Relevance 기준:

- `0`: 무관
- `1`: 부분 관련
- `2`: 명확히 관련

## RunResult

파일 예시: `benchmarks/runs/run_triaxis_20260501.jsonl`

```json
{"run_id":"20260501_001","engine_id":"triaxis","query_id":"q001","rank":1,"item_id":"123","score":0.91,"latency_ms":120,"error":null,"cost_usd":null}
```

필수 필드:

- `run_id`: 실행 ID
- `engine_id`: `triaxis`, `vv`, `mv`, `fts`, `structure` 등
- `query_id`: QuerySet의 ID
- `rank`: 1-based rank
- `item_id`: 검색 결과 item/file ID
- `score`: 엔진 점수
- `latency_ms`: query 처리 시간, 모르면 `null`
- `error`: 실패 사유, 성공이면 `null`
- `cost_usd`: 외부 API 비용, 없으면 `null`

## 기본 Metric

- `P@k`: top-k 중 relevance > 0 비율
- `Recall@k`: 전체 관련 문서 중 top-k에서 찾은 비율
- `MRR@k`: top-k 안 첫 관련 결과의 reciprocal rank
- `nDCG@k`: relevance 등급과 순위를 함께 반영한 ranking quality

기본 k:

- `5`
- `10`
- `50`

## 명령

고정 QuerySet을 실제 검색 엔진에 실행하고 평가까지 생성:

```bash
python3 tools/run_search_benchmark.py \
  --queries benchmarks/data/queries/queryset_v1.jsonl \
  --labels benchmarks/data/labels/labels_v1.jsonl \
  --engines vv,mv,fts,triaxis \
  --top-k 50
```

이미 생성된 RunResult를 평가:

```bash
python3 tools/evaluate_search_quality.py \
  --labels benchmarks/data/labels/labels_v1.jsonl \
  --run benchmarks/runs/run_triaxis_20260501.jsonl \
  --queries benchmarks/data/queries/queryset_v1.jsonl \
  --output-json benchmarks/reports/summary_20260501.json \
  --output-md benchmarks/reports/evaluation_report_20260501.md
```

Baseline 비교:

```bash
python3 tools/compare_search_evaluation.py \
  --baseline benchmarks/baselines/search_eval_triaxis.json \
  --candidate benchmarks/reports/summary_20260501.json \
  --engines triaxis \
  --metrics nDCG@10,P@10,Recall@10,MRR@10 \
  --min-delta -0.01 \
  --output-json benchmarks/reports/compare_20260501.json \
  --output-md benchmarks/reports/compare_20260501.md
```

Human review task 생성:

```bash
python3 tools/build_search_label_review.py \
  --queries benchmarks/data/queries/queryset_v1.jsonl \
  --run benchmarks/runs/run_triaxis_20260501.jsonl \
  --labels benchmarks/data/labels/weak_labels_v1.jsonl \
  --top-k 10 \
  --db-path imageparser.db \
  --output-jsonl benchmarks/reviews/review_tasks.jsonl \
  --output-csv benchmarks/reviews/review_tasks.csv
```

완료된 review task를 gold LabelSet으로 변환:

```bash
python3 tools/finalize_search_label_review.py \
  --review benchmarks/reviews/review_tasks.csv \
  --output-labels benchmarks/data/labels/scoped_gold_v1.jsonl \
  --label-version scoped_gold_v1 \
  --reviewer-id reviewer_name
```

## 판정 원칙

- 기준선 비교 전에는 단일 점수를 “개선”으로 표현하지 않는다.
- `query_type`별 breakdown 없이 전체 평균만 보고하지 않는다.
- gold set과 refresh set은 분리한다.
- baseline은 자동 덮어쓰지 않는다.
- weak label 기반 개선은 gold LabelSet 샘플로 spot-check한 뒤 다음 랭킹 변경에 사용한다.
- 기본 gate는 candidate metric이 baseline 대비 `-0.01` 이상이면 통과로 본다.
- 평가 query 수가 baseline보다 줄어든 실행은 회귀로 본다.
- latency gate는 명시적으로 `--max-latency-ratio`를 준 경우에만 적용한다.
