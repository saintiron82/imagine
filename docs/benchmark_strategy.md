# Imagine Benchmark Strategy

> 이 문서는 `image_search_auto_benchmark_master_plan_2026-02-18.md`를 현실 프로젝트에 맞게 조정한 실행 전략이다.

## 1. 목표

검색 코드(RRF 가중치, 모델, 파이프라인) 변경 시 **품질 회귀를 자동 검출**하는 로컬 벤치마크 시스템.

## 1.1 핵심 설계 원칙: 독립 실행 + 점수 기록

모든 엔진/설정을 한번에 비교하는 것이 아니라, **각 엔진을 독립적으로 실행하고 점수를 기록**한다. 나중에 원하는 기록끼리 비교한다.

```bash
# 각 엔진/설정을 독립 실행 → 점수 기록
python benchmarks/run.py --engine triaxis --tag "v3.7.0_balanced"
python benchmarks/run.py --engine triaxis --tag "v3.7.0_visual_heavy"
python benchmarks/run.py --engine fts_only --tag "bm25_baseline"

# 기록된 점수끼리 비교
python benchmarks/run.py --compare "v3.7.0_balanced,v3.7.0_visual_heavy"

# 점수 이력 조회
python benchmarks/run.py --history triaxis
```

**장점**:
- 엔진 추가가 점진적 (CLIP 준비 안 됐어도 Triaxis부터 기록 가능)
- 설정 변경 전/후 비교 자유로움
- 시계열 추적으로 품질 추세 파악

## 2. 원본 계획 대비 변경사항

| 항목 | 원본 | 변경 | 사유 |
|------|------|------|------|
| 비교 대상 | 상용 1 + 오픈소스 1 + 자사 1 | Random + BM25 + CLIP + Jina + SigLIP2 + Triaxis (5~7개) | 상용 API에 자사 이미지 인덱싱 불가, 동일 코퍼스 비교 필요 |
| 라벨 생성 | 클릭/체류 weak label | LLM Judge + DB 메타데이터 + 수동 검수 하이브리드 | 행동 로그 수집 인프라 없음 |
| CI/CD | PR check + 야간 스케줄 + 배포 게이트 | Git pre-push hook + Makefile gate + macOS launchd | CI 인프라 없음 |
| 쿼리셋 소스 | 검색 로그 자동 추출 | DB 메타데이터 기반 자동 생성 + 수동 설계 | 검색 로그 없음 |
| 비용 | 외부 API 월 $300 | LLM Judge ~$5/회 + 오픈소스 무료 | 상용 API 제외로 비용 대폭 절감 |

## 3. 비교 대상 엔진 (동일 로컬 코퍼스)

| # | 엔진 | 역할 | 구현 |
|---|------|------|------|
| 1 | Random | 하한선 | DB에서 무작위 N개 |
| 2 | BM25-Only (FTS) | 키워드 baseline | 기존 `fts_search()` |
| 3 | SigLIP2 VV 단독 | 시각 벡터 ablation | RRF v=1.0, s=0, m=0 |
| 4 | OpenCLIP ViT-B/32 | 업계 표준 CLIP | `open_clip_torch` (151M, ~290MB) |
| 5 | Triaxis | 자사 전체 시스템 | `triaxis_search()` |
| 6 | Jina CLIP v2 (확장) | 한국어 네이티브 | `jinaai/jina-clip-v2` (865M, ~2-4GB) |
| 7 | OpenCLIP ViT-L/14 (확장) | SigLIP2 규모 비교 | `open_clip_torch` (428M, ~816MB) |

**기대 계층**: Random < BM25 < CLIP-B/32 < SigLIP2 VV < Triaxis

## 4. 정답셋(Label) 구축 - 3단계 하이브리드

### Stage 1: Weak Label (무료, 즉시)
- DB 메타데이터 규칙: `image_type`, `ai_tags`, `mc_caption` 매칭
- `label_source = "weak"`

### Stage 2: LLM Judge (~$5, 수 시간)
- Gemini 2.0 Flash Vision, 썸네일 전송
- 3회 반복 다수결
- `label_source = "llm"`

### Stage 3: 수동 검수 (2일)
- Micro 골드셋: 30쿼리 x 10결과 = 300쌍 수동 라벨
- Weak과 LLM 불일치 건만 추가 검수
- `label_source = "adjudicated"`

### 라벨셋 확장 로드맵

| 버전 | 쿼리 수 | 방식 | 용도 |
|------|--------|------|------|
| v0.1 | 30 | 수동 골드셋 | LLM judge 보정 |
| v0.2 | 100 | LLM + 수동 검수 | PR 스모크 벤치 |
| v1.0 | 300 (6,000쌍) | 하이브리드 | 야간 풀 벤치 |

### LLM Judge 비용 참고

| 모델 | 6,000쌍 비용 |
|------|-------------|
| Gemini 2.0 Flash (Vision) | ~$0.46 |
| Gemini 2.0 Flash x3 다수결 | ~$1.38 |
| GPT-4o (Vision) | ~$8.78 |

## 5. 측정 지표

### Primary
- **nDCG@10**: 상위 10개 결과의 graded relevance (0/1/2)

### Secondary
- **Recall@50**: 상위 50개에서 관련 이미지 재현율
- **MRR@10**: 첫 번째 관련 이미지의 역순위

### 성능
- **p95 Latency**: 95번째 백분위 응답 시간
- **error_rate**: 검색 실패율

### Gate 규칙 (기준선 대비)
- nDCG@10 하락 > 1.0%p → **차단**
- Recall@50 하락 > 2.0%p → **차단**
- p95 latency 증가 > 20% → **차단**
- error_rate 증가 > 0.5%p → **차단**

## 6. 자동화 (CI 없이)

```
[git push] → pre-push hook → 검색 파일 변경 감지
                               ├─ Yes → smoke bench (50쿼리, ~1분)
                               │         ├─ PASS → push
                               │         └─ FAIL → 차단
                               └─ No  → 스킵

[make build] → Makefile gate → full bench 통과 → electron:build
                               └─ 실패 → 차단

[03:00 daily] → macOS launchd → full bench (300쿼리)
                                 ├─ Markdown + JSON 리포트
                                 ├─ git commit
                                 └─ macOS 알림 / Slack
```

## 7. 디렉터리 구조

```
benchmarks/
  run.py                       # CLI (smoke/full/compare/gate/save-baseline)
  config/benchmark.yaml         # 설정
  data/
    queries/                    # 쿼리셋 JSONL
    labels/                     # 라벨셋 JSONL
  baselines/                    # 버전별 기준선 JSON
  reports/                      # 날짜별 Markdown + JSON
  lib/
    metrics.py                  # nDCG, Recall, MRR
    runner.py                   # 엔진 어댑터
    comparator.py               # 기준선 비교 + bootstrap
    reporter.py                 # 리포트 생성
```

## 8. 통계 검정

- 단위: query-level paired 비교
- 방법: bootstrap 10,000회
- 판정: nDCG@10 delta의 95% CI 하한 > 0이면 개선
- gate는 통계 유의 + 가드레일 동시 만족 필요

## 9. 데이터 스키마

### QuerySet (JSONL)
```json
{"query_id": "q001", "query_text": "판타지 전사 캐릭터", "query_type": "semantic", "locale": "ko-KR"}
```

### LabelSet (JSONL)
```json
{"query_id": "q001", "item_id": "123", "relevance": 2, "label_source": "adjudicated"}
```

### RunResult (JSONL)
```json
{"run_id": "run_001", "engine_id": "triaxis", "query_id": "q001", "rank": 1, "item_id": "123", "score": 0.156, "latency_ms": 234}
```
