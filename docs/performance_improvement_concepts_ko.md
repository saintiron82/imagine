# Imagine 성능 개선 개념 정리

작성일: 2026-05-14

이 문서는 아직 전부 적용되지는 않은 Imagine 성능 개선 후보를 개념 단위로 정리한다. 여기서 말하는 성능은 하나가 아니다.

- 검색 품질: 원하는 이미지를 상위 결과에 잘 올리는가
- 검색 응답 속도: 사용자가 검색 후 결과를 받기까지 얼마나 빠른가
- 분석 처리량: 많은 파일을 MC/VV/MV 단계로 얼마나 빨리 처리하는가
- 운영 안정성: 수정 후 품질이나 속도가 떨어졌는지 바로 알 수 있는가

## 1. 평가 게이트

검색 개선은 느낌으로 판단하면 안 된다. QuerySet, LabelSet, RunResult를 고정하고 `nDCG@10`, `P@10`, `Recall@50`, `MRR@10`, latency를 비교해야 한다.

Imagine에는 이미 `tools/run_search_benchmark.py`, `tools/evaluate_search_quality.py`, `tools/compare_search_evaluation.py` 경로가 있다. 다음 단계는 모든 랭킹/가중치 변경을 이 게이트에 묶는 것이다.

적용 우선순위: 최상.

이유:
- 검색 가중치, rerank, BM25 weight 수정은 일부 예시에서는 좋아 보이고 전체 품질은 나빠질 수 있다.
- baseline 대비 하락 폭을 자동으로 막아야 안전하게 실험할 수 있다.

검증:
```bash
.venv/bin/python tools/run_search_benchmark.py --help
.venv/bin/python tools/evaluate_search_quality.py --help
.venv/bin/python tools/compare_search_evaluation.py --help
```

## 2. Query Type 기반 Fusion 튜닝

Imagine 검색은 여러 축을 섞는다.

- VV: 이미지 자체의 시각 유사도
- MV: 캡션/태그를 의미 벡터로 검색
- FTS: 정확한 텍스트 키워드 검색
- Structure: 구도/형태 유사도

RRF는 이 축들의 순위를 합치는 방식이다. 현재는 `backend/search/rrf.py`에서 query type별 preset weight를 사용한다. 더 높은 품질을 내려면 query type별로 실제 LabelSet에서 최적 weight를 찾아야 한다.

적용 우선순위: 높음.

예:
- 짧은 키워드: FTS와 MV 비중을 높이고 VV noise를 줄인다.
- 분위기/스타일 검색: VV와 MV 비중을 높인다.
- 폴더 범위가 있는 검색: scope filter를 먼저 고정하고, 그 안에서 MV/VV/FTS를 비교한다.
- 부정 조건이 있는 검색: exclude filter와 negative visual penalty를 별도로 본다.

검증:
- query type별 `nDCG@10`, `P@10`, `Recall@50`을 따로 본다.
- 전체 평균만으로 통과시키지 않는다.

## 3. Top-N Reranker

RRF는 후보를 잘 모으는 데 강하지만, 최종 1~20위 정렬은 더 정교한 점수가 필요할 수 있다.

현재 `backend/search/scoring.py`에는 `quality_rerank()`가 있다. 이 함수는 후보의 축별 점수, 다중 축 일치, 메타데이터 완성도, 토큰 매칭을 조합한다. 다음 단계는 상위 80~200개 후보에 대해 더 강한 reranker를 붙이는 것이다.

가능한 방식:
- 규칙 기반 reranker 고도화
- 작은 local cross-encoder
- LLM 기반 rerank는 비용과 latency가 크므로 제한적으로만 사용

적용 우선순위: 높음.

주의:
- reranker는 recall을 만들지 못한다. 먼저 VV/MV/FTS 후보 풀에 정답이 들어와야 한다.
- 후보 풀을 키우면 품질은 좋아질 수 있지만 latency도 늘어난다.

검증:
- `Recall@50`이 유지되는지 먼저 확인한다.
- 이후 `nDCG@10`, `P@10`, `MRR@10` 상승을 본다.

## 4. 검색 Latency Cache

검색 응답 속도는 여러 단계의 합이다.

- query decomposition
- VV query embedding
- MV query embedding
- FTS 검색
- RRF merge
- axis score enrichment
- rerank

현재 decomposition cache는 `backend/search/query_decomposer.py`에 있다. 하지만 query embedding cache, 최종 결과 cache, diagnostic 집계 cache는 더 보강할 여지가 있다.

적용 우선순위: 높음.

좋은 적용 후보:
- 동일 query의 VV/MV embedding cache
- 같은 query + filter + top_k 조합의 짧은 TTL result cache
- load more 시 decomposition과 embedding 재계산 방지
- axis별 timeout 후 fallback

주의:
- 파일 DB가 갱신되면 result cache는 무효화되어야 한다.
- embedding cache는 모델 버전, dimension, instruction prefix를 cache key에 포함해야 한다.

검증:
- 평균 latency보다 p95 latency를 우선 본다.
- 품질 metric이 cache 전후로 동일해야 한다.

## 5. SQLite Writer 병목

분석 파이프라인은 GPU/CPU 추론만 빠르면 끝나지 않는다. 결과를 SQLite에 쓰는 단계가 병목이 될 수 있다.

현재 `backend/db/write_queue.py`에는 단일 writer queue와 batch transaction 구조가 있다. `backend/pipeline/ingest_engine.py`도 이 writer를 사용한다. 다만 server router, analysis save, 일부 maintenance 경로에는 직접 commit이 남아 있다.

적용 우선순위: 중상.

다음 개선:
- 대량 분석 결과 저장 경로를 writer queue로 통일
- batch size 500~1000 범위에서 실제 처리량 측정
- WAL checkpoint와 `ANALYZE` 실행 시점 관리
- DB locked retry 로그를 지표화

검증:
- files/min
- DB commit count
- DB locked 발생 횟수
- worker idle time

## 6. Observability

성능 개선은 병목을 찍어서 고쳐야 한다. 단순히 “검색이 느림”이라고 보면 원인을 찾기 어렵다.

현재 `SEARCH_DIAGNOSTIC`과 `logs/search_diagnostic.jsonl` 경로가 있다. 다음 단계는 이 로그를 운영 지표로 집계하는 것이다.

필요한 지표:
- decomposition_ms
- vector_ms
- text_vec_ms
- fts_ms
- rrf_merge_ms
- rerank_ms
- total_ms
- result_count
- query_type
- active axes
- cache hit 여부

적용 우선순위: 최상.

이유:
- latency 개선 전에 어디가 느린지 알아야 한다.
- search quality regression과 latency regression을 같은 run에서 같이 봐야 한다.

검증:
- p50, p95, p99 리포트 생성
- 느린 query sample 자동 추출
- axis별 timeout/오류율 집계

## 7. BM25와 FTS Weight

BM25는 고려해야 한다. 다만 새로 도입할 대상이 아니라, 이미 FTS 축에서 사용 중인 랭킹 기법으로 봐야 한다.

현재 코드 기준:
- `backend/search/sqlite_search.py`의 `fts_search()`에서 SQLite FTS5 `bm25(files_fts, ?, ?, ?, ?, ?)`를 사용한다.
- `config.yaml`의 `search.fts.bm25_weights`에서 5개 컬럼 weight를 설정한다.
- 현재 weight는 `meta_strong`, `meta_weak`, `caption`, `ai_tags`, `classification`이다.
- `backend/db/sqlite_client.py`의 현재 FTS 컬럼도 이 5개를 기준으로 한다.

따라서 결론은 다음과 같다.

- BM25는 반드시 유지한다.
- 단독 검색 엔진으로 키우기보다 Triaxis의 FTS 축으로 둔다.
- 개선 대상은 BM25 자체보다 컬럼 구성, weight, query tokenization, FTS 후보 수다.
- BM25 weight 변경은 반드시 benchmark gate를 통과해야 한다.

튜닝 후보:
- `meta_strong`: 파일명, 레이어명, 폰트, user tag, OCR처럼 정확한 식별 단서. 높은 weight 유지.
- `meta_weak`: 경로, 폴더, 크기 같은 약한 단서. 너무 높이면 폴더명 noise가 커진다.
- `caption`: VLM 설명. 의미 검색에는 좋지만 hallucination 가능성이 있으므로 MV와 역할을 나눠야 한다.
- `ai_tags`: 짧은 태그 매칭. keyword query에서 효과가 크다.
- `classification`: image_type, scene_type, art_style 같은 분류 필드. scoped/filter query에서 보조 신호로 쓰기 좋다.

검증:
- FTS 단독 평가와 Triaxis 통합 평가를 모두 본다.
- FTS 단독이 좋아져도 Triaxis 전체가 나빠지면 적용하지 않는다.
- keyword/scoped query에서 `P@10`, `MRR@10`을 중점 확인한다.

## 8. FTS Vocabulary 확장

FTS는 텍스트가 실제로 들어 있어야 검색된다. 따라서 어떤 텍스트를 어떤 컬럼에 넣을지 중요하다.

확장 후보:
- OCR text
- PSD layer names
- translated layer names
- user note
- user tags
- folder tags
- classification fields
- caption_ko 같은 한국어 display field

주의:
- 모든 텍스트를 한 컬럼에 밀어 넣으면 BM25 weight 제어가 안 된다.
- 사람이 입력한 user metadata와 AI 생성 caption/tag는 신뢰도가 다르다.
- scope 용어와 find 용어가 섞이면 폴더명만 맞는 이미지가 상위로 올라올 수 있다.

검증:
- FTS 단독 metric
- Triaxis 통합 metric
- false positive sample review

## 9. Evidence-Centric Model

현재 파일 row에 최신 caption, tag, vector가 덮어쓰기되는 구조는 단순하지만, 모델 버전 비교와 rollback에는 약하다.

Evidence-Centric Model은 다음을 분리한다.

- asset: 원본 파일의 정체성
- analysis_run: 어떤 모델/설정으로 분석했는지
- evidence_text: caption, tag, OCR, classification
- evidence_vector: VV/MV/Structure vector
- search_materialized: 현재 검색에 사용할 대표 snapshot

적용 우선순위: 중간.

장점:
- 모델 A/B 비교 가능
- confidence 기반 rerank 가능
- 검색 결과가 왜 나왔는지 추적 가능
- 나쁜 모델 결과 rollback 가능

단점:
- DB schema와 ingestion 경로 영향이 크다.
- 먼저 benchmark와 observability가 있어야 효과를 검증할 수 있다.

## 10. Image Decode 가속

분석 처리량에서 이미지 decoding이 병목이 될 수 있다. 특히 JPEG/PNG가 많으면 PIL만으로는 충분하지 않을 수 있다.

후보:
- TurboJPEG
- OpenCV
- pillow-simd
- thumbnail/preprocess cache

적용 우선순위: 중간.

주의:
- PSD, WebDAV, thumbnail 생성 경로와 분리해서 측정해야 한다.
- decode가 병목인지 먼저 확인해야 한다.

검증:
- decode_ms
- preprocess_ms
- end-to-end files/min
- 이미지 품질/색상 차이 smoke test

## 11. Feedback 기반 Label Refresh

사용자의 재검색, 클릭, 저장, 컬렉션 추가, 수정된 태그는 약한 relevance signal이 될 수 있다.

적용 우선순위: 중간.

가능한 signal:
- 검색 후 바로 클릭한 결과
- 오래 본 결과
- 컬렉션에 저장한 결과
- 검색어를 바꿔 다시 찾은 패턴
- 사용자가 직접 수정한 tag/category

주의:
- feedback은 gold label이 아니다.
- weak label로만 쓰고, 일부는 human review로 승격해야 한다.

검증:
- weak label 기반 개선 후 gold LabelSet spot-check
- query type별 overfitting 확인

## 12. Structure Axis 고도화

Structure axis는 색/의미보다 구도와 형태가 중요한 검색에서 유리하다.

예:
- 같은 레이아웃의 UI 화면
- 비슷한 구도의 배경
- 인물 배치가 비슷한 이미지
- PSD 구조가 중요한 파일

적용 우선순위: 중간.

주의:
- 모든 query에서 structure weight를 높이면 오히려 noise가 된다.
- query intent가 “구도/레이아웃/비슷한 화면”일 때만 켜는 편이 좋다.

검증:
- structure query subset을 따로 만든다.
- 일반 semantic query에서 regression이 없는지 본다.

## 권장 실행 순서

1. Observability p50/p95/p99 집계
2. Benchmark gate 강제
3. BM25/FTS weight와 RRF weight ablation
4. Query embedding/result cache
5. Top-N reranker 강화
6. Server save 경로 writer queue 통일
7. FTS vocabulary 확장
8. ECM 설계

핵심 원칙은 간단하다. 먼저 측정하고, 작은 단위로 바꾸고, 같은 LabelSet에서 통과한 것만 남긴다.
