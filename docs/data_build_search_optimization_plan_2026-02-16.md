# 데이터 구축/검색 최적화 계획서 (코드 기반)

작성일: 2026-02-16
대상: ImageParser (Phase 1/2/3a/3b 파이프라인 + SQLite Triaxis 검색)

## 1) 목적
- 실제 코드 기준으로 데이터 구축 파이프라인의 문제를 식별한다.
- 검색 단계에서 생성된 데이터를 최대한 활용하되, 계산/메모리 비용을 합리적으로 통제한다.
- 즉시 적용 가능한 수정과 중기 구조 개선안을 분리해 실행한다.

## 2) 현재 코드 기반 진단

### A. 정확도/정합성 리스크
1. Structure 검색 키 불일치(치명)
- 증상: `vec_structure` 스키마는 `file_id`를 사용하지만, 검색 쿼리 일부가 `id`를 사용.
- 영향: Structure 유사도 검색 누락/오류 가능.
- 조치: 즉시 수정 완료.

2. 이미지+텍스트 통합 검색에서 Structure 축 미활용(상)
- 증상: 기존 `triaxis_image_search`는 VV/MV/FTS 3축만 결합.
- 영향: 레이아웃/구도/배치 유사성 정보가 랭킹에 반영되지 않음.
- 조치: 즉시 수정 완료(4축: VV/X/MV/FTS).

3. 결과 페이로드에서 구조/분류 필드 전달 부족(중)
- 증상: API 포맷 결과에서 일부 구조/분류 필드 미노출.
- 영향: UI/필터/후처리에서 데이터 재활용 제한.
- 조치: 즉시 수정 완료(구조 점수 + 분류 필드 전달 확장).

### B. 구축(ingest) 성능/안정성 리스크
1. DB 커밋 단위가 과도하게 작음(상)
- 증상: `update_vision_fields`, `upsert_vectors`가 호출 단위 커밋.
- 영향: 대량 구축 시 fsync/락 오버헤드 증가.
- 상태: 미해결(설계 개선 필요).

2. 전용 Async DB Writer 부재(상)
- 증상: 추론 루프가 DB write 경로와 강하게 결합.
- 영향: GPU 처리와 저장 경합 시 전체 처리율 하락.
- 상태: 미해결(구조 개선 필요).

3. sqlite-vec 로드 실패 시 fail-open(중)
- 증상: 확장 로드 실패 후 경고만 출력하고 진행.
- 영향: 벡터 저장/검색 시점 런타임 오류 가능.
- 상태: 미해결(초기 health gate 필요).

### C. 데이터 활용 범위 리스크
1. 텍스트 검색 축의 어휘 활용 제한(중)
- 증상: FTS는 메타 중심, MC/ai_tags의 어휘 매칭을 적극 활용하지 않음.
- 영향: exact keyword 탐색 회수율 저하 가능.
- 상태: 미해결(정책적 확장 필요).

## 3) 이번 턴 즉시 반영한 코드 수정

1. Structure 검색 정합성 수정
- `search_structure` JOIN 키를 `vs.file_id`로 수정.
- `find_similar_structure` 조회 키를 `file_id`로 수정.

2. 이미지+텍스트 검색의 4축 결합
- `triaxis_image_search`에 Structure(DINOv2) 축 추가.
- RRF 병합에서 `structure` 축 점수 반영.
- 활성 구조축에 대한 가중치가 없을 때 visual weight 일부를 구조축으로 재배분.

3. 점수/필드 전달 확장
- 결과에 `structure_score` 포함.
- 결과 payload에 `image_type`, `color_palette`, `time_of_day`, `weather`, `character_type`, `item_type`, `ui_type` 등 구조화 필드 확장.
- UI 배지에 Structure 점수(SV) 노출.

4. 마이그레이션 중복 호출 제거
- `_migrate_v3_columns()` 중복 호출 제거.

## 4) 목표 아키텍처 (데이터 최대 활용 + 합리적 소비)

### 4.1 구축 파이프라인
- Parse/Decode Queue (CPU)
- Vision Queue (VLM)
- Embed Queue (VV + X + MV)
- DB Writer Queue (단일 전용 writer)

원칙:
- 추론 경로와 DB 저장 경로를 분리한다.
- DB writer는 batch transaction(예: 500~1000건) 단위로 커밋한다.
- 모든 단계는 idempotent key(file_id + phase + model_version) 기반 재시도한다.

### 4.2 검색 파이프라인
- Candidate 생성:
  - VV 후보
  - X(Structure) 후보
  - MV 후보
  - FTS 후보
- 가중 RRF로 1차 병합
- 필터 적용(LLM lenient -> user strict)
- Top-K 후처리(미싱 축 점수 enrichment)

원칙:
- 축별 score range가 달라도 rank 기반 병합(RRF) 유지.
- 구조축은 이미지 입력이 있을 때만 활성화(불필요 연산 방지).
- 후보 풀 크기(candidate_multiplier)와 축별 threshold를 외부 설정으로 제어.

## 5) 남은 개선 백로그 (우선순위)

P0 (즉시)
- sqlite-vec health gate: 앱 시작 시 미로딩이면 벡터 모드 차단 + 명확한 복구 안내.
- Structure 결합 A/B 토글(온/오프) + 진단 로그 분리.

P1 (단기)
- Async DB Writer 도입(Queue + bulk transaction + retry).
- write path에서 commit 배치화.
- WAL checkpoint 주기 제어 + ANALYZE 자동 실행.

P2 (중기)
- FTS에 MC/ai_tags 어휘 축 옵션 추가(과적합 방지용 가중치 분리).
- 쿼리 타입별 동적 가중치 학습(offline eval 기반).

P3 (고도화)
- 질의 의도 분류(형태/구도 중심 vs 의미 중심)에 따라 Structure 축 가중 자동 조정.
- top-N 재정렬기 도입(비용 상한 포함).

## 6) 검증 계획 (필수)
- Offline: nDCG@10, Recall@20, MRR@10, Precision@5
- Online: P95 latency, 처리량(img/s), 오류율, 재시도 성공률
- 승격 기준:
  - 정확도: baseline 대비 동등 이상(최소 -2% 이내)
  - 성능: P95 2s 이하, 구축 처리량 1.3x 이상
  - 안정성: 재처리 후 누락률 0에 수렴

## 7) 결론
- 현재 코드베이스는 4단계 구축 파이프라인의 뼈대가 맞다.
- 다만 검색 결합/저장 경로의 일부 병목과 정합성 리스크가 있었고, 이번 턴에서 핵심 정확도 리스크(Structure 키/축 활용)는 먼저 해소했다.
- 다음 단계의 본질은 `DB writer 분리 + 배치 트랜잭션 + health gate`이며, 이것이 대량 구축 안정성과 실효 성능을 결정한다.
