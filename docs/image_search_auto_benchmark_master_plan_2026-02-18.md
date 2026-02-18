# 이미지 검색 자동 벤치마크 개발문서 (Lean, 초자동화) - 2026-02-18

## 1. 문서 목적
- 최소 인력(2명)과 짧은 기간(6주) 안에, 이미지 검색 품질/속도/비용을 자동 검증하는 벤치마크 시스템을 구축한다.
- 비교 대상 3축(상용 서비스, 오픈소스 baseline, 자사 버전)을 동일 조건에서 주기적으로 측정하고 배포 게이트에 연동한다.
- 수동 대시보드 구축보다 자동 실행/자동 리포트/자동 차단을 우선한다.

## 2. 범위 (In/Out)

### 2.1 In Scope
- 텍스트->이미지 검색 1개 유즈케이스 우선 지원
- 오프라인 벤치마크(정답셋 기반) 자동 실행
- 상용 1개 + 오픈소스 1개 + 자사 1개 비교
- PR 스모크 벤치 + 야간 풀 벤치 + 배포 전 게이트
- 결과 Markdown 리포트 자동 생성

### 2.2 Out of Scope (1차 제외)
- 다중 유즈케이스(이미지->이미지, 멀티턴 검색)
- 실시간 사용자 A/B 실험 자동화
- 전용 시각화 대시보드 신설
- 다수 상용 API 동시 계약/통합

## 3. 제약 조건
- 인력: 검색/ML 1명, 백엔드/플랫폼 1명 (총 2명)
- 기간: 6주
- 운영 비용 상한: 외부 API 벤치 비용 월 예산 고정 (예: USD 300)
- 실패 허용: 벤치 실패가 서비스 장애를 유발하면 안 됨(벤치와 서비스 경로 분리)

## 4. 성공 기준 (Go/No-Go)

### 4.1 품질 지표
- Primary: `nDCG@10`
- Secondary: `Recall@50`, `MRR@10`

### 4.2 성능/비용 가드레일
- `p95 latency`
- `error rate`
- `cost_per_1k_queries`

### 4.3 배포 게이트 규칙 (초기값)
- `nDCG@10`가 직전 프로덕션 기준 대비 1.0%p 이상 하락하면 배포 차단
- `Recall@50` 하락폭이 2.0%p 초과면 배포 차단
- `p95 latency` 증가율이 20% 초과면 배포 차단
- `error rate` 증가가 0.5%p 초과면 배포 차단
- `cost_per_1k_queries` 증가율이 25% 초과면 경고(차단 아님)

## 5. 아키텍처 개요

### 5.1 구성 요소
1. `dataset_builder`
- 로그/샘플 데이터에서 평가 쿼리셋(JSONL) 자동 생성
- 고정 골드셋 + 월간 리프레시셋 분리 관리

2. `label_builder`
- 약한 라벨(클릭/체류 기반) 자동 생성
- 주간 샘플(100건)만 수동 검수해 라벨 품질 보정

3. `engine_adapters`
- `commercial_adapter`
- `oss_adapter`
- `internal_adapter`
- 공통 I/O 계약으로 교체 가능하게 구현

4. `benchmark_runner`
- 동일 쿼리셋을 3개 엔진에 실행
- 결과 저장, metric 계산, 통계 검정

5. `gate_evaluator`
- 기준선과 비교해 pass/fail 결정
- CI/CD에 실패 코드 반환

6. `report_generator`
- 단일 Markdown 리포트 생성
- 핵심 표: 종합 점수, 유형별 점수, 비용/속도, pass/fail

### 5.2 실행 흐름
1. 쿼리셋/라벨셋 버전 선택
2. 3개 엔진 순차 또는 제한 병렬 실행
3. metric 계산 + 기준선 비교 + 통계 검정
4. gate 판정
5. 리포트 생성 및 아티팩트 업로드

## 6. 데이터 계약 (스키마)

### 6.1 QuerySet (`benchmark_queries_v1.jsonl`)
- `query_id` (string, unique)
- `query_text` (string)
- `query_type` (`exact|semantic|longtail|ambiguous`)
- `locale` (string, default `ko-KR`)
- `created_at` (ISO8601)

### 6.2 LabelSet (`relevance_labels_v1.jsonl`)
- `query_id` (string)
- `item_id` (string)
- `relevance` (`0|1|2`)
- `label_source` (`weak|human|adjudicated`)
- `label_version` (string)

### 6.3 RunResult (`run_<engine>_<timestamp>.jsonl`)
- `run_id` (string)
- `engine_id` (`commercial|oss|internal`)
- `query_id` (string)
- `rank` (int)
- `item_id` (string)
- `score` (float)
- `latency_ms` (int)
- `error` (nullable string)
- `cost_usd` (float, nullable)

### 6.4 Summary (`evaluation_report_<date>.md` + `summary.json`)
- 종합 지표/유형별 지표
- 기준선 대비 delta
- gate 결과
- 실패 사유

## 7. 디렉터리 표준
- `benchmarks/config/benchmark.yaml`
- `benchmarks/data/queries/`
- `benchmarks/data/labels/`
- `benchmarks/runs/`
- `benchmarks/reports/`
- `benchmarks/baselines/`

## 8. 설정 파일 명세 (`benchmarks/config/benchmark.yaml`)
- `engines`: 비교 엔진 목록 및 인증정보 참조키
- `dataset`: 쿼리셋/라벨셋 버전
- `limits`: timeout, max_concurrency, max_budget_usd
- `metrics`: primary/secondary/gates
- `schedules`: `pr_smoke`, `nightly_full`, `pre_release`
- `output`: report 경로, artifact 보관일

## 9. 자동화 설계

### 9.1 PR 스모크 벤치
- 대상: 샘플 50쿼리
- 목적: 명백한 회귀 차단
- 실행 시간 목표: 10분 이내
- 실패 시: PR check fail

### 9.2 야간 풀 벤치
- 대상: 전체 500쿼리(초기 300 가능)
- 목적: 정확한 품질 추세 추적
- 실패 시: 슬랙/이메일 알림 + 이슈 자동 생성

### 9.3 배포 전 게이트
- 대상: 최신 풀 벤치 결과
- 목적: 릴리즈 차단/통과
- 실패 시: 배포 파이프라인 중단

## 10. 통계 검정 규칙
- 단위: query-level paired 비교
- 방법: bootstrap 10,000회
- 판정:
- `nDCG@10` delta의 95% CI 하한 > 0 이면 개선
- gate는 통계 유의 + 가드레일 동시 만족 필요

## 11. 실패 처리/복구
- 엔진 호출 실패: 재시도 2회 (지수 백오프)
- 외부 API 장애: 해당 엔진 `partial_failed`로 표시하고 리포트에 분리 기록
- 예산 초과: 실행 중단 후 현재 결과만으로 리포트 생성
- 벤치 자체 장애: 서비스 배포와 독립 실패로 처리

## 12. 보안/컴플라이언스
- 외부 API key는 환경변수 또는 시크릿 매니저 사용
- 쿼리 로그는 익명화 후 저장(개인식별자 제거)
- 결과 아티팩트 보관기간 기본 90일

## 13. 구현 계획 (6주)

### Week 1
- 벤치마크 스키마 확정 (`QuerySet`, `LabelSet`, `RunResult`)
- 고정 골드셋 300쿼리 자동 생성 스크립트 작성
- 기준선 버전 고정 정책 문서화

### Week 2
- 엔진 어댑터 3종 구현(상용/오픈소스/자사)
- 공통 인터페이스 테스트 작성

### Week 3
- metric 계산기 + bootstrap 검정 구현
- 기준선 비교기 구현

### Week 4
- PR 스모크 벤치 CI 연동
- Markdown 리포트 자동 생성기 구현

### Week 5
- 야간 풀 벤치 스케줄 + 알림 연결
- 배포 전 gate_evaluator 연동

### Week 6
- 운영 안정화(재시도/예산상한/타임아웃)
- 문서/런북/인수 체크 완료

## 14. 역할 분담 (2명)
- 엔지니어 A (검색/ML)
- metric, 라벨 품질 보정, 기준선 해석, 회귀 분석
- 엔지니어 B (플랫폼/백엔드)
- runner, CI/CD, 스케줄, 리포트, 알림

## 15. 완료 조건 (Definition of Done)
- PR마다 스모크 벤치가 자동 실행되고 fail/pass가 체크에 반영됨
- 야간 풀 벤치 결과가 매일 보고서로 생성됨
- 배포 전 gate가 실제로 릴리즈 차단 가능함
- 최근 2주간 수동 실행 없이 자동 벤치가 안정 동작함(성공률 95% 이상)

## 16. 운영 런북 (요약)
1. 실패 알림 수신
2. `benchmarks/reports/latest.md`에서 실패 축 확인
3. `summary.json`의 `failed_gate` 확인
4. 모델/설정 롤백 또는 임계치 재검토(승인 프로세스 필요)
5. 재실행 후 gate 통과 시 배포 재개

## 17. 리스크와 대응
- 라벨 노이즈: 주간 100건 수동 샘플 검수 유지
- 외부 API 비용 급증: 예산 상한 + 샘플 다운스케일
- 오픈소스 baseline 품질 저하: baseline 버전 pinning
- 벤치 과적합: 월 1회 리프레시셋 교체

## 18. 즉시 착수 체크리스트
- [ ] `benchmarks/` 표준 디렉터리 생성
- [ ] `benchmark.yaml` 초안 작성
- [ ] 300쿼리 골드셋 자동 추출 스크립트 작성
- [ ] 어댑터 인터페이스(3종) 스텁 구현
- [ ] PR 스모크 CI job 생성
- [ ] 리포트 템플릿(`evaluation_report.md`) 추가

---
이 문서는 "소수 인력 + 짧은 일정" 제약에서 자동화 중심 벤치마크를 운영 가능한 수준으로 만들기 위한 1차 개발 기준서다.
