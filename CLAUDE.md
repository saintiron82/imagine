# CLAUDE.md

이 문서는 Imagine의 **동작 원리, 모델 접근 논리, 처리 구조, 핵심 알고리즘**을 빠르게 파악하기 위한 기술 개요다.

## 프로젝트의 기본 문제 설정

Imagine은 PSD, PNG, JPG 같은 시각 자산을 단순 파일이 아니라 **검색 가능한 분석 단위**로 바꾸는 시스템이다.

이 시스템이 푸는 문제는 세 가지다.

1. 파일 내부의 구조와 팩트를 뽑아내야 한다.
2. 이미지의 시각적 유사성과 의미적 유사성을 함께 다뤄야 한다.
3. 단일 로컬 환경과 서버-워커 분산 환경에서 같은 처리 모델을 유지해야 한다.

이 때문에 Imagine은 파싱, 비전 분석, 벡터화, 검색, 스케줄링을 각각 분리된 계층으로 두고, 그 사이를 명시적인 데이터 구조로 연결한다.

## 핵심 개념

### VV

`VV`는 Visual Vector다.  
이미지 픽셀에서 직접 추출한 시각 임베딩으로, “이것과 비슷하게 생긴 이미지”를 찾기 위한 축이다.

주로 담당하는 신호:

- 구도
- 실루엣
- 색감
- 밀도감
- 전반적 비주얼 스타일

### MC

`MC`는 Meta-Context Caption이다.  
비전 모델이 이미지와 파싱 컨텍스트를 함께 보고 만든 해석 결과다.

MC는 단순 캡션이 아니라 다음을 묶는다.

- 핵심 설명 텍스트
- 태그
- 분류 결과
- 도메인별 구조화 정보

### MV

`MV`는 Meaning Vector다.  
MC 텍스트를 다시 텍스트 임베딩 모델에 넣어 만든 의미 임베딩이다.

VV가 픽셀의 유사성을 보는 축이라면, MV는 설명 가능한 의미와 용도를 본다.

### FTS

`FTS`는 SQLite FTS5 기반의 팩트 검색 축이다.  
이 축은 생성형 추론이 아니라 **정확한 텍스트 단서**를 다룬다.

대표 입력:

- 파일명
- 경로
- 레이어명
- 텍스트 레이어
- OCR
- 사용자 메모/태그
- 구조화 메타데이터

### Structure

검색 엔진은 기본적으로 Triaxis를 중심으로 설명할 수 있지만, 실제 구현은 필요 시 **구조 축**도 함께 다룬다.

구조 축은 주로 다음 성질을 본다.

- 큰 형태 배치
- 구조적 균형
- 레이아웃의 골격
- 이미지 간 형태적 배열 유사성

코드상으로는 `vec_structure`, `structure_score`, `structural_similarity` 같은 이름으로 드러난다.

## 전체 구조

Imagine의 큰 흐름은 아래와 같다.

```text
원본 파일
  -> Parse
  -> Vision
  -> VV Encode
  -> MV Encode
  -> SQLite 저장
  -> Triaxis Search
```

실제로는 이 흐름이 단일 머신, Electron 내장 서버, 브라우저 서버 모드, 외부 워커 모드에서 같은 논리로 반복된다.

## 파싱 계층

### 목적

파싱 계층은 원본 파일을 검색 가능한 기본 재료로 바꾸는 단계다.

여기서 만들어지는 것은 대략 다음과 같다.

- 파일 메타데이터
- 썸네일
- PSD 레이어 트리
- 텍스트 레이어 내용
- 폰트 정보
- 구조 태그와 의미 단서

### PSD 접근 논리

PSD는 단순 이미지가 아니라 계층 구조를 가진 문서이므로, 파서는 최종 합성 이미지뿐 아니라 내부 레이어 구조를 함께 읽는다.

`backend/parser/psd_parser.py` 기준 흐름:

1. PSD 열기
2. 캔버스 크기 확인
3. 레이어 트리 재귀 순회
4. 텍스트 레이어에서 문자열과 폰트 추출
5. 합성 이미지 또는 대체 경로로 썸네일 생성
6. 레이어명 기반 의미 태그 생성
7. `AssetMeta` 구조로 저장

### PSD 파서의 핵심 판단

레이어 파서는 레이어를 단순히 나열하지 않는다.  
다음 판단을 함께 만든다.

- 레이어 이름 정제
- 레이어 위치와 크기
- 그룹/비그룹 구조
- content type 추론

즉 PSD 파싱은 “포토샵 문서를 이미지 검색 재료로 바꾸는 구조화 단계”다.

## 비전 분석 계층

### 역할 분리

비전 분석 계층의 목적은 이미지에 대해 사람이 검색에 사용할 수 있는 설명을 만드는 것이다.

이 계층은 대체로 두 가지 역할을 한다.

1. 이미지에서 MC를 생성
2. 이후 MV 생성을 위한 의미 입력을 정리

### 모델 접근 논리

Imagine은 비전 모델을 단순히 “좋은 캡션 생성기”로 쓰지 않는다.  
핵심은 **검색 중간 표현 생성기**로 쓴다는 점이다.

즉 비전 모델의 목표는 문학적 설명이 아니라 다음에 가깝다.

- 검색에 유리한 짧고 밀도 높은 설명
- 분류 가능한 태그
- 후속 MV 임베딩에 적합한 텍스트
- 도메인별로 통제 가능한 출력

### 구현 구조

모델 선택과 생명주기 관리는 분리되어 있다.

- 모델 생명주기: `backend/pipeline/model_manager.py`
- 비전 분석 인터페이스: `backend/vision/base.py`
- 기본 구현 축: `backend/vision/analyzer.py`
- 플랫폼/백엔드 선택: `backend/vision/vision_factory.py`
- 플랫폼별 어댑터: `mlx_adapter.py`, `ollama_adapter.py`, `vllm_adapter.py`

### 핵심 설계 포인트

비전 계층은 다음 원칙으로 움직인다.

- lazy load: 필요할 때만 모델 적재
- explicit unload: phase 종료 후 메모리 해제
- backend abstraction: MLX, transformers, Ollama, vLLM을 같은 인터페이스 뒤로 숨김
- prompt shaping: 장황한 자유 생성이 아니라 구조화된 검색 재료 생성
- domain injection: 이미지 유형에 따라 분류 기준과 태그 공간을 바꿈

## VV 계층

VV 계층은 이미지 자체를 벡터로 바꾸는 부분이다.

이 계층의 목표는 자연어 의미가 아니라 “픽셀 레벨에서 닮음”을 보존하는 것이다.

대표적으로 잘 잡는 것:

- 동일한 구도
- 비슷한 배색
- 유사한 톤
- 반복되는 디자인 패턴

VV는 검색에서 다음 상황에 강하다.

- 유사 레퍼런스 찾기
- 업로드 이미지 기반 검색
- 말로 설명하기 어려운 스타일 매칭

## MV 계층

MV 계층은 MC를 다시 임베딩해 의미 공간으로 보내는 단계다.

핵심 아이디어는 단순하다.

- 원본 이미지를 바로 텍스트 임베딩하지 않는다.
- 먼저 비전 모델이 해석한 MC를 만든다.
- 그 MC를 의미 벡터로 바꾼다.

이 구조를 쓰는 이유는 이미지의 의미를 직접 임베딩하는 대신, **해석된 중간 언어 표현**을 거쳐 더 안정적인 검색 신호를 만들기 위해서다.

즉 MV는 “설명 가능한 의미 공간”을 담당한다.

## 파이프라인 실행 구조

### Phase 분리

Imagine의 파이프라인은 크게 아래처럼 본다.

```text
DL: Download
P: Parse
V: Vision
VV: Visual Vector
MV: Meaning Vector
```

`backend/pipeline/phase_runner.py`는 Vision, VV, MV 단계의 공통 실행기를 담당한다.  
Parse는 입력 경로와 실행 모드에 따라 처리 형태가 다르기 때문에 별도 풀과 별도 경로로 관리된다.  
원격 소스에서는 Download 단계가 먼저 개입하며, 실제 서버 상태 모델은 `file_tasks`의 phase status로 유지된다.

### 왜 이런 구조를 쓰는가

이 구조는 세 가지 문제를 동시에 해결한다.

1. 실패 복구를 단순하게 만든다.
2. 각 단계의 비용이 매우 다르다는 점을 반영한다.
3. 모델 메모리 점유를 단계별로 분리한다.

Vision은 무겁고, VV와 MV는 상대적으로 가볍다.  
따라서 한 번에 모든 모델을 상주시켜 돌리기보다 단계별로 적재하고 해제하는 쪽이 더 안정적이다.

### 모델 생명주기

`ModelManager`의 핵심 철학은 간단하다.

- VLM, VV, MV를 모두 lazy load
- phase 완료 후 unload
- GPU cache clear + gc를 함께 수행

이 방식은 속도만이 아니라 **VRAM 경쟁 회피**가 목적이다.

### 작업 상태 모델

현재 서버의 작업 단위는 `analysis_jobs`와 `file_tasks`로 구성된다.

- `analysis_jobs`: 사용자가 인식하는 작업 묶음
- `file_tasks`: 파일별 실제 처리 상태

`file_tasks`는 대체로 아래 상태들을 가진다.

- `download_status`
- `parse_status`
- `mc_status`
- `vv_status`
- `mv_status`

즉 Imagine의 파이프라인은 “파일 하나를 순차적으로 완성하는 상태 기계”로 볼 수 있다.

## 검색 구조: Triaxis

### 기본 사고방식

Imagine 검색은 단일 점수 함수보다 **여러 독립 축을 먼저 만들고 나중에 합치는 구조**를 선택한다.

이유는 축마다 잘 잡는 것이 다르기 때문이다.

- VV는 시각 유사도에 강하다.
- MV는 의미와 용도에 강하다.
- FTS는 정확 키워드와 팩트에 강하다.

단일 모델로 이 셋을 모두 완전히 해결하려고 하면 제어력이 떨어진다.  
그래서 Imagine은 아예 세 축을 분리하고 융합한다.

### Query Decomposition

`backend/search/query_decomposer.py`는 자연어 질의를 바로 검색하지 않고 중간 구조로 바꾼다.

출력의 핵심은 다음과 같다.

- `vector_query`
- `negative_query`
- `fts_keywords`
- `exclude_keywords`
- `filters`
- `query_type`

이 단계의 목적은 검색을 “질문 하나”가 아니라 “여러 축에 걸친 실행 계획”으로 바꾸는 것이다.

즉 사용자의 한 문장을 다음처럼 쪼갠다.

- 시각 축에서 쓸 설명
- 의미 축에서 쓸 설명
- FTS에서 쓸 키워드
- 제외해야 할 조건
- 구조화 필터
- 어떤 축을 더 믿어야 하는지에 대한 힌트

Query Decomposer 자체도 단일 구현이 아니라 backend resolution을 가진다.

- Codex CLI
- MLX text LLM
- Ollama
- 규칙 기반 fallback

즉 질의 분해 역시 “LLM 1개에 전적으로 묶인 기능”이 아니라, 사용 가능한 backend에 따라 실행 경로가 달라지는 계층이다.

### Candidate-First 구조

Triaxis 검색은 일반적으로 모든 축을 끝까지 전수 계산하지 않는다.  
먼저 후보군을 만들고, 그 뒤에 재정렬과 보정을 한다.

핵심 흐름은 대략 이렇다.

```text
Query
  -> Decompose
  -> Per-axis retrieval
  -> RRF merge
  -> Negative filter
  -> Quality rerank
  -> Axis score enrichment
  -> Final top-k
```

### RRF 융합 알고리즘

`backend/search/scoring.py`의 `rrf_merge`와 `rrf_merge_multi`는 축별 랭킹을 합친다.

핵심 아이디어는 점수 절대값을 직접 섞기보다 **순위 기반으로 결합**하는 것이다.

RRF의 장점:

- 각 축의 점수 스케일 차이에 덜 민감함
- 어느 한 축의 극단값에 덜 끌려감
- 독립적인 후보 생성기의 장점을 유지하기 쉬움

Imagine은 단순 2축이 아니라 다축 버전 `rrf_merge_multi`를 사용해 visual, text_vec, fts, structure 축을 합칠 수 있다.

즉 개념 설명은 Triaxis지만, 실제 점수 결합 계층은 이미 **N-axis merge**를 받아들일 수 있게 설계되어 있다.

### Negative Filter

질의에 “제외” 조건이 있으면 `apply_negative_filter`가 후속 필터/패널티로 개입한다.

이 구조를 분리한 이유는, 긍정 질의와 부정 질의를 같은 벡터 공간 점수에 섞는 대신 **후처리 패널티**로 다루는 것이 더 제어 가능하기 때문이다.

### Quality Rerank

`quality_rerank`는 RRF 이후의 후보 풀을 다시 정리한다.

이 단계의 목적은 단순 회수율이 아니라 **교차 축 합의가 강한 결과를 위로 올리는 것**이다.

여기서 보는 신호는 예를 들면 다음과 같다.

- 각 축의 정규화된 점수
- 쿼리 토큰과의 부드러운 일치
- 경로 힌트
- 메타데이터 밀도
- 필터와의 합치 여부

즉 RRF가 “축별 회수력 확보”라면, quality rerank는 “최종 체감 품질 정리”에 가깝다.

### Axis Score Enrichment

최종 결과에 대해 `enrich_axis_scores`가 추가 축 점수를 보충한다.

이 단계는 순위를 바꾸기 위한 것이 아니라, UI에서 결과를 설명할 수 있게 하기 위한 성격이 강하다.

즉 “왜 이 결과가 나왔는가”를 더 잘 보여주기 위한 display layer다.

## 스케줄링 구조

### 문제 설정

워커 시스템의 핵심 문제는 단순 큐 소비가 아니다.

- MC는 무겁다.
- VV/MV는 상대적으로 가볍다.
- GPU 성능이 워커마다 다르다.
- phase 전환은 모델 스위칭 비용을 만든다.

이 때문에 Imagine은 단순 round-robin보다 **pressure-based scheduling**을 택한다.

### Scheduler의 핵심 모델

`backend/server/queue/scheduler.py`는 다음 아이디어를 사용한다.

1. 워커를 GPU class로 분류
2. phase별 pending 압력을 계산
3. MC는 GPU class에 따라 penalty 적용
4. 현재 phase를 유지하는 안정성 보정 적용
5. 완료 직전 단계(MV)에 보너스 부여
6. 측정된 처리량으로 batch size를 동적으로 조절

즉 스케줄러는 “어떤 작업이 남았는가”만 보지 않고, “누가 어떤 일을 가장 싸게 처리할 수 있는가”를 함께 본다.

### Pressure-based Scheduling의 직관

압력은 대체로 다음 성격을 가진다.

```text
pressure ≈ pending / (workers_on + 1) × phase_weight
```

여기에 MC penalty, phase stability, completion bias가 추가된다.

이 구조는 다음 효과를 노린다.

- 느린 워커가 MC에 과투입되는 것을 방지
- 같은 phase를 계속 처리해 모델 스위칭 비용 완화
- 끝나기 직전 작업을 빨리 닫아 전체 완료 체감 향상

## 서버와 워커의 역할 분리

현재 구조에서는 서버와 워커가 완전히 같은 일을 하지 않는다.

### 서버가 강한 영역

- Analysis Job 생성
- file task 상태 관리
- DB 커밋
- 다운로드 선행 처리
- 파싱 선행 처리
- 스케줄링과 세션 관리
- phase pause/resume 제어
- 오류 조회와 진행률 집계

### 워커가 강한 영역

- MC
- VV
- MV
- 실제 추론 처리
- 하트비트와 성능 보고

즉 서버는 orchestration 쪽이고, 워커는 inference executor 쪽이다.

단, 이 경계는 완전히 절대적이지 않다.  
현재 구현에는 **embedded worker**가 있어서 서버 프로세스 내부에서도 추론 실행 경로가 존재한다.

따라서 구조적으로는 다음처럼 이해하는 것이 맞다.

- 외부 워커: 분산 inference executor
- embedded worker: 서버 내부 inference executor
- 서버 본체: 상태 관리와 orchestration

## WebDAV와 원격 처리 논리

Imagine은 로컬 폴더만 가정하지 않는다.  
WebDAV/NAS 원격 경로도 현재 구조의 일부다.

핵심 접근은 다음과 같다.

- 원본 대용량 파일을 즉시 전부 가져오지 않는다.
- 필요한 다운로드를 선행 처리 풀에서 관리한다.
- 파싱 풀은 준비된 파일만 잡아 처리한다.
- 가능한 한 썸네일/메타데이터 중심으로 먼저 움직인다.

즉 원격 처리 구조의 핵심은 “원본 파일 이동 최소화”다.

여기서 중요한 점은 다운로드와 파싱이 같은 단계가 아니라는 것이다.

- DownloadAheadPool은 원격 원본 확보를 맡고
- FileTaskParsePool은 준비된 입력을 파싱하며
- 이후 AI 단계는 별도 phase로 이어진다

즉 원격 처리 경로는 `download -> parse -> ai`의 분리 구조를 가진다.

상태 전이, 파일 맵, 설정 키를 포함한 상세 흐름은 `docs/nas_processing_flow.md`가
단일 기준 문서다.

## 프런트엔드 구조

프런트엔드는 하나의 React 앱이 두 실행 표면을 가진다.

### Electron 모드

- 로컬 프로세스 제어
- IPC 브리지
- 로컬 서버 제어
- 데스크톱 앱 경험

### Web 모드

- HTTP API 호출
- 인증 기반 서버 접속
- 브라우저 UI

핵심은 UI를 두 벌 만드는 것이 아니라, **실행 표면만 다르게 두고 상태/화면 구조는 공유**한다는 점이다.

## 인증 구조

인증은 단순 로그인 화면이 아니라 두 층으로 구성된다.

- 개인 신원: Firebase 계층
- 서버 역할/세션: JWT 계층

즉 “누구인가”와 “이 서버에서 무엇을 할 수 있는가”를 분리한다.

이 구조는 개인 계정 체계와 서버별 권한 체계를 따로 관리하기 위한 것이다.

## 이 문서를 읽을 때 중요한 관점

Imagine은 다음 네 가지를 동시에 만족시키려는 설계다.

1. 로컬 우선
2. 분산 가능
3. 검색 품질 중심
4. 모델/플랫폼 교체 가능

그래서 구현이 다소 층화되어 있다.

- parser는 구조를 만든다.
- VLM은 해석을 만든다.
- embedding은 검색 축을 만든다.
- scheduler는 계산 자원을 배분한다.
- search는 다축 후보를 융합한다.

이 다섯 층을 따로 이해하면 전체 구조가 빠르게 보인다.

## 빠른 진입 파일

처음 읽을 때는 아래 순서가 가장 효율적이다.

1. `backend/server/app.py`
2. `backend/server/routers/analysis.py`
3. `backend/server/queue/analysis_manager.py`
4. `backend/pipeline/phase_runner.py`
5. `backend/pipeline/model_manager.py`
6. `backend/search/sqlite_search.py`
7. `backend/search/scoring.py`
8. `backend/search/query_decomposer.py`
9. `frontend/src/App.jsx`
10. `frontend/src/contexts/AuthContext.jsx`

## 한 줄 요약

Imagine은 **파싱으로 구조를 만들고, 비전 모델로 해석을 만들고, 두 종류의 임베딩으로 검색 축을 만들고, RRF와 재정렬로 최종 결과를 만드는 로컬 우선 멀티모달 자산 검색 시스템**이다.
