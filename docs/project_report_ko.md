# Imagine — 프로젝트 종합 기술 보고서

**버전**: v0.6.4.20260312_08
**보고 기준일**: 2026-03-12
**현재 상태**: Phase 4.6 완료 · Phase 5 진행 중

---

## 1. 프로젝트 개요

### 1.1 배경과 문제 정의
게임 스튜디오, 애니메이션 팀, 프리랜스 디자이너 등 시각 에셋 중심의 크리에이티브 작업 환경에서는 수만~수십만 장의 PSD, PNG, JPG 파일이 깊은 폴더 구조 안에 비정형 데이터로 축적됩니다. 기존의 파일명 기반 검색은 "색감이 따뜻한 도시 배경" 혹은 "로그인 버튼이 포함된 UI 목업" 같은 크리에이터의 직관적인 검색 의도를 전혀 처리하지 못합니다.

### 1.2 솔루션: Imagine
Imagine은 이미지를 **AI가 검색 가능한 고차원 벡터 데이터**로 변환하는 멀티모달 에셋 인텔리전스 시스템입니다. 시각적 유사성·의미적 맥락·텍스트 키워드를 동시에 검색할 수 있는 독자적인 **Triaxis(3축) 아키텍처**를 적용하여, 창작자의 의도에 가장 부합하는 에셋을 정확하게 찾아냅니다.

### 1.3 핵심 가치
| 가치 | 설명 |
|------|------|
| **프라이버시 우선** | 모든 AI 추론과 데이터 저장이 100% 로컬에서 수행. 외부 클라우드 API 의존 없음 |
| **제로 인프라** | Docker, PostgreSQL 등 외부 종속성 없이 SQLite 단일 파일로 전체 시스템 동작 |
| **하드웨어 무관성** | 6GB VRAM 노트북부터 24GB+ 워크스테이션까지, Apple Silicon(MLX)과 NVIDIA(CUDA) 모두 지원 |
| **분산 확장** | 네트워크 내 유휴 장비들을 토큰 한 줄로 클러스터에 합류시켜 처리 속도를 선형적으로 확장 |

### 1.4 핵심 기능 목록
- **멀티모달 검색**: 텍스트 쿼리, 이미지 쿼리, 또는 둘의 조합으로 이미지 검색
- **3축 랭킹**: VV(시각) + MV(의미) + FTS(키워드) 점수를 RRF로 융합하여 정밀 순위 산출
- **PSD 심층 파싱**: 레이어 트리, 텍스트 콘텐츠, 폰트 정보, 구조 벡터(Structure Vector) 추출
- **AI 비전 분석**: Qwen3-VL을 통한 자동 캡션(MC), 태그, 분류 생성
- **데스크톱 + 웹 듀얼 모드**: 공유 React UI를 사용하는 Electron 앱(로컬) 및 브라우저(원격) 접속
- **클러스터링 분산 처리**: 작업 큐 기반으로 여러 PC에서 이미지를 병렬 인덱싱
- **외부 접속**: LAN 자동 감지, QR 코드, Cloudflare Quick Tunnel을 통한 원격 접속

---

## 2. 시스템 아키텍처

### 2.1 데이터 인제스트 파이프라인 (4단계 원자적 처리)

```
이미지 파일 (PSD/PNG/JPG)
    │
    ▼
Phase P (Parse)    ─── 메타데이터 추출, 썸네일 생성, 레이어 트리 파싱
    │                   (PSD: psd-tools, PNG/JPG: Pillow)
    ▼
Phase V (Vision)   ─── Qwen3-VL: MC 캡션 + 태그 + 분류 생성
    │                   (2-Stage: 구조 설명 → 태그/분류 추출)
    ▼
Phase E (Embed)    ─── SigLIP2 → VV (시각 벡터)
    │                   Qwen3-Embedding → MV (의미 벡터)
    │                   DINOv2 → Structure Vector (구조 벡터, PSD 전용)
    ▼
Phase S (Summary)  ─── 결과 DB 커밋, 인덱스 갱신, 완료 마킹
```

**핵심 설계 원리:**
- **원자적 트랜잭션**: 각 Phase가 완료될 때마다 즉시 DB에 커밋. 중간에 크래시가 발생해도 이미 완료된 Phase의 결과는 보존됨
- **Smart Skip**: 파일별로 마지막 실패 Phase를 기록하여, 재처리 시 이미 완료된 Phase를 자동 건너뜀 → 대용량 PSD VLM 분석(수십 초) 중복 방지
- **서브배치 저장**: 대규모 파일 처리 시 배치 내에서도 중간 결과를 주기적으로 디스크에 기록

### 2.2 Triaxis 검색 엔진

```
사용자 쿼리
    │
    ▼
QueryDecomposer (LLM 기반 쿼리 분해)
    │
    ├──▶ VV축: SigLIP2 텍스트 인코더 → 코사인 유사도 (시각적 유사성)
    │         "이 이미지와 비슷한 색감/구도를 가진 이미지"
    │
    ├──▶ MV축: Qwen3-Embedding → 코사인 유사도 (의미론적 관련성)
    │         "사이버펑크 도시의 외로운 분위기"
    │
    └──▶ FTS축: SQLite FTS5 BM25 → 키워드 매칭 (16개 컬럼)
              "login_button, blue_bg, 300x200"
    │
    ▼
RRF Merge (Reciprocal Rank Fusion)
    │  가중치 프리셋: balanced / visual / semantic / fact
    ▼
최종 랭킹 결과 (Top-K)
```

**검색 최적화:**
- **Candidate-First 전략**: 전체 DB 풀 스캔 대신 FTS5 후보를 먼저 2,000건으로 추려낸 뒤 벡터 유사도 재정렬 (10만+ 에셋에서 100ms 이내 응답)
- **가중치 자동 선택**: QueryDecomposer가 쿼리 특성을 분석하여 visual/semantic/balanced/fact 프리셋을 자동 선택
- **이미지 기반 검색**: 단일/다중 이미지 입력, AND/OR 조합 검색 지원

### 2.3 클라이언트-서버 듀얼 모드 아키텍처

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│  Electron 모드 (데스크톱)   │    │  웹 모드 (브라우저 접속)    │
│  ─ 인증 우회 (로컬 전용)    │    │  ─ JWT 토큰 인증 필수      │
│  ─ IPC → Python 직접 통신   │    │  ─ HTTP API → FastAPI      │
│  ─ 로컬 DB 직접 접근        │    │  ─ 원격 DB 접근            │
│  ─ Embedded Worker 내장     │    │  ─ 다중 사용자 동시 접속   │
│  ─ [서버 구동] 토글 활성화  │    │  ─ RBAC (admin/user 분리)  │
└─────────────────────────────┘    └─────────────────────────────┘
              │                                   │
              ├── 그룹(Group) 기반 프로젝트 단위 관리 ──┤
              │                                   │
              └───── 공유 React 19 프론트엔드 ─────┘
```

**최근 추가된 사용자 관리 기능 (Firebase 2-Layer Auth):**
- **2계층 인증 구조**: 1계층(Firebase Auth)으로 개인 신원(구글/이메일 로그인)을 확인하고, 2계층(Local JWT)으로 개별 서버의 역할을 인가(admin/user)받는 듀얼 인증
- **클라우드 기반 그룹(Group) 탐색**: 초대 코드를 폐기하고, Firebase Firestore를 통해 서버 정보를 조회하여 팀 단위 그룹으로 신속히 합류
- **앱 모드 지속 및 빠른 재접속**: 재시작 시 SetupPage를 건너뛰고, 최근 접속한 그룹 카드를 제공해 클릭 한번으로 재로그인

### 2.4 클러스터링 기반 분산 워커 시스템

대규모 에셋(수만~수십만 장) 초기 인덱싱 병목을 해소하기 위한 독자적 분산 처리 시스템입니다.

```
┌═══════════════════════════════════════════════════════════════════┐
│  마스터 서버 (Master Server · FastAPI)                            │
│  ─ Phase P (Parse)를 직접 수행하여 썸네일 + 메타데이터 추출      │
│  ─ Job Queue (SQLite 기반) 관리 및 작업 분배                    │
│  ─ 워커별 세션 추적, 하트비트 수신, 처리량 모니터링              │
│  ─ Embedded Worker: 서버 자체도 유휴 시 워커로서 작업 수행       │
└═══════════════════════════════════════════════════════════════════┘
        ▲                ▲                ▲                ▲
   네트워크(썸네일     네트워크          네트워크          네트워크
    ~200KB 전송)
        ▼                ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ 워커 노드 1  │ │ 워커 노드 2  │ │ 워커 노드 3  │ │ 워커 노드 N  │
│ RTX 4090     │ │ Mac M3 Ultra │ │ RTX 3060     │ │ 남는 노트북  │
│ Prefetch: 20 │ │ Prefetch: 10 │ │ Prefetch: 5  │ │ Prefetch: 2  │
│ Phase V+E    │ │ Phase V+E    │ │ Phase V+E    │ │ Phase V+E    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

**병목 제거 핵심 전략: "Payload Offloading"**

기가바이트 단위의 원본 PSD 파일을 네트워크로 전송하면 대역폭 병목이 발생합니다. 따라서 Imagine은 **서버와 워커의 역할을 완전히 분리**했습니다:

| 역할 | 서버 (마스터) | 워커 (노드) |
|------|--------------|-------------|
| **담당 Phase** | Phase P (Parse) | Phase V (Vision) + Phase E (Embed) |
| **처리 내용** | PSD 파싱, 레이어 추출, 썸네일 생성, 메타데이터 추출 | AI 비전 분석 (VLM), 벡터 인코딩 (VV/MV) |
| **네트워크 부하** | — (로컬 처리) | 썸네일 ~200KB + JSON 메타데이터만 수신 |
| **리소스 요구** | CPU 중심 (GPU 불필요) | GPU(VRAM) 중심 |

**분산 제어 기능 상세:**

1. **동적 부하 분산 (Prefetch Pool)**
   - 각 워커가 자체 VRAM/Tier에 맞게 작업을 선점(Prefetch). 고성능 노드(RTX 4090)는 20개, 저성능 노드(CPU Only)는 2개 작업을 미리 가져감
   - 작업 처리 중에 **백그라운드 스레드에서 다음 작업 파일을 병렬 다운로드** (GPU idle 시간 최소화)

2. **하드웨어 자원 실시간 모니터링 (Resource Monitor)**
   - 30초 간격 하트비트마다 워커의 **CPU 사용률, 시스템 메모리, GPU VRAM 사용량, GPU 온도**를 수집
   - CUDA(NVIDIA) 및 MPS(Apple Silicon) 환경 모두 대응
   - 4단계 스로틀 레벨 (normal → warning → danger → critical)에 따라 자동으로 배치 속도 조절

3. **토큰 기반 간편 연결 (One-Click Setup)**
   - 복잡한 네트워크 설정 없이, 마스터 서버에서 발급한 `Worker Token`과 IP 주소만 입력하면 즉시 클러스터 합류
   - Worker Token → JWT 자동 교환, 세션 자동 갱신

4. **VRAM 효율화를 위한 단계적 모델 교체 (Phased Batching)**
   - 배치 내 모든 파일에 대해 VLM 분석 완료 → VLM 모델 언로드 → SigLIP2 로드 → VV 벡터화 완료 → SigLIP2 언로드 → Qwen3-Embedding 로드 → MV 벡터화
   - 단일 GPU에 여러 모델이 동시에 상주하는 VRAM 경합을 원천 방지

5. **Embedded Worker (서버 자체 워커)**
   - 외부 워커 노드가 없는 단독 운영 환경에서도, 서버 프로세스 자체가 유휴 상태일 때 워커 역할을 겸하여 작업 처리

---

## 3. 기술 스택

### 3.1 백엔드 (Python · ~31,900 라인)

| 컴포넌트 | 적용 기술 | 상세 |
|----------|----------|------|
| 파서 | psd-tools, Pillow | PSD 레이어 트리/텍스트/폰트 추출, PNG/JPG EXIF |
| VLM | Qwen3-VL (2B/4B/8B) | 2-Stage MC: 구조 캡션 → 태그/분류 추출 |
| VV 인코더 | SigLIP2 시리즈 | 시각 유사도 벡터 (768d / 1152d / 1664d) |
| MV 인코더 | Qwen3-Embedding (0.6B/8B) | 의미론적 벡터 (256d / 1024d / 4096d) |
| 데이터베이스 | SQLite + sqlite-vec | 메타데이터, 벡터, FTS 인덱스 통합 단일 파일 |
| 검색 엔진 | FTS5 BM25 + RRF | 16컬럼 전문 검색 + 3축 순위 융합 |
| 서버 | FastAPI + uvicorn | REST API 14개 라우터, JWT 인증, SPA 서빙 |
| 인증 | JWT (access + refresh) | 역할 기반 접근 제어 (admin/user), Rate Limiting |
| 분산 | 자체 구현 워커 데몬 | Prefetch Pool, Resource Monitor, 스케줄러 |
| 외부 접속 | Cloudflare Tunnel, mDNS | 원클릭 인터넷 접속, LAN 자동 감지 |

### 3.2 프론트엔드 (React/Electron · ~14,700 라인)

| 컴포넌트 | 적용 기술 | 상세 |
|----------|----------|------|
| UI | React 19 | 19개 컴포넌트, 5개 페이지 (Main/Admin/Worker/Login/Setup) |
| 데스크톱 | Electron 40 | IPC 브릿지, Embedded Worker 통합 |
| 빌드 | Vite 6.x | HMR 개발 서버 + 프로덕션 빌드 |
| 스타일링 | Tailwind CSS 4 | 유틸리티 퍼스트 CSS |
| 다국어 | 자체 LocaleContext | 한국어 / 영어 실시간 전환 |
| QR 접속 | qrcode.react | LAN/Tunnel URL 공유 |

### 3.3 AI 모델 매트릭스 (하드웨어 적응형 Tier 시스템)

가용 VRAM에 따라 모델 조합을 자동 보정합니다:

| 티어 | 필요 VRAM | VLM (MC) | VV Encoder | VV 차원 | MV Encoder | MV 차원 |
|------|----------|----------|-----------|---------|-----------|---------|
| **Standard** | ~6GB | moondream2 | siglip2-base | 768d | gemma-2b | 256d |
| **Pro** | 8-16GB | Qwen3-VL-4B | siglip2-so400m | 1152d | qwen3-embedding:0.6b | 1024d |
| **Ultra** | 20GB+ | qwen3-vl:8b | siglip2-giant-opt | 1664d | qwen3-embedding:8b | 4096d |

### 3.4 크로스 플랫폼 VLM 폴백 체인

| 티어 | macOS (Apple Silicon) | Windows (NVIDIA) | Linux |
|------|----------------------|------------------|-------|
| Standard | transformers | transformers | transformers |
| Pro | mlx → transformers | transformers | transformers |
| Ultra | mlx → transformers | ollama → transformers | vllm → ollama → transformers |

---

## 4. 관리 기능 (Admin & Operations)

### 4.1 Admin 대시보드 (AdminPage · 84KB)
- **워커 모니터링**: 연결된 모든 워커의 실시간 상태(GPU 온도, VRAM, CPU, 처리 속도) 통합 대시보드
- **워커별 상세 뷰**: 개별 워커의 초당/분당 처리량, 현재 작업 파일, 세션 시간 표시
- **다중 워커 계정 관리**: Admin 패널에서 워커 전용 계정 생성 및 토큰 발급
- **작업 큐 관리 (QueueManagerPanel)**: Job 생성/claim/완료/실패 상태 추적, 재처리 명령
- **등록 폴더 관리 (RegisteredFoldersPanel)**: 감시 대상 폴더 등록/해제, 폴더별 인덱싱 통계

### 4.2 리소스 모니터링 시스템
```
Resource Monitor (30초 주기 하트비트)
├── CPU 사용률 (%)
├── 시스템 메모리 (사용량/전체, %)
├── GPU VRAM (사용량/전체, %)
├── GPU 온도 (°C)
└── Throttle Level 판정
    ├── normal   : 정상 속도 처리
    ├── warning  : GPU 75°C / Memory 75% 이상 → 경고 로그
    ├── danger   : GPU 85°C / Memory 85% 이상 → 30초 대기 삽입
    └── critical : GPU 90°C / Memory 95% 이상 → 처리 일시 중단
```

### 4.3 서버 외부 접속 관리
- **LAN IP 자동 감지**: os.networkInterfaces를 통해 접속 가능 주소 자동 탐색
- **ServerInfoPanel**: Local / LAN / Tunnel URL을 드롭다운으로 한 눈에 확인
- **QR 코드 실시간 생성**: 모바일 기기에서 즉시 접속 가능한 QR 코드 자동 생성
- **Cloudflare Quick Tunnel**: 계정 불필요, 원클릭으로 인터넷 접속 터널 활성화

### 4.4 그룹 기반 사용자 관리
- 기존 초대 코드(Invite Code) 방식을 **그룹(Group) 단위로 전환**하여 조직 구조를 반영
- 관리자는 그룹 생성 시 자동으로 admin 권한 획득
- 팀원은 그룹 URL 접속 → 계정 생성으로 자동 합류
- 세션 간 앱 모드 영속화 (재시작 시 초기 설정 화면 자동 스킵)

---

## 5. 코드베이스 통계

| 지표 | 현재 값 |
|------|--------|
| 최신 버전 | v0.6.4.20260312_08 |
| 백엔드 (Python) | ~35,800 라인 |
| 프론트엔드 (JSX/JS/CSS) | ~17,500 라인 |
| 서버 API 라우터 | 14개 모듈 |
| 프론트엔드 페이지 | 5개 (Main, Admin, Worker, Login, Setup) |
| 프론트엔드 컴포넌트 | 19개 |
| DB 구조 | SQLite 단일 파일 (Docker 불필요) |
| FTS 인덱스 컬럼 | 16개 |
| 다국어 지원 | 2개 (ko-KR, en-US) |

---

## 6. 성능 벤치마크

### 6.1 배치 처리 효율성 (Standard Tier, 단일 PC)

| 배치 크기 | 20장 총 시간 | 장당 평균 | 속도 향상 |
|----------|-------------|----------|----------|
| 1 (순차) | ~244초 | 12.2초 | 1.0x |
| 5 | 14.9초 | 3.0초 | 4.1x |
| 10 | 17.0초 | 1.7초 | 7.2x |
| **20** | **17.4초** | **0.9초** | **13.6x** |

### 6.2 검색 응답 시간

| 검색 유형 | 예상 지연 시간 (10만+ 에셋) |
|----------|---------------------------|
| FTS5 키워드 매칭 | < 10ms |
| 벡터 유사도 검색 (sqlite-vec) | < 50ms |
| Triaxis 통합 랭킹 (RRF) | < 100ms 이내 E2E |

### 6.3 클러스터링 확장 효과

단일 PC의 배치 최적화(13.6배)에 더하여, 워커 노드 추가 시 처리량이 선형적으로 증가합니다:
- **워커 1대**: 100장 → ~1분 30초
- **워커 3대**: 100장 → ~30초 이내
- **워커 N대**: 처리량 ≈ N × 단일 워커 처리량 (네트워크 오버헤드 < 5%)

---

## 7. 완료된 마일스톤

### Phase 1: 구조적 파싱 (Structural Parsing)
- PSD/PNG/JPG 전용 파서 (BaseParser, PSDParser, ImageParser)
- 표준 데이터 스키마 (AssetMeta, LayerInfo, ParseResult)
- 레이어 이름 정제기(Cleaner), 썸네일 생성기
- 4단계 원자성 Ingest Pipeline

### Phase 2: 시각 벡터화 (Visual Vectorization)
- SigLIP2 VV 인코더
- SQLite + sqlite-vec 마이그레이션 (ChromaDB, PostgreSQL 완전 제거)
- FTS5 전문 검색 인덱스 (16컬럼)

### Phase 3: 서술적 비전 (Descriptive Vision)
- Qwen3-VL 2-Stage 캡션/태그/분류 생성 (MC)
- Qwen3-Embedding MV 벡터
- Triaxis 검색 (VV + MV + FTS, RRF 결합)
- 3단계 Tier 시스템 (Standard/Pro/Ultra)
- 크로스플랫폼 VLM 폴백 체인 (mlx → transformers → ollama)

### Phase 4: Electron GUI + 클라이언트-서버
- React 19 + Electron 40 데스크톱
- 가상 스크롤 그리드, Triaxis 검색 UI, 메타데이터 모달
- 이미지 기반 검색 (단일/다중, AND/OR)
- i18n (한국어/영어), Electron/Web 듀얼 모드
- FastAPI 서버 (JWT 인증, SPA 서빙, RBAC)

### Phase 4.5: 분산 워커 시스템
- 워커 데몬 (Prefetch Pool + 하트비트 + 명령 피기백)
- Resource Monitor (CPU/GPU/VRAM/온도 + 4단계 스로틀)
- 워커 세션 관리 API, 토큰 원클릭 셋업
- 단계적 배치 처리 (Phased Batching), VRAM 모델 스와핑
- Admin 워커 모니터링 (전체 통합/개별 상세)
- Embedded Worker (서버 자체 워커 겸용)

### Phase 4.6: 서버 외부 접속
- LAN IP 자동 감지, ServerInfoPanel, QR 코드 생성
- CORS 완화 (JWT 보호), Cloudflare Quick Tunnel

### v0.6.4 신규 아키텍처 및 통신 최적화
- **통합 파이프라인 (PhaseRunner)**: 기존 산발적인 인제스트 엔진을 하나로 통합, 
- **WebDAV & BufferPool 원격 처리**: 로컬 파일 시스템을 넘어서 NAS/WebDAV 서버의 원본 폴더를 직접 읽고, `BufferPool` 기반으로 썸네일만 캐시하여 디스크 낭비 없이 브라우저 단에서 즉시 처리 지원
- **Firebase Auth (2-Layer Identity)**: Google 로그인 및 SaaS 형태의 멤버 관리
- **동적 워커 세션 관리 강화**: Embedded Worker 구동 시 외부 워커를 능동 방어하고 자동 폴백하는 안전 장치
- **WebDAV Remote 썸네일 브라우징**: 데스크톱/웹 무관하게 원격지 PSD/이미지를 다운로드 없이 썸네일로 바로 열람

---

## 8. 향후 로드맵

| Phase | 목표 | 주요 항목 |
|-------|------|----------|
| **5** | UI/UX 개선 | 라이트박스 뷰어, 검색 히스토리, 드래그 앤 드롭, 키보드 단축키 |
| **6** | 검색 고도화 | 고급 필터(해상도/날짜), 스마트 컬렉션, 태그 일괄 편집 |
| **7** | 성능 최적화 | 증분 인덱싱, VLM 결과 캐싱, WebP 썸네일, 모델 자동 언로드 |
| **8** | 패키징/배포 | Windows NSIS 인스톨러, macOS DMG, 자동 업데이트, 모델 다운로드 위자드 |
| **9** | 데이터/협업 | DB 자동 백업, 내보내기/가져오기, 코멘트 히스토리, 읽기 전용 공유 |

---

## 9. 설계 의사결정

### SQLite를 택한 이유 (vs PostgreSQL)
- 개인/소규모 팀 대상 → Docker 등 인프라 진입 장벽 제거
- `sqlite-vec` 확장으로 벡터 검색까지 단일 파일 내 구현
- DB 파일 하나만 복사하면 백업/이전 완료 → 압도적 이식성

### SigLIP2를 택한 이유 (vs PE-Core, CLIP)
- Apache 2.0 라이선스 (PE-Core는 CC-BY-NC 상업 제한)
- HuggingFace transformers 네이티브 통합 + Apple Silicon MPS 검증 완료
- NaFlex: 다양한 이미지 종횡비에서도 임베딩 품질 보존

### Triaxis 3축 검색 (vs 단일 벡터)
- VV(시각): 색감, 구도, 무드 → "이것과 비슷하게 생긴 이미지"
- MV(의미): 추상적 개념, 맥락 → "사이버펑크 도시의 고독한 분위기"
- FTS(팩트): 파일명, 레이어명, 태그 → "login_button이 포함된 PSD"
- RRF 융합으로 단일 축 대비 검색 적합도 대폭 향상

---

## 10. 개발 환경

| 항목 | 값 |
|------|-----|
| 주 개발 장비 | macOS 26.2, Apple M5 Max, 32GB |
| 운용 Tier | Pro (8-16GB VRAM) |
| Python | 3.12.12 (.venv) |
| Node | v24.13.0 |
| 설정 파일 | config.yaml (티어/검색/워커/스로틀 통합 관리) |
| Repository | github.com/saintiron82/imagine |
