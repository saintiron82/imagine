# CLAUDE.md

## 언어 규칙 (MANDATORY)

**모든 응답은 반드시 한국어로 작성합니다.** 코드 주석, 커밋 메시지, 변수명 등 코드 자체는 영어를 사용하되, 사용자에게 보여주는 설명·요약·보고는 항상 한국어로 합니다.

## 금지 용어 (MANDATORY)

- **"고아(orphan)"**: 어떤 맥락에서도 사용 금지. 대신 "참조 없는(unreferenced)", "매칭되지 않는(unmatched)", "잔여(residual)" 등으로 표현할 것.

## 커밋 규칙 (MANDATORY)

**작업 단위(논리적으로 완결된 변경)마다 커밋합니다.** 사용자가 별도로 커밋을 요청하지 않아도, 하나의 작업 단위가 끝나면 자동으로 커밋합니다.

### 작업 단위 기준

| 작업 단위 | 커밋 시점 | 예시 |
|-----------|----------|------|
| 새 기능/모듈 추가 | 기능이 동작하는 상태에 도달했을 때 | i18n 인프라 생성, 새 파서 추가 |
| 기존 파일 수정 (일괄) | 동일 목적의 수정이 모든 대상 파일에 적용되었을 때 | 7개 컴포넌트 하드코딩 문자열 치환 |
| 설정/문서 변경 | 해당 변경이 완료되었을 때 | CLAUDE.md 규칙 추가, 스킬/워크플로우 생성 |
| 버그 수정 | 수정 및 검증 완료 시 | 빌드 에러 수정, 런타임 오류 수정 |
| 리팩토링 | 리팩토링 단위 완료 시 | 함수 분리, 모듈 재구성 |

### 커밋 메시지 형식

```
<type>: <간결한 설명 (영어)>

<변경 내용 상세 (선택, 영어)>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**type 종류**: `feat`, `fix`, `refactor`, `docs`, `chore`, `style`, `test`

### 규칙

- **하나의 커밋 = 하나의 논리적 작업 단위** (너무 크지도, 너무 작지도 않게)
- **빌드가 깨지는 상태로 커밋 금지** (커밋 전 빌드 확인)
- **관련 없는 변경을 하나의 커밋에 섞지 않기**
- **사용자에게 커밋 내용을 간략히 보고** (커밋 후 한국어로 요약)

## 버전 규칙 (MANDATORY)

**모든 수정 커밋 후 반드시 버전을 올립니다.**

### 형식

```
M.m.p.YYYYMMDD_NN
```

| 필드 | 설명 | 예시 |
|------|------|------|
| `M` | Major — 대규모 아키텍처 변경 | `3` → `4` |
| `m` | Minor — 새 기능 추가 | `3.5` → `3.6` |
| `p` | Patch — 버그 수정, 소규모 개선 | `3.5.0` → `3.5.1` |
| `YYYYMMDD` | 수정 날짜 | `20260213` |
| `NN` | 해당 날짜의 수정 순번 (01부터) | `01`, `02`, `03` |

### 표기 위치

- `frontend/src/components/StatusBar.jsx` — UI 하단에 표시

### 절차

1. 작업 단위 커밋 완료
2. `StatusBar.jsx`의 버전 문자열 업데이트
3. `chore: version bump to vX.X.X.YYYYMMDD_NN` 커밋

---

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 프로젝트 개요

**ImageParser**는 PSD, PNG, JPG 파일을 AI 검색 가능한 데이터로 변환하는 멀티모달 이미지 데이터 추출 및 벡터화 시스템입니다. **Triaxis 아키텍처** (VV + MV + FTS)를 사용합니다:

1. **VV (Visual Vector)**: SigLIP2로 시각적 임베딩 (이미지 픽셀 유사도 검색, `vec_files`)
2. **MV (Meaning Vector)**: Qwen3-Embedding으로 MC를 벡터화 (의미 기반 검색, `vec_text`)
3. **MC (Meta-Context Caption)**: VLM(Qwen3-VL)이 생성한 캡션/태그 (`mc_caption`, `ai_tags`)
4. **FTS (Full-Text Search)**: FTS5 BM25로 파일명/레이어명/태그 키워드 검색 (`files_fts`)

**기술 스택**:
- **Backend**: Python 3.x + `psd-tools`, `Pillow`, `transformers`, `sqlite-vec`
- **Server**: FastAPI (JWT 인증, 분산 워커 지원, REST API)
- **Frontend**: React 19 + Electron 40 + Vite + Tailwind CSS (데스크탑 + 웹 듀얼 모드)
- **Database**: SQLite + sqlite-vec (통합 메타데이터 + 벡터 저장소, Docker 불필요)
- **AI 모델**: SigLIP2 (VV), Qwen3-VL (VLM/MC), Qwen3-Embedding (MV)

## 설계 근거 (Design Rationale)

**이 프로젝트의 핵심 아키텍처 결정과 그 이유를 기록합니다.**

### 플랫폼 구조: 왜 Electron + 웹 듀얼 모드인가?

**대상 사용자**: 게임/일러스트 스튜디오의 아티스트.
**핵심 제약**: 아티스트에게 Docker, PostgreSQL, CLI 설치를 요구할 수 없다.

| 시나리오 | 필요한 것 | 해결 |
|---------|----------|------|
| 개인 아티스트 1명 | 내 PC에서 바로 실행, 설치 간편 | **Electron 앱** — 더블클릭으로 실행, 로컬 DB, 로컬 GPU |
| 스튜디오 팀 (5-20명) | 공유 에셋 검색, 한 명이 처리하면 전원이 검색 | **서버 모드** — Electron 앱에서 [Server] 토글, 나머지는 브라우저 접속 |
| 처리량 확장 | GPU 머신 여러 대로 분산 처리 | **워커 시스템** — 별도 머신에서 워커 데몬 연결 |

**같은 React 코드베이스**가 Electron(IPC) / 브라우저(HTTP API) 양쪽 모드로 동작 → 유지보수 비용 1x.
**Python 백엔드**는 Electron에서 subprocess로 직접 호출하거나, FastAPI 서버로 독립 실행 가능.

### Triaxis 검색: 왜 3축인가?

이미지 에셋 검색에서 단일 벡터는 한계가 있다:

| 축 | 검색 의도 | 예시 쿼리 | 단독으로 부족한 이유 |
|----|---------|---------|-----------------|
| **VV** (시각) | "이것과 비슷하게 생긴 이미지" | 참조 이미지 업로드 | 의미("검 든 기사") 검색 불가 |
| **MV** (의미) | "검 든 판타지 기사" | 텍스트 쿼리 | 시각적 유사성 무시 (같은 설명이라도 스타일 다름) |
| **FTS** (키워드) | "character_hero_v2.psd" | 파일명/태그 직접 검색 | AI 이해 없이 정확한 키워드 매칭만 |

**RRF (Reciprocal Rank Fusion)** 으로 3축 결과를 결합하면, 텍스트 쿼리 하나로 의미+시각+키워드를 동시에 검색 가능.
각 축의 가중치는 쿼리 유형(visual/semantic/keyword)에 따라 자동 조절 (`search.rrf.presets`).

### SQLite: 왜 PostgreSQL이 아닌가?

| 기준 | SQLite | PostgreSQL |
|------|--------|------------|
| **설치** | 없음 (Python 내장) | Docker 또는 시스템 설치 필요 |
| **배포** | `.db` 파일 1개 복사 = 전체 백업 | 덤프/복원 절차 필요 |
| **벡터 검색** | sqlite-vec (vec0 가상 테이블) | pgvector |
| **전문 검색** | FTS5 (내장) | pg_trgm / tsquery |
| **동시성** | 단일 writer (서버 모드 uvicorn 1 worker) | 다중 writer |
| **대상 규모** | ~100만 파일 (충분) | 수천만+ |

**결정**: 대상 사용자(아티스트)에게 Docker 설치를 요구하는 것은 채택 장벽.
**트레이드오프**: 서버 모드에서 동시 쓰기 제한 → uvicorn 단일 워커로 해결. 아티스트 스튜디오 규모(수만~수십만 파일)에서 SQLite 성능은 충분.

### 3-Tier 시스템: 왜 standard / pro / ultra인가?

**현실의 GPU 다양성**에 대응:

| Tier | 타겟 하드웨어 | VRAM | VLM | 품질 |
|------|-------------|------|-----|------|
| **standard** | 통합 GPU 노트북, 저사양 | ~6GB | Qwen3-VL-**2B** | 기본 캡션/태그 |
| **pro** | Mac M-시리즈, RTX 3060+ | 8-16GB | Qwen3-VL-**4B** | 상세 캡션, 정확한 분류 |
| **ultra** | RTX 4090, A100 서버 | 20GB+ | Qwen3-VL-**8B** | 최고 품질 |

**핵심 설계**: VV(SigLIP2)와 MV(0.6B)를 standard↔pro에서 **동일 모델**로 통일 → Tier 전환 시 벡터 재생성 불필요.
VLM만 Tier에 따라 교체되므로, Tier 업그레이드 = MC만 재생성 (VV/MV 유지).

**왜 3단계인가**: 2단계(light/full)로는 8GB Mac과 24GB RTX를 같은 설정으로 쓸 수 없고, 4단계 이상은 관리 복잡도 대비 이점이 없음.

### VLM 선택: 왜 Qwen3-VL인가?

| 모델 | 크기 범위 | 구조화 출력 | 다국어(한/영) | 라이선스 | 크로스 플랫폼 |
|------|----------|-----------|-------------|---------|------------|
| **Qwen3-VL** | 2B/4B/8B | ✅ JSON | ✅ | Apache 2.0 | transformers + MLX + Ollama |
| LLaVA-Next | 7B/13B | 제한적 | △ | 혼합 | transformers만 |
| InternVL2 | 2B/8B/26B | ✅ | △ 영어 중심 | 혼합 | transformers만 |
| Phi-4-Vision | 14B | ✅ | △ | MIT | transformers만 |

**결정 근거**:
1. **2B/4B/8B 3단 라인업** — Tier 시스템과 1:1 매핑 가능 (다른 모델은 크기 간격이 큼)
2. **한국어+영어 캡션 품질** — 한국 아티스트 대상, 한국어 MC가 MV 검색 품질에 직접 영향
3. **크로스 플랫폼** — transformers(범용) + MLX(macOS) + Ollama(Windows) 모두 지원
4. **Apache 2.0** — 상업적 제약 없음

### MV 선택: 왜 Qwen3-Embedding인가?

| 모델 | 크기 | 차원 | MRL | 다국어 | 라이선스 |
|------|------|------|-----|--------|---------|
| **Qwen3-Embedding** | 0.6B / 8B | 1024 / 4096 | ✅ | ✅ 한/영 | Apache 2.0 |
| BGE-M3 | 0.6B | 1024 | ❌ | ✅ | MIT |
| Jina-Embeddings-v3 | 0.6B | 1024 | ✅ | ✅ | CC-BY-NC |
| E5-Mistral | 7B | 4096 | ❌ | △ | MIT |

**결정 근거**:
1. **MRL 지원** — 차원 truncation으로 저장 공간/속도 최적화 가능 (미래 활용)
2. **0.6B + 8B 이중 라인업** — standard/pro는 0.6B(가벼움), ultra는 8B(고품질)
3. **한국어 임베딩 품질** — MC가 한국어일 때 MV 검색 정확도에 직접 영향
4. **Qwen3-VL과 같은 패밀리** — 토크나이저/언어 모델 기반이 유사하여 MC→MV 변환 시 의미 손실 최소화 기대

### FTS5: 왜 Elasticsearch가 아닌가?

| 기준 | FTS5 (SQLite 내장) | Elasticsearch |
|------|-------------------|---------------|
| **설치** | 없음 (SQLite에 포함) | JVM + 별도 서비스 |
| **메모리** | ~0 (인덱스만) | 최소 1GB+ |
| **역할** | Triaxis 3축 중 보조축 (RRF 결합) | 단독 검색 엔진 |
| **기능** | BM25 + prefix match | 형태소 분석, 퍼지 매칭 |

**결정**: FTS는 Triaxis에서 **보조축**. VV+MV가 AI 검색의 핵심이고, FTS는 파일명/태그 **정확 키워드 매칭** 전담.
이 역할에 Elasticsearch의 복잡도는 과잉. FTS5 BM25 + RRF 결합으로 충분.

### 워커 배치: 왜 서버 Phase 큐 분리가 아닌 워커 내부 배치인가?

**검토한 대안**:
- (A) 서버가 Phase별 큐를 관리: Job을 P큐 → V큐 → VV큐 → MV큐로 이동
- (B) 워커 내부에서 Claim한 N개를 Phase별로 묶어 처리

**사용자 피드백**: *"로컬에 있지도 않는 파일을 처리하려고 모델 스위칭 비용이 더 들 것 같다"*

| 기준 | (A) 서버 Phase 큐 | (B) 워커 내부 배치 ✅ |
|------|------------------|-------------------|
| **서버 복잡도** | 큐 4개 + Phase 간 상태 전이 관리 | 기존 1 Job = 1 파일 구조 유지 |
| **네트워크** | Phase마다 중간 결과 서버↔워커 전송 | 최초 다운로드 1회, 최종 업로드 1회 |
| **모델 언로드** | 워커가 Phase 전담이면 모델 상주 가능하나, 1워커=1Phase → 워커 4배 필요 | 워커 내부에서 Phase 순서대로 로드/언로드 |
| **장애 복구** | Phase 간 중간 상태가 서버 DB에 → 복잡 | Job 단위 실패 = 단순 재시도 |

**결정**: 서버 변경 없이 워커 내부에서 `process_batch_phased()`로 Phase별 배치 처리.
서버는 "Job 할당/완료" 만 관리하므로 단순하고 안정적.

### SigLIP2 so400m-naflex: 왜 이 VV 모델인가?

> 이 항목은 **모델 선택 근거 (2026-02-20 조사 결과)** 섹션에 상세 기록됨.

요약:
1. **Apache 2.0** — PE-Core(최강 성능)는 CC-BY-NC 비상업
2. **transformers 네이티브** — PE-Core는 자체 라이브러리/OpenCLIP 필요
3. **MPS 검증** — macOS M5에서 동작 확인 (PE-Core는 미검증)
4. **NaFlex** — 다양한 종횡비 PSD/PNG에 유리 (고정 크기 리사이즈 불필요)
5. **성능 차이 0.5% 미만** — 라이선스+생태계 이점이 크게 상회
6. **재평가 시점**: PE-Core transformers 통합 + 라이선스 변경 시, 또는 SigLIP3 출시 시

## 용어 사전 (MANDATORY)

**이 프로젝트의 공식 용어입니다. 코드 주석, 문서, 대화에서 반드시 이 용어를 사용하세요.**

### 핵심 약어

| 약어 | 정식 명칭 | 설명 | DB 테이블/컬럼 |
|------|----------|------|---------------|
| **VV** | Visual Vector | SigLIP2가 이미지 픽셀로부터 생성하는 시각 임베딩 벡터. 이미지↔이미지 유사도 검색에 사용 | `vec_files.embedding` |
| **MV** | Meaning Vector | Qwen3-Embedding이 MC 텍스트로부터 생성하는 의미 임베딩 벡터. 텍스트↔텍스트 유사도 검색에 사용 | `vec_text.embedding` |
| **MC** | Meta-Context Caption | VLM이 이미지 + 파일 메타데이터(경로, 레이어 등) 컨텍스트를 보고 생성한 캡션과 태그. "Meta-Context"는 단순 AI 캡션이 아닌 메타데이터 맥락 포함을 의미. MV의 입력 소스 | `files.mc_caption`, `files.ai_tags` |
| **FTS** | Full-Text Search | FTS5 BM25 기반 키워드 전문 검색. 파일명, 레이어명, 태그 등 메타데이터 검색 | `files_fts` |
| **VLM** | Vision-Language Model | 이미지를 보고 자연어를 생성하는 AI 모델 (현재: Qwen3-VL). MC를 생성하는 주체 | — |
| **RRF** | Reciprocal Rank Fusion | VV, MV, FTS 3축 검색 결과를 하나로 결합하는 랭킹 알고리즘 | — |
| **MRL** | Matryoshka Representation Learning | 고차원 임베딩을 저차원으로 잘라도 품질이 유지되는 학습 기법. MV 차원 조절에 사용 | — |

### 데이터 흐름 관계

```
이미지 파일 ──→ [Parser] ──→ 메타데이터 (AssetMeta)
                                 │
                                 ▼
                            [VLM: Qwen3-VL]
                                 │
                                 ├──→ MC (mc_caption + ai_tags)  ──→ [Qwen3-Embedding] ──→ MV (vec_text)
                                 │
이미지 픽셀  ──→ [SigLIP2] ──→ VV (vec_files)
                                 │
메타데이터   ──→ [FTS5 Indexer] ──→ FTS (files_fts)
                                 │
                                 ▼
                         [Triaxis Search: VV + MV + FTS → RRF 결합]
```

### 모델 역할 매핑

| 모델 | 입력 | 출력 | 역할 |
|------|------|------|------|
| **SigLIP2** | 이미지 픽셀 | VV 벡터 | 이미지를 시각적으로 벡터화 (비전 인코더) |
| **Qwen3-VL** | 이미지 + 프롬프트 | MC 텍스트 | 이미지를 보고 캡션/태그 생성 (VLM, 생성형) |
| **Qwen3-Embedding** | MC 텍스트 | MV 벡터 | 텍스트를 의미 벡터로 변환 (텍스트 인코더, 비전 없음) |

### 금지 용어 → 올바른 용어

| 금지 | 올바른 표현 |
|------|-----------|
| V-axis, V축 | **VV** (Visual Vector) |
| S-axis, S축, Semantic축 | **MV** (Meaning Vector) |
| M-axis, M축 | **FTS** (Full-Text Search) |
| ai_caption | **mc_caption** (MC = Meta-Context Caption, AI 캡션뿐 아니라 파일 메타데이터 컨텍스트 포함) |
| 시각 임베딩, visual embedding | **VV** |
| 텍스트 임베딩, text embedding | **MV** |
| 캡션/태그 | **MC** (VLM이 생성한 경우) |

## 개발 명령어

### Backend (Python)

```powershell
# 의존성 설치
python -m pip install -r requirements.txt

# 단일 파일 처리
python backend/pipeline/ingest_engine.py --file "path/to/image.psd"

# 여러 파일 배치 처리
python backend/pipeline/ingest_engine.py --files "[\"file1.psd\", \"file2.png\"]"

# 디렉토리 DFS 탐색 (하위 폴더 재귀 스캔 + 스마트 스킵)
python backend/pipeline/ingest_engine.py --discover "C:\path\to\assets"

# DFS 탐색 (스마트 스킵 비활성화, 전체 재처리)
python backend/pipeline/ingest_engine.py --discover "C:\path\to\assets" --no-skip

# 디렉토리 감시 (초기 DFS 스캔 + 실시간 변경 감지)
python backend/pipeline/ingest_engine.py --watch "C:\path\to\assets"

# Triaxis 검색 (VV + MV + FTS, SQLite + sqlite-vec)
# 프론트엔드 Electron 앱에서 검색 UI 사용
# 또는 백엔드 API 직접 호출:
python -c "from backend.search.sqlite_search import SqliteVectorSearch; s=SqliteVectorSearch(); print(s.triaxis_search('fantasy character'))"

# 특정 테스트 실행
python test_image_parser.py
python test_psd_parser_mock.py
```

### Frontend (Electron + React)

```powershell
cd frontend

# 의존성 설치
npm install

# 개발 모드 실행 (Electron + 핫 리로드)
npm run electron:dev

# 프로덕션 빌드
npm run build

# Electron 실행 파일 빌드
npm run electron:build
```

### Server (FastAPI)

```bash
# 서버 시작 (개발 모드, 핫 리로드)
python -m backend.server.app
# 또는
uvicorn backend.server.app:app --host 0.0.0.0 --port 8000 --reload

# 워커 데몬 시작 (서버에 접속하여 작업 처리)
python -m backend.worker.worker_daemon --server http://서버IP:8000 --username USER --password PASS

# 헬스 체크
curl http://localhost:8000/api/v1/health
```

### 진단 스크립트

```powershell
# 디렉토리 배치 분석
python scripts/batch_analyze.py

# 단일 이미지 진단
python scripts/diagnose_image.py "path/to/image.psd"

# CLI에서 이미지 검색
python scripts/search_images.py "검색어"
```

## 클라이언트-서버 아키텍처 (v4.x)

**하나의 React 프론트엔드가 두 가지 모드로 동작합니다.**

### 듀얼 모드 구조

```
┌─────────────────────────────────────────────────────────────┐
│                   동일한 React 프론트엔드                       │
│                                                             │
│  ┌───────────────────────┐   ┌────────────────────────────┐ │
│  │   Electron 모드 (앱)    │   │   Web 모드 (브라우저)       │ │
│  │                       │   │                            │ │
│  │ • Auth 바이패스         │   │ • JWT 로그인 필수           │ │
│  │ • 자동 admin 권한       │   │ • 역할 기반 (admin/user)   │ │
│  │ • IPC → Python 직접    │   │ • HTTP API → FastAPI       │ │
│  │ • 로컬 DB 직접 접근     │   │ • 서버 DB 간접 접근         │ │
│  │                       │   │                            │ │
│  │ [Server] 버튼으로      │   │  서버에 접속하여 사용        │ │
│  │  FastAPI 서버 내장 시작  │   │                            │ │
│  └───────────────────────┘   └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 모드 판별

- **`isElectron`** (`api/client.js:14`): `window.electron` 존재 여부로 판별
- **`skipAuth`** (`AuthContext.jsx:20`): Electron이면 `true` → 인증 생략, `{ username: 'local', role: 'admin' }` 자동 부여
- **`bridge.js`**: `isElectron`이면 IPC, 아니면 HTTP API 호출 (검색, 메타데이터 등)

### 서버 모드 (Electron → FastAPI)

- **위치**: Electron 앱 헤더 바 우측 `[Server]` 토글 버튼
- **동작**: `window.electron.server.start({ port })` → FastAPI 프로세스 기동
- **SPA 서빙**: `app.py:122-137`에서 `frontend/dist` 빌드 결과를 정적 파일로 서빙
- 서버가 켜지면 다른 사용자가 `http://서버IP:포트`로 브라우저 접속 가능

### 인증 시스템 (JWT)

| 구성요소 | 파일 | 설명 |
|---------|------|------|
| 라우터 | `backend/server/auth/router.py` | 로그인/회원가입/토큰 갱신 API |
| 스키마 | `backend/server/auth/schemas.py` | Pydantic 모델 (LoginRequest 등) |
| JWT | `backend/server/auth/jwt.py` | 토큰 발급/검증 |
| 의존성 | `backend/server/deps.py` | `get_current_user`, `require_admin` |
| DB | `backend/db/sqlite_schema_auth.sql` | `users`, `invite_codes`, `worker_tokens`, `worker_sessions` 테이블 |
| 프론트 | `frontend/src/contexts/AuthContext.jsx` | React 인증 Context |
| API | `frontend/src/api/client.js` | JWT 자동 첨부 + 401 시 refresh |

### 역할 기반 접근

| 역할 | 접근 가능 탭 | 서버 API 접근 |
|------|------------|--------------|
| **admin** | Search, Archive, Worker, Admin | 전체 API + 사용자/워커/초대 관리 |
| **user** | Search, Archive, Worker | 파이프라인 실행, 검색, 내 워커 관리 |
| **Electron (local)** | 전체 (auth 바이패스) | IPC 직접 호출 (서버 불필요) |

---

## 워커 시스템 (v10.x)

**분산 이미지 처리를 위한 워커 풀 시스템. Phase별 배치 처리 + 모델 언로드로 GPU 메모리 최적화.**

### 개요

서버가 작업(Job) 큐를 관리하고, 워커가 서버에 접속하여 작업을 가져가 처리하는 구조.

```
Server (FastAPI)                    Worker (Python daemon / Electron IPC)
┌──────────────┐                   ┌──────────────────────┐
│ Job Queue    │◄── claim(N) ────  │ Prefetch Pool        │
│ (SQLite)     │── jobs[] ───────► │ (deque, capacity×2)  │
│              │                   │                      │
│ worker_      │◄── heartbeat ──── │ 30초마다 하트비트      │
│ sessions     │── command ──────► │ stop/block 명령 수신   │
│              │                   │                      │
│ job_queue    │◄── complete ────  │ Phase별 배치 처리     │
│ (throughput) │                   │ (P→V→VV→MV→Upload)   │
└──────────────┘                   └──────────────────────┘
```

### Phase별 배치 처리 (v10.0, process_batch_phased)

**기존**: 파일 1개씩 4 Phase 순차 처리 → 3개 모델 동시 메모리 상주
**변경**: N개 파일을 Phase별로 묶어 배치 처리 → Phase당 모델 1개만 메모리

```
Claim 5 jobs → Download/resolve all →
  Phase P:  Parse(1,2,3,4,5)       → report progress
  Phase V:  Vision(1,2,3,4,5)      → report progress    [VLM only in memory]
  ── unload VLM ──
  Phase VV: embed_vv(1,2,3,4,5)    → report progress    [SigLIP2 only in memory]
  ── unload SigLIP2 ──
  Phase MV: embed_mv(1,2,3,4,5)    → report progress    [Qwen3-Embed only in memory]
  ── unload Qwen3-Embed ──
  Upload: complete_job(1), complete_job(2), ...
```

**이점**:
- Phase당 모델 1개만 메모리 → GPU 메모리 효율 극대화
- 서브배치 추론 활용: VV `encode_image_batch(batch=8)`, MV `encode_batch(batch=16)`
- MC(VLM)는 MLX/transformers 제약으로 batch_size=1 (순차 처리)

### 모델 언로드

| 메서드 | 대상 | 시점 |
|--------|------|------|
| `_unload_vlm()` | VLM (Qwen3-VL) | Phase V 완료 후 |
| `_unload_vv()` | SigLIP2 인코더 | Phase VV 완료 후 |
| `_unload_mv()` | Qwen3-Embedding | Phase MV 완료 후 |

각 언로드 후 `gc.collect()` + `torch.cuda.empty_cache()` / `torch.mps.empty_cache()` 호출.

### IPC 이벤트 프로토콜 (Electron ↔ Python)

**배치 이벤트** (worker_ipc.py → main.cjs → preload.cjs → App.jsx):

| 이벤트 | 데이터 | 설명 |
|--------|--------|------|
| `batch_start` | `{batch_size: N}` | 배치 처리 시작 |
| `batch_phase_start` | `{phase, count}` | Phase 시작 ("parse"/"vision"/"embed_vv"/"embed_mv") |
| `batch_file_done` | `{phase, file_name, index, count, success}` | Phase 내 파일 1개 완료 |
| `batch_phase_complete` | `{phase, count}` | Phase 완료 |
| `batch_complete` | `{batch_size, completed, failed}` | 배치 전체 완료 |
| `job_done` | `{job_id, file_path, file_name, success}` | Job 최종 완료 (기존 호환) |

**데이터 흐름**:
```
worker_daemon.py::process_batch_phased(5 jobs)
  ↓ progress_callback("file_done", {...})
worker_ipc.py::_batch_progress_cb()
  ↓ stdout: {"event":"batch_file_done","phase":"vision","file_name":"hero.psd","index":3,"count":5}
main.cjs::processWorkerOutput()
  ↓ sendWorkerEvent('worker-batch-file-done', parsed)
preload.cjs → App.jsx::onBatchFileDone
  ↓ setWorkerProgress({currentPhase: "vision", phaseIndex: 3, phaseCount: 5})
StatusBar: "MC 3/5 | 15/741 | 4.2/min | ~12m"
```

### 스마트 Prefetch 풀

- **풀 크기**: `batch_capacity × 2` (예: capacity=8 → 풀에 16개 유지)
- **동작**: 현재 배치 처리 중 백그라운드 스레드가 부족분만큼 claim
- **설정**: `config.yaml > worker.batch_capacity` (기본값: 5)

```
기존: claim(5) → process(5) → claim(5) → ...
변경: fill_pool(16) → take_batch(8) + refill_thread → process(8) → ...
```

### 하트비트 + 명령 피기백

- 워커가 30초마다 서버에 하트비트 전송 (메트릭 보고)
- 서버는 응답에 `pending_command`를 포함 (stop/pause/block)
- 명령은 **일회성 소비**: 한 번 전달되면 DB에서 NULL로 리셋
- `pool_hint`: 서버가 권장하는 풀 크기 (batch_capacity × 2)
- **중요**: IPC 모드(Electron)에서도 `_connect_session()` / `_heartbeat()` / `_disconnect_session()` 호출 필수

### Admin 모니터링 API (v10.2+)

**큐 통계 + 처리속도** (`GET /api/v1/admin/queue/stats`):
- `throughput`: 슬라이딩 윈도우 처리속도 (files/min) — 1분 우선, 5분 폴백
- `recent_1min`, `recent_5min`: 각 윈도우 내 완료 파일 수
- `pending`, `assigned`, `processing`, `completed`, `failed`: 상태별 카운트

**워커별 처리속도** (`GET /api/v1/admin/workers`):
- 각 워커에 `throughput` 필드 추가 (개별 files/min)
- `job_queue.assigned_to` 기준으로 per-user 집계
- 프론트엔드: WorkersPanel에 종합 속도 + 워커별 속도 컬럼 표시

### 워커 세션 관리 API

| 엔드포인트 | 역할 | 권한 |
|-----------|------|------|
| `POST /api/v1/workers/connect` | 워커 세션 시작 | 인증된 사용자 |
| `POST /api/v1/workers/heartbeat` | 하트비트 + 명령 수신 | 인증된 사용자 |
| `POST /api/v1/workers/disconnect` | 정상 종료 알림 | 인증된 사용자 |
| `GET /api/v1/workers/my` | 내 워커 목록 | 인증된 사용자 |
| `POST /api/v1/workers/{id}/stop` | 내 워커 정지 | 본인 소유만 |
| `GET /api/v1/admin/workers` | 전체 워커 목록 | Admin |
| `POST /api/v1/admin/workers/{id}/stop` | 워커 정지 명령 | Admin |
| `POST /api/v1/admin/workers/{id}/block` | 워커 차단 (재접속 불가) | Admin |

### 워커 토큰 (원클릭 셋업)

- Admin이 워커 토큰 생성 → 토큰 포함 셋업 스크립트 제공
- 외부 PC에서 스크립트 실행 → 자동 환경 설정 + 워커 데몬 시작
- `backend/server/routers/worker_setup.py`: 토큰 생성/관리 API

### 프론트엔드 UI

| 위치 | 컴포넌트 | 설명 |
|------|---------|------|
| Admin 탭 → Workers | `WorkersPanel` | 전체 워커 목록 (이름, 상태, 배치용량, 작업수, 현재태스크, 정지/차단) |
| Worker 탭 상단 | `MyWorkersSection` | 내 워커 현황 (이름, 하트비트, 작업수, 정지) |
| Worker 탭 | `ConnectMyPC` | 워커 토큰 기반 원클릭 셋업 스크립트 다운로드 |
| Admin 탭 → Worker Tokens | 토큰 관리 | 토큰 생성/폐기/목록 |

### DB 테이블 (`sqlite_schema_auth.sql`)

```sql
worker_sessions (id, user_id, worker_name, hostname, status, batch_capacity,
                 jobs_completed, jobs_failed, current_job_id, current_file,
                 current_phase, pending_command, connected_at, last_heartbeat,
                 disconnected_at)
-- status: 'online' | 'offline' | 'blocked'
-- pending_command: NULL | 'stop' | 'pause' | 'block'
```

### 워커 설정 (`config.yaml`)

```yaml
worker:
  batch_capacity: 5        # 배치 처리 능력 (파일 수)
  heartbeat_interval: 30   # 하트비트 주기 (초)
  poll_interval: 10        # 작업 없을 때 대기 (초)
```

---

## 아키텍처 & 데이터 흐름

### 데이터 파이프라인 (Ingest → Vector DB)

```
원본 이미지 파일 (PSD/PNG/JPG)
    ↓
[--discover] DFS 폴더 탐색 (discover_files)
    ├─ 재귀 DFS로 지원 파일 수집
    ├─ 폴더 메타데이터 계산 (folder_path, folder_depth, folder_tags)
    └─ 스마트 스킵 (modified_at 비교)
    ↓
ParserFactory.get_parser(file_path)
    ↓
[BaseParser 하위 클래스: PSDParser | ImageParser]
    ├─ 메타데이터, 레이어, 텍스트 추출
    ├─ 썸네일 생성 (utils/thumbnail_generator.py)
    ├─ 레이어 이름 정제 (parser/cleaner.py)
    └─ AssetMeta 반환 (parser/schema.py)
    ↓
폴더 메타데이터 주입 (process_file)
    ├─ folder_path: 상대 경로 (e.g., "Characters/Hero")
    ├─ folder_depth: 깊이 (0 = 루트)
    └─ folder_tags: 폴더명 태그 (e.g., ["Characters", "Hero"])
    ↓
자동 번역 (deep-translator)
    ├─ semantic_tags → translated_tags (ko/en)
    ├─ text_content → translated_text (ko/en)
    └─ layer_tree → translated_layer_tree (ko/en)
    ↓
Phase V: AI Vision (vision_factory.py)
    ├─ VLM 캡션/태그/분류 생성 (Qwen3-VL, tier별 backend)
    ├─ 2-Stage Pipeline: 빠른 분류 → 상세 캡션
    └─ 서브배치마다 즉시 DB 저장 (mc_caption, ai_tags → files)
    ↓
Phase E: Embedding (siglip2_encoder.py + text_embedding.py)
    ├─ SigLIP2 → VV 생성 (이미지 시각 벡터, tier별 차원)
    ├─ Qwen3-Embedding → MV 생성 (MC 텍스트 의미 벡터)
    └─ 서브배치마다 즉시 DB 저장 (VV → vec_files, MV → vec_text)
    ↓
Phase S: Summary (완료 확인)
    ├─ [OK] emit (프론트엔드 processedCount 추적용)
    └─ 실제 저장은 P/V/E에서 이미 완료
```

### 파서 선택 (Factory Pattern)

`backend/pipeline/ingest_engine.py`:
- **PSDParser**: `.psd` 파일 처리 (`psd-tools` 사용)
- **ImageParser**: `.png`, `.jpg`, `.jpeg` 처리 (`Pillow` 사용)
- 각 파서는 `can_parse(file_path)` 클래스 메서드로 자동 감지
- **CLI 입력 모드**: `--file` (단일), `--files` (배치 JSON), `--discover` (DFS 폴더 탐색), `--watch` (감시+초기스캔)
- **스마트 스킵**: `--discover`/`--watch` 시 `modified_at` 비교로 변경되지 않은 파일 자동 건너뜀 (`--no-skip`으로 비활성화)

### 스키마 (단일 진실 공급원, Single Source of Truth)

`backend/parser/schema.py`:
- **AssetMeta**: 모든 이미지 타입에 대한 통합 메타데이터 모델
- **LayerInfo**: 개별 레이어 구조 (PSD 전용)
- **ParseResult**: 성공 상태, 오류, 경고를 포함하는 래퍼

모든 파서는 `AssetMeta`를 포함하는 `ParseResult`를 반환해야 합니다.

### SQLite Database (v3.1: Triaxis Data Storage)

> 상세 DB/검색/모델/Tier 스펙은 아래 **인프라 스펙 (MANDATORY)** 섹션을 참조하세요.

`backend/db/sqlite_client.py` (SQLiteDB):
- **files 테이블**: 파일 메타데이터 + AI 생성 필드
  - `mc_caption`, `ai_tags`: VLM이 생성한 MC (2-Stage Vision)
  - `image_type`, `scene_type`, `art_style`: VLM 분류 필드
  - `folder_path`, `folder_depth`, `folder_tags`: DFS 폴더 탐색 메타데이터
- **vec_files**: SigLIP2 VV (sqlite-vec)
- **vec_text**: Qwen3-Embedding MV (sqlite-vec)
- **files_fts**: FTS5 전문 검색 인덱스 (BM25 키워드)

`backend/search/sqlite_search.py` (SqliteVectorSearch):
- **triaxis_search()**: VV + MV + FTS 3축 통합 검색 (RRF 결합)
- **vector_search()**: SigLIP2 VV 시각 유사도 검색
- **text_vector_search()**: Qwen3-Embedding MV 의미 검색
- **fts_search()**: FTS5 BM25 키워드 검색

## 유닛 개발 프로토콜 (필수)

**이 프로젝트는 엄격한 5단계 유닛 개발 워크플로우를 사용합니다**. 모든 개발 작업은 이 프로토콜을 따라야 합니다 (`INSTRUCT.md` 및 `.agent/skills/unit_dev_agent/SKILL.md` 참조).

### 필수 워크플로우 명령어

```
/unit-start    # 새 유닛 시작 (5단계 프로세스 트리거)
/unit-status   # 현재 진행 상태 확인
/unit-done     # 유닛 완료 및 회고 작성
/troubleshoot  # 문제 및 해결책 기록
/build         # 의존성 설치
```

### 5단계 프로세스 (유닛당)

각 유닛(U-001, U-002 등)은 모든 단계를 완료해야 합니다:

1. **목표 (Goal)**: 측정 가능한 성공 기준 정의
2. **정의 (Definition)**: 상세 명세 작성
3. **개발 (Development)**: 코드 구현
4. **테스트 (Test)**: 실행 가능한 테스트 명령어로 검증
5. **회고 (Retrospective)**: 배운 점 및 개선 사항 문서화

**핵심 규칙**:
- ❌ 테스트나 회고를 절대 건너뛰지 마세요
- ❌ 현재 유닛이 테스트를 통과하기 전에 다음 유닛을 시작하지 마세요
- ❌ `docs/troubleshooting.md`에 실패 사항을 기록하지 않고 진행하지 마세요
- ✅ 항상 워크플로우 명령어를 사용하세요 (`/unit-start`, `/unit-done`)

### 현재 유닛 목록

- **U-001**: 프로젝트 초기화 ✅
- **U-002**: 데이터 스키마 정의 ✅
- **U-003**: 데이터 정제 모듈 ✅
- **U-004**: 기본 파서 인터페이스 ✅
- **U-005**: 이미지 파서 (PNG/JPG) ✅
- **U-006**: PSD 파서 ✅
- **U-007**: 파이프라인 통합 ✅
- **U-008**: 통합 테스트 (진행 중)

## 주요 기술 패턴

### PSD 파싱 (`psd-tools`)

```python
from psd_tools import PSDImage

psd = PSDImage.open('file.psd')

# 전체 합성 이미지 (모든 효과, 레이어, 마스크 적용)
composite = psd.composite()
composite.save('output.png')

# 레이어 순회
for layer in psd.descendants():
    print(f"이름: {layer.name}, 종류: {layer.kind}")
    # 종류: 'group', 'pixel', 'type', 'shape', 'smartobject', 'adjustment'

    if layer.kind == 'type':
        # 텍스트 레이어에서 텍스트 추출
        text_data = layer.engine_dict
        text = text_data.get('Editor', {}).get('Text', '')
```

**중요**: `layer.composite()`는 조정 레이어 효과 없이 개별 레이어만 렌더링합니다. 전체 충실도 미리보기는 `psd.composite()`를 사용하세요.

### 번역 청킹 전략

`backend/pipeline/ingest_engine.py`는 API 호출을 최소화하기 위해 **배치 번역**을 사용합니다:
- 모든 텍스트를 구분자 `" ||| "`로 결합
- 4000자 청크로 분할
- 청크 번역 후 다시 분할
- DFS 순회를 통해 레이어 트리의 순서 유지

### 메모리 관리

- **VV 인코더**: 첫 사용 시 SigLIP2 모델 지연 로드
- **CUDA 정리**: 이미지 10개마다 `torch.cuda.empty_cache()` 실행
- **전역 싱글톤**: `_global_indexer`가 장시간 실행 프로세스에서 모델 재로딩 방지

## 주요 파일 위치

| 경로 | 목적 |
|------|------|
| `backend/parser/schema.py` | **표준 데이터 스키마** (AssetMeta) |
| `backend/pipeline/ingest_engine.py` | **4단계 처리 파이프라인** (메인 진입점) |
| `backend/db/sqlite_client.py` | **SQLite 클라이언트** (메타데이터 + VV + MV 저장) |
| `backend/db/sqlite_schema.sql` | SQLite 스키마 정의 |
| `backend/search/sqlite_search.py` | **Triaxis 검색 엔진** (VV + MV + FTS) |
| `backend/search/rrf.py` | RRF 가중치 프리셋 (query_type별) |
| `backend/search/query_decomposer.py` | LLM 쿼리 분류기 (query_type 판별) |
| `backend/vision/vision_factory.py` | VLM 백엔드 자동 선택 (Factory) |
| `backend/vision/analyzer.py` | Transformers VLM 어댑터 (2-Stage) |
| `backend/vector/siglip2_encoder.py` | SigLIP2 VV 인코더 |
| `backend/vector/text_embedding.py` | Qwen3-Embedding MV 인코더 (Transformers/Ollama) |
| `backend/api_search.py` | 프론트엔드 검색 API 브리지 |
| `backend/server/app.py` | **FastAPI 서버** (라우터 등록, SPA 서빙) |
| `backend/server/deps.py` | 서버 의존성 (`get_current_user`, `require_admin`) |
| `backend/server/auth/router.py` | JWT 인증 API (로그인/회원가입/갱신) |
| `backend/server/routers/workers.py` | **워커 세션 API** (connect/heartbeat/admin) |
| `backend/server/routers/pipeline.py` | 파이프라인 API (업로드/claim/완료) |
| `backend/server/routers/worker_setup.py` | 워커 토큰 + 원클릭 셋업 API |
| `backend/server/queue/manager.py` | **작업 큐 관리자** (Job 생성/claim/완료) |
| `backend/worker/worker_daemon.py` | **워커 데몬** (prefetch 풀 + 배치 처리 + 하트비트) |
| `backend/worker/worker_ipc.py` | **워커 IPC 브리지** (Electron ↔ Python JSON 프로토콜) |
| `backend/worker/config.py` | 워커 설정 (batch_capacity, heartbeat 등) |
| `backend/db/sqlite_schema_auth.sql` | 인증 DB 스키마 (users, worker_sessions 등) |
| `frontend/src/api/client.js` | API 클라이언트 (JWT 자동 첨부, isElectron 판별) |
| `frontend/src/api/admin.js` | Admin/Worker API 함수 |
| `frontend/src/contexts/AuthContext.jsx` | React 인증 Context |
| `frontend/src/services/bridge.js` | Electron/Web 모드 브리지 (IPC ↔ HTTP) |
| `backend/setup/installer.py` | **통합 설치 프로그램** |
| `config.yaml` | **Tier/검색/배치 설정** (단일 소스) |
| `output/thumbnails/` | 썸네일 이미지 (gitignore됨) |
| `docs/troubleshooting.md` | **모든 문제에 대한 필수 로깅** |
| `INSTALLATION.md` | **신규 설치 가이드** |
| `frontend/src/i18n/` | **프론트엔드 로컬라이제이션 시스템** |
| `frontend/src/i18n/locales/en-US.json` | 영어 번역 파일 |
| `frontend/src/i18n/locales/ko-KR.json` | 한국어 번역 파일 |

**핵심 문서** (작업 시 반드시 참조):

| 문서 | 용도 | 참조 시점 |
|------|------|----------|
| `phase.md` | **개발 로드맵** — Phase 1~9 체크리스트, 완료/미완료 현황 | 새 기능 개발 전, 진행 상황 파악 시 |
| `docs/project_report.md` | **프로젝트 종합 보고서** — 아키텍처, 기술 스택, 완료 Phase 상세, 코드 통계 | 프로젝트 전체 구조 파악 시 |
| `docs/V3.1.md` | **핵심 설계 명세** — VV/MC/MV/FTS 파이프라인, MC 2-Stage, MV 입력 포맷, 검색 로직 | 파이프라인/검색 로직 수정 시 |
| `docs/triaxis_search_architecture.md` | **검색 시스템 해설** — 3축(VV/MV/FTS) 동작 원리, RRF 공식, 점수 범위, DB 스키마 | 검색 관련 작업 시 |
| `docs/future_roadmap.md` | **미래 계획 통합** — 클라우드 확장, DB 최적화, 벤치마크, ECM 설계 | 아키텍처 확장/최적화 논의 시 |
| `docs/platform_optimization.md` | **플랫폼별 VLM 최적화** — MLX/vLLM/Ollama/Transformers 폴백 체인 | VLM 백엔드 관련 작업 시 |
| `docs/troubleshooting.md` | **트러블슈팅 기록** — 알려진 문제와 해결책 | 에러 발생 시 |

## 테스트 전략

### 테스트 데이터

테스트 파일 위치:
- `test_assets/`: PSD, PNG, JPG 샘플 파일
- `output_mock/`: 유닛 테스트용 모의 데이터

### 테스트 실행

```powershell
# 유닛 테스트 (개별 파서)
python test_base_parser.py
python test_image_parser.py
python test_psd_parser_mock.py

# 통합 테스트
python -m pytest tests/
```

### 테스트 검증 사항

모든 테스트는 다음을 검증해야 합니다:
1. **스키마 준수**: 출력이 `AssetMeta` 구조와 일치
2. **데이터 무결성**: 레이어 개수, 텍스트 추출 정확도
3. **파일 출력**: JSON 파일이 올바른 위치에 저장됨
4. **오류 처리**: 적절한 로깅과 함께 우아한 실패

## 일반적인 문제 및 해결책

### 문제: SigLIP2 모델 로드 실패
**원인**: PyTorch 누락 또는 CUDA 불일치
**해결**: `python backend/setup/installer.py --check` 실행하여 진단

### 문제: Ollama 연결 실패 ("connection refused")
**원인**: Ollama 서버가 실행되지 않음
**해결**:
```powershell
# Ollama 상태 확인
ollama list

# Ollama 서비스 시작 (Windows: 자동 시작됨)
# 필요 모델 확인
ollama pull qwen3-embedding:0.6b
```

### 문제: 검색 결과가 너무 적음
**원인**: SigLIP2 점수 범위 (0.06~0.17)와 threshold 불일치
**해결**: 프론트엔드 threshold를 0으로 설정 (기본값). config.yaml의 `search.thresholds` 확인.

### 문제: MV 결과 없음 (vec_text 비어있음)
**원인**: STEP 2 (AI Vision)가 실행되지 않은 파일은 MC가 없으므로 MV도 생성 불가
**해결**: `--no-skip` 옵션으로 파일 재처리
```powershell
python backend/pipeline/ingest_engine.py --discover "경로" --no-skip
```

### 문제: 번역 API 속도 제한
**원인**: Google Translate에 대한 요청이 너무 많음
**해결**: 배치 크기가 이미 최적화됨; 재시도 로직이나 지연 추가 고려

### 문제: PSD 레이어 텍스트 추출 시 빈 값 반환
**원인**: 비표준 PSD 필드에 텍스트 저장됨
**해결**: `layer.engine_dict` 구조 확인; `layer.text` 속성으로 폴백

### 문제: Electron 앱이 시작되지 않음
**원인**: 프론트엔드 개발 서버가 준비되지 않음
**해결**: `wait-on`이 Electron 전에 Vite가 시작되도록 보장; 포트 5173 가용성 확인

### 문제: 검색 그리드 깜빡임 (스크롤바 진동)
**원인**: `overflow-y: auto` 컨테이너에서 ResizeObserver 무한 루프 발생. 스크롤바 등장 → 컨테이너 ~48px 축소 → 카드 축소 → 콘텐츠 줄어듬 → 스크롤바 소멸 → 컨테이너 확대 → 카드 확대 → 콘텐츠 넘침 → 스크롤바 등장 (반복)
**해결**: `useResponsiveColumns` 훅에서 **최대 관측 너비 래치** 적용. 60px 이내 너비 감소(스크롤바)는 무시하고 넓은 값 유지. 500ms 안정 후 리셋하여 실제 창 크기 변경은 정상 반영.
**교훈**: `overflow-y: auto` + `ResizeObserver` 조합은 스크롤바 toggle로 인한 무한 루프 위험이 있다. 항상 스크롤바 폭 변동을 고려해야 한다.

## 단계별 로드맵

- ✅ **Phase 1**: 구조적 파싱 (PSD 레이어, 메타데이터 추출)
- ✅ **Phase 2**: VV 벡터화 (SigLIP2 임베딩, ChromaDB → SQLite 마이그레이션 완료)
- ✅ **Phase 3**: SQLite + sqlite-vec 통합 (Triaxis Data Storage)
- ✅ **Phase 4**: VLM + MC 생성 + Electron GUI
- ⏳ **Phase 5**: UI/UX 개선 (라이트박스 뷰어, 검색 히스토리, 뷰 모드)
- ⏳ **Phase 6**: 검색 고도화 (고급 필터, 컬렉션, 유사 이미지)
- ⏳ **Phase 7**: 성능 최적화 (증분 인덱싱, 병렬 파싱, 캐싱)
- ⏳ **Phase 8**: 패키징/배포 (인스톨러, 자동 업데이트)
- ⏳ **Phase 9**: 협업 기능 (DB 공유, 코멘트)

자세한 로드맵은 `phase.md`, 기능 명세는 `Spec.md` 참조.

## 인프라 스펙 (MANDATORY)

**이 섹션은 DB, 검색, 모델, 배포 구조의 단일 진실 공급원(Single Source of Truth)입니다.**

### DB: SQLite + sqlite-vec (단일 소스)

| 항목 | 값 |
|------|------|
| **주력 DB** | SQLite (`imageparser.db`, 프로젝트 루트) |
| **벡터 확장** | sqlite-vec (vec0 가상 테이블) |
| **전문 검색** | FTS5 (BM25, Triaxis FTS축) |
| **스키마 파일** | `backend/db/sqlite_schema.sql` |
| **클라이언트** | `backend/db/sqlite_client.py` (SQLiteDB) |
| **자동 마이그레이션** | 연결 시 자동 스키마 업그레이드 |

**PostgreSQL은 미사용** (레거시 마이그레이션 코드만 보존). Docker 불필요.

### 검색: Triaxis (VV + MV + FTS)

| 축 | 역할 | 모델 | DB 테이블 |
|----|------|------|----------|
| **VV** (Visual Vector) | 이미지 픽셀 유사도 | SigLIP2 (tier별, HuggingFace) | `vec_files` |
| **MV** (Meaning Vector) | MC 캡션/태그 텍스트 유사도 | Qwen3-Embedding (Transformers/Ollama) | `vec_text` |
| **FTS** (Full-Text Search) | 파일명/레이어명/태그 키워드 | FTS5 BM25 | `files_fts` |

**결합**: RRF (Reciprocal Rank Fusion), 가중치는 `config.yaml` > `search.rrf.presets` 설정.
**검색 엔진**: `backend/search/sqlite_search.py` (SqliteVectorSearch)

### Tier 시스템 (config.yaml) — 2026-02-20 기준

**VV/MV 크로스 티어 호환성 확보 완료.**

| Tier | VRAM | VV 모델 (SigLIP2) | VLM (MC 생성) | MV 모델 (Qwen3-Embedding) |
|------|------|-------------------|---------------|----------------------|
| **standard** | ~6GB | `siglip2-so400m-patch16-naflex` (1152d) | `Qwen3-VL-2B` (transformers) | `Qwen3-Embedding-0.6B` (1024d) |
| **pro** | 8-16GB | `siglip2-so400m-patch16-naflex` (1152d) | `Qwen3-VL-4B` (auto: mlx/transformers) | `Qwen3-Embedding-0.6B` (1024d) |
| **ultra** | 20GB+ | `siglip2-so400m-patch16-naflex` (1152d) | `Qwen3-VL-8B` (auto: mlx/ollama/vllm/transformers) | `Qwen3-Embedding-8B` (4096d) |

**핵심 설계 결정 (2026-02-20):**
- **VV 모델 통일**: 모든 Tier에서 동일한 `siglip2-so400m-patch16-naflex` (1152d) 사용. Tier 전환 시 VV 재생성 불필요.
- **MV 모델 통일 (standard↔pro)**: 동일한 `qwen3-embedding:0.6b` (1024d). standard↔pro 전환이 **완전 무중단**.
- **MV ultra 분리**: ultra는 `qwen3-embedding:8b` (4096d)로 모델 자체가 다름. 전환 시 MV만 재생성 (빠름, ~0.5s/file).
- **SigLIP2 so400m NaFlex 선택 근거**: Meta PE-Core(2025.04)가 성능 최강이나 CC-BY-NC 라이선스 + transformers 미통합 + MPS 미검증. SigLIP2는 Apache 2.0, transformers 네이티브, MPS 검증됨. 동급 파라미터 대비 성능 차이 0.5% 미만.

**Tier 전환 호환성 매트릭스:**

| 전환 | VV | MV | MC | FTS | 판정 |
|------|----|----|----|----|------|
| standard ↔ pro | 호환 | 호환 | 호환 | 호환 | **완전 호환** |
| standard/pro ↔ ultra | 호환 | MV만 재생성 | 호환 | 호환 | MV만 재생성 |

**설정 파일**: `config.yaml` > `ai_mode.override` (현재: `pro`)
**Tier 로더**: `backend/utils/tier_config.py` > `get_active_tier()`
**호환성 매트릭스**: `backend/utils/tier_compatibility.py`

### 파이프라인 4단계 (process_batch_phased, ingest_engine.py)

```
Phase P (Parse)   → PSD/PNG/JPG 파싱, 썸네일 생성, 메타데이터 추출 → 즉시 DB 저장
Phase V (Vision)  → VLM으로 MC(캡션/태그/분류) 생성 → 서브배치마다 즉시 DB 저장
Phase E (Embed)   → SigLIP2→VV, Qwen3-Embedding→MV 벡터화 → 서브배치마다 즉시 DB 저장
Phase S (Summary) → 완료 확인 ([OK] emit, 프론트엔드 카운트용, 실제 저장 아님)
```

**핵심 원칙**:
- **각 Phase는 서브배치 단위로 즉시 저장**. 1000개 파일이라도 배치(2~16개)씩 처리→저장→다음 배치. 중간 크래시 시 이미 저장된 파일은 Smart Skip으로 건너뜀.
- Phase S는 별도 저장이 아닌 **요약 단계** (모든 데이터는 P/V/E에서 이미 저장 완료).
- Tier 메타데이터(mode_tier, embedding_model 등)는 Phase V 전에 설정되므로 Vision 실패와 무관하게 항상 기록됨.

### Adaptive Batch Controller

```yaml
# config.yaml
batch_processing:
  adaptive:
    enabled: true
    memory_budget_gb: 20        # 메모리 예산
    vlm_initial_batch: 2        # Vision 초기 배치 (→ 메모리에 따라 자동 증감)
    vv_initial_batch: 4         # VV 초기 배치
    mv_initial_batch: 16        # MV 초기 배치
```

- **Triangular-step 성장**: 배치 크기가 메모리 여유에 따라 점진적으로 증가
- **메모리 압박 시 자동 축소**: OOM 방지
- **Phase별 독립 배치**: VLM(무거움)은 작게, MV(가벼움)는 크게 시작
- **StatusBar 표시**: 노란색 `B:N` 뱃지에서 현재 배치 크기 실시간 확인

### Smart Skip (Phase별 재개)

```python
_check_phase_skip(parsed_files)  # 파일별로 이미 완료된 Phase 확인
```

- **파일별 Phase 완료 추적**: MC 존재→Vision 스킵, VV 존재→VV 스킵, MV 존재→MV 스킵
- **Resume 시 미완료 Phase부터 이어서 처리**: 이전 세션에서 Parse 완료 → 재시작 시 Vision부터 시작
- Discover 모드와 Pipeline 모드 모두 동일한 `process_batch_phased` 사용

### Discover 모드 (Resume/Auto-scan)

- `--discover` CLI 또는 Electron IPC `run-discover`로 실행
- DFS 폴더 탐색 → Smart Skip 필터링 → `process_batch_phased` 호출
- **프론트엔드 진행 UI**: Pipeline과 동일한 4-Phase Pills (P/V/E/S) + processed/total + 배치 크기 표시
- **세션 추적**: `config.yaml > last_session.folders`에 작업 대상 기록, 완료 시 초기화
- **앱 재시작 시 Resume Dialog**: 미완료 작업이 있으면 팝업으로 이어하기 제안

### 배포 구조

**A) 로컬 데스크탑 모드 (Electron)**

| 구성요소 | 기술 |
|---------|------|
| **프론트엔드** | Electron 40 + React 19 + Vite + Tailwind CSS |
| **백엔드 통신** | IPC → Python subprocess (stdio JSON) |
| **DB** | SQLite (로컬 파일, Docker 불필요) |
| **VLM** | transformers (standard/pro) 또는 Ollama (ultra) |
| **VV 인코더** | SigLIP2 (HuggingFace, 로컬 캐시) |
| **서버** | 선택적 — [Server] 버튼으로 FastAPI 내장 시작 가능 |

**B) 서버 모드 (FastAPI + 분산 워커)**

| 구성요소 | 기술 |
|---------|------|
| **서버** | FastAPI (uvicorn, 단일 워커 — SQLite 제약) |
| **프론트엔드** | `frontend/dist` SPA 정적 서빙 |
| **인증** | JWT (access + refresh 토큰) |
| **DB** | SQLite (`imageparser_server.db`) |
| **작업 큐** | SQLite `job_queue` 테이블 |
| **워커** | Python 데몬 (`worker_daemon.py`), 여러 대 연결 가능 |
| **워커 통신** | REST API (claim → process → complete) + 하트비트 |

### 필수 설치 요소 (standard tier 기준)

```powershell
# 1. Python 3.11+ (venv)
python -m venv .venv && .venv\Scripts\activate

# 2. Ollama 설치 (https://ollama.com/download) - MV 모델용
# 3. Ollama 모델 pull (MV용만, VLM은 HuggingFace 자동 다운로드)
ollama pull qwen3-embedding:0.6b

# 4. Python 패키지 + SigLIP2/Qwen3-VL 모델 + DB 초기화
python backend/setup/installer.py --full-setup

# 또는 개별 실행:
python backend/setup/installer.py --install          # pip 패키지
python backend/setup/installer.py --download-model    # SigLIP2 + Qwen3-VL (HuggingFace)
python backend/setup/installer.py --setup-ollama      # Ollama MV 모델 확인/pull
python backend/setup/installer.py --init-db           # SQLite 스키마

# 5. 상태 진단
python backend/setup/installer.py --check
```

### 관련 파일 인덱스

| 파일 | 역할 | 수정 가능 |
|------|------|----------|
| `config.yaml` | Tier/검색/배치 설정 (단일 소스) | ✅ |
| `backend/db/sqlite_client.py` | SQLite 클라이언트 | ❌ |
| `backend/db/sqlite_schema.sql` | DB 스키마 정의 | ❌ |
| `backend/search/sqlite_search.py` | Triaxis 검색 엔진 | ❌ |
| `backend/pipeline/ingest_engine.py` | 4단계 처리 파이프라인 | ❌ |
| `backend/vision/vision_factory.py` | VLM 백엔드 자동 선택 | ❌ |
| `backend/vision/analyzer.py` | Transformers VLM 어댑터 (2-Stage) | ❌ |
| `backend/vision/ollama_adapter.py` | Ollama VLM 어댑터 (2-Stage) | ❌ |
| `backend/vector/siglip2_encoder.py` | SigLIP2 VV 인코더 | ❌ |
| `backend/utils/tier_config.py` | Tier 설정 로더 | ❌ |
| `backend/setup/installer.py` | 통합 설치 프로그램 | ❌ |
| `backend/server/app.py` | FastAPI 서버 진입점 | ❌ |
| `backend/server/routers/workers.py` | 워커 세션 관리 API | ❌ |
| `backend/server/queue/manager.py` | 작업 큐 관리자 | ❌ |
| `backend/worker/worker_daemon.py` | 워커 데몬 (배치 처리 + prefetch 풀) | ❌ |
| `backend/worker/worker_ipc.py` | 워커 IPC 브리지 (Electron ↔ Python) | ❌ |
| `backend/worker/config.py` | 워커 설정 | ✅ |
| `backend/db/sqlite_schema_auth.sql` | 인증 DB 스키마 | ❌ |
| `tools/setup_models.py` | Ollama 모델 설치 스크립트 | ❌ |

### 금지 사항

- ❌ **PostgreSQL 사용** (SQLite 단일 소스)
- ❌ **Docker 의존** (로컬 SQLite + Ollama로 충분)
- ❌ **config.yaml 외 하드코딩** (Tier/모델 설정은 config.yaml에서만)
- ❌ **CLIP ViT-L-14 사용** (SigLIP2로 교체 완료)
- ❌ **ChromaDB 사용** (deprecated, 삭제 예정)

---

## Windows 관련 참고사항

- 명령어는 **PowerShell** 또는 **Git Bash** 사용
- 파일 경로는 백슬래시(`C:\Users\...`) 사용하지만 Python은 슬래시로 정규화
- CUDA는 호환되는 NVIDIA GPU + 드라이버 필요 (`nvidia-smi`로 확인)
- **인코딩은 무조건 UTF-8** (MANDATORY): 모든 파일 I/O, subprocess 출력, DB 읽기/쓰기에서 `encoding='utf-8'`을 명시해야 함. Windows 기본 인코딩(cp949)에 의존하면 한글 경로/데이터가 깨짐

## 개발 흐름 예시

```powershell
# 1. 현재 상태 확인
/unit-status

# 2. 새 유닛 시작
/unit-start

# 3. 기능 개발 (예: 새 파서)
# backend/parser/ 에서 파일 편집

# 4. 테스트
python test_new_parser.py
# 실패 시:
/troubleshoot  # 문제 문서화

# 5. 통과하면:
/unit-done  # 회고 작성

# 6. 테스트 파일 처리
python backend/pipeline/ingest_engine.py --file "test.psd"

# 7. Triaxis 검색 검증 (프론트엔드 Electron 앱 사용)
npm run electron:dev  # frontend/ 에서 실행
```

## 프론트엔드 로컬라이제이션 (i18n) 규칙

**모든 프론트엔드 UI 문자열은 반드시 i18n 키를 사용해야 합니다.**

### 시스템 구조

```
frontend/src/i18n/
├── LocaleContext.jsx   ← React Context + useLocale 훅
├── index.js            ← 진입점 (re-export)
└── locales/
    ├── en-US.json      ← 영어 번역
    └── ko-KR.json      ← 한국어 번역
```

### 사용법

```jsx
import { useLocale } from '../i18n';

const { t } = useLocale();
// 기본: t('app.title')
// 보간: t('status.selected', { count: 5 })
```

### 새 문자열 추가 절차

1. 키 이름 결정 (컨벤션: `prefix.name`)
2. `en-US.json`에 영어 값 추가
3. `ko-KR.json`에 한국어 값 추가
4. 코드에서 `t('key')` 사용

### 키 접두사 컨벤션

| 접두사 | 용도 | 예시 |
|--------|------|------|
| `app.` | 앱 전역 | `app.title` |
| `tab.` | 탭 이름 | `tab.search` |
| `action.` | 버튼/액션 | `action.process` |
| `label.` | 폼 라벨 | `label.notes` |
| `placeholder.` | 입력 힌트 | `placeholder.search` |
| `status.` | 상태 표시 | `status.loading` |
| `msg.` | 알림/메시지 | `msg.no_results` |

### 금지 사항

- **하드코딩 UI 텍스트 금지**: `<span>Search</span>` 대신 `<span>{t('tab.search')}</span>`
- **한쪽만 업데이트 금지**: en-US와 ko-KR 양쪽 파일 동시 업데이트 필수
- **중첩 키 금지**: 플랫 도트 표기법만 사용 (`app.title`, `{ app: { title } }` 아님)

### 검증 워크플로우

- `/localize-scan` - 하드코딩 문자열 탐색 리포트
- `/localize-add` - 새 번역 키 추가 (양쪽 동시)

## 프로젝트 핵심 원칙

1. **5단계 유닛 프로토콜 준수**: 모든 개발은 정의된 워크플로우를 따라야 함
2. **AssetMeta 스키마 준수**: 모든 파서 출력은 표준 스키마를 따라야 함
3. **Triaxis 데이터 분해**: VV (시각 벡터) + MV (의미 벡터) + FTS (키워드) 3축 검색
4. **Factory Pattern 사용**: 새 파서는 BaseParser를 상속하고 can_parse() 구현
5. **문제 발생 시 기록 필수**: troubleshooting.md에 모든 이슈와 해결책 문서화
6. **UI 문자열 로컬라이제이션 필수**: 모든 프론트엔드 텍스트는 i18n 키 사용
7. **플랫폼별 최적화 우선**: AUTO 모드를 사용하여 플랫폼에 맞는 백엔드 자동 선택

---

## 크로스 플랫폼 VLM 폴백 체인 (v8.1, 2026-02-20 기준)

**VLM 팩토리(`vision_factory.py`)는 명시적 폴백 체인을 사용합니다.**

### 폴백 체인 매트릭스

모든 플랫폼 × 티어 조합에서 `transformers`가 최종 안전망.

| 티어 | macOS | Windows | Linux |
|------|-------|---------|-------|
| **standard** | `[transformers]` | `[transformers]` | `[transformers]` |
| **pro** | `[mlx → transformers]` | `[transformers]` | `[transformers]` |
| **ultra** | `[mlx → transformers]` | `[ollama → transformers]` | `[vllm → ollama → transformers]` |

### 동작 원리

1. `_resolve_backend_chain()`: config.yaml의 `backend` + `fallback` 필드에서 체인 구성
2. `_check_backend_available()`: 사전 가용성 검사 (is_mlx_vlm_available, is_vllm_available, is_ollama_available)
3. `_instantiate_backend()`: 체인 순서대로 시도, 실패 시 다음으로
4. `transformers`: 항상 마지막에 보장 (torch는 필수 의존성)

### config.yaml 폴백 설정

```yaml
# pro/ultra tier의 backends.{platform} 섹션에 fallback 필드 사용
backends:
  darwin:
    backend: mlx
    fallback: transformers    # MLX 불가 시 transformers로 폴백
  windows:
    backend: ollama
    fallback: transformers    # Ollama 불가 시 transformers로 폴백
  linux:
    backend: vllm
    fallback: ollama          # vLLM 불가 시 ollama, 그래도 불가 시 transformers (암묵적)
```

### 로그 패턴

```
INFO VLM backend chain (tier: pro): mlx → transformers
INFO [SKIP] mlx: not available, trying next       # 또는
INFO [OK] VLM backend: mlx (tier: pro)            # 또는
WARNING [FAIL] mlx: <error>, trying next
```

### VV/MV는 폴백 불필요

- **VV (SigLIP2)**: 모든 플랫폼에서 `transformers` 직접 사용 — 폴백 체인 없음
- **MV (Qwen3-Embedding)**: `text_embedding.py`에 자체 `Transformers → Ollama` 폴백 구현됨

### 개발 가이드라인

```python
# ✅ 올바른 방법 - Factory 사용 (폴백 체인 자동 작동)
from backend.vision.vision_factory import get_vision_analyzer
analyzer = get_vision_analyzer()

# ❌ 잘못된 방법 - 직접 adapter 초기화
from backend.vision.ollama_adapter import OllamaVisionAdapter
analyzer = OllamaVisionAdapter()  # 폴백 체인 무시
```

### 핵심 파일

| 파일 | 역할 |
|------|------|
| `backend/vision/vision_factory.py` | VLM 폴백 체인 팩토리 (v8.0 리팩토링) |
| `backend/utils/platform_detector.py` | 플랫폼 감지 + 가용성 검사 함수 |
| `backend/utils/tier_config.py` | Tier 설정 로더 |
| `config.yaml` > `ai_mode.tiers.{tier}.vlm.backends` | 플랫폼별 백엔드 + 폴백 설정 |

### 성능 참고 (2026-02-20 기준)

| 백엔드 | 장점 | 단점 | 적합 플랫폼 |
|--------|------|------|------------|
| **MLX** | Apple Silicon 네이티브, 3-4x TTFT 향상 | macOS 전용, 배치 불가 (batch_size=1) | macOS |
| **vLLM** | 8-16x 배치 처리, 최고 처리량 | CUDA 필수, Windows 미지원 | Linux (GPU) |
| **Ollama** | 설치 간편, 모델 자동 관리 | 배치 성능 저하, batch_size=1 권장 | Windows, 범용 |
| **Transformers** | 항상 사용 가능, MPS/CUDA/CPU 자동 | MLX/vLLM 대비 느릴 수 있음 | 최종 폴백 |

### 모델 선택 근거 (2026-02-20 조사 결과)

**조사 대상**: SigLIP2 (Google, 2025.02), Meta PE-Core (2025.04), OpenVision (UCSC, 2025.05), Jina-CLIP-v2 (2024.12), AIMv2 (Apple, 2024.11)

**SigLIP2 so400m-naflex 유지 결정 이유:**
1. **라이선스**: Apache 2.0 (PE-Core는 CC-BY-NC 비상업)
2. **생태계**: HuggingFace transformers 네이티브 (PE-Core는 자체 라이브러리/OpenCLIP)
3. **MPS 호환**: macOS M5에서 검증됨 (PE-Core는 MPS 미검증, xformers 의존)
4. **성능 차이**: PE-Core-L14 vs SigLIP2-so400m — ImageNet ZS 0.5% 차이 (83.5% vs ~83%)
5. **NaFlex**: 종횡비 보존 (PSD/PNG 다양한 비율에 적합), HuggingFace 다운로드 1위 (1.31M+)
6. **가성비**: So400m(400M) → giant-opt(1B): 파라미터 2.5배, 메모리 2배, 검색 성능 +0.3%

**재평가 시점**: PE-Core가 transformers 통합 + 라이선스 변경 시, 또는 SigLIP3 출시 시

## 개발/테스트 환경 (macOS)

### 현재 시스템
- **HW**: Apple M5, 32GB Unified Memory, Metal 4
- **OS**: macOS 26.2 (Darwin 25.2.0, arm64)
- **Tier**: pro (8-16GB VRAM range)
- **Python**: 3.12.12 (.venv)
- **Node**: v24.13.0
- **Vite dev port**: 9274

### 테스트 이미지 경로
- **테스트 폴더**: `/Users/saintiron/imageDB/마캬베리즈무/실내소품`
  - PSD 파일 다수 (실내 소품 배경 원화)
  - 파이프라인 E2E 테스트용

### 알려진 이슈
- **Ollama + M5 Metal 4**: `mlx_metal_device_info` 심볼 로드 실패 → 모델 로딩 불가
  - 원인: Ollama 0.15.5의 Metal shader가 M5 Metal 4 bfloat 미지원
  - 해결: MV 생성을 TransformersEmbeddingProvider로 전환 (Ollama 우회)
  - Ollama는 현재 MV 생성에 사용하지 않음
- **VLM (Qwen3-VL-4B)**: transformers 5.1.0의 video_processing_auto.py TypeError 가능성
  - 모델 파일은 캐시됨, 런타임 로딩 시 확인 필요
- **멀티워커 동일 GPU 경합**: 같은 머신에서 워커 N개 실행 시 각 워커 속도 1/N (총 처리량 이득 없음)
  - 원인: GPU 시분할로 인한 경합
  - 멀티워커는 **별도 GPU 머신에 분산 배치할 때만 효과적**

### 양자화 옵션 (조사 완료, 미적용)

Mac MPS (Apple Silicon)에서 사용 가능한 양자화 방법 3가지. 현재는 적용하지 않음.

| 방법 | 지원 | 장점 | 단점 |
|------|------|------|------|
| **optimum-quanto** | MPS int4/int8 | HuggingFace 공식, `QuantoConfig` 추가만으로 적용, 메모리 ~40-50% 절감 | 속도 동일~약간 느림 |
| **torchao** | MPS int4 weight-only | PyTorch 공식, `quantize_(model, int4_weight_only())` | quanto보다 생태계 작음 |
| **MLX** | M-시리즈 네이티브 int4 | Apple 공식, Neural Engine TensorOps 활용, 최고 속도 | transformers API 비호환, 코드 전면 재작성 필요 |

**현재 상태**: pro tier fp16 기준 ~11GB 사용, 32GB 시스템에서 여유 충분 → 양자화 불필요
**적용 시점**: ultra tier 지원 또는 저메모리(8GB) 기기 타겟 시 검토
**참고**: bitsandbytes는 여전히 CUDA 전용 (MPS 미지원)
