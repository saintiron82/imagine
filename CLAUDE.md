# CLAUDE.md

## 언어 규칙 (MANDATORY)

**모든 응답은 반드시 한국어로 작성합니다.** 코드 주석, 커밋 메시지, 변수명 등 코드 자체는 영어를 사용하되, 사용자에게 보여주는 설명·요약·보고는 항상 한국어로 합니다.

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
- **Frontend**: React 19 + Electron 40 + Vite + Tailwind CSS
- **Database**: SQLite + sqlite-vec (통합 메타데이터 + 벡터 저장소, Docker 불필요)
- **AI 모델**: SigLIP2 (VV), Qwen3-VL (VLM/MC), Qwen3-Embedding (MV)

## 용어 사전 (MANDATORY)

**이 프로젝트의 공식 용어입니다. 코드 주석, 문서, 대화에서 반드시 이 용어를 사용하세요.**

### 핵심 약어

| 약어 | 정식 명칭 | 설명 | DB 테이블/컬럼 |
|------|----------|------|---------------|
| **VV** | Visual Vector | SigLIP2가 이미지 픽셀로부터 생성하는 시각 임베딩 벡터. 이미지↔이미지 유사도 검색에 사용 | `vec_files.embedding` |
| **MV** | Meaning Vector | Qwen3-Embedding이 MC 텍스트로부터 생성하는 의미 임베딩 벡터. 텍스트↔텍스트 유사도 검색에 사용 | `vec_text.embedding` |
| **MC** | Meta-Context Caption | VLM(Qwen3-VL)이 이미지를 보고 생성한 캡션 텍스트와 태그. MV의 입력 소스 | `files.mc_caption`, `files.ai_tags` |
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
| ai_caption | **mc_caption** (MC) |
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

### 진단 스크립트

```powershell
# 디렉토리 배치 분석
python scripts/batch_analyze.py

# 단일 이미지 진단
python scripts/diagnose_image.py "path/to/image.psd"

# CLI에서 이미지 검색
python scripts/search_images.py "검색어"
```

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
STEP 2/4: AI Vision (vision_factory.py)
    ├─ VLM 캡션/태그/분류 생성 (Qwen3-VL, tier별 backend)
    └─ 2-Stage Pipeline: 빠른 분류 → 상세 캡션
    ↓
STEP 3/4: Embedding (siglip2_encoder.py + text_embedding.py)
    ├─ SigLIP2 → VV 생성 (이미지 시각 벡터, tier별 차원)
    └─ Qwen3-Embedding → MV 생성 (MC 텍스트 의미 벡터)
    ↓
STEP 4/4: SQLite Storage (db/sqlite_client.py)
    ├─ 메타데이터 + MC → files 테이블 (JSON 필드)
    ├─ VV → vec_files 테이블 (sqlite-vec)
    ├─ MV → vec_text 테이블 (sqlite-vec)
    └─ FTS 인덱스 자동 동기화 (files_fts)
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
| `backend/setup/installer.py` | **통합 설치 프로그램** |
| `config.yaml` | **Tier/검색/배치 설정** (단일 소스) |
| `output/thumbnails/` | 썸네일 이미지 (gitignore됨) |
| `docs/troubleshooting.md` | **모든 문제에 대한 필수 로깅** |
| `INSTALLATION.md` | **신규 설치 가이드** |
| `frontend/src/i18n/` | **프론트엔드 로컬라이제이션 시스템** |
| `frontend/src/i18n/locales/en-US.json` | 영어 번역 파일 |
| `frontend/src/i18n/locales/ko-KR.json` | 한국어 번역 파일 |

**Legacy (Deprecated)**:
- `backend/db/pg_client.py` - PostgreSQL 클라이언트 (deprecated, SQLite로 교체)
- `backend/search/pg_search.py` - pgvector 검색 (deprecated, Triaxis로 교체)
- `backend/vector/chroma_indexer.py` - ChromaDB 모듈 (deprecated)

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

### Tier 시스템 (config.yaml)

| Tier | VRAM | VV 모델 (SigLIP2) | VLM (MC 생성) | MV 모델 (Qwen3-Embedding) |
|------|------|-------------------|---------------|----------------------|
| **standard** | ~6GB | `siglip2-base-patch16-224` (768d) | `Qwen3-VL-2B` (transformers) | `Qwen3-Embedding-0.6B` (256d) |
| **pro** | 8-16GB | `siglip2-so400m-patch14-384` (1152d) | `Qwen3-VL-4B` (transformers) | `Qwen3-Embedding-0.6B` (1024d) |
| **ultra** | 20GB+ | `siglip2-giant-opt-patch16-256` (1664d) | `Qwen3-VL-8B` (auto: ollama/vllm) | `Qwen3-Embedding-8B` (4096d) |

**설정 파일**: `config.yaml` > `ai_mode.override` (현재: `standard`)
**Tier 로더**: `backend/utils/tier_config.py` > `get_active_tier()`

### 파이프라인 4단계 (ingest_engine.py)

```
STEP 1/4: Parse       → PSD/PNG/JPG 파싱, 썸네일 생성, 메타데이터 추출
STEP 2/4: AI Vision   → VLM으로 캡션/태그/분류 생성 (tier별 backend: transformers/ollama/auto)
STEP 3/4: Embedding   → SigLIP2로 VV 생성, Qwen3-Embedding으로 MV 생성
STEP 4/4: Storing     → SQLite 저장 (메타데이터 + VV + MV)
```

**핵심**: Tier 메타데이터(mode_tier, embedding_model 등)는 STEP 2 전에 설정되므로 Vision 실패와 무관하게 항상 기록됨.

### Windows 배포 구조

| 구성요소 | 기술 |
|---------|------|
| **프론트엔드** | Electron 40 + React 19 + Vite + Tailwind CSS |
| **백엔드 통신** | IPC → Python subprocess (stdio JSON) |
| **DB** | SQLite (로컬 파일, Docker 불필요) |
| **VLM** | transformers (standard/pro) 또는 Ollama (ultra) |
| **VV 인코더** | SigLIP2 (HuggingFace, 로컬 캐시) |
| **API 서버** | 없음 (subprocess 직접 호출) |

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

## 플랫폼별 최적화 규칙 (v3.1.1)

**모든 vision 처리는 플랫폼별로 최적화된 백엔드를 사용합니다.**

### 핵심 원칙

1. **항상 AUTO 모드 사용**
   ```yaml
   # config.yaml
   ai_mode:
     tiers:
       ultra:
         vlm:
           backend: auto  # 플랫폼 자동 감지
   ```

2. **플랫폼별 권장 설정**
   - **Windows**: Ollama (batch_size=1, 순차 처리)
   - **Mac**: vLLM 우선 (batch_size=16, 병렬 처리)
   - **Linux**: vLLM 우선 (batch_size=16, 병렬 처리)

3. **성능 특성 이해**
   - Ollama Vision API: 배치 처리 시 성능 저하 (0.6x)
   - vLLM: 배치 처리 시 8.5배 향상
   - Transformers: 배치 처리 시 4-14배 향상

### 실측 벤치마크 데이터

**Ultra Tier (Qwen3-VL-8B):**

| 플랫폼 | 백엔드 | 1개 이미지 | 10개 이미지 | Speedup |
|--------|--------|-----------|-------------|---------|
| Windows | Ollama | 51초 | 510초 | 1.0x (기준) |
| Mac/Linux | vLLM | ~6초 | ~60초 | 8.5x |
| Mac/Linux | Ollama | 51초 | 510초 | 1.0x |

**Windows + Ollama 배치 처리 특성:**
- batch_size=1: 평균 68.9초/파일 (일관적)
- batch_size=3: 평균 127.8초/파일 (2배 느림)
- **결론**: 순차 처리가 최적

### 개발 가이드라인

**새 기능 개발 시:**
```python
# ✅ 올바른 방법 - Factory 사용 (플랫폼 자동 감지)
from backend.vision.vision_factory import get_vision_analyzer
analyzer = get_vision_analyzer()  # AUTO 모드 작동

# ❌ 잘못된 방법 - 직접 adapter 초기화
from backend.vision.ollama_adapter import OllamaVisionAdapter
analyzer = OllamaVisionAdapter()  # 플랫폼 무시
```

**배치 처리 구현 시:**
```python
# 플랫폼별 최적 batch_size 자동 감지
from backend.utils.platform_detector import get_optimal_batch_size

optimal_batch = get_optimal_batch_size(backend='auto', tier='ultra')
# Windows: 1, Mac/Linux: 16
```

### 문제 해결 체크리스트

**Vision 처리가 느린 경우:**
1. [ ] 플랫폼 확인: `python -m backend.utils.platform_detector`
2. [ ] Tier 확인: config.yaml의 `override` 설정
3. [ ] Backend 확인: 로그에서 "Using ... backend" 검색
4. [ ] Batch size 확인: Windows는 1이 최적

**Windows에서 속도 개선 원하는 경우:**
- **단기**: Tier 낮추기 (ultra → pro → standard)
- **장기**: Mac/Linux 서버로 마이그레이션 (8.5배 향상)
- **대안**: WSL2 + vLLM (고급 사용자)

### 관련 문서

- [플랫폼별 최적화 가이드](docs/platform_optimization.md) - 상세 설정
- [빠른 시작 가이드](docs/quick_start_guide.md) - 플랫폼별 사용법
- [Ollama 배치 분석](docs/ollama_batch_processing_analysis.md) - 벤치마크 데이터

**심각도**: ⚠️ HIGH - 성능에 직접적인 영향
**적용 시기**: v3.1.1부터 필수

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
