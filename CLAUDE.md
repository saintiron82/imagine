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

**ImageParser**는 PSD, PNG, JPG 파일을 AI 검색 가능한 데이터로 변환하는 멀티모달 이미지 데이터 추출 및 벡터화 시스템입니다. **3-Axis 아키텍처**를 사용하여 이미지를 다음과 같이 분해합니다:

1. **Structural Axis (구조적 축)**: 레이어 계층 구조, 텍스트 내용, 메타데이터 (PSD 파싱)
2. **Latent Axis (잠재적 축)**: CLIP-ViT-L-14를 사용한 시각적 임베딩 (의미 기반 유사도 검색)
3. **Descriptive Axis (서술적 축)**: AI 생성 캡션 및 태그 (Phase 4, 진행 중)

**기술 스택**:
- **Backend**: Python 3.x + `psd-tools`, `Pillow`, `sentence-transformers`, `psycopg2`
- **Frontend**: React 19 + Electron 40 + Vite + Tailwind CSS
- **Database**: PostgreSQL 16+ + pgvector (통합 메타데이터 + 벡터 저장소)
- **AI 모델**: CLIP ViT-L-14 (이미지 임베딩), Qwen-VL/Florence-2 (예정)

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

# 텍스트 쿼리로 이미지 검색 (PostgreSQL + pgvector)
python backend/cli_search_pg.py "fantasy character with sword"

# 하이브리드 검색 (벡터 + 메타데이터 필터)
python backend/cli_search_pg.py "cartoon city" --mode hybrid --format PSD --min-width 2000

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
PostgreSQL Storage (db/pg_client.py)
    ├─ CLIP 모델 로드 (lazy loading)
    ├─ 썸네일 인코딩 → CLIP 임베딩 벡터 (768차원)
    ├─ Metadata → JSONB (nested layer_tree 완벽 지원)
    └─ PostgreSQL INSERT (files 테이블)
        ├─ metadata JSONB (구조적 데이터)
        └─ embedding vector(768) (pgvector)
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

### PostgreSQL Database (Phase 3: Vision Data Storage)

**통합 스토리지**: 메타데이터 + CLIP 벡터를 단일 데이터베이스에서 관리

`backend/db/schema.sql`:
- **files 테이블**: 파일 레벨 메타데이터 + CLIP 임베딩
  - `metadata JSONB`: Nested 구조 완벽 지원 (layer_tree, translated_layer_tree 등)
  - `embedding vector(768)`: pgvector를 사용한 CLIP 벡터 (768차원)
  - `ai_caption`, `ai_tags`, `ocr_text`: AI 생성 필드 (Phase 4)
  - `folder_path`, `folder_depth`, `folder_tags`: 폴더 탐색 메타데이터 (DFS Discovery)
- **layers 테이블** (선택): 주요 레이어 레벨 메타데이터 (30% 선별)

`backend/db/pg_client.py`:
- **PostgresDB**: PostgreSQL 클라이언트 (psycopg2 래퍼)
- **insert_file()**: 파일 메타데이터 + CLIP 벡터 저장
- **get_file_by_path()**: 파일 경로로 조회
- **get_stats()**: 데이터베이스 통계

`backend/search/pg_search.py`:
- **PgVectorSearch**: pgvector 기반 CLIP 유사도 검색
- **vector_search()**: 텍스트 쿼리 → CLIP 임베딩 → 벡터 유사도 검색
- **hybrid_search()**: 벡터 검색 + 메타데이터 필터 (한 번의 SQL 쿼리)
- **metadata_query()**: 순수 메타데이터 필터링
- **jsonb_query()**: Nested JSON 구조 쿼리 (layer_tree 등)

**성능**:
- 검색 속도: ~20ms (194 파일), ~40ms (10,000 파일)
- 60배 향상: 기존 JSON 파일 방식 대비 (1.2초 → 20ms)
- HNSW 인덱스: O(log n) 벡터 유사도 검색
- GIN 인덱스: JSONB 고속 쿼리

**설치 및 마이그레이션**:
```powershell
# PostgreSQL 설치 (Docker 권장)
docker-compose up -d

# 스키마 초기화
python backend/setup/installer.py --init-db

# 기존 데이터 마이그레이션 (ChromaDB → PostgreSQL)
python tools/migrate_to_postgres.py

# 검증
python tools/verify_migration.py
```

**Legacy (Deprecated)**:
- `backend/vector/` 모듈: ChromaDB 기반 (deprecated, 마이그레이션 지원용으로 보존)
- 자세한 내용: `backend/vector/README.md`

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

- **Vector Indexer**: 첫 사용 시 CLIP 모델 지연 로드
- **CUDA 정리**: 이미지 10개마다 `torch.cuda.empty_cache()` 실행
- **전역 싱글톤**: `_global_indexer`가 장시간 실행 프로세스에서 모델 재로딩 방지

## 주요 파일 위치

| 경로 | 목적 |
|------|------|
| `backend/parser/schema.py` | **표준 데이터 스키마** (AssetMeta) |
| `backend/pipeline/ingest_engine.py` | **모든 처리의 메인 진입점** |
| `backend/db/schema.sql` | **PostgreSQL 스키마** (테이블, 인덱스 정의) |
| `backend/db/pg_client.py` | PostgreSQL 클라이언트 (메타데이터 + 벡터 저장) |
| `backend/search/pg_search.py` | pgvector 검색 엔진 (벡터 + 하이브리드 검색) |
| `backend/cli_search_pg.py` | CLI 검색 도구 |
| `tools/migrate_to_postgres.py` | **마이그레이션 스크립트** (ChromaDB → PostgreSQL) |
| `tools/verify_migration.py` | 마이그레이션 검증 |
| `backend/setup/installer.py` | **통합 설치 프로그램** |
| `output/thumbnails/` | 썸네일 이미지 (gitignore됨) |
| `docker-compose.yml` | PostgreSQL + pgvector (Docker 설정) |
| `docs/postgresql_setup.md` | PostgreSQL 상세 설치 가이드 |
| `docs/troubleshooting.md` | **모든 문제에 대한 필수 로깅** |
| `INSTALLATION.md` | **신규 설치 가이드** |
| `.agent/skills/unit_dev_agent/SKILL.md` | 개발 프로토콜 세부사항 |
| `frontend/src/i18n/` | **프론트엔드 로컬라이제이션 시스템** |
| `frontend/src/i18n/locales/en-US.json` | 영어 번역 파일 |
| `frontend/src/i18n/locales/ko-KR.json` | 한국어 번역 파일 |
| `.agent/skills/localize/SKILL.md` | 로컬라이제이션 에이전트 정의 |

**Legacy (Deprecated)**:
- `backend/vector/` - ChromaDB 모듈 (마이그레이션 후 삭제 예정)
- `chroma_db/` - ChromaDB 데이터 (백업용 보존, 1개월 후 삭제)
- `output/json/` - JSON 메타데이터 (PostgreSQL로 이동됨)

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

### 문제: CLIP 모델 로드 실패
**원인**: PyTorch 누락 또는 CUDA 불일치
**해결**: `python backend/setup/installer.py` 실행하여 진단 및 의존성 설치

### 문제: PostgreSQL 연결 실패 ("connection refused")
**원인**: PostgreSQL 서버가 실행되지 않음
**해결**:
```powershell
# Docker: 컨테이너 상태 확인
docker-compose ps

# Docker: 컨테이너 재시작
docker-compose restart

# 로컬 설치: PostgreSQL 서비스 확인 (services.msc)
```

### 문제: pgvector extension not found
**원인**: pgvector 확장이 설치되지 않음
**해결**:
```powershell
# Docker: 데이터베이스 재생성
docker-compose down -v
docker-compose up -d
python backend/setup/installer.py --init-db

# 로컬: pgvector 설치 (docs/postgresql_setup.md 참조)
```

### 문제: 마이그레이션 실패
**원인**: ChromaDB 데이터 손상 또는 PostgreSQL 연결 문제
**해결**:
```powershell
# 진단 실행
python backend/setup/installer.py --check

# PostgreSQL 연결 확인
python -c "from backend.db.pg_client import PostgresDB; db = PostgresDB(); print('OK')"

# ChromaDB 없이 마이그레이션 (zero embeddings 사용)
# 이후 파일 재처리로 embeddings 생성
python tools/migrate_to_postgres.py
```

### 문제: ChromaDB 권한 오류 (Legacy)
**원인**: 다른 프로세스가 데이터베이스를 잠금
**참고**: ChromaDB는 deprecated. PostgreSQL로 마이그레이션 권장

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
- ✅ **Phase 2**: 잠재적 벡터화 (CLIP 임베딩)
- ✅ **Phase 3**: PostgreSQL + pgvector 통합 (Vision Data Storage)
  - ChromaDB → PostgreSQL 마이그레이션
  - 통합 스토리지 (메타데이터 + 벡터)
  - 60배 성능 향상 (1.2s → 20ms)
  - JSONB 지원 (nested layer_tree)
  - 하이브리드 검색 (벡터 + 메타데이터)
- 🚧 **Phase 4**: 서술적 비전 (Qwen-VL/Florence-2 캡션 생성) - **진행 중**
- ⏳ **Phase 5**: 최적화 (레이어 단위 인덱싱, 전체 패키징)

자세한 마일스톤 추적은 `docs/phase_roadmap.md` 참조.

## Windows 관련 참고사항

- 명령어는 **PowerShell** 또는 **Git Bash** 사용
- 파일 경로는 백슬래시(`C:\Users\...`) 사용하지만 Python은 슬래시로 정규화
- PostgreSQL은 NTFS에서 가장 잘 작동; 네트워크 드라이브 피하기
- CUDA는 호환되는 NVIDIA GPU + 드라이버 필요 (`nvidia-smi`로 확인)
- Docker Desktop 권장 (PostgreSQL 간편 설치)

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

# 7. PostgreSQL 검색 검증
python backend/cli_search_pg.py "테스트 쿼리"
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
3. **3-Axis 데이터 분해**: 구조적, 잠재적, 서술적 데이터를 모두 추출
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
