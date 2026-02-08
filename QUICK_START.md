# ImageParser Quick Start Guide

**최소 5분 설치 가이드** - 새 사용자를 위한 빠른 시작

## 사전 요구사항

- Python 3.11+ 설치됨
- Git 설치됨

## 설치 (5분)

### 1. 저장소 클론 및 가상환경 설정

```powershell
# 저장소 클론
git clone <repository-url>
cd ImageParser

# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate
```

### 2. 자동 설치 실행

```powershell
# 모든 의존성 설치 + AI 모델 다운로드 + PostgreSQL 가이드
python backend/setup/installer.py --full-setup
```

이 명령은 다음을 수행합니다:
- ✅ Python 패키지 설치 (torch, psycopg2, sentence-transformers 등)
- ✅ CLIP AI 모델 다운로드 (~1.7GB, 최초 1회)
- ✅ PostgreSQL 설치 가이드 표시

### 3. PostgreSQL 설정

#### Option A: Docker (추천 - 가장 쉬움)

```powershell
# Docker Desktop 설치: https://www.docker.com/products/docker-desktop/

# PostgreSQL 시작 (백그라운드)
docker-compose up -d

# 10초 대기
timeout /t 10

# 데이터베이스 스키마 초기화
python backend/setup/installer.py --init-db
```

#### Option B: 수동 설치

[docs/postgresql_setup.md](docs/postgresql_setup.md) 참조

### 4. 설치 확인

```powershell
python backend/setup/installer.py --check
```

**성공 시 출력:**
```
✅ Python Dependencies: OK
✅ CLIP Model Cached: Yes
✅ PostgreSQL: Connected
✅ pgvector Extension: Active
```

## 첫 이미지 처리

```powershell
# 단일 파일 처리
python backend/pipeline/ingest_engine.py --file "path\to\image.psd"

# 성공 시 메시지:
# ✅ Stored to PostgreSQL: image.psd
```

## 검색 테스트

```powershell
# 텍스트로 이미지 검색
python backend/cli_search_pg.py "cartoon city"

# 하이브리드 검색 (벡터 + 필터)
python backend/cli_search_pg.py "fantasy character" --mode hybrid --format PSD
```

## 디렉토리 감시 (자동 처리)

```powershell
# 폴더 내 파일 변경 시 자동 처리
python backend/pipeline/ingest_engine.py --watch "C:\path\to\assets"
```

## 문제 해결

### PostgreSQL 연결 실패

```powershell
# Docker 상태 확인
docker-compose ps

# 컨테이너 재시작
docker-compose restart

# 로그 확인
docker-compose logs -f postgres
```

### 의존성 설치 실패

```powershell
# 캐시 삭제 후 재설치
python -m pip cache purge
python -m pip install -r requirements.txt --force-reinstall
```

### CLIP 모델 다운로드 느림

```powershell
# Hugging Face 캐시 삭제 후 재다운로드
rmdir /s /q "%USERPROFILE%\.cache\huggingface"
python backend/setup/installer.py --download-model
```

## 다음 단계

- **상세 설치 가이드**: [INSTALLATION.md](INSTALLATION.md)
- **PostgreSQL 설정**: [docs/postgresql_setup.md](docs/postgresql_setup.md)
- **프로젝트 구조**: [CLAUDE.md](CLAUDE.md)
- **마이그레이션** (기존 사용자): `python tools/migrate_to_postgres.py`

## 주요 명령어 요약

| 작업 | 명령어 |
|------|--------|
| 설치 확인 | `python backend/setup/installer.py --check` |
| PostgreSQL 시작 | `docker-compose up -d` |
| 파일 처리 | `python backend/pipeline/ingest_engine.py --file "path"` |
| 검색 | `python backend/cli_search_pg.py "query"` |
| 감시 모드 | `python backend/pipeline/ingest_engine.py --watch "path"` |

## 성능 기대치

- **파일 처리**: 2-5초/PSD 파일
- **검색 속도**: <50ms (10,000 이미지)
- **저장 공간**: ~50KB/이미지
- **메모리**: ~1.7GB (CLIP 모델)

---

**문의사항**: [docs/troubleshooting.md](docs/troubleshooting.md) 참조
