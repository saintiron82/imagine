# DB/AI 독립 시스템 구축 가이드

## 📋 현재 설치 완료

```
✓ PyTorch 2.5.1+cu121 (CUDA 지원)
✓ ChromaDB 1.4.1
✓ sentence-transformers 5.2.2 (CLIP 포함)
✓ transformers 5.1.0 (Florence-2/Qwen용)
✓ psd-tools, watchdog, exifread, tqdm
```

---

## 🎯 사용자님의 요구사항

> "DB나 AI는 좀 이 GUI 시스템과 독립시켜서 돌릴건데?"

**이해한 내용:**
1. GUI (Electron 앱)와 **분리된** DB/AI 시스템 운영
2. CLI 또는 API 서버로 독립 실행
3. MacBook M5에서 별도로 Qwen + CLIP 구축

---

## 🏗️ 시스템 분리 아키텍처

### 현재 통합 구조 (All-in-One)
```
┌─────────────────────────────────────┐
│   Electron GUI (Frontend)           │
│   ├─ 파일 탐색기                     │
│   ├─ 썸네일 뷰어                     │
│   └─ 검색 UI                        │
└─────────────────────────────────────┘
              ↕ IPC
┌─────────────────────────────────────┐
│   Python Backend                    │
│   ├─ Parser (PSD 분석)              │
│   ├─ Vision Analyzer (AI)           │
│   ├─ Vector Indexer (CLIP)          │
│   └─ ChromaDB (저장)                 │
└─────────────────────────────────────┘
```

### 제안: 독립 시스템 (Standalone)
```
┌─────────────────────────────────────┐
│   독립 DB/AI 서버 (Backend)          │
│   ├─ FastAPI REST API               │
│   ├─ Parser Service                 │
│   ├─ Vision Service (Qwen/Florence) │
│   ├─ CLIP Service                   │
│   └─ ChromaDB (로컬/리모트)          │
│   포트: 8000                         │
└─────────────────────────────────────┘
              ↕ HTTP API
┌─────────────────────────────────────┐
│   GUI 클라이언트 (Optional)          │
│   - Electron 앱                     │
│   - 또는 웹 브라우저                 │
│   - 또는 CLI 도구                   │
└─────────────────────────────────────┘
```

---

## 🚀 구현 방법 (3가지 옵션)

### 옵션 1: CLI 독립 실행 (가장 간단)

**현재 구조를 그대로 활용, GUI 없이 CLI만 사용**

```powershell
# 1. 파일 처리 (GUI 없이)
python backend/pipeline/ingest_engine.py --file "image.psd"
python backend/pipeline/ingest_engine.py --watch "C:\Assets"

# 2. 검색 (GUI 없이)
python backend/cli_search.py "fantasy character"

# 3. 배치 처리
python backend/pipeline/ingest_engine.py --files "[\"file1.psd\", \"file2.png\"]"
```

**장점:**
- 즉시 사용 가능
- GUI 설치 불필요
- 가볍고 빠름

**단점:**
- 시각적 UI 없음
- 수동으로 명령 실행

---

### 옵션 2: FastAPI REST 서버 (권장)

**독립 API 서버로 구축, 어디서든 HTTP로 접근**

#### 서버 구현 예시

```python
# backend/api_server.py
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from backend.pipeline.ingest_engine import process_file
from backend.vector.searcher import VectorSearcher

app = FastAPI()
searcher = VectorSearcher()

@app.post("/api/ingest")
async def ingest_image(file: UploadFile = File(...)):
    """이미지 업로드 및 분석"""
    temp_path = Path(f"temp/{file.filename}")
    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = process_file(temp_path)
    return {"success": True, "file": file.filename}

@app.get("/api/search")
async def search_images(query: str, top_k: int = 20):
    """이미지 검색"""
    results = searcher.search(query, top_k)
    return {"query": query, "results": results}

@app.get("/api/stats")
async def get_stats():
    """DB 통계"""
    return {
        "total_images": searcher.collection.count(),
        "db_size": "672KB"
    }
```

#### 서버 실행

```powershell
# FastAPI 설치
pip install fastapi uvicorn python-multipart

# 서버 시작
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000

# 브라우저에서 접속
# http://localhost:8000/docs (자동 생성된 API 문서)
```

#### 클라이언트 사용 (어디서든 접근)

```python
# Python 클라이언트
import requests

# 이미지 업로드
files = {"file": open("image.psd", "rb")}
response = requests.post("http://localhost:8000/api/ingest", files=files)

# 검색
response = requests.get("http://localhost:8000/api/search", params={"query": "sword"})
results = response.json()
```

```bash
# curl로도 사용 가능
curl -X POST "http://localhost:8000/api/ingest" \
  -F "file=@image.psd"

curl "http://localhost:8000/api/search?query=fantasy"
```

**장점:**
- GUI와 완전 분리
- 네트워크 접근 가능
- 다른 클라이언트에서 사용 가능 (웹, 앱, CLI)
- 확장성 높음

**단점:**
- 추가 구현 필요 (FastAPI 코드 작성)
- 서버 운영 관리

---

### 옵션 3: Docker 컨테이너 (프로덕션)

**완전히 독립된 환경으로 패키징**

```dockerfile
# Dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY chroma_db/ ./chroma_db/

EXPOSE 8000
CMD ["uvicorn", "backend.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```powershell
# 빌드
docker build -t imageparser-api .

# 실행
docker run -d -p 8000:8000 -v C:\Assets:/assets imageparser-api

# 다른 컴퓨터에서 접근
curl http://192.168.1.10:8000/api/search?query=sword
```

**장점:**
- 환경 독립성 (어디서든 동일하게 실행)
- 배포 간편
- 확장 용이 (Kubernetes 등)

**단점:**
- Docker 학습 필요
- 설정 복잡도 증가

---

## 🖥️ MacBook M5 별도 시스템 구축

**사용자님의 경우: Mac에서 Qwen, Windows에서 GUI**

### Mac 서버 (Qwen + CLIP)

```bash
# MacBook M5에서
brew install ollama
ollama pull qwen2.5-vl:7b

pip install chromadb sentence-transformers fastapi uvicorn

# API 서버 실행
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

### Windows 클라이언트 (GUI만)

```python
# frontend에서 Mac 서버에 HTTP 요청
const response = await fetch('http://192.168.1.100:8000/api/search', {
  method: 'GET',
  params: { query: searchQuery }
});
```

**장점:**
- Mac M5의 NPU 활용 (Qwen 최적화)
- Windows는 GUI만 담당 (가벼움)
- 네트워크로 연결

---

## 📂 디렉토리 분리 전략

### 현재 구조
```
ImageParser/
├── backend/        # Python (Parser, Vision, DB)
├── frontend/       # Electron (GUI)
└── chroma_db/      # 벡터 DB
```

### 독립 시스템 구조
```
# 서버 (독립 실행)
ImageParser-Server/
├── backend/
│   ├── api_server.py      # FastAPI 서버
│   ├── parser/
│   ├── vision/
│   └── vector/
├── chroma_db/             # DB (여기에 존재)
└── requirements.txt

# 클라이언트 (선택적)
ImageParser-Client/
├── frontend/              # GUI (선택)
└── cli_client.py          # CLI 도구
```

---

## 🎯 권장 솔루션 (사용자님 상황)

### 단계 1: CLI로 시작 (즉시 사용)
```powershell
# GUI 없이 CLI로 테스트
python backend/pipeline/ingest_engine.py --watch "C:\Assets"
python backend/cli_search.py "검색어"
```

### 단계 2: FastAPI 서버 구축 (1-2일)
```powershell
# API 서버 작성
# backend/api_server.py 생성

# 서버 실행
uvicorn backend.api_server:app --reload
```

### 단계 3: Mac에서 별도 실행 (선택)
```bash
# MacBook M5에서 Qwen + CLIP 서버
# Windows GUI에서 HTTP로 접근
```

---

## 🛠️ 다음 단계

### 즉시 실행 가능 (GUI 없이)
```powershell
# 1. 기존 ChromaDB 확인
python -c "
import chromadb
client = chromadb.PersistentClient(path='chroma_db')
print('Collections:', [c.name for c in client.list_collections()])
"

# 2. CLI 검색 테스트
python backend/cli_search.py "test"

# 3. 새 파일 처리 (GUI 없이)
python backend/pipeline/ingest_engine.py --file "new_image.psd"
```

### FastAPI 서버 구축 (선택)
```powershell
# 1. FastAPI 설치
pip install fastapi uvicorn python-multipart

# 2. backend/api_server.py 작성
# (위의 예시 코드 사용)

# 3. 서버 실행
uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
```

---

## 💡 결론

**현재 상태:**
- ✅ 모든 의존성 설치 완료
- ✅ CLI로 독립 실행 가능
- ✅ GUI와 분리 가능한 구조

**권장 순서:**
1. **지금 바로**: CLI로 테스트 (`python backend/cli_search.py`)
2. **1주일 후**: FastAPI 서버 구축
3. **필요시**: MacBook M5 별도 서버

**핵심:**
> "Backend는 이미 독립적으로 실행 가능합니다. GUI는 선택 사항이며, CLI나 API 서버로 언제든지 사용할 수 있습니다."

---

**작성일**: 2026-02-06
**다음 문서**: `api_server_implementation.md` (FastAPI 상세 가이드)
