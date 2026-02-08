---
description: 프로젝트 빌드 및 테스트 실행
---
# 빌드 및 테스트 워크플로우

## 의존성 설치
// turbo
1. 가상환경 생성 및 활성화
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

// turbo
2. 패키지 설치
```powershell
pip install -r requirements.txt
```

## 테스트 실행
// turbo
3. 단위 테스트
```powershell
python -m pytest tests/ -v --tb=short
```

## 수동 테스트
4. 단일 파일 처리 테스트
```powershell
python -m backend.pipeline.ingest_engine --file "path/to/test.psd"
```

5. Watchdog 모드 실행
```powershell
python -m backend.pipeline.ingest_engine --watch "path/to/watch/folder"
```
