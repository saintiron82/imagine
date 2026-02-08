---
description: 유닛 개발 시작 (5단계 프로세스 실행)
---
# 유닛 개발 시작

이 워크플로우는 지정된 유닛의 5단계 프로세스(목표/정의/개발/테스트/회고)를 순차적으로 실행합니다.

## 사전 조건
- 현재 진행할 유닛 ID 확인 (예: U-001, U-002)
- 선행 유닛이 완료되었는지 확인

## 실행 단계

### 1️⃣ 목표 단계
1. `task.md`에서 현재 유닛 항목 확인
2. 목표를 명확히 선언 (달성할 결과물, 완료 조건)
3. 해당 유닛 항목을 `[/]` (진행 중)으로 마킹

### 2️⃣ 정의 단계
4. 입력/출력/의존성/제약 사항 정의
5. 생성할 파일 경로 확정

### 3️⃣ 개발 단계
6. 정의에 따라 코드 작성
7. 파일 생성 또는 수정

### 4️⃣ 테스트 단계
8. 유닛별 테스트 명령어 실행:

**U-001 (프로젝트 초기화)**:
// turbo
```powershell
pip install -r requirements.txt
```

**U-002 (스키마)**:
// turbo
```powershell
python -c "from backend.parser.schema import AssetMeta; print('Schema OK')"
```

**U-003 (정제 모듈)**:
// turbo
```powershell
python -c "from backend.parser.cleaner import clean_layer_name; assert clean_layer_name('Layer 1 copy') == 'Layer'; print('Cleaner OK')"
```

**U-004 (기본 파서)**:
// turbo
```powershell
python -c "from backend.parser.base_parser import BaseParser; print('BaseParser OK')"
```

**U-005 (이미지 파서)**:
// turbo
```powershell
python -c "from backend.parser.image_parser import ImageParser; print('ImageParser OK')"
```

**U-006 (PSD 파서)**:
// turbo
```powershell
python -c "from backend.parser.psd_parser import PSDParser; print('PSDParser OK')"
```

**U-007 (파이프라인)**:
// turbo
```powershell
python -m backend.pipeline.ingest_engine --help
```

9. 테스트 결과 확인:
   - ✅ PASS → 회고 단계로 진행
   - ❌ FAIL → `/troubleshoot` 실행

### 5️⃣ 회고 단계
10. 잘된 점, 개선점, 소요 시간 기록
11. `task.md`에서 해당 유닛 `[x]` (완료)로 마킹
12. 다음 유닛으로 진행
