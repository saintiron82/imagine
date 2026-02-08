---
description: ImageParser 프로젝트 개발 워크플로우
---
# ImageParser 개발 워크플로우

## 개요
이 워크플로우는 멀티모달 이미지 데이터화 시스템(ImageParser)의 구조화된 개발 절차를 안내합니다.

> [!IMPORTANT]
> **유닛 기반 개발 에이전트 프로토콜을 따릅니다.**
> 스킬 파일: `.agent/skills/unit_dev_agent/SKILL.md`
> 
> 핵심 원칙: "하나의 유닛을 완성하고, 테스트를 통과한 후에만 다음으로 진행한다."

## 사전 준비
1. Python 3.10+ 설치 확인
2. 프로젝트 루트: `c:\Users\saint\ImageParser`

## 개발 순서

### Phase 1: 프로젝트 초기화
// turbo
1. requirements.txt 생성
```
psd-tools>=2.0.0
Pillow>=10.0.0
pydantic>=2.0.0
watchdog>=3.0.0
exifread>=3.0.0
numpy
tqdm
```

// turbo
2. 디렉토리 구조 생성
```powershell
mkdir backend\parser
mkdir backend\pipeline
mkdir output\thumbnails
mkdir output\json
```

### Phase 2: 스키마 정의
3. `backend/parser/schema.py` 생성
   - AssetMeta Pydantic 모델 정의
   - Literal['PSD', 'PNG', 'JPG'] 포맷 타입
   - Vector Source 필드 정의

### Phase 3: 정제 모듈
4. `backend/parser/cleaner.py` 생성
   - clean_layer_name(): 정규화 (Regex로 "copy", 숫자 제거)
   - infer_content_type(): 휴리스틱 태깅 (Stub)

### Phase 4: 파서 구현
5. `backend/parser/base_parser.py` - 추상 기본 클래스
6. `backend/parser/psd_parser.py` - PSD 심층 파싱
7. `backend/parser/image_parser.py` - PNG/JPG 처리

### Phase 5: 파이프라인
8. `backend/pipeline/ingest_engine.py` - Watchdog + Factory Pattern

### Phase 6: 검증
// turbo
9. 테스트 실행
```powershell
python -m pytest tests/ -v
```

## 참고 문서
- 명세서: `spec.md`
- 구현 계획: `.gemini/antigravity/brain/.../implementation_plan.md`
- 작업 목록: `.gemini/antigravity/brain/.../task.md`
