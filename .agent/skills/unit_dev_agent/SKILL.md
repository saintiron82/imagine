---
description: 단계 분리 개발 에이전트 프로토콜 (5단계 구조)
---
# 단계 분리 개발 에이전트 프로토콜

## 핵심 원칙
각 유닛(Unit)은 반드시 **5단계 프로세스**를 순차적으로 통과해야 합니다.
문제 발생 시 `troubleshooting.md`에 기록하고, 해결 후 다음 단계로 진행합니다.

---

## 유닛 실행 5단계 프로세스

### 1️⃣ 목표 (Goal)
현재 유닛에서 달성해야 할 **명확한 결과물**을 정의합니다.

```markdown
## U-XXX: [유닛명]
### 1. 목표
- 달성할 결과: [구체적인 산출물]
- 완료 조건: [측정 가능한 기준]
```

### 2️⃣ 정의 (Definition)
목표를 달성하기 위한 **상세 명세**를 작성합니다.

```markdown
### 2. 정의
- 입력: [필요한 선행 조건 또는 데이터]
- 출력: [생성될 파일 또는 결과]
- 의존성: [선행 유닛 목록]
- 제약 사항: [기술적 제약 또는 규칙]
```

### 3️⃣ 개발 (Development)
정의에 따라 **코드를 작성**합니다.

```markdown
### 3. 개발
- 생성 파일: `path/to/file.py`
- 주요 로직: [핵심 구현 내용]
- 코드 라인 수: ~XX lines
```

### 4️⃣ 테스트 (Test)
개발된 코드가 **목표를 달성했는지 검증**합니다.

```markdown
### 4. 테스트
- 테스트 명령어:
  ```powershell
  python -c "from module import func; assert func('input') == 'expected'"
  ```
- 결과: ✅ PASS / ❌ FAIL
- 실패 시: → troubleshooting.md에 기록 후 재개발
```

### 5️⃣ 회고 (Retrospective)
유닛 완료 후 **배운 점과 개선점**을 기록합니다.

```markdown
### 5. 회고
- 잘된 점: [효과적이었던 접근법]
- 개선점: [다음에 주의할 사항]
- 소요 시간: ~XX분
- 다음 유닛: U-XXX
```

---

## 트러블슈팅 프로토콜

### 문제 발생 시 필수 기록
문제가 발생하면 **즉시** `troubleshooting.md`에 기록합니다.

```markdown
## [날짜] U-XXX: [문제 제목]

### 증상
- 에러 메시지 또는 예상과 다른 동작

### 원인 분석
- 근본 원인 파악

### 해결 방법
- 적용한 해결책

### 예방책
- 재발 방지를 위한 조치
```

---

## 유닛 전환 규칙

| 조건 | 행동 |
|------|------|
| 테스트 PASS | → 회고 작성 → 다음 유닛으로 진행 |
| 테스트 FAIL | → troubleshooting.md 기록 → 재개발 → 재테스트 |
| 3회 연속 FAIL | → 사용자에게 알림 → 방향 재검토 요청 |

---

## 유닛 정의서 (Phase 1)

### U-001: 프로젝트 초기화
- **목표**: Python 환경 및 디렉토리 구조 생성
- **정의**: requirements.txt, backend/, output/ 폴더
- **테스트**: `pip install -r requirements.txt` 성공

### U-002: 데이터 스키마
- **목표**: AssetMeta Pydantic 모델 정의
- **정의**: schema.py 파일, 모든 필드 타입 힌트
- **테스트**: `from backend.parser.schema import AssetMeta`

### U-003: 데이터 정제 모듈
- **목표**: 레이어 이름 정규화 함수
- **정의**: cleaner.py, clean_layer_name(), infer_content_type()
- **테스트**: `assert clean_layer_name("Layer 1 copy") == "Layer"`

### U-004: 기본 파서 인터페이스
- **목표**: 추상 기본 클래스 정의
- **정의**: base_parser.py, ABC, abstract parse() method
- **테스트**: 상속 시 parse() 미구현 에러 발생

### U-005: 이미지 파서
- **목표**: PNG/JPG 파일 처리
- **정의**: image_parser.py, Pillow/exifread 활용
- **테스트**: 샘플 PNG → JSON 출력

### U-006: PSD 파서
- **목표**: PSD 심층 파싱
- **정의**: psd_parser.py, psd-tools 활용
- **테스트**: 샘플 PSD → JSON + 썸네일 출력

### U-007: 파이프라인 통합
- **목표**: CLI 기반 통합 엔진
- **정의**: ingest_engine.py, Factory Pattern
- **테스트**: CLI로 3종 파일 처리

### U-008: 통합 테스트
- **목표**: E2E 검증
- **정의**: 테스트 스크립트, 스키마 일치 확인
- **테스트**: 전체 파이프라인 정상 동작
