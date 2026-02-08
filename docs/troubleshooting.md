# 트러블슈팅 기록 (Troubleshooting Log)

개발 과정에서 발생한 문제와 해결 방법을 기록합니다.

---

<!-- 
템플릿:

## [YYYY-MM-DD] U-XXX: 문제 제목

### 증상
- 에러 메시지 또는 예상과 다른 동작

### 원인 분석
- 근본 원인 파악

### 해결 방법
- 적용한 해결책

### 예방책
- 재발 방지를 위한 조치
-->

---

## [2026-02-05] U-001: psd-tools 버전 오류

### 증상
- `pip install -r requirements.txt` 실패
- ERROR: No matching distribution found for psd-tools>=2.0.0

### 원인 분석
- psd-tools 라이브러리의 최신 버전이 2.x가 아닌 1.9.x

### 해결 방법
- requirements.txt에서 `psd-tools>=2.0.0` → `psd-tools>=1.9.0`으로 수정

### 예방책
- 패키지 버전 지정 전 PyPI에서 실제 버전 확인

---

## [2026-02-05] U-003: clean_layer_name 테스트 케이스 오류

### 증상
- `clean_layer_name("Layer 1 copy")` 결과가 "Layer"가 아닌 빈 문자열

### 원인 분석
- "Layer 1 copy" → 정규화 후 "Layer" → MEANINGLESS_NAMES에 포함 → 빈 문자열 반환
- 이는 **의도된 동작**임 (완전히 의미 없는 이름은 제거)

### 해결 방법
- 테스트 케이스를 의미 있는 이름으로 변경: `"Character_Body_01 copy"` → `"Character Body"`

### 예방책
- 테스트 설계 시 기능의 의도를 명확히 이해한 후 케이스 작성

---

## [2026-02-06] Phase 2~3 Analysis: 레이어 분석 및 한글화 오류

### 증상
- 복잡한 PSD 레이어 구조 분석 시 일부 누락이나 잘못된 계층 인식.
- 자동 번역 적용 시 한글 깨짐 현상 또는 문맥에 맞지 않는 오역 발생.

### 원인 분석
- **레이어 분석**: 중첩된 그룹(Nested Group) 처리 로직의 재귀 호출 부분에서 엣지 케이스 발생 가능성.
- **한글화**: 
    1. Python(`cp949`)과 Node.js(`utf-8`) 간의 IPC 통신 문자셋 불일치.
    2. `deep-translator` 라이브러리가 API 제한이나 네트워크 이슈로 실패 시 에러 처리 미흡.

### 해결 방법 (Pending)
- [ ] 레이어 파서의 재귀 로직을 단위 테스트로 재검증 및 강화.
- [ ] IPC 통신 시 모든 문자열을 명시적으로 UTF-8 인코딩/디코딩 처리.
- [ ] 번역 실패 시 원본 영문 텍스트를 그대로 사용하는 Fallback 로직 강화.

### 교훈
- "자동 번역"이나 "자동 분석"은 편의 기능이지만, 핵심 데이터 무결성을 해칠 수 있으므로 항상 원본(Raw Data)을 함께 보관해야 함.

---

## [2026-02-07] Windows 한글 인코딩 반복 이슈 (cp949/UTF-8 충돌)

### 증상

1. **Python stdout/stderr 출력 깨짐**: 한글 로그 메시지 출력 시 `UnicodeEncodeError` 또는 깨진 문자
2. **Windows 명령어 한글 깨짐**: `findstr`, `type`, `echo` 등 네이티브 명령어에서 한글 검색/출력 실패
   - 예: `findstr /C:"DONE"` → `FINDSTR: C:DONE은(는) 열 수 없습니다.`
3. **subprocess/IPC 인코딩 불일치**: Python↔Electron, Python↔CLI 간 한글 데이터 전달 시 깨짐
4. **datetime ISO 포맷 불일치**: DB 저장값(`2024-01-01 12:00:00`)과 Python `isoformat()`(`2024-01-01T12:00:00`)의 `T` vs 공백 차이

### 원인 분석

**근본 원인: Windows 기본 인코딩이 `cp949`(EUC-KR)**

| 환경 | 기본 인코딩 | 문제 |
|------|------------|------|
| Windows 콘솔 (cmd/PowerShell) | `cp949` | 한글 출력 깨짐 |
| Python `sys.stdout` | 콘솔 코드페이지 따름 (`cp949`) | `UnicodeEncodeError` |
| Windows `findstr` | `cp949` | UTF-8 파일/패턴 검색 실패 |
| Node.js / Electron | `utf-8` | Python↔Node IPC 불일치 |
| SQLite / JSON | `utf-8` | Python 읽기 시 인코딩 지정 필요 |

**발생 패턴**:
- Python 스크립트에서 한글 포함 문자열을 `print()` / `logger.info()` 출력
- Windows 명령줄에서 한글 키워드로 파일 내용 검색
- subprocess로 외부 프로세스 호출 시 stdin/stdout 인코딩 불일치
- datetime 포맷 비교 시 구분자 차이 (`T` vs ` `)

### 해결 방법

**1. Python stdout/stderr 강제 UTF-8 (이미 적용됨)**:
```python
# ingest_engine.py 상단에 이미 적용
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

**2. Windows 명령어 대신 Python/전용 도구 사용**:
```powershell
# ❌ 금지: Windows 네이티브 명령어로 한글 검색
findstr /C:"완료" output.log
type result.txt | findstr "성공"

# ✅ 권장: Python으로 직접 처리
python -c "
with open('output.log', encoding='utf-8') as f:
    for line in f:
        if '완료' in line: print(line.strip())
"

# ✅ 또는 Claude Code 전용 도구 사용
# Grep 도구 (ripgrep 기반, UTF-8 네이티브)
# Read 도구 (파일 직접 읽기)
```

**3. datetime 비교 시 정규화**:
```python
# ❌ 직접 비교 (포맷 불일치)
stored == current_mtime  # "2024-01-01 12:00:00" != "2024-01-01T12:00:00"

# ✅ 정규화 후 비교
stored.replace('T', ' ') == current.replace('T', ' ')
```

**4. 파일 I/O 시 명시적 인코딩**:
```python
# ❌ 인코딩 미지정 (cp949로 열릴 수 있음)
with open('data.json') as f: ...

# ✅ UTF-8 명시
with open('data.json', encoding='utf-8') as f: ...
```

**5. subprocess 호출 시 인코딩 지정**:
```python
# ❌ 기본 인코딩
result = subprocess.run(cmd, capture_output=True, text=True)

# ✅ UTF-8 명시
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
```

### 예방책 (재발 방지 체크리스트)

**Claude Code 에이전트용**:
- [ ] `findstr`, `type` 등 Windows 네이티브 명령어로 한글 검색 금지 → Grep/Read 도구 사용
- [ ] Bash에서 한글 포함 출력을 파이프할 때 `findstr` 대신 Python 스크립트 사용
- [ ] datetime 비교 시 반드시 `T`/공백 정규화 적용

**개발자용**:
- [ ] 모든 `open()` 호출에 `encoding='utf-8'` 명시
- [ ] 모든 `subprocess.run()` 호출에 `encoding='utf-8'` 명시
- [ ] 새 Python 스크립트 작성 시 상단에 `sys.stdout` UTF-8 래핑 고려
- [ ] IPC 통신(Electron↔Python) 시 양쪽 모두 UTF-8 명시

### 교훈

Windows 환경에서 한글 처리는 **기본이 깨진다고 가정**하고, 모든 I/O 경계에서 명시적으로 `utf-8`을 지정해야 한다. 특히 Claude Code 환경에서는 Bash 도구로 한글 포함 출력을 처리할 때 Windows 네이티브 명령어(`findstr`, `type`)를 절대 사용하지 말고 전용 도구(Grep, Read)를 사용해야 한다.
