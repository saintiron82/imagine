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

---

## [2026-02-21] v10.0: 워커 MC 데이터 손실 (VLM 키 매핑 오류)

### 증상
- 워커가 파일 처리 완료 후 서버 DB에서 `mc_caption`, `ai_tags` 필드가 NULL
- Vision(MC) Phase는 정상 실행되었으나 최종 메타데이터에 MC 데이터가 누락됨

### 원인 분석
- `worker_daemon.py`의 `_run_vision()` 반환값 키가 VLM 출력 키와 불일치
- VLM은 `caption`, `tags` 키를 반환하지만, `complete_job()`은 `mc_caption`, `ai_tags` 키를 기대
- `vision_fields`의 키 매핑이 누락되어 서버에 MC 데이터가 전달되지 않음

### 해결 방법
- `_run_vision()` 내부에서 VLM 반환값을 DB 스키마 키로 매핑:
```python
# VLM returns: {"caption": "...", "tags": [...], ...}
# DB expects: {"mc_caption": "...", "ai_tags": [...], ...}
key_map = {"caption": "mc_caption", "tags": "ai_tags", ...}
vision_fields = {key_map.get(k, k): v for k, v in raw_fields.items()}
```

### 예방책
- 워커와 로컬 파이프라인의 VLM 호출 경로가 다르므로, 키 매핑 테스트를 워커 전용으로 추가 필요
- `complete_job()` API에서 필수 키 누락 시 경고 로그 추가 검토

---

## [2026-02-21] v10.0: 워커 MV 임포트 오류 (text_embedding 함수명 불일치)

### 증상
- 워커에서 Phase MV(Qwen3-Embedding) 실행 시 `ImportError` 발생
- `from backend.vector.text_embedding import get_text_embedder` 실패

### 원인 분석
- `text_embedding.py` 모듈의 실제 함수명은 `get_text_embedding_provider()`
- 워커 코드에서 `get_text_embedder()`라는 존재하지 않는 함수명을 임포트 시도

### 해결 방법
```python
# ❌ 잘못된 임포트
from backend.vector.text_embedding import get_text_embedder

# ✅ 올바른 임포트
from backend.vector.text_embedding import get_text_embedding_provider
```

### 예방책
- 새 모듈 연동 시 실제 모듈의 `__all__` 또는 함수 시그니처를 확인
- IDE 자동완성이 없는 환경에서는 `grep` 등으로 실제 정의된 함수명 검증

---

## [2026-02-21] v10.2: IPC 워커 세션이 서버에 등록되지 않음

### 증상
- Electron Worker 앱에서 워커 시작 → Admin 패널 Workers 탭에 워커가 표시되지 않음
- 워커는 정상 동작 (Job claim + 처리 + 완료) 하지만 서버 DB의 `worker_sessions`에 레코드 없음

### 원인 분석
- `worker_ipc.py`의 `_worker_loop()`가 `WorkerDaemon()` 인스턴스를 생성하지만 **세션 등록 호출을 누락**
- CLI 워커의 `run()` 메서드는 `_connect_session()` → `_heartbeat()` → `_disconnect_session()`을 호출하지만, IPC 모드는 이 경로를 타지 않음
- IPC 모드에서는 `_authenticate()` + `claim_jobs()` + `process_batch_phased()` 만 호출

### 해결 방법
- `worker_ipc.py::_worker_loop()`에 세션 관리 호출 추가:
```python
# 1. 워커 인증 후 세션 등록
daemon._connect_session()

# 2. 메인 루프에서 주기적 하트비트
if time.time() - last_heartbeat >= heartbeat_interval:
    hb = daemon._heartbeat()
    cmd = hb.get("command")
    if cmd in ("stop", "block"):
        self._running = False

# 3. 종료 시 세션 해제
daemon._disconnect_session()
```

### 예방책
- CLI 워커(`run()`)와 IPC 워커(`_worker_loop()`)의 기능 패리티 체크리스트 유지
- 새 워커 기능 추가 시 양쪽 경로 모두 적용 확인
- 서버 Admin 패널에서 워커 등록 여부를 E2E 테스트 항목에 포함

---

## [2026-02-21] v10.x: 동일 머신 멀티워커 GPU 경합 (처리속도 이득 없음)

### 증상
- 같은 Mac(M5, 32GB)에서 워커 3개를 동시 실행
- 각 워커의 처리속도가 정확히 1/3로 감소 (예: 단독 4.2/min → 3개 각각 1.4/min)
- 총 처리량 ≈ 단독 처리량 (실질적 이득 없음)

### 원인 분석
- 3개 워커가 동일 GPU(Metal/MPS)를 공유하여 시분할(time-sharing)
- VLM(Qwen3-VL), SigLIP2, Qwen3-Embedding 모두 GPU 연산 → 경합 발생
- GPU 메모리도 3개 프로세스가 공유하므로 배치 크기 축소 가능성

### 해결 방법
- **멀티워커는 별도 GPU 머신에서만 의미 있음** — 단일 머신에서는 워커 1개 권장
- GPU가 없는 머신에서 CPU 전용 워커를 추가하는 것은 VLM Phase에서 극도로 느려 비실용적

### 예방책
- 문서화: "멀티워커는 별도 GPU 탑재 머신에 분산 배치할 때만 효과적"
- Admin 패널에서 워커별 처리속도를 모니터링하여 GPU 경합 여부 판단 가능
- 향후 과제: CPU-only 워커에서 VLM 스킵 + VV/MV만 처리하는 partial 모드 검토
