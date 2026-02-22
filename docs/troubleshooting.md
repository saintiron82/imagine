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

## [2026-02-22] v9.3: Windows 워커 Parse Phase 무한 행(Hang) — stdin 파이프 I/O 락 데드락

### 증상
- Windows Electron 앱에서 워커 시작 → 서버 접속, 파일 다운로드 모두 정상
- "Phase parse — 3 files" 로그 출력 후 **영구적으로 멈춤** (CPU 0%, 타임아웃까지 무반응)
- PSD 파싱 중 `psd.composite()` 호출 시점에서 정확히 멈춤
- macOS에서는 동일 코드가 정상 동작 (**Windows 전용 문제**)

### 원인 분석

**근본 원인: Windows CRT(C Runtime) I/O 락과 piped stdin의 충돌**

```
Electron (main.cjs)
  └─ spawn("python", ["-u", "worker_ipc.py"], { stdio: ['pipe', 'pipe', 'pipe'] })
       ├─ stdin: Electron → Python (JSON 명령)
       ├─ stdout: Python → Electron (JSON 이벤트)
       └─ stderr: Python → Electron (로그)
```

1. `worker_ipc.py`의 메인 스레드가 `for line in sys.stdin`으로 **stdin을 블로킹 읽기**
2. 워커 로직은 **백그라운드 스레드**에서 `psd.composite()` 실행 (numpy C-extension 호출)
3. **Windows에서만 발생**: 파이프된 stdin의 블로킹 `ReadFile()`이 CRT/커널 I/O 락을 보유
4. 백그라운드 스레드의 numpy/C-extension이 같은 I/O 서브시스템을 사용하려 할 때 **데드락**
5. macOS/Linux에서는 이 락 경합이 발생하지 않음 (POSIX I/O 모델이 다름)

**검증 과정** (체계적 격리 테스트):

| 테스트 조건 | stdin 방식 | 스레드 | 결과 |
|------------|-----------|--------|------|
| subprocess, 메인 스레드 | pipe | 메인 | ✅ 0.3초 정상 |
| subprocess, 백그라운드 스레드 | pipe (stdin 안 읽음) | 백그라운드 | ✅ 0.3초 정상 |
| subprocess, 백그라운드 스레드 | pipe (stdin 블로킹 읽기) | 백그라운드 | ❌ **데드락** |
| subprocess, 백그라운드 스레드 | pipe (os.read 블로킹) | 백그라운드 | ❌ **데드락** |
| subprocess, 백그라운드 스레드 | pipe (별도 스레드에서 블로킹) | 백그라운드 | ❌ **데드락** |
| subprocess, 백그라운드 스레드 | pipe (PeekNamedPipe 논블로킹) | 백그라운드 | ✅ **0.3초 정상** |

**핵심 발견**: Windows에서는 어떤 스레드든 파이프된 stdin을 블로킹 읽기하면, 다른 스레드의 C-extension I/O가 멈춤.
유일한 해결책: 블로킹 읽기를 아예 하지 않는 것 (논블로킹 폴링).

### 해결 방법

**Win32 `PeekNamedPipe` 논블로킹 stdin 리더** (`worker_ipc.py`):

```python
def _make_win32_stdin_reader():
    """Windows 전용: PeekNamedPipe로 stdin을 논블로킹 폴링."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    buf = bytearray()

    def readline():
        nonlocal buf
        while True:
            available = wintypes.DWORD(0)
            ok = kernel32.PeekNamedPipe(
                handle, None, 0, None, ctypes.byref(available), None
            )
            if not ok:
                return _STDIN_CLOSED  # 파이프 닫힘

            if available.value > 0:
                # 데이터 있을 때만 ReadFile (논블로킹)
                read_buf = ctypes.create_string_buffer(available.value)
                bytes_read = wintypes.DWORD(0)
                kernel32.ReadFile(handle, read_buf, available.value,
                                  ctypes.byref(bytes_read), None)
                buf.extend(read_buf.raw[:bytes_read.value])
                if b"\n" in buf:
                    # 라인 완성 시 반환
                    idx = buf.index(b"\n")
                    line = bytes(buf[:idx])
                    del buf[:idx + 1]
                    return line.decode("utf-8", errors="replace").strip() or None
            else:
                time.sleep(0.05)  # 50ms 폴링 (CPU 부하 최소)
    return readline
```

**플랫폼 분기** (`main()` 함수):
```python
if sys.platform == "win32":
    _read_line = _make_win32_stdin_reader()   # 논블로킹 폴링
else:
    _line_iter = iter(sys.stdin)              # 블로킹 (Unix에서는 안전)
    def _read_line():
        try:
            return next(_line_iter).strip() or None
        except StopIteration:
            return _STDIN_CLOSED
```

**추가 조치 (이전 커밋에서 적용)**:
- 무거운 모듈(numpy, psd_tools, PIL, torch)을 메인 스레드에서 사전 import
  - Windows `LoadLibrary` + Python import lock = 백그라운드 스레드 DLL 로딩 데드락 방지
- `ingest_engine.py`의 `sys.stdout` UTF-8 래핑을 `__main__` 가드로 보호
  - 모듈로 import될 때 stdout 스트림 교체 방지

### 플랫폼별 차이 요약

| 항목 | Windows | macOS / Linux |
|------|---------|---------------|
| **stdin 읽기** | Win32 PeekNamedPipe (논블로킹 50ms 폴링) | `for line in sys.stdin` (블로킹) |
| **DLL/SO 로딩** | 메인 스레드 사전 import 필수 (LoadLibrary 락) | 사전 import 불필요 (안전) |
| **stdout 인코딩** | UTF-8 강제 래핑 필요 (기본 cp949) | UTF-8 기본 |
| **프로세스 종료** | SIGKILL 필요 (SIGTERM 무시 가능) | SIGTERM 정상 동작 |

### 예방책

- **Windows에서 piped subprocess 개발 시**: 블로킹 stdin 읽기 + 백그라운드 C-extension 조합 금지
- **크로스 플랫폼 IPC 코드**: `sys.platform` 분기로 Windows/Unix 별도 처리
- **새 워커 기능 추가 시**: Windows/macOS 양쪽에서 테스트 필수
- **C-extension 사용 모듈** (numpy, torch, psd-tools): Windows에서는 메인 스레드 사전 import

### 후속 조치 (v0.5.0)

- **공용 유틸리티 추출**: 인라인 Win32 코드를 `backend/utils/win32_stdin.py`로 추출.
  `make_stdin_reader()` 함수가 플랫폼을 자동 감지하여 적절한 리더 반환.
- **검색 데몬 분석**: `api_search.py`의 `run_daemon()`도 `for line in sys.stdin` 사용하나,
  단일 스레드 구조이므로 CRT I/O 락 경합 위험 없음 (백그라운드 스레드 부재). 수정 불필요.
- **규칙 명시**: 향후 새 Python subprocess가 stdin 파이프 + 백그라운드 C-extension 조합을
  사용할 경우 반드시 `make_stdin_reader()` 사용 (CLAUDE.md에 규칙 추가).

### 교훈

macOS에서 정상 동작하는 코드가 Windows에서 데드락을 일으킬 수 있다. 파이프된 stdin의 I/O 락 동작이 OS별로 근본적으로 다르며, Windows CRT의 I/O 직렬화는 문서화가 거의 되어있지 않아 디버깅이 극도로 어렵다.
**"동일 코드, 다른 OS = 다른 동작"을 항상 가정**해야 한다.

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

---

## [2026-02-22] v0.5: 부모 프로세스 사망 시 자식 잔류 — 3가지 버그 수정

### 증상
- Electron 앱을 강제 종료(taskkill /F)하면 Python subprocess(Server, Pipeline, Discover)가 잔류
- 잔류 프로세스가 포트(8000)와 GPU 메모리를 점유하여 앱 재시작 실패
- Windows에서만 프로세스 트리 킬이 동작하지 않음

### 원인 분석

**버그 1 (CRITICAL): Pipeline/Discover의 stdin 파이프 누락**
- `main.cjs`에서 `spawn(..., { detached: true })` 시 `stdio` 옵션 미지정
- Node.js 기본값으로 부모의 stdio를 상속 → stdin이 파이프가 아닌 부모의 stdin
- `parent_watchdog.py`의 stdin 모니터링이 작동하지 않음
- 더블클릭으로 앱 실행 시 stdin이 `/dev/null` → `read()` 즉시 EOF → 프로세스 즉시 종료 위험

**버그 2 (MEDIUM): parent_watchdog.py의 BufferedReader 락**
- `sys.stdin.buffer.read()` 사용 → Python BufferedReader 내부 락을 블로킹 중 보유
- 백그라운드 C-extension 스레드(numpy, torch)와 경합 가능성

**버그 3 (MEDIUM): Windows에서 프로세스 그룹 킬 실패**
- `process.kill(-pid, 'SIGKILL')` — 음수 PID는 POSIX 프로세스 그룹 시그널로, Windows 미지원
- Windows에서 항상 예외 → catch 블록이 루트 프로세스만 종료, 자식 트리 미종료

### 해결 방법

**3계층 방어 체계 구축**:

| 계층 | 메커니즘 | OS | 반응 속도 |
|------|---------|-------|----------|
| **Layer 1** | `prctl(PR_SET_PDEATHSIG, SIGKILL)` | Linux 전용 | 즉시 (커널 레벨) |
| **Layer 2** | stdin 파이프 EOF 감지 (`os.read()`) | 전체 | ~수ms |
| **Layer 3** | PID 폴링 (2초 간격) | 전체 (폴백) | ~2초 |

**구체적 수정**:
1. Pipeline/Discover spawn에 `stdio: ['pipe', 'pipe', 'pipe']` 추가 (stdin lifeline 확보)
2. `parent_watchdog.py` 리라이트: `sys.stdin.buffer.read()` → `os.read(fd, 1024)` (BufferedReader 락 회피)
3. Linux `prctl(PR_SET_PDEATHSIG, SIGKILL)` 추가 (커널이 직접 SIGKILL 전달)
4. `killProcessTree()` 헬퍼 추가: Windows `taskkill /F /T /PID`, Unix `process.kill(-pid)`

### 예방책
- 새 Python subprocess spawn 시 반드시 `stdio: ['pipe', 'pipe', 'pipe']` 명시
- stdin 읽기에는 `sys.stdin.buffer.read()` 대신 `os.read()` 사용
- Windows 프로세스 종료에는 `taskkill /F /T /PID` 사용 (프로세스 트리 전체)
- `parent_watchdog.py`는 stdin 파이프가 있는 모든 장수명 subprocess에서 호출

---

## [2026-02-22] v0.5: 서버-워커 시스템 4가지 이슈 수정

### 이슈 1: VisionAnalyzerFactory.reset() 메모리 누수

**증상**: 워커 데몬에서 반복 배치 처리 시 GPU 메모리 폭증

**원인**: `VisionAnalyzerFactory.reset()`이 `cls._cached_analyzer = None`만 하고 `unload_model()` 미호출. GC에 의존하지만 순환 참조로 즉시 해제 안 됨. `_unload_vlm_verified()`가 `get_vision_analyzer()` 호출 → 캐시가 이미 None이면 새 인스턴스 생성하는 버그.

**해결**: `reset()`에 `unload_model()` 호출 추가. `_unload_vlm_verified()`에서 `VisionAnalyzerFactory._cached_analyzer` 직접 접근.

### 이슈 2: 다중 워커 throughput 합산 문제

**증상**: 같은 유저의 워커 N대가 Admin 패널에서 동일한 throughput 표시

**원인**: `job_queue.assigned_to`가 `user_id`만 저장. `admin_list_workers()`에서 `assigned_to`로 GROUP BY → 같은 유저의 워커 throughput 합산.

**해결**: `job_queue`에 `worker_session_id` 컬럼 추가. `claim_jobs()`에서 session_id 함께 저장. throughput 계산에서 `worker_session_id`로 GROUP BY.

### 이슈 3: complete_job 중복 upsert_metadata

**증상**: pre_parsed 잡에서 불필요한 DB 쓰기 + FTS 리빌드

**원인**: `complete_job`이 `parse_status`를 확인하지 않고 항상 `upsert_metadata()` 호출. ParseAheadPool이 이미 저장한 메타데이터를 다시 쓰는 중복.

**해결**: `complete_job`에서 `parse_status == 'parsed'`이면 `upsert_metadata` 스킵, 기존 file_id 조회만 수행.

### 이슈 4: Windows 배치 모드 현황 (수정 불필요)

**현황 확인 결과**:
- **VLM**: batch_size=1 고정 — Qwen3-VL의 구조적 제약 (이미지+텍스트 혼합 입력). 모든 플랫폼 동일.
- **VV (SigLIP2)**: ✅ Adaptive batch discovery로 CUDA에서 배치 자동 증가.
- **MV (Qwen3-Embedding)**: ✅ `encode_batch()` 배치 정상 지원.

**결론**: Windows 특유 문제 아님. VV/MV는 정상 배치 운영.
