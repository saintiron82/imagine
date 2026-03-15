# 내장 워커 모드 설계 (Builtin Worker Mode)

**날짜**: 2026-03-04
**접근 방식**: C. 하이브리드 (ParseAheadPool 확장 + 가상 워커 세션)

## 목적

서버 머신에서 Electron을 하나 더 워커로 띄우는 번거로움 제거.
전역 처리 모드에 `builtin_worker`를 추가하여, 서버가 외부 워커 유무와 무관하게
항상 전체 파이프라인(P→V→VV→MV)을 직접 처리.

## 현재 vs 변경

| 항목 | 현재 auto 모드 | 내장 워커 모드 |
|------|---------------|--------------|
| 워커 없을 때 | 서버 자동 처리 (ON이면) | 항상 서버 처리 |
| 워커 연결 시 | 서버→워커 위임 (distribute) | 서버 계속 처리 (워커도 병행 가능) |
| UI 표시 | 워커 테이블에 안 보임 | 가상 워커로 테이블에 표시 |
| 진행 정보 | 제한적 | 워커와 동일 (Phase, 속도, 현재 파일) |
| 제어 | 서버 자동 처리 OFF | 모드 전환으로 제어 |

## UI 변경

### 전역 처리 모드 버튼

```
현재: [MC Only] [Parse Only] [자동]
변경: [MC Only] [Parse Only] [내장 워커] [자동]
```

- 색상: 보라색 (`purple-600`)
- 설명: "서버가 직접 전체 파이프라인(P→V→VV→MV)을 처리합니다. 외부 워커 없이 독립 운영."

### 서버 자동 처리 섹션

- `builtin_worker` 모드일 때: 숨김 (내장 워커가 항상 처리하므로 불필요)
- 다른 모드일 때: 기존대로 표시

### 워커 테이블

- `builtin_worker` 모드 활성 시 가상 워커 행 표시:
  - 이름: "내장 워커" (Built-in Worker)
  - 사용자명: "server"
  - 상태: online (초록 뱃지)
  - 능력: "full"
  - 현재 작업/Phase: ParseAheadPool에서 갱신
  - 속도: throughput 계산

## 백엔드 변경

### 1. `backend/server/queue/manager.py`

`get_processing_mode()` 허용값에 `"builtin_worker"` 추가.

```python
def get_processing_mode():
    mode = cfg.get("server.processing_mode") or "auto"
    if mode not in ("mc_only", "parse_only", "auto", "builtin_worker"):
        mode = "auto"
    return mode
```

### 2. `backend/server/routers/workers.py`

#### `admin_update_global_config()`

`"builtin_worker"` 허용. 선택 시:
- 가상 워커 세션 생성/활성화
- ParseAheadPool `_processing_mode = "auto"` 고정
- 서버 자동 처리 강제 활성화

```python
elif mode == "builtin_worker":
    # 가상 워커 세션 생성/활성화
    _ensure_builtin_worker_session(db)
    # 외부 워커 override는 건드리지 않음 (병행 가능)
```

#### `_recalculate_server_pools()`

`builtin_worker` 모드일 때:
- 외부 워커 유무 무시
- `ParseAheadPool._processing_mode = "auto"` 항상 유지
- `EmbedAheadPool` 불필요 (ParseAhead가 MV까지 처리)

```python
if global_mode == "builtin_worker":
    app.state.parse_ahead._processing_mode = "auto"
    # EmbedAhead 불필요 — ParseAhead가 P→V→VV→MV 전체 처리
    return
```

#### 새 함수: `_ensure_builtin_worker_session(db)`

```python
def _ensure_builtin_worker_session(db):
    """가상 워커 세션을 생성하거나 기존 것을 online으로 복원."""
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT id FROM worker_sessions WHERE worker_name = '__builtin__' AND status = 'online'"
    )
    if cursor.fetchone():
        return  # 이미 활성
    # 기존 offline 세션 재활성화 또는 신규 생성
    cursor.execute(
        "UPDATE worker_sessions SET status = 'online', last_heartbeat = ? "
        "WHERE worker_name = '__builtin__' AND status = 'offline'",
        (_utcnow_sql(),)
    )
    if cursor.rowcount == 0:
        # admin user_id=1 사용
        cursor.execute(
            """INSERT INTO worker_sessions
               (user_id, worker_name, hostname, batch_capacity, status, connected_at, last_heartbeat)
               VALUES (1, '__builtin__', 'server', 5, 'online', ?, ?)""",
            (_utcnow_sql(), _utcnow_sql())
        )
    db.conn.commit()
```

#### 새 함수: `_deactivate_builtin_worker_session(db)`

모드 전환 시 가상 세션 offline 처리.

### 3. `backend/server/queue/parse_ahead.py`

ParseAheadPool이 `auto` 모드에서 처리할 때, 가상 워커 세션의 상태를 갱신:

- `current_file`: 현재 처리 중인 파일명
- `current_phase`: 현재 Phase (parse/vision/embed_vv/embed_mv)
- `jobs_completed`: 완료 카운트 증가
- `last_heartbeat`: 주기적 갱신

이를 위해 기존 `_process_auto_batch()` 내에 콜백 추가:

```python
def _update_builtin_session(self, phase, file_name=None, completed=False):
    """Update virtual worker session for UI visibility."""
    if self._processing_mode != "auto" or not self._is_builtin_mode():
        return
    # UPDATE worker_sessions SET current_phase=?, current_file=?, ...
    # WHERE worker_name = '__builtin__'
```

### 4. 하트비트 워치독 예외

`_start_heartbeat_watchdog()`에서 `__builtin__` 세션은 타임아웃 체크 제외.
(서버 내부 스레드이므로 하트비트 불필요 — ParseAheadPool이 직접 갱신)

## 프론트엔드 변경

### 1. `frontend/src/pages/AdminPage.jsx`

- 버튼 그룹에 `builtin_worker` 추가 (보라색)
- `builtin_worker` 모드일 때 서버 자동 처리 섹션 숨김
- 워커 테이블에서 `__builtin__` 세션 특수 표시 (아이콘/색상 차별화)

### 2. `frontend/src/i18n/locales/ko-KR.json`

```json
"admin.worker_mode_builtin": "내장 워커",
"admin.worker_mode_builtin_desc": "서버가 직접 전체 파이프라인(P→V→VV→MV)을 처리합니다. 외부 워커 없이 독립 운영."
```

### 3. `frontend/src/i18n/locales/en-US.json`

```json
"admin.worker_mode_builtin": "Built-in Worker",
"admin.worker_mode_builtin_desc": "Server processes the full pipeline (P→V→VV→MV) directly. Operates independently without external workers."
```

## 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `backend/server/queue/manager.py` | `get_processing_mode()` 허용값 추가 |
| `backend/server/routers/workers.py` | 모드 처리, 가상 세션 관리, 풀 재계산 |
| `backend/server/queue/parse_ahead.py` | 가상 세션 상태 갱신 콜백 |
| `backend/server/app.py` | 하트비트 워치독 예외 |
| `frontend/src/pages/AdminPage.jsx` | UI 버튼/조건부 렌더링 |
| `frontend/src/i18n/locales/ko-KR.json` | 번역 키 |
| `frontend/src/i18n/locales/en-US.json` | 번역 키 |

## 동작 시나리오

### 시나리오 1: 내장 워커 모드 선택 (워커 없음)
1. Admin이 "내장 워커" 버튼 클릭
2. API: `PATCH /admin/workers/global-config {processing_mode: "builtin_worker"}`
3. 서버: config 저장, 가상 세션 생성, ParseAheadPool → auto 모드
4. UI: 워커 테이블에 "내장 워커" 행 표시, 서버 자동 처리 섹션 숨김
5. ParseAheadPool: P→V→VV→MV 전체 처리, 가상 세션에 진행 상태 기록

### 시나리오 2: 내장 워커 모드 → 다른 모드 전환
1. Admin이 "자동" 버튼 클릭
2. 서버: 가상 세션 offline 처리, ParseAheadPool 모드 재계산
3. UI: 워커 테이블에서 가상 워커 사라짐

### 시나리오 3: 내장 워커 + 외부 워커 병행
1. 내장 워커 모드 활성 상태에서 외부 워커 연결
2. 서버 ParseAheadPool: 계속 auto 처리 (중단 없음)
3. 외부 워커: claim_jobs()로 Job 청구 가능 (병행 처리)
4. 워커 테이블: 내장 워커 + 외부 워커 모두 표시
