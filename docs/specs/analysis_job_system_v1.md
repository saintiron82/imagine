# 큐 시스템 완전 재설계 — 분석작업 (Analysis Job)

## 핵심 정의

### 분석작업 (Analysis Job)
사용자가 폴더를 지정하여 분석 요청하는 단위.
지정 폴더 + 모든 하위 폴더의 파일을 포함하는 **단일 플랫 큐**.

- 하위 큐 없음 (work_subtasks 폐기)
- Recovery 큐 없음 (실패 파일은 같은 작업 안에서 재시도)
- 분석작업 1개 = 파일 N개

### 파일 1개의 처리 흐름

```
Download(WebDAV) 또는 즉시(로컬)
    ↓
Parse (썸네일 생성 + PSD/PNG 메타데이터)
    ↓
    ├─ MC (이미지→캡션/태그) ──→ MV (MC텍스트→벡터)
    │                              ↑ MC 완료 대기
    └─ VV (이미지→벡터)     ← MC와 독립, 동시 가능
    ↓
Done (MC + VV + MV 모두 존재)
```

**의존 관계:**
- MC ← Parse (썸네일)
- VV ← Parse (썸네일) — MC와 독립
- MV ← MC (캡션 텍스트 필요)
- Done = MC ∧ VV ∧ MV

**실패 처리:**
- 롤백 가능한 지점에서 이어서 처리
- MC 실패 → MC만 재시도 (Parse 안 다시 함)
- VV 실패 → VV만 재시도
- Download 실패 → Download부터 재시도
- 최대 N회 후 permanent fail

### 분석작업 생명주기
```
생성 (사용자가 폴더 지정)
  → 진행 중 (파일들이 단계별로 처리됨)
  → 사용자가 닫을 때까지 유지
```

### 워커 배정 (병렬 활용)
```
워커 A (GPU strong): MC 전담 → MC 끝난 파일 MV도 처리
워커 B (GPU weak):   VV 전담 → Parse만 끝나면 바로 시작 (MC 대기 불필요)
임베디드:            병목 Phase 자동 배정
```

---

## 폐기 대상

| 현재 | 상태 | 대체 |
|------|:----:|------|
| `work_subtasks` 테이블 | 폐기 | 불필요 (플랫 큐) |
| `job_completions` 테이블 | 폐기 | job_queue에서 직접 계산 |
| Recovery WR 자동 생성 | 폐기 | 같은 작업 안에서 retry |
| `work_requests.completed_count` | 폐기 | files DB 실시간 계산 |
| `work_requests.failed_count` | 폐기 | files DB 실시간 계산 |
| `phase_completed` JSON | 폐기 | files DB로 판별 |
| `_update_wr_counters()` | 폐기 | 카운터 자체 없음 |
| audit의 job DELETE | 폐기 | job 보존 |

## 신규 데이터 모델

### `analysis_jobs` (분석작업 — work_requests 대체)

```sql
CREATE TABLE analysis_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                    -- "크랑베르무"
    source_path TEXT NOT NULL,             -- "webdav://13730b09/예비/크랑베르무"
    status TEXT DEFAULT 'active'           -- active / paused / completed / cancelled
        CHECK (status IN ('active','paused','completed','cancelled')),
    total_files INTEGER NOT NULL DEFAULT 0,
    created_by INTEGER REFERENCES users(id),
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
);
```

### `file_tasks` (파일별 작업 — job_queue 대체)

```sql
CREATE TABLE file_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_job_id INTEGER NOT NULL REFERENCES analysis_jobs(id),
    file_id INTEGER NOT NULL REFERENCES files(id),
    file_path TEXT NOT NULL,
    
    -- 준비 단계
    download_status TEXT DEFAULT 'pending'    -- pending/done/failed (WebDAV만)
        CHECK (download_status IN ('pending','done','failed','n/a')),
    parse_status TEXT DEFAULT 'pending'       -- pending/done/failed
        CHECK (parse_status IN ('pending','done','failed')),
    
    -- AI 단계 (독립적 — MC/VV 동시 가능)
    mc_status TEXT DEFAULT 'pending'          -- pending/done/failed
        CHECK (mc_status IN ('pending','done','failed')),
    vv_status TEXT DEFAULT 'pending'          -- pending/done/failed
        CHECK (vv_status IN ('pending','done','failed')),
    mv_status TEXT DEFAULT 'pending'          -- pending/done/failed (MC 완료 후)
        CHECK (mv_status IN ('pending','done','failed')),
    
    -- 메타
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    assigned_worker INTEGER REFERENCES worker_sessions(id),
    priority INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    
    UNIQUE(analysis_job_id, file_id)
);
```

**핵심 차이**: `phase_completed` JSON 대신 **각 Phase별 독립 status 컬럼**.
MC/VV가 독립이므로 `mc_status=done, vv_status=pending` 상태가 가능.

### Phase 상태 판별

파일 1개의 현재 Phase = **가장 뒤처진 단계**:

```sql
CASE
    WHEN mc_status='done' AND vv_status='done' AND mv_status='done' THEN 'done'
    WHEN mc_status='done' AND vv_status='done' AND mv_status!='done' THEN 'mv'
    WHEN mc_status='done' AND vv_status!='done' THEN 'vv'
    WHEN vv_status='done' AND mc_status!='done' THEN 'mc'
    WHEN parse_status='done' THEN 'ai'     -- MC/VV 둘 다 대기
    WHEN download_status IN ('done','n/a') THEN 'parse'
    ELSE 'download'
END
```

### 분석작업 전체 진행률 (단일 쿼리)

```sql
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN download_status IN ('done','n/a') THEN 1 ELSE 0 END) AS downloaded,
    SUM(CASE WHEN parse_status = 'done' THEN 1 ELSE 0 END) AS parsed,
    SUM(CASE WHEN mc_status = 'done' THEN 1 ELSE 0 END) AS mc_done,
    SUM(CASE WHEN vv_status = 'done' THEN 1 ELSE 0 END) AS vv_done,
    SUM(CASE WHEN mv_status = 'done' THEN 1 ELSE 0 END) AS mv_done,
    SUM(CASE WHEN mc_status='done' AND vv_status='done' AND mv_status='done'
        THEN 1 ELSE 0 END) AS complete
FROM file_tasks
WHERE analysis_job_id = ?
```

**합계 검증**: `complete <= mv_done, mc_done, vv_done <= parsed <= downloaded <= total`

---

## API

### `GET /api/v1/analysis-jobs`
분석작업 목록 + 파일 기반 진행률

### `POST /api/v1/analysis-jobs`
새 분석작업 생성 (폴더 지정)

### `GET /api/v1/analysis-jobs/{id}/progress`
특정 분석작업의 Phase별 진행률

### `POST /api/v1/analysis-jobs/{id}/pause`
### `POST /api/v1/analysis-jobs/{id}/resume`
### `POST /api/v1/analysis-jobs/{id}/cancel`

### 워커 claim:
### `POST /api/v1/tasks/claim`
워커가 자신의 mode(mc/vv/mv)에 맞는 task를 가져감

---

## 프론트엔드 대시보드

```
활성 분석작업
┌─────────────────────────────────────────────────┐
│ 크랑베르무 (3190)                    93.5%       │
│ ██████████████████████████████░░░░               │
│                                                   │
│ Download  Parse   MC    VV    MV    Done          │
│   OK       OK    2992  3054  2694  2694           │
│                   /3190 /3190 /3190               │
│                                                   │
│ 실패: 5  ·  남은: 496  ·  3.6/min  ·  ETA 2m     │
│                                          [일시정지]│
└─────────────────────────────────────────────────┘
```

---

## 마이그레이션

### 1. 새 테이블 생성
`analysis_jobs` + `file_tasks` 생성. 기존 테이블 유지.

### 2. 데이터 마이그레이션
`work_requests` (is_recovery=false) → `analysis_jobs`
`job_queue` → `file_tasks` (Phase 상태는 files DB에서 역산)

### 3. 코드 전환
`manager.py` 새로 작성 (기존 2931줄 → 간결한 새 구현)
워커 claim 로직 교체
프론트엔드 교체

### 4. 레거시 제거
`work_requests`, `work_subtasks`, `job_completions`, `job_queue` 미사용 처리
(당장 DROP 안 하고, 새 코드가 안정화된 후 제거)

---

## 정합성 검증 결과

### 코드 레벨 확인

**1. VV는 실제로 MC에 독립적인가?** → **YES ✓**
- `PhaseRunner.run_vv()`: 썸네일만 사용, `mc_raw` 참조 없음
- `SigLIP2Encoder.encode_image()`: PIL Image만 받음
- 현재 `_PHASE_FILTERS["vv"]`가 `vision=1` 요구 → **이것이 불필요한 제약이었음**

**2. MV는 MC에 의존하는가?** → **YES ✓**
- `PhaseRunner.run_mv()`: `item.mc_raw` 필터링 — MC 없으면 skip
- `_compose_mv_text()`: `mc_caption` + `ai_tags`로 텍스트 생성
- **MV는 반드시 MC 완료 후**

**3. `run_all()`은 순차 강제인가?** → **YES, 하지만 불필요**
- 현재: `run_vision()` → `run_vv()` → `run_mv()` (직렬)
- 문제: VV가 MC 끝날 때까지 대기 — 불필요한 직렬화
- 새 구조: 워커별 독립 claim이므로 `run_all()` 자체가 불필요

**4. 기존 `_PHASE_FILTERS`의 오류**
```
현재 VV 필터: vision=1 AND vv=0  ← MC 완료를 요구 (잘못됨!)
올바른 VV:    parse=done AND vv=pending  ← Parse만 되면 VV 가능
```
이것이 기존 시스템에서 VV가 MC를 기다리던 근본 원인.

### 논리적 문제

**5. 워커 배치 모드와 새 구조의 충돌**
- 현재: `process_batch_phased(jobs)` — 같은 job 리스트를 V→VV→MV 순차
- 새 구조: 워커는 한 Phase만 처리 (`_process_batch_mc`, `_process_batch_vv_only` 등)
- **`process_batch_phased()`(full pipeline 모드)를 폐기하고 단일 Phase 배치만 유지**

**6. `file_tasks` vs `files` DB 이중 상태**
- `file_tasks.mc_status = 'done'`이지만 `files.mc_caption`이 빈 경우 발생 가능?
- **규칙**: `mc_status`는 `files` DB에 실제로 기록된 후에만 'done'으로 전환
- 즉 `mc_status = 'done'` ↔ `files.mc_caption IS NOT NULL` 항상 일치해야 함
- **검증 쿼리로 불일치 감지 필수**

**7. 동시성: MC와 VV를 다른 워커가 동시에 처리할 때**
- 워커A: file_id=100의 MC claim → 처리 중
- 워커B: file_id=100의 VV claim → 동시 처리 가능해야 함
- **같은 파일의 다른 Phase를 다른 워커가 동시에 claim 가능해야 함**
- `file_tasks`는 파일당 1행이므로, `assigned_worker`가 1개밖에 못 담음
- **해결**: `assigned_worker` 제거, 대신 `mc_assigned_to`, `vv_assigned_to` 등 Phase별 lock

### file_tasks 테이블 수정

```sql
CREATE TABLE file_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_job_id INTEGER NOT NULL REFERENCES analysis_jobs(id),
    file_id INTEGER NOT NULL REFERENCES files(id),
    file_path TEXT NOT NULL,
    
    -- 준비 단계
    download_status TEXT DEFAULT 'pending'
        CHECK (download_status IN ('pending','done','failed','n/a')),
    parse_status TEXT DEFAULT 'pending'
        CHECK (parse_status IN ('pending','done','failed')),
    
    -- AI 단계 (각각 독립 claim 가능)
    mc_status TEXT DEFAULT 'pending'
        CHECK (mc_status IN ('pending','assigned','done','failed')),
    mc_assigned_to INTEGER,                -- MC를 처리 중인 워커
    
    vv_status TEXT DEFAULT 'pending'
        CHECK (vv_status IN ('pending','assigned','done','failed')),
    vv_assigned_to INTEGER,                -- VV를 처리 중인 워커
    
    mv_status TEXT DEFAULT 'pending'
        CHECK (mv_status IN ('pending','assigned','done','failed')),
    mv_assigned_to INTEGER,                -- MV를 처리 중인 워커
    
    -- 메타
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    priority INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    
    UNIQUE(analysis_job_id, file_id)
);
```

### 폐기 확정

| 기존 | 이유 |
|------|------|
| `process_batch_phased()` (full pipeline) | 단일 Phase 배치로 대체 |
| `_PHASE_FILTERS` JSON 기반 | `file_tasks` 컬럼 기반으로 교체 |
| `phase_completed` JSON | Phase별 독립 status 컬럼으로 교체 |
| `run_all()` (V→VV→MV 직렬) | 워커별 독립 Phase claim으로 교체 |
| `assigned_worker` 단일 컬럼 | Phase별 `*_assigned_to`로 교체 |

## 속도 측정 + 성능 지표

### Phase별 타임스탬프 (file_tasks에 기록)

```sql
-- file_tasks에 각 Phase 시작/종료 시각
download_started_at TEXT,
download_completed_at TEXT,
parse_started_at TEXT,
parse_completed_at TEXT,
mc_started_at TEXT,
mc_completed_at TEXT,
vv_started_at TEXT,
vv_completed_at TEXT,
mv_started_at TEXT,
mv_completed_at TEXT
```

**측정 가능 항목 (파일별)**:
- Download 시간 = `download_completed_at - download_started_at`
- Parse 시간 = `parse_completed_at - parse_started_at`
- MC 시간 = `mc_completed_at - mc_started_at`
- VV 시간 = `vv_completed_at - vv_started_at`
- MV 시간 = `mv_completed_at - mv_started_at`
- 전체 시간 = 마지막 완료 - 최초 시작

### 실시간 throughput (슬라이딩 윈도우)

```sql
-- 최근 5분간 MC 완료 속도
SELECT COUNT(*) FROM file_tasks
WHERE mc_completed_at > datetime('now', '-5 minutes')
  AND analysis_job_id = ?

-- Phase별 평균 처리 시간 (이번 분석작업)
SELECT
    AVG(julianday(mc_completed_at) - julianday(mc_started_at)) * 86400 AS mc_avg_s,
    AVG(julianday(vv_completed_at) - julianday(vv_started_at)) * 86400 AS vv_avg_s,
    AVG(julianday(mv_completed_at) - julianday(mv_started_at)) * 86400 AS mv_avg_s
FROM file_tasks
WHERE analysis_job_id = ? AND mc_status = 'done'
```

### 성능 지표 API: `GET /api/v1/analysis-jobs/{id}/metrics`

```json
{
  "job_id": 45,
  "total": 3190,
  "complete": 2983,
  "throughput": {
    "current_fpm": 3.6,         // 최근 5분 files/min
    "avg_per_file_s": 17.2,     // 전체 평균
    "eta_seconds": 120
  },
  "phase_metrics": {
    "download": { "avg_s": 12.3, "max_s": 45.0, "failed": 5 },
    "parse":    { "avg_s": 0.5,  "max_s": 3.2,  "failed": 2 },
    "mc":       { "avg_s": 7.4,  "max_s": 15.1, "failed": 3 },
    "vv":       { "avg_s": 0.7,  "max_s": 2.1,  "failed": 0 },
    "mv":       { "avg_s": 0.5,  "max_s": 1.8,  "failed": 1 }
  },
  "bottleneck": "mc",           // 가장 느린 Phase
  "design_speed": {             // 벤치마크 기준
    "mc_fpm": 7.8,
    "vv_fpm": 81.1,
    "mv_fpm": 119.3
  }
}
```

### 벤치마크 대비 실측 비교

```
Phase    설계속도    실측속도    Gap
MC       7.8/m      3.6/m     -54%  ← 조사 필요
VV       81.1/m     80.2/m    -1%   OK
MV       119.3/m    115.0/m   -4%   OK
```

## 데이터 검증 (분석 완료 후)

### 파일별 결과 검증 (file_tasks 완료 시 자동 실행)

```python
def verify_file_result(file_id):
    """Phase 완료 후 실제 데이터 존재 확인"""
    
    # MC 완료 검증
    if mc_status == 'done':
        assert files.mc_caption IS NOT NULL AND mc_caption != ''
        assert files.ai_tags IS NOT NULL
        assert files.image_type IS NOT NULL
    
    # VV 완료 검증
    if vv_status == 'done':
        assert EXISTS(vec_files WHERE file_id = ?)
        assert vec dimension == expected (1152)
    
    # MV 완료 검증  
    if mv_status == 'done':
        assert EXISTS(vec_text WHERE file_id = ?)
        assert vec dimension == expected (1024)
    
    # 불일치 시: status를 'failed'로 롤백 + error_message 기록
```

### 분석작업 완료 시 종합 검증

```python
def verify_analysis_job(job_id):
    """분석작업 전체 완료 시 종합 검증"""
    
    results = {
        # 1. Phase status ↔ 실제 데이터 일치
        "mc_mismatch": COUNT WHERE mc_status='done' BUT mc_caption IS NULL,
        "vv_mismatch": COUNT WHERE vv_status='done' BUT vec_files 없음,
        "mv_mismatch": COUNT WHERE mv_status='done' BUT vec_text 없음,
        
        # 2. 합계 검증
        "total_check": total == download+parse+mc+vv+mv+done,
        
        # 3. 빈 데이터
        "empty_caption": COUNT WHERE mc_caption = '',
        "empty_tags": COUNT WHERE ai_tags = '[]',
        
        # 4. 벡터 차원 검증 (샘플)
        "vv_dim_ok": sample vec_files dimension == 1152,
        "mv_dim_ok": sample vec_text dimension == 1024,
    }
    
    if any mismatch:
        log warning + 해당 파일 재처리 큐에 추가
```

### 검증 실행 시점

| 시점 | 검증 | 자동/수동 |
|------|------|:--------:|
| Phase 1개 완료 시 | 해당 Phase 결과 존재 확인 | 자동 |
| 배치 완료 시 | 배치 내 전체 파일 검증 | 자동 |
| 분석작업 완료 시 | 종합 검증 (위 전체) | 자동 |
| 사용자 요청 시 | 전체 DB 정합성 검사 | 수동 (API) |

## 추가 기능

### 분석작업 일시정지/재개

분석작업 단위로 일시정지 가능. "이 폴더 나중에, 저 폴더 먼저".

```sql
-- analysis_jobs.status = 'paused' → 해당 작업의 file_tasks claim 안 됨
-- claim 쿼리에 조건 추가:
WHERE aj.status = 'active'  -- paused인 작업의 파일은 건너뜀
```

API:
- `POST /api/v1/analysis-jobs/{id}/pause`
- `POST /api/v1/analysis-jobs/{id}/resume`

### 워커별 Phase 처리 이력

어떤 워커가 어떤 Phase를 얼마나 처리했는지 기록.

```sql
-- file_tasks에 Phase별 처리 워커 기록 (이미 있음)
mc_assigned_to, vv_assigned_to, mv_assigned_to

-- 워커별 성능 집계
SELECT 
    mc_assigned_to AS worker_id,
    COUNT(*) AS mc_count,
    AVG(julianday(mc_completed_at) - julianday(mc_started_at)) * 86400 AS avg_mc_s
FROM file_tasks
WHERE mc_status = 'done'
GROUP BY mc_assigned_to
```

워커 성능 비교 → 자동 배정 최적화:
- 워커 A의 MC 평균 7.4s, 워커 B의 MC 평균 15s → A에 MC 우선 배정
- 워커별 Phase 친화도 학습 가능

### 분석작업별 네트워크 상태

WebDAV 다운로드 실패는 분석작업 단위로 추적.

```sql
-- analysis_jobs에 네트워크 상태 컬럼
ALTER TABLE analysis_jobs ADD COLUMN network_status TEXT DEFAULT 'ok'
    CHECK (network_status IN ('ok', 'degraded', 'paused'));
```

판정 로직:
```
연속 3개 다운로드 실패 → network_status = 'degraded'
헬스체크 실패 → network_status = 'paused' (다운로드 중단)
헬스체크 복구 → network_status = 'ok' (자동 재개)
```

대시보드 표시:
```
크랑베르무 (3190)  93.5%  ⚠ 네트워크 불안정
```

### Phase 완료 → 즉시 다음 Phase 트리거

현재: 5초 폴링으로 다음 작업 탐색
개선: Phase 완료 시 이벤트로 즉시 다음 Phase 큐에 push

```python
def on_phase_complete(file_task_id, phase):
    """Phase 완료 후 다음 Phase를 즉시 활성화"""
    if phase == 'parse':
        # MC와 VV 둘 다 즉시 claimable
        notify_workers('mc_available')
        notify_workers('vv_available')
    elif phase == 'mc':
        # MV 즉시 claimable
        notify_workers('mv_available')
    elif phase == 'download':
        # Parse 즉시 claimable
        notify_workers('parse_available')
```

구현 방식:
- 임베디드 워커: 직접 함수 호출 (같은 프로세스)
- 외부 워커: heartbeat 응답에 `urgent_phase` 힌트 포함 → 폴링 주기 단축
- 또는: 완료 시 `file_tasks` UPDATE → DB trigger → 워커가 감지

**가장 간단한 구현**: Phase 완료 후 워커가 즉시 다음 claim 시도 (sleep 없이). 현재 rest_after_batch 30초가 주 지연 원인이므로, **Phase 완료 직후 다음 배치 즉시 시작**이면 대부분 해결.

## 검증 체크리스트

- [ ] `file_tasks`의 Phase별 status로 합계 = total 보장
- [ ] MC와 VV 동시 claim/처리 가능 (같은 파일, 다른 워커)
- [ ] MV claim 조건: `mc_status = 'done'` 필수
- [ ] VV claim 조건: `parse_status = 'done'` 만 (MC 불필요)
- [ ] `file_tasks.mc_status = 'done'` ↔ `files.mc_caption IS NOT NULL` 일치
- [ ] 실패 파일은 같은 분석작업에서 retry
- [ ] Recovery WR 생성 안 됨
- [ ] `process_batch_phased()` 미사용
- [ ] 사용자 분석작업이 대시보드에 보임
- [ ] 로컬 파일 삭제 안 됨, WebDAV temp만 삭제
