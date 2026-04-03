# Queue System Redesign v2

## 1. 현재 시스템 문제

### 1.1 테이블 난립 (4개 + 1개 보조)

| 테이블 | 역할 | 행 수 | 문제 |
|--------|------|:-----:|------|
| `job_queue` | 파일별 작업 추적 | 600 | 27컬럼, audit가 완료 job 삭제 → 진행률 추적 불가 |
| `work_requests` | 큐 단위 관리 | 59 | 이벤트 기반 카운터 오염, Recovery 부모 참조 없음 |
| `work_subtasks` | 폴더별 분할 | 96 | 카운터 초과 3건, 독립 카운터 관리 부담 |
| `job_completions` | throughput 계산 | 568 | 1시간 후 삭제, 영속성 없음 |
| `files` | 결과 저장 | 7848 | **유일하게 정확**, Phase 결과가 여기에 있음 |

### 1.2 데이터 소스 혼재

같은 "완료 수"를 3곳에서 다르게 계산:
- `work_requests.completed_count` → 이벤트 기반 (중복 카운팅, 3211 > 3190)
- `job_completions` COUNT → 1시간 후 삭제됨
- `files` DB 3축 완결 COUNT → 정확하지만 별도 쿼리 필요

### 1.3 구조적 결함

1. **job 삭제**: audit가 completed/failed job을 DELETE → 전체 대비 진행률 계산 불가
2. **WR 카운터**: `failed_count + 1`이 이벤트마다 → 같은 파일 retry 시 중복
3. **Recovery WR**: `parent_wr_id` 없음 → 사용자 큐와 자동 큐 구분 불가
4. **551개 WR-없는 job**: audit 자동 생성 job에 `work_request_id = NULL`
5. **Phase 상태 이원화**: `phase_completed` JSON(job_queue) vs 실제 데이터(files DB) 불일치 가능

---

## 2. 설계 원칙

### 2.1 진실 공급원 (SSOT)

| 정보 | SSOT | 이유 |
|------|------|------|
| **파일 존재** | `files` 테이블 | 등록 시점에 생성, 삭제 안 됨 |
| **Phase 완료 여부** | `files` + `vec_files` + `vec_text` | 실제 결과가 여기에 있음 |
| **작업 진행 상태** | `job_queue` | 현재 어떤 Phase를 처리 중인지 |
| **큐 범위** | `work_requests.source_path` | 사용자가 등록한 폴더 경로 |

### 2.2 핵심 규칙

1. **파일 단위 카운팅**: 1파일 = 5단계(DL→Parse→MC→VV→MV). "Done" = 5단계 전부 완결
2. **합계 항상 일치**: `download + parse + mc + vv + mv + done + failed = total`
3. **job 보존**: 큐 전체가 완료될 때까지 job 삭제 안 함
4. **카운터 = 쿼리 결과**: 별도 카운터 컬럼 불필요. 매번 실시간 계산
5. **로컬 파일 삭제 금지**: WebDAV temp만 삭제 대상

### 2.3 파일 1개의 Phase 판별

```
Phase 결정 = 가장 뒤처진 단계 (파일은 항상 정확히 1개 Phase에 속함)

if mc+vv+mv 전부 있음          → Done
elif mc+vv 있고 mv 없음        → MV 대기
elif mc 있고 vv 없음            → VV 대기
elif files에 mc 없음:
    if job_queue에서 parsed     → MC 대기
    elif job_queue에서 ready=1  → Parse 대기
    elif job_queue에서 ready=0  → Download 대기
    else                        → Failed (job 없고 mc도 없음)
```

---

## 3. 새 테이블 구조

### 3.1 `job_queue` 간소화

불필요한 중복 컬럼 제거. Phase 상태는 files DB에서 유도.

**유지할 컬럼** (작업 상태 추적):
```sql
id, file_id, file_path, status,
file_ready,          -- 0: download 대기, 1: 다운로드 완료, -1: 실패
parse_status,        -- NULL/pending/parsing/parsed/failed
assigned_to, assigned_at, worker_session_id,
work_request_id,
error_message, error_code,
retry_count, max_retries,
priority, created_at, updated_at,
archived_at
```

**제거 후보** (files DB에서 유도 가능):
```sql
phase_completed      -- files DB로 판별 가능 (mc_caption, vec_files, vec_text)
mc_completed_at      -- throughput은 job_completions 또는 실시간 측정으로
vv_completed_at      -- 상동
mv_completed_at      -- 상동
started_at           -- 사용처 없음
completed_at         -- status='completed'로 충분
parsed_metadata      -- parse_ahead 전용, 별도 관리
parsed_at            -- parse_status로 충분
work_subtask_id      -- subtask 테이블 폐기 시
```

**참고**: 기존 컬럼은 당장 삭제하지 않고, 새 로직에서 참조하지 않으면 됨. 마이그레이션은 점진적.

### 3.2 `work_requests` 간소화

```sql
id, name, source_path, status, parent_wr_id,  -- ← NEW: Recovery 부모 참조
total_files,  -- 등록 시 설정, 이후 불변
created_by, created_at, completed_at,
sort_order
```

**제거**:
```sql
completed_count  -- files DB에서 실시간 계산
failed_count     -- files DB에서 실시간 계산
started_at       -- 사용처 미미
```

**추가**:
```sql
parent_wr_id INTEGER REFERENCES work_requests(id)  -- Recovery WR의 부모
```

### 3.3 `work_subtasks` — 폐기 검토

현재 역할: 폴더별 분할. 하지만 카운터가 부정확하고, files DB에서 폴더별 GROUP BY로 대체 가능.
**당장 폐기하지 않되, 새 로직에서 참조하지 않음.**

### 3.4 `job_completions` — 폐기 검토

현재 역할: throughput 계산 (1시간 window). 
대체: `job_queue`의 `updated_at` + `status='completed'` 타임스탬프로 슬라이딩 윈도우.
**job이 삭제 안 되므로 job_queue 자체가 완료 로그 역할.**

---

## 4. 새 API

### 4.1 `GET /api/v1/admin/queue-progress`

사용자 WR별 파일 기반 진행률. **단일 엔드포인트로 대시보드의 모든 카운터 제공.**

```json
{
  "queues": [
    {
      "wr_id": 45,
      "name": "크랑베르무",
      "source_path": "webdav://13730b09/예비/크랑베르무",
      "is_recovery": false,
      "total": 3190,
      "phases": {
        "download": 181,
        "parse": 15,
        "mc": 2,
        "vv": 0,
        "mv": 9,
        "done": 2983
      },
      "pct": 93.5
    }
  ],
  "recovery_queues": [
    {"wr_id": 46, "name": "[Recovery] 안나의집", "parent_wr_id": 45, ...}
  ],
  "throughput": {
    "files_per_min": 3.6,
    "per_file_s": 17,
    "eta_seconds": 120,
    "bottleneck": "mv"
  }
}
```

### 4.2 Phase 계산 SQL (단일 쿼리)

```sql
-- WR source_path별 파일 Phase 분류 (합계 = total 보장)
SELECT
    COUNT(*) AS total,
    
    -- Done: 3축 완결
    COUNT(*) FILTER (WHERE 
        (f.mc_caption IS NOT NULL AND f.mc_caption != '')
        AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
        AND EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
    ) AS done,
    
    -- MV 대기: mc+vv 있고 mv 없음
    COUNT(*) FILTER (WHERE
        (f.mc_caption IS NOT NULL AND f.mc_caption != '')
        AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
        AND NOT EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
    ) AS mv_pending,
    
    -- VV 대기: mc 있고 vv 없음
    COUNT(*) FILTER (WHERE
        (f.mc_caption IS NOT NULL AND f.mc_caption != '')
        AND NOT EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
    ) AS vv_pending,
    
    -- MC 이전 (download/parse/mc) — job_queue JOIN으로 세분화
    COUNT(*) FILTER (WHERE
        (f.mc_caption IS NULL OR f.mc_caption = '')
    ) AS pre_mc

FROM files f
WHERE f.file_path LIKE :source_path_pattern
```

`pre_mc` 세분화:
```sql
SELECT
    SUM(CASE WHEN jq.file_ready = 0 THEN 1 ELSE 0 END) AS download,
    SUM(CASE WHEN jq.file_ready = 1 AND jq.parse_status != 'parsed' THEN 1 ELSE 0 END) AS parse,
    SUM(CASE WHEN jq.parse_status = 'parsed' THEN 1 ELSE 0 END) AS mc_pending
FROM files f
LEFT JOIN job_queue jq ON jq.file_id = f.id AND jq.archived_at IS NULL
WHERE f.file_path LIKE :pattern
  AND (f.mc_caption IS NULL OR f.mc_caption = '')
```

### 4.3 Throughput 계산

job이 보존되므로 `job_queue`에서 직접:
```sql
-- 최근 5분간 completed된 job 수
SELECT COUNT(*) FROM job_queue
WHERE status = 'completed'
  AND updated_at > datetime('now', '-5 minutes')
  AND file_path LIKE :pattern
```

---

## 5. 프론트엔드

### 5.1 WorkersPanel 대시보드

**데이터 소스**: `GET /api/v1/admin/queue-progress` 1개만 사용

```
활성 큐
┌─────────────────────────────────────────────────┐
│ 크랑베르무 (3190)                    93.5%       │
│ ███████████████████████████████░░░░              │
│ DL:181  Parse:15  MC:2  VV:0  MV:9  Done:2983  │
├─────────────────────────────────────────────────│
│ ▸ 자동 복구 큐 (6개)                ← 접힘      │
└─────────────────────────────────────────────────┘

3.6/min · 17s/file · 남은 2m · workers 1
```

### 5.2 Phase 파이프라인

```
Download → Parse → MC → VV → MV → Done
  181       15      2    0    9   2983

합계 검증: 181+15+2+0+9+2983 = 3190 ✓
```

---

## 6. 마이그레이션 전략

### Phase 1: 새 API + 프론트엔드 교체 (이번 작업)
- `queue-progress` API 추가
- WorkersPanel이 새 API 사용
- 기존 `get_stats()` 유지 (StatusBar 등 다른 소비자)
- 기존 테이블 구조 변경 없음

### Phase 2: 카운터 폐기 (다음 작업)
- `work_requests`에서 `completed_count`/`failed_count` 컬럼 미사용 처리
- `work_subtasks` 미참조 처리
- `job_completions` → `job_queue` 기반 throughput으로 전환
- `_update_wr_counters` 호출 제거

### Phase 3: 스키마 정리 (장기)
- `parent_wr_id` 추가 (Recovery WR 부모 참조)
- 미사용 컬럼 DROP
- `work_subtasks` 테이블 폐기

---

## 7. 검증 체크리스트

- [ ] `total == download + parse + mc + vv + mv + done` 항상 성립
- [ ] 사용자 큐 "크랑베르무" 카드가 보임
- [ ] Recovery 큐 기본 접힘
- [ ] 서버 재시작해도 카운터 정확
- [ ] job 삭제 안 됨 (큐 전체 끝나기 전)
- [ ] WebDAV temp만 삭제, 로컬 파일 보존
- [ ] throughput/ETA 정상 동작
- [ ] Phase 합계 = total (UI에서 검증 표시)
