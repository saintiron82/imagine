# 워커 대시보드 완전 재작성 스펙

## 현재 문제

대시보드의 모든 카운터가 부정확하고, 사용자 큐가 안 보임.
부분 수정을 반복하면서 데이터 소스가 3곳 혼재 → 더 꼬임.

### 근본 원인: 진실 공급원(SSOT) 부재

| 데이터 | 현재 소스 | 문제 |
|--------|----------|------|
| 전체 큐 | WR.total_files 합산 | Recovery WR 중복 카운팅 |
| 완료 | files DB 3축 완결 | WR 범위 밖 파일 포함 |
| 실패 | WR.failed_count | 이벤트 기반 중복 카운팅 (3211>3190) |
| Phase별 | job_queue pipe_* | 완료된 job이 삭제되어 합계 안 맞음 |
| 진행률 | WR 카운터 | 카운터 오염 |

## 사용자가 원하는 것

```
사용자가 "크랑베르무" 폴더(3190파일) 등록

대시보드:
  크랑베르무 (3190)                                84.5%
  ┌─────────────────────────────────────────────────────┐
  │ Download  Parse   MC    VV    MV    Done            │
  │   204      0      27    63     7    2694            │
  │                                      ← 3축 완결    │
  └─────────────────────────────────────────────────────┘
  합계: 204 + 0 + 27 + 63 + 7 + 2694 = 2995... + 진행중 + 실패 = 3190

파일 1개 = 5단계 전부 끝나야 "Done" 1개
Download 204 = 아직 NAS에서 안 받은 파일 204개
Done 2694 = MC+VV+MV 3축 모두 완결된 파일 2694개
```

**핵심 원칙:**
1. **전체 = 각 Phase 합계** — 항상 일치해야 함
2. **Done = 3축(MC+VV+MV) 전부 완결** — 부분 완료 카운팅 안 함
3. **파일 단위** — 이벤트 단위 아님
4. **사용자 큐 중심** — Recovery 큐는 접혀서 숨김
5. **job은 큐 전체가 끝나기 전까지 삭제 안 함**

## SSOT: files DB 기준 단일 쿼리

```sql
-- 사용자 큐 "크랑베르무"의 모든 파일 Phase 상태를 한 번에
SELECT
    COUNT(*) AS total,
    -- Done: 3축 전부 완결
    COUNT(*) FILTER (WHERE 
        (mc_caption IS NOT NULL AND mc_caption != '')
        AND EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)
        AND EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)
    ) AS done,
    -- MC 완료 (vision 끝남)
    COUNT(*) FILTER (WHERE mc_caption IS NOT NULL AND mc_caption != '') AS mc_done,
    -- VV 완료
    COUNT(*) FILTER (WHERE EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id)) AS vv_done,
    -- MV 완료
    COUNT(*) FILTER (WHERE EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id)) AS mv_done
FROM files f
WHERE f.file_path LIKE 'webdav://13730b09/예비/크랑베르무%'
```

Phase별 "대기" 계산:
```
download_waiting = total - (job_queue에서 file_ready=1인 수)
parse_pending    = file_ready=1이고 parse 안 된 수
mc_pending       = parsed인데 MC 없는 수
vv_pending       = MC 있는데 VV 없는 수
mv_pending       = MC 있는데 MV 없는 수
done             = 3축 완결
```

**합계 검증: download + parse + mc + vv + mv + done + failed = total**
(진행 중인 파일은 "가장 뒤처진 Phase"에 카운트)

## 구현 계획

### 1. 백엔드: 새 API `GET /api/v1/admin/queue-progress`

사용자 WR별로 files DB 기반 진행률 반환:

```json
{
  "queues": [
    {
      "wr_id": 45,
      "name": "크랑베르무",
      "source_path": "webdav://13730b09/예비/크랑베르무",
      "is_recovery": false,
      "total": 3190,
      "done": 2694,
      "phases": {
        "download": 204,
        "parse": 0,
        "mc": 27,
        "vv": 63,
        "mv": 7,
        "done": 2694,
        "failed": 195
      },
      "pct": 84.5,
      "created_at": "2026-03-30T03:10:37"
    }
  ],
  "totals": { ... }
}
```

### 2. 프론트엔드: WorkersPanel 대시보드 재작성

기존 `queueStats` + `pipe_*` 기반 로직을 **새 API 기반**으로 교체.

```
┌─ 활성 큐 ────────────────────────────────────┐
│ 크랑베르무 (3190)              84.5% ████░░  │
│ Download:204 Parse:0 MC:27 VV:63 MV:7        │
│ Done:2694  Failed:195                         │
├──────────────────────────────────────────────│
│ ▸ 자동 복구 큐 (6개)                          │
└──────────────────────────────────────────────┘

속도: 3.6/min | 17s/file | 남은: 2m | workers: 1
```

### 3. 제거할 코드

- `get_stats()`의 `complete_files`, `queue_total`, `queue_completed` 계산
- WR 카운터 기반 상단 카운터 (wrTotal/wrDone/wrFail)
- `pipe_*` Phase 카운터 (job_queue 기반 → files DB 기반으로 교체)
- `job_completions` 기반 완료 카운팅

### 4. 유지할 코드

- `throughput` / `eta` 계산 (sliding window — 실시간 속도)
- `embedded_worker` 상태
- 워커 테이블
- 서버 자동 처리 설정

## 검증

1. `total == download + parse + mc + vv + mv + done + failed` 항상 성립
2. 사용자 큐(크랑베르무) 카드가 대시보드에 보임
3. Recovery 큐는 접혀서 기본 숨김
4. 서버 재시작해도 카운터 정확 (files DB 기반이므로)
5. 실시간 갱신 (5초 폴링)
