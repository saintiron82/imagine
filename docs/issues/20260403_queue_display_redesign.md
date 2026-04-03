# Bug Report: 큐 표시 구조 재설계 필요

| 항목 | 내용 |
|------|------|
| **보고일** | 2026-04-03 |
| **보고자** | saintiron |
| **발견 경로** | 워커 탭에서 큐 진행 상태 확인 |
| **환경** | Mac M5, Electron dev 모드 |
| **심각도** | **높음** — 사용자가 큐 진행 상태를 볼 수 없음 |
| **상태** | Open |

---

## BUG-001: 사용자 큐의 진행 상태가 보이지 않음

| 항목 | 내용 |
|------|------|
| **심각도** | **높음** |
| **재현** | 워커 탭에서 큐 진행 상태 확인 |
| **현상** | 사용자가 등록한 "크랑베르무" 큐(3190개)의 진행률이 안 보임. 대신 자동 생성된 [Recovery] 소형 큐들만 보이거나, 카운터가 부정확 |

### 현재 구조 문제

```
사용자가 만든 큐:
  WR-45 "크랑베르무" (3190개) → status=completed (실제론 미완)
    ├─ job_queue: 0개 (audit가 전부 삭제)
    ├─ subtasks 31개 (일부 fail 수백개)
    └─ Recovery WR 14개 (WR-46~59) → 실패분 재처리

실제 상태 (files DB 기준):
  total: 3190, 3축 완결: 2694 (84.5%), 큐 남은 job: 600
```

### 문제 원인

1. **WR 카운터 이벤트 기반**: `completed_count`, `failed_count`가 파일 단위가 아니라 이벤트 단위로 증가 → `done+fail > total` (3211 > 3190)
2. **WR-45 `status=completed`**: `done+fail >= total`이면 completed 판정이지만 실제 600개 미완
3. **Recovery WR이 별도 큐로 분리**: 사용자 큐의 자식인데 별개 WR로 표시
4. **job 삭제**: 완료/실패 job이 audit에서 DELETE → WR에서 실제 진행률 역산 불가

### 설계 방향

**사용자 큐(WR-45) 중심 표시:**
```
크랑베르무 (3190)                    84.5%  ███████████░░░
  MC: 2992/3190  VV: 3054/3190  MV: 2694/3190
  큐 남은: 600  |  3축 완결: 2694
```

**데이터 소스**: WR 카운터가 아닌 **files DB에서 source_path 기반 실시간 계산**

```sql
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN mc_caption IS NOT NULL AND mc_caption != '' THEN 1 ELSE 0 END) as mc_done,
    SUM(CASE WHEN EXISTS(SELECT 1 FROM vec_files WHERE file_id = f.id) THEN 1 ELSE 0 END) as vv_done,
    SUM(CASE WHEN EXISTS(SELECT 1 FROM vec_text WHERE file_id = f.id) THEN 1 ELSE 0 END) as mv_done
FROM files f
WHERE f.file_path LIKE 'webdav://13730b09/예비/크랑베르무%'
```

**WR 구분**:
- `name LIKE '[Recovery]%'` → 자동 생성, 기본 숨김
- 나머지 → 사용자 생성, 항상 표시

**Recovery WR → 부모 WR 하위로 그룹화**: `source_path` prefix 매칭

### 관련 파일

| 파일 | 역할 |
|------|------|
| `backend/server/queue/manager.py` | WR 카운터 로직, get_stats() |
| `backend/server/routers/pipeline.py` | WR API |
| `frontend/src/components/admin/WorkersPanel.jsx` | 워커 탭 대시보드 |
| `frontend/src/components/WRCards.jsx` | WR 카드 컴포넌트 |

### 필요 작업

1. **API**: 사용자 WR별 files DB 기반 진행률 계산 엔드포인트
2. **WR 그룹화**: Recovery WR을 부모 WR 하위로 묶기
3. **대시보드**: 사용자 큐 카드에 실제 진행률 표시 (files DB 기준)
4. **카운터 보정**: WR `completed_count`/`failed_count`를 files DB 기반으로 재계산하는 audit

---

## 우선순위

| 순위 | 버그 | 사용자 영향 | 구현 난이도 |
|:----:|------|:---------:|:---------:|
| 1 | BUG-001 큐 진행 상태 미표시 | **높음** | **높음** |
