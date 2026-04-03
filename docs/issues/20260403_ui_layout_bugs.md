# Bug Report: UI 레이아웃 이슈

| 항목 | 내용 |
|------|------|
| **보고일** | 2026-04-03 |
| **보고자** | saintiron |
| **발견 경로** | 워커 탭 하단 확인 |
| **환경** | Mac M5, Electron dev 모드 |
| **심각도** | 중간 |
| **상태** | Open |

---

## BUG-001: 워커 탭 하단이 StatusBar에 가려서 안 보임

| 항목 | 내용 |
|------|------|
| **심각도** | 중간 |
| **재현** | 워커 탭에서 워커 테이블 하단까지 스크롤 |
| **현상** | 워커 테이블 마지막 행이 하단 StatusBar(출력/오류/진행률 바)에 가려져서 잘림. "자동 / saintironui-MacBookPro.local" 행의 호스트명이 반쯤 잘려 보임 |
| **원인** | WorkersPanel 컨테이너에 하단 마진/패딩이 없어서 StatusBar 높이만큼 콘텐츠가 가려짐 (추정) |
| **근본 해결** | WorkersPanel 또는 FactoryPage의 콘텐츠 영역에 `pb-16` (또는 StatusBar 높이만큼) 하단 패딩 추가 |
| **관련 파일** | `frontend/src/components/admin/WorkersPanel.jsx` (컨테이너), `frontend/src/pages/FactoryPage.jsx:147` (overflow-auto 영역) |

---

---

## BUG-002: Parse 실패 파일이 다운로드 버퍼를 영구 점유 → 워커 무한 대기

| 항목 | 내용 |
|------|------|
| **심각도** | **높음** |
| **재현** | Parse 실패 파일이 다수 발생한 후 워커 상태 확인 |
| **현상** | 워커가 resting/대기 상태에서 무한 멈춤. pending 589개 있으나 MC claimable 0. NAS 접속은 정상 |
| **원인** | Parse 실패한 파일이 `file_ready=1`로 남으면서 temp 디렉토리에 PSD 원본이 유지됨. DownloadAheadPool의 `active_files`(59) > `max_files`(10) → 새 다운로드 차단 → 처리 가능한 job 0 → 워커 무한 대기 |
| **임시 해결** | `UPDATE job_queue SET file_ready=0, parse_status=NULL WHERE file_ready=1 AND parse_status='failed'` + `status='failed'` → `'pending'` 리셋 |
| **근본 해결** | DownloadAheadPool에서 parse 실패/job 실패한 파일의 temp 자동 정리. `_reset_stale_file_ready()`에 parse_status='failed' 케이스 추가. 또는 parse 실패 시 즉시 temp 삭제 + file_ready=0 리셋 |
| **관련 파일** | `backend/server/queue/download_ahead.py` (_reset_stale_file_ready, _download_batch), `backend/server/queue/manager.py` (parse 실패 처리) |
| **상태** | Open |

---

## 우선순위

| 순위 | 버그 | 사용자 영향 | 구현 난이도 |
|:----:|------|:---------:|:---------:|
| 1 | BUG-002 다운로드 버퍼 점유 | **높음** (워커 멈춤) | 중간 |
| 2 | BUG-001 하단 잘림 | 중간 | 낮음 |
