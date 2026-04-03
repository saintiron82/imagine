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

## 우선순위

| 순위 | 버그 | 사용자 영향 | 구현 난이도 |
|:----:|------|:---------:|:---------:|
| 1 | BUG-001 하단 잘림 | 중간 | 낮음 |
