# Bug Report: Electron 앱 시작 시 콘솔 에러

| 항목 | 내용 |
|------|------|
| **보고일** | 2026-04-03 |
| **보고자** | saintiron |
| **발견 경로** | Electron dev 모드 시작 시 콘솔 확인 |
| **환경** | Mac M5, Electron 40, dev 모드 (`npm run electron:dev`) |
| **심각도** | 낮음 — 기능 동작에 영향 없으나 콘솔 오염 |
| **상태** | Open |

---

## BUG-001: auth/refresh, auth/me 접속 에러 (ERR_CONNECTION_REFUSED)

| 항목 | 내용 |
|------|------|
| **심각도** | 낮음 |
| **재현** | Electron 앱 시작 직후 콘솔 확인 |
| **현상** | `POST http://localhost:8000/api/v1/auth/refresh net::ERR_CONNECTION_REFUSED` + `GET http://localhost:8000/api/v1/auth/me net::ERR_CONNECTION_REFUSED` |
| **원인** | Electron 모드에서 서버가 아직 시작되지 않았거나, 서버 모드가 아닌데 `localStorage`에 이전 세션의 서버 URL(`localhost:8000`)이 남아있어 API 호출 시도. `client.js:68`에서 `getServerUrl()`이 빈 문자열이 아닌 이전 URL 반환 |
| **근본 해결** | Electron 로컬 모드에서는 서버 API 호출을 하지 않도록 가드. 또는 앱 시작 시 서버 연결 상태 확인 후 API 호출 |
| **관련 파일** | `frontend/src/api/client.js:68` (refresh), `frontend/src/api/auth.js:29` (getMe), `frontend/src/contexts/AuthContext.jsx` (초기화 시 호출) |

---

## BUG-002: Electron Security Warning 3건

| 항목 | 내용 |
|------|------|
| **심각도** | 낮음 (dev 모드 전용, 패키징 시 표시 안 됨) |
| **재현** | Electron dev 모드 시작 |
| **현상** | `Disabled webSecurity`, `allowRunningInsecureContent`, `Insecure Content-Security-Policy` 경고 3건 |
| **원인** | `main.cjs`에서 `webPreferences.webSecurity: false` 등 dev 편의를 위한 설정. 패키징 빌드에서는 자동 숨김 |
| **근본 해결** | dev 모드에서도 보안 설정 강화하거나, dev 모드 전용 CSP 설정. 또는 무시 (패키징 시 표시 안 됨) |
| **관련 파일** | `frontend/electron/main.cjs` (BrowserWindow webPreferences) |

---

## 우선순위

| 순위 | 버그 | 사용자 영향 | 구현 난이도 |
|:----:|------|:---------:|:---------:|
| 1 | BUG-001 auth 접속 에러 | 낮음 (콘솔 오염) | 낮음 |
| 2 | BUG-002 Security Warning | 낮음 (dev 전용) | 낮음 |
