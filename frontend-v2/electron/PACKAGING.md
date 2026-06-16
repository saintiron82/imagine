# 패키징 & 자동 업데이트 (IMGV2-15 / IMGV2-27)

v2 얇은 셸은 **자동 업데이트 코드 통합이 완료**됐다(`main.cjs` `setupAutoUpdater`,
`preload.cjs` `onUpdateEvent/checkForUpdates/installUpdate`,
`src/shell/UpdateNotification.jsx`). 다만 실제로 동작하는 **서명된 설치본 + 업데이트
피드**는 아래 운영 결정이 선행돼야 한다. 코드 자체는 dev 를 깨지 않도록 `app.isPackaged`
일 때만 `electron-updater` 를 require 한다.

## 남은 단계 (운영 결정 필요)

### 1. Python 백엔드 번들 — **기존 backend_cli.spec 재사용(신규 발명 아님)**
구 frontend 와 **동일한 검증된 경로**를 쓴다(재설계 금지):
- `scripts/build-backend.sh`(=`npm run build:backend`) 가 `backend_cli.spec` 으로
  PyInstaller onedir 번들 → `dist/backend_cli/backend_cli`(+ `.exe`) 생성.
- `package.json build.extraResources` 가 `../dist/backend_cli` → `resources/backend`
  로 동봉. 런타임에 `main.cjs` 가 `resources/backend/backend_cli server --port 8000`
  으로 실행(`backend/backend_cli.py` 의 `server` 서브커맨드 = `uvicorn backend.server.app:app`).
- `npm run dist:electron` = build:backend → vite build → electron-builder 순서.

주의: PyInstaller 번들은 **빌드한 OS/아키텍처 전용**(mac arm64 / mac x64 / win / linux 각각).
`sqlite-vec`/MLX/torch 네이티브는 `backend_cli.spec` 의 hiddenimports/datas 에 이미
반영돼 있음. 크로스 플랫폼 산출물은 각 OS(혹은 CI 매트릭스)에서 빌드해야 한다.

### 2. 코드 서명 / 공증 — **설정은 배선됨, 인증서만 주입**
`package.json build.mac` 에 `hardenedRuntime`+`entitlements(electron/entitlements.mac.plist)`+
`notarize:true` 가 들어가 있다(번들 torch/MLX 가 하드닝 런타임에서 돌도록 JIT/library-
validation 완화 엔타이틀먼트 포함). 인증서/계정은 **CI 시크릿**으로만 주입(코드/리포에 없음):
- **macOS**: `CSC_LINK`(Developer ID .p12 base64)·`CSC_KEY_PASSWORD` + 공증용 `APPLE_ID`·
  `APPLE_APP_SPECIFIC_PASSWORD`·`APPLE_TEAM_ID`.
- **Windows**: `CSC_LINK`·`CSC_KEY_PASSWORD`(Authenticode .p12).
시크릿 미설정 시 서명/공증은 실패(또는 미서명) — 미서명 dev 산출만 원하면
`CSC_IDENTITY_AUTO_DISCOVERY=false` 로 로컬 빌드.

### 3. 업데이트 피드 채널 — **GitHub Releases(턴키)로 확정**
`build.publish` = `{provider: github, owner: saintiron82, repo: imagine}`. 별도 호스팅
불필요 — 리포의 GitHub Releases 가 곧 피드. electron-builder `--publish always` 가
설치본 + `latest*.yml` 매니페스트를 릴리스에 올리고, `electron-updater`(github provider)
가 최신 릴리스의 `latest*.yml` 을 읽어 새 버전을 판단한다.

## 릴리스 절차 (CI — `.github/workflows/release-v2.yml`)
1. 위 시크릿을 리포 Settings → Secrets 에 등록(macOS 공증·win 서명).
2. 버전 태그 푸시 — **구 frontend 의 `v*` 와 충돌 안 나게 `v2-` 접두**:
   ```sh
   git tag v2-2.0.0 && git push origin v2-2.0.0
   ```
3. 워크플로가 mac/win/linux 에서: PyInstaller 백엔드 → vite build → electron-builder
   `--publish always` → GitHub Release 에 설치본 + `latest*.yml` 게시.
   (태그에서 버전을 추출해 `frontend-v2/package.json` version 에 주입 — 현재 0.0.0.)

로컬 단일 OS 검증(미서명):
```sh
cd frontend-v2 && npm ci && npm run dist:electron   # build:backend → vite → electron-builder
```

## 동작 확인 포인트
- dev(`npm run electron:dev`): updater 비활성(require 안 함) — 영향 없음.
- 패키징본: 시작 5초 후 + 6시간 주기로 `checkForUpdates`. 새 버전 발견 시
  자동 다운로드 → `UpdateNotification` 토스트 "지금 재시작" → `quitAndInstall`.
- 피드/릴리스 없을 때: `checkForUpdates` 가 에러 → 토스트 없이 조용히 무시(앱 정상).

## 검증 한계 (정직)
CI 워크플로·서명 설정·entitlements·피드 배선은 모두 **구 frontend 의 검증된 release.yml
패턴을 그대로 따른 표준 레시피**다. 다만 **실제 서명 릴리스는 인증서·공증 계정이 있어야
처음 한 번 돌려봐야** 검증된다(이 환경에선 시크릿이 없어 미실행). 첫 실 릴리스 때
확인할 것: GitHub Release 에 `latest-mac.yml`/`latest.yml` 이 올라갔는지, 설치본이
Gatekeeper/SmartScreen 을 통과하는지, 구버전 앱이 자동 업데이트를 받는지.
