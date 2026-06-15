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

### 2. 코드 서명 / 공증
- **macOS**: Developer ID 인증서 + notarization(`notarize`). 미서명 시 Gatekeeper 차단.
- **Windows**: Authenticode 코드 서명 인증서. 미서명 시 SmartScreen 경고.
electron-builder 환경변수(`CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_ID` 등)로 주입.

### 3. 업데이트 피드 채널
`build.publish` 가 현재 placeholder(`https://REPLACE-ME...`)다. 택1:
- **generic**: 정적 호스팅(S3/CF R2/자체 서버)에 `latest.yml`+설치본 업로드.
- **github**: GitHub Releases.
electron-updater 는 이 피드의 `latest*.yml` 을 읽어 새 버전을 판단한다.

## 빌드 명령 (위 1~3 구성 후)
```sh
npm run dist:electron   # vite build && electron-builder
```

## 동작 확인 포인트
- dev(`npm run electron:dev`): updater 비활성(require 안 함) — 영향 없음.
- 패키징본: 시작 5초 후 + 6시간 주기로 `checkForUpdates`. 새 버전 발견 시
  자동 다운로드 → `UpdateNotification` 토스트 "지금 재시작" → `quitAndInstall`.
- 피드 미구성 시: `checkForUpdates` 가 에러 → 토스트 없이 조용히 무시(앱 정상).
