# 패키징 & 자동 업데이트 (IMGV2-15 / IMGV2-27)

v2 얇은 셸은 **자동 업데이트 코드 통합이 완료**됐다(`main.cjs` `setupAutoUpdater`,
`preload.cjs` `onUpdateEvent/checkForUpdates/installUpdate`,
`src/shell/UpdateNotification.jsx`). 다만 실제로 동작하는 **서명된 설치본 + 업데이트
피드**는 아래 운영 결정이 선행돼야 한다. 코드 자체는 dev 를 깨지 않도록 `app.isPackaged`
일 때만 `electron-updater` 를 require 한다.

## 남은 단계 (운영 결정 필요)

### 1. Python 백엔드 번들
`main.cjs` 는 prod 에서 `process.resourcesPath/python/python3` 를 실행한다.
`package.json` 의 `build.extraResources` 가 `../.dist-python` → `resources/python`
로 복사하도록 돼 있다. 이 `.dist-python` 디렉터리를 만드는 빌드 훅이 필요하다(택1):
- **PyInstaller**: `backend.server.app:app` 진입점을 onedir 로 빌드 → `.dist-python` 에 배치.
- **임베디드 파이썬 + venv**: 플랫폼별 파이썬 런타임 + `pip install -r requirements.txt`
  결과를 `.dist-python` 에 복사.
주의: `sqlite-vec`, MLX/torch 등 네이티브 의존성은 플랫폼별로 따로 빌드해야 한다.

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
