---
description: 릴리스 빌드 및 배포 자동화 — 버전 bump, Mac/Win 빌드, Firebase Storage 업로드, Firestore 릴리스 등록, 웹사이트 배포
---
# Release Skill

## Overview
전체 릴리스 워크플로우를 자동화합니다:
1. 버전 업데이트 (3곳 동기화)
2. Mac 로컬 빌드 (PyInstaller + Electron)
3. Windows CI 빌드 (GitHub Actions)
4. Firebase Storage 업로드 + Firestore 릴리스 문서 생성
5. Firebase Hosting 웹사이트 배포

## Trigger
`/release`, "릴리스", "빌드 배포", "버전 올려"

---

## Workflow

### Step 1: 사전 검증

```bash
# 1-1. 커밋되지 않은 변경 확인
git status

# 1-2. 현재 브랜치 확인
git branch --show-current
```

**main이 아닌 경우**: 사용자에게 확인 → 병합 진행:
```bash
git checkout main
git merge <현재브랜치>
git branch -d <현재브랜치>
```

```bash
# 1-3. 프론트엔드 빌드 확인
cd frontend && npm run build
```

빌드 실패 시 → 수정 후 재시도. 진행하지 않음.

---

### Step 2: 버전 결정

```bash
# CLAUDE.md에서 현재 M.m.p 읽기
grep "현재 버전" CLAUDE.md
# → 현재: v0.6.3

# 오늘 기존 태그 확인 → 순번 결정
git tag -l "v0.6.3.$(date +%Y%m%d)_*" | sort -V | tail -1
# 없으면 → _01, 마지막이 _03이면 → _04
```

**AskUserQuestion으로 확인:**
- M.m.p 변경이 필요한가? (Major/Minor/Patch bump)
- 필요하면 CLAUDE.md의 "현재 버전" 섹션도 업데이트

최종 버전: `vM.m.p.YYYYMMDD_NN` (예: `v0.6.3.20260306_03`)

---

### Step 3: 버전 업데이트 (3곳)

| # | 파일 | 수정 내용 |
|---|------|----------|
| 1 | `frontend/package.json` 라인 4 | `"version": "M.m.p"` |
| 2 | `frontend/vite.config.js` 라인 14 | `__BUILD_ID__: JSON.stringify('YYYYMMDD_NN')` |
| 3 | `frontend/src/components/StatusBar.jsx` 라인 364 | fallback `'vM.m.p.YYYYMMDD_NN'` |

**M.m.p 변경 시 추가**: CLAUDE.md의 `현재 버전` 값도 업데이트.

---

### Step 4: 커밋 + 태그

```bash
git add frontend/package.json frontend/vite.config.js \
        frontend/src/components/StatusBar.jsx
# CLAUDE.md도 변경했으면 추가
git commit -m "$(cat <<'EOF'
chore: version bump to vM.m.p.YYYYMMDD_NN

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"

git tag vM.m.p.YYYYMMDD_NN
```

---

### Step 5: Push (사용자 확인 필수)

**반드시 사용자 확인 후 실행:**

```bash
git push origin main --tags
```

- 태그 push → `.github/workflows/release.yml` 자동 실행
- **build-win**: PyInstaller 백엔드 + Electron 앱 빌드
- **release**: Firebase Storage 업로드 + Firestore 릴리스 문서 생성

---

### Step 6: Mac 로컬 빌드

**주의**: 모든 경로를 절대경로로 사용 (zsh glob 문제 방지)

```bash
# 6-1. Python 백엔드 번들링 (프로젝트 루트에서)
cd /Users/saintiron/Projects/Imagine
python -m PyInstaller backend_cli.spec --noconfirm

# 6-2. 백엔드 빌드 확인
ls -la /Users/saintiron/Projects/Imagine/dist/backend_cli/backend_cli

# 6-3. Electron 앱 빌드 (코드 사이닝 없이)
cd /Users/saintiron/Projects/Imagine/frontend
CSC_IDENTITY_AUTO_DISCOVERY=false npm run electron:build
```

**출력 확인** (파일명에 `arm64` 포함):
```bash
ls -lh /Users/saintiron/Projects/Imagine/frontend/dist-electron/Imagine-M.m.p-arm64.dmg
ls -lh /Users/saintiron/Projects/Imagine/frontend/dist-electron/Imagine-M.m.p-arm64-mac.zip
```

---

### Step 7: CI 완료 대기 + Mac DMG Firebase 업로드

```bash
# CI 상태 확인 (완료될 때까지)
gh run list --workflow=release.yml --limit=1 --json status,conclusion,databaseId
```

CI 완료 후, Mac DMG를 Firebase에 수동 업로드:

```bash
# firebase-release.mjs로 macOS 아티팩트 추가 업로드
# (기존 Firestore 문서의 assets.macos만 업데이트)
cd /Users/saintiron/Projects/Imagine
FIREBASE_SERVICE_ACCOUNT="$(cat /path/to/service-account.json)" \
  node scripts/firebase-release.mjs \
    --tag vM.m.p.YYYYMMDD_NN \
    --macos "/Users/saintiron/Projects/Imagine/frontend/dist-electron/Imagine-M.m.p-arm64.dmg"
```

**또는** Firebase Console에서 직접 업로드:
1. Firebase Console > Storage > `releases/vM.m.p.YYYYMMDD_NN/`
2. DMG 파일 업로드
3. Firestore > `releases` 컬렉션 > 해당 문서 > `assets.macos` URL 수동 입력

---

### Step 8: 웹사이트 배포

```bash
cd /Users/saintiron/Projects/Imagine/website
firebase deploy --only hosting
```

배포 후 URL: `https://imagine-b1e9c.web.app`

**웹사이트 동작 원리:**
- `index.html` → Firestore `releases` 컬렉션 조회 → 최신 다운로드 링크 표시
- `release.html` → Firestore `releases` 컬렉션 조회 → 버전 이력 표시
- Firestore 문서가 업데이트되면 웹사이트가 자동 반영 (별도 배포 불필요)
- 웹사이트 코드 자체를 수정한 경우에만 `firebase deploy` 필요

---

### Step 9: 검증

```bash
# Firestore 릴리스 문서 확인 (Firebase Console)
# https://console.firebase.google.com/project/imagine-b1e9c/firestore/databases/-default-/data/~2Freleases

# GitHub Actions 로그 확인
gh run list --workflow=release.yml --limit=1

# 웹사이트 확인
open https://imagine-b1e9c.web.app
open https://imagine-b1e9c.web.app/release.html
```

사용자에게 안내:
- 웹사이트: `https://imagine-b1e9c.web.app`
- 릴리스 이력: `https://imagine-b1e9c.web.app/release.html`
- Firebase Console: `https://console.firebase.google.com/project/imagine-b1e9c`

---

## 1회성 셋업 (최초 실행 시)

### GitHub Secret 등록

1. Firebase Console > 프로젝트 설정 > 서비스 계정 > **새 비공개 키 생성**
2. JSON 파일 다운로드
3. GitHub repo > Settings > Secrets and variables > Actions > **New repository secret**
   - 이름: `FIREBASE_SERVICE_ACCOUNT`
   - 값: JSON 파일 내용 전체 붙여넣기

### Firebase Storage 규칙

Firebase Console > Storage > Rules에서 `releases/` 경로 공개 읽기 허용:

```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Release artifacts: public read
    match /releases/{allPaths=**} {
      allow read;
      allow write: if request.auth != null;
    }
    // Board attachments: authenticated users
    match /board/{allPaths=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

### Firebase CLI 설치 (1회)

```bash
npm install -g firebase-tools
firebase login
```

### 로컬 Mac 빌드용 서비스 계정 키 (선택)

Mac DMG를 `firebase-release.mjs`로 업로드하려면 로컬에 서비스 계정 키 필요:

```bash
# 안전한 위치에 저장 (gitignore됨)
cp ~/Downloads/imagine-b1e9c-*.json ~/.config/firebase/imagine-sa.json

# 사용 시:
FIREBASE_SERVICE_ACCOUNT="$(cat ~/.config/firebase/imagine-sa.json)" \
  node scripts/firebase-release.mjs --tag vX.X.X --macos /path/to/file.dmg
```

---

## 롤백 (빌드 실패 시)

```bash
# 로컬 태그 삭제
git tag -d vM.m.p.YYYYMMDD_NN

# 리모트 태그 삭제 (CI 중단됨)
git push origin :refs/tags/vM.m.p.YYYYMMDD_NN

# Firestore 릴리스 문서 삭제 (Firebase Console에서 수동)
# https://console.firebase.google.com/project/imagine-b1e9c/firestore

# Firebase Storage 파일 삭제 (Firebase Console에서 수동)
# Storage > releases/vM.m.p.YYYYMMDD_NN/ 폴더 삭제

# 버전 bump 커밋 되돌리기
git revert HEAD
```

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `frontend/package.json:4` | npm 버전 |
| `frontend/vite.config.js:14` | `__BUILD_ID__` |
| `frontend/src/components/StatusBar.jsx:364` | UI 표시 버전 |
| `backend_cli.spec` | PyInstaller 설정 |
| `.github/workflows/release.yml` | Windows CI + Firebase 업로드 |
| `scripts/firebase-release.mjs` | Firebase Storage 업로드 + Firestore 문서 생성 |
| `website/public/index.html` | 웹사이트 다운로드 (Firestore 조회) |
| `website/public/release.html` | 웹사이트 릴리스 이력 (Firestore 조회) |
| `website/firebase.json` | Firebase Hosting 설정 |
| `CLAUDE.md` | 현재 버전 정책 |

## 안전장치

- **push 전 반드시 사용자 확인** (AskUserQuestion 사용)
- **main 브랜치에서만 실행** (아니면 병합 먼저)
- **빌드 실패 시 태그 삭제** 안내 제공
- **force push 절대 금지**
- **서비스 계정 키 커밋 금지** (환경변수 또는 GitHub Secret으로만 사용)

## 실전 주의사항

1. **zsh glob 문제**: `ls frontend/dist-electron/Imagine-*` 같은 zsh glob은 매칭 실패 시 에러 발생. 항상 `ls -lh "절대경로"` 또는 `ls -lh /path/to/dir/` 전체 디렉토리 나열 사용.
2. **코드 사이닝**: Mac 개발자 인증서가 없으면 `CSC_IDENTITY_AUTO_DISCOVERY=false` 환경변수 필수. ad-hoc 서명으로 빌드됨 (Gatekeeper 경고 있지만 실행 가능).
3. **파일명 패턴**: arm64 Mac 빌드 시 파일명에 `arm64`가 포함됨 (예: `Imagine-0.6.3-arm64.dmg`).
4. **작업 디렉토리**: PyInstaller는 프로젝트 루트에서, electron-builder는 `frontend/`에서 실행. `cd` 명확히 지정.
5. **CI와 로컬 빌드 병렬**: Push 후 CI 대기 중 Mac 로컬 빌드 진행 가능 (Step 5 → Step 6 순차 불필요, 병렬 실행).
6. **Firestore 문서 merge**: `firebase-release.mjs`는 `set(..., { merge: true })`를 사용하므로, Mac DMG를 나중에 업로드해도 Windows URL이 덮어쓰이지 않음.
7. **웹사이트 자동 반영**: Firestore 릴리스 문서가 업데이트되면 웹사이트가 실시간 반영. `firebase deploy`는 HTML/JS 코드 변경 시에만 필요.

## 요약 흐름도

```
/release
  ↓
Step 1: 사전 검증 (git status, build)
  ↓
Step 2: 버전 결정 (YYYYMMDD_NN)
  ↓
Step 3: 3곳 버전 업데이트
  ↓
Step 4: 커밋 + 태그
  ↓
Step 5: Push (→ GitHub Actions 자동 실행)
  │
  ├──→ CI: build-win → firebase-release.mjs → Firebase Storage + Firestore
  │
  └──→ Step 6: Mac 로컬 빌드 (병렬)
         ↓
       Step 7: Mac DMG Firebase 업로드
         ↓
       Step 8: 웹사이트 배포 (코드 변경 시만)
         ↓
       Step 9: 검증
```
