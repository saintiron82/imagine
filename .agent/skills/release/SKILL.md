---
description: 릴리스 빌드 및 배포 자동화 — 버전 bump, tag push → GitHub Actions (Win+Mac 빌드) → GitHub Release → Firebase 웹사이트 자동 반영
---
# Release Skill

## Overview

```
/release
  ↓
Step 1: 사전 검증 (git status, build)
  ↓
Step 2: 버전 결정 (YYYYMMDD_NN)
  ↓
Step 3: 버전 업데이트 (2곳)
  ↓
Step 4: 커밋 + 태그
  ↓
Step 5: Push (→ GitHub Actions 자동 실행)
  ├─ build-win (windows-latest): PyInstaller + Electron → .zip
  ├─ build-mac (macos-latest):   PyInstaller + Electron → .dmg + .zip
  └─ release: 양쪽 아티팩트 수집 → GitHub Release 자동 생성
  ↓
Step 6: 빌드 완료 대기 + 검증
  ↓
Firebase 웹사이트는 GitHub Releases API를 읽으므로 자동 반영 (재배포 불필요)
```

## Trigger
`/release`, "릴리스", "빌드 배포", "버전 올려"

---

## Workflow

### Step 1: 사전 검증

```bash
# 1-1. 커밋되지 않은 변경 확인
git status

# 1-2. 현재 브랜치 확인 (main이어야 함)
git branch --show-current

# 1-3. 프론트엔드 빌드 확인
cd frontend && npm run build
```

**main이 아닌 경우**: 사용자에게 확인 → 병합 진행:
```bash
git checkout main
git merge <현재브랜치>
git branch -d <현재브랜치>
```

빌드 실패 시 → 수정 후 재시도. 진행하지 않음.

---

### Step 2: 버전 결정

```bash
# CLAUDE.md에서 현재 M.m.p 읽기
grep "현재 버전" CLAUDE.md
# → 현재: v0.1.0

# 오늘 기존 태그 확인 → 순번 결정
git tag -l "v0.1.0.$(date +%Y%m%d)_*" | sort -V | tail -1
# 없으면 → _01, 마지막이 _03이면 → _04
```

**AskUserQuestion으로 확인:**
- M.m.p 변경이 필요한가? (Major/Minor/Patch bump)
- 필요하면 CLAUDE.md의 "현재 버전" 섹션도 업데이트

최종 버전: `vM.m.p.YYYYMMDD_NN` (예: `v0.1.0.20260325_01`)

---

### Step 3: 버전 업데이트 (2곳)

| # | 파일 | 수정 내용 |
|---|------|----------|
| 1 | `frontend/package.json` 라인 4 | `"version": "M.m.p"` |
| 2 | `frontend/vite.config.js` 라인 14 | `__BUILD_ID__: JSON.stringify('YYYYMMDD_NN')` |

**M.m.p 변경 시 추가**: CLAUDE.md의 `현재 버전` 값도 업데이트.

> `StatusBar.jsx`는 `__APP_VERSION__` + `__BUILD_ID__`를 자동으로 읽으므로 별도 수정 불필요.

---

### Step 4: 커밋 + 태그

```bash
git add frontend/package.json frontend/vite.config.js
# CLAUDE.md도 변경했으면 추가
git commit -m "$(cat <<'EOF'
chore: version bump to vM.m.p.YYYYMMDD_NN

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"

git tag vM.m.p.YYYYMMDD_NN
```

---

### Step 5: Push (사용자 확인 필수)

**반드시 AskUserQuestion으로 사용자 확인 후 실행:**

```bash
git push origin main --tags
```

- 태그 push → `.github/workflows/release.yml` 자동 실행
- **build-win**: Windows PyInstaller + Electron 빌드
- **build-mac**: macOS PyInstaller + Electron 빌드 (unsigned)
- **release**: 양쪽 아티팩트 수집 → GitHub Release 자동 생성

---

### Step 6: 빌드 완료 대기 + 검증

```bash
# Actions 실행 상태 확인
gh run list --workflow=release.yml --limit=1

# 상세 job 상태
gh run view <RUN_ID> --json jobs --jq '.jobs[] | "\(.name): \(.status) \(.conclusion // "running")"'

# 완료 후 릴리즈 아티팩트 확인
gh release view vM.m.p.YYYYMMDD_NN --json tagName,name,assets \
  --jq '{tag: .tagName, assets: [.assets[] | "\(.name) (\(.size / 1048576 | floor)MB)"]}'
```

**예상 아티팩트 3개:**
- `Imagine-M.m.p-arm64-mac.zip` (macOS)
- `Imagine-M.m.p-arm64.dmg` (macOS)
- `Imagine-M.m.p-win.zip` (Windows)

---

### Step 7: 검증

```bash
# 웹사이트 확인 (GitHub Releases API 자동 반영)
open https://imagine-b1e9c.web.app
open https://imagine-b1e9c.web.app/release.html

# GitHub Release 확인
open https://github.com/saintiron82/imagine/releases/tag/vM.m.p.YYYYMMDD_NN
```

**Firebase 웹사이트는 GitHub Releases API를 읽으므로 별도 배포 불필요.**
웹사이트 HTML/JS/CSS 자체를 수정한 경우에만:
```bash
cd website && firebase deploy --only hosting
```

사용자에게 안내:
- 웹사이트: `https://imagine-b1e9c.web.app`
- 릴리스 이력: `https://imagine-b1e9c.web.app/release.html`
- GitHub Releases: `https://github.com/saintiron82/imagine/releases`

---

## 롤백 (빌드 실패 시)

```bash
# 리모트 태그 삭제 (CI 중단됨)
git push origin :refs/tags/vM.m.p.YYYYMMDD_NN

# 로컬 태그 삭제
git tag -d vM.m.p.YYYYMMDD_NN

# GitHub Release 삭제 (이미 생성된 경우)
gh release delete vM.m.p.YYYYMMDD_NN --yes

# 버전 bump 커밋 되돌리기
git revert HEAD
```

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `frontend/package.json:4` | npm 버전 (`M.m.p`) |
| `frontend/vite.config.js:14` | `__BUILD_ID__` (`YYYYMMDD_NN`) |
| `backend_cli.spec` | PyInstaller 설정 |
| `.github/workflows/release.yml` | CI/CD: Win+Mac 빌드 → GitHub Release |
| `website/public/index.html` | 웹사이트 (GitHub Releases API 조회) |
| `website/public/release.html` | 릴리스 이력 (GitHub Releases API 조회) |
| `CLAUDE.md` | 현재 버전 정책 |

## 안전장치

- **push 전 반드시 사용자 확인** (AskUserQuestion 사용)
- **main 브랜치에서만 실행** (아니면 병합 먼저)
- **빌드 실패 시 태그 + Release 삭제** 안내 제공
- **force push 절대 금지**

## 실전 주의사항

1. **GitHub API Rate Limit**: 미인증 GitHub API는 60req/hour. 웹사이트 방문자가 많으면 캐싱 고려.
2. **코드 사이닝**: Mac은 `CSC_IDENTITY_AUTO_DISCOVERY=false`로 ad-hoc 서명 (Gatekeeper 경고 있지만 실행 가능).
3. **파일명 패턴**: arm64 Mac 빌드는 `Imagine-M.m.p-arm64.dmg`, Windows는 `Imagine-M.m.p-win.zip`.
4. **CI 빌드 시간**: Win ~8-12분, Mac ~6-10분. 병렬 실행이므로 최대 ~12분.
5. **웹사이트 asset 매핑**: `index.html`에서 파일명에 `mac`/`darwin` 포함 → macOS, `win` 포함 → Windows로 자동 매핑.
6. **Firestore 미사용**: 릴리스 정보는 GitHub Releases API에서 직접 읽음. Firestore `releases` 컬렉션은 레거시 (더 이상 사용하지 않음).
