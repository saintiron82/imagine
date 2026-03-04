---
description: 릴리스 빌드 및 배포 자동화 — 버전 bump, Mac/Win 빌드, GitHub Release, Pages 갱신
---
# Release Skill

## Overview
전체 릴리스 워크플로우를 자동화합니다:
1. 버전 업데이트 (3곳 동기화)
2. Mac 로컬 빌드 (PyInstaller + Electron)
3. Windows CI 빌드 (GitHub Actions)
4. GitHub Release 생성 + Pages 갱신

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
# → 현재: v0.6.2

# 오늘 기존 태그 확인 → 순번 결정
git tag -l "v0.6.2.$(date +%Y%m%d)_*" | sort -V | tail -1
# 없으면 → _01, 마지막이 _03이면 → _04
```

**AskUserQuestion으로 확인:**
- M.m.p 변경이 필요한가? (Major/Minor/Patch bump)
- 필요하면 CLAUDE.md의 "현재 버전" 섹션도 업데이트

최종 버전: `vM.m.p.YYYYMMDD_NN` (예: `v0.6.2.20260304_01`)

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
- Windows: PyInstaller 백엔드 + Electron 앱 빌드
- GitHub Release 자동 생성 (Windows 파일 포함)

---

### Step 6: Mac 로컬 빌드

```bash
# 6-1. Python 백엔드 번들링
cd /Users/saintiron/Projects/Imagine
python -m PyInstaller backend_cli.spec --noconfirm

# 6-2. 백엔드 빌드 확인
ls -la dist/backend_cli/backend_cli

# 6-3. Electron 앱 빌드 (DMG + ZIP)
cd frontend
npm run electron:build
```

**출력 확인:**
```bash
ls frontend/dist-electron/Imagine-*-mac.*
# → Imagine-M.m.p-mac.dmg, Imagine-M.m.p-mac.zip
```

---

### Step 7: CI 완료 대기 + Mac 업로드

```bash
# CI 상태 확인 (완료될 때까지)
gh run list --workflow=release.yml --limit=1 --json status,conclusion,databaseId

# CI 완료 확인 후 Mac DMG 업로드
gh release upload vM.m.p.YYYYMMDD_NN \
  frontend/dist-electron/Imagine-*-mac.dmg \
  frontend/dist-electron/Imagine-*-mac.zip
```

CI가 아직 진행 중이면 → Step 8 먼저 실행하고 돌아옴.

---

### Step 8: `release` 브랜치 갱신 (GitHub Pages)

```bash
# release 브랜치로 전환 (없으면 생성)
git checkout release 2>/dev/null || git checkout -b release

# main의 최신 내용 반영
git merge main --ff-only

# push
git push origin release

# main으로 복귀
git checkout main
```

**GitHub Pages 동작 원리:**
- `docs/release.html` → GitHub Releases API 자동 호출 → 최신 릴리스 표시
- `docs/index.html` → GitHub Releases API → 최신 다운로드 링크 표시
- 페이지 내용 수동 수정 불필요, `release` 브랜치 push만 하면 됨

---

### Step 9: 검증

```bash
# Release 확인 (Win + Mac 파일 모두 있는지)
gh release view vM.m.p.YYYYMMDD_NN

# 아티팩트 목록
gh release view vM.m.p.YYYYMMDD_NN --json assets --jq '.assets[].name'
```

사용자에게 안내:
- GitHub Release: `https://github.com/saintiron82/imagine/releases/tag/vM.m.p.YYYYMMDD_NN`
- Download Page: `https://saintiron82.github.io/imagine/release.html`

---

## 1회성 셋업 (최초 실행 시)

스킬 첫 실행 시 `release` 브랜치와 Pages 설정이 없으면 자동 생성:

```bash
# release 브랜치 존재 확인
git rev-parse --verify release 2>/dev/null || {
  git checkout -b release
  git push origin release
  git checkout main
}

# GitHub Pages 소스를 release/docs로 변경
gh api repos/saintiron82/imagine/pages -X PUT \
  -f source[branch]=release -f source[path]=/docs
```

---

## 롤백 (빌드 실패 시)

```bash
# 로컬 태그 삭제
git tag -d vM.m.p.YYYYMMDD_NN

# 리모트 태그 삭제 (CI 중단됨)
git push origin :refs/tags/vM.m.p.YYYYMMDD_NN

# Release가 이미 생성되었으면 삭제
gh release delete vM.m.p.YYYYMMDD_NN --yes

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
| `.github/workflows/release.yml` | Windows CI + Release 생성 |
| `docs/release.html` | GitHub Pages 릴리스 (API 자동) |
| `docs/index.html` | GitHub Pages 다운로드 (API 자동) |
| `CLAUDE.md` | 현재 버전 정책 |

## 안전장치

- **push 전 반드시 사용자 확인** (AskUserQuestion 사용)
- **main 브랜치에서만 실행** (아니면 병합 먼저)
- **빌드 실패 시 태그 삭제** 안내 제공
- **force push 절대 금지**
