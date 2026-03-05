# /release — 빌드 및 릴리스

## 버전 3곳 동기화

| # | 파일 | 필드 |
|---|------|------|
| 1 | `frontend/package.json:4` | `"version": "M.m.p"` |
| 2 | `frontend/vite.config.js:14` | `__BUILD_ID__: JSON.stringify('YYYYMMDD_NN')` |
| 3 | `frontend/src/components/StatusBar.jsx:364` | fallback `'vM.m.p.YYYYMMDD_NN'` |

M.m.p는 CLAUDE.md의 **현재 버전** 참조. YYYYMMDD는 오늘 날짜, NN은 `git tag -l "v*$(date +%Y%m%d)*"` 로 순번 결정.

## 워크플로우

### 1. 사전 검증
```bash
git status                    # 커밋되지 않은 변경 확인
git branch --show-current     # main 확인 (아니면 병합 먼저)
cd frontend && npm run build  # 빌드 확인
```

### 2. 버전 업데이트 + 커밋 + 태그
3곳 버전 동기화 후:
```bash
git add frontend/package.json frontend/vite.config.js frontend/src/components/StatusBar.jsx
git commit -m "chore: version bump to vM.m.p.YYYYMMDD_NN"
git tag vM.m.p.YYYYMMDD_NN
```

### 3. Push (사용자 확인 필수)
```bash
git push origin main --tags
```
태그 push → GitHub Actions `release.yml` 자동 실행 → Windows CI 빌드

### 4. Mac 로컬 빌드 (push 후 병렬 실행 가능)
```bash
# PyInstaller 백엔드
cd /Users/saintiron/Projects/Imagine
python -m PyInstaller backend_cli.spec --noconfirm

# Electron 앱 (코드 사이닝 없이)
cd /Users/saintiron/Projects/Imagine/frontend
CSC_IDENTITY_AUTO_DISCOVERY=false npm run electron:build
```

출력: `frontend/dist-electron/Imagine-M.m.p-arm64.dmg`

### 5. CI 완료 후 Mac 업로드
```bash
gh run list --workflow=release.yml --limit=1
gh release upload vM.m.p.YYYYMMDD_NN \
  "/Users/saintiron/Projects/Imagine/frontend/dist-electron/Imagine-M.m.p-arm64.dmg" \
  "/Users/saintiron/Projects/Imagine/frontend/dist-electron/Imagine-M.m.p-arm64-mac.zip"
```

### 6. GitHub Pages 갱신
```bash
git checkout release && git merge main --ff-only && git push origin release && git checkout main
```

## 롤백
```bash
git tag -d vTAG && git push origin :refs/tags/vTAG
gh release delete vTAG --yes
git revert HEAD
```
