# /release — 빌드 및 릴리스

## 버전 3곳 동기화

| # | 파일 | 필드 |
|---|------|------|
| 1 | `frontend/package.json:4` | `"version": "M.m.p"` |
| 2 | `frontend/vite.config.js:14` | `__BUILD_ID__: JSON.stringify('YYYYMMDD_NN')` |
| 3 | `frontend/src/components/StatusBar.jsx:364` | fallback `'vM.m.p.YYYYMMDD_NN'` |

M.m.p는 CLAUDE.md의 **현재 버전** 참조. YYYYMMDD는 오늘 날짜, NN은 `git tag -l "v*$(date +%Y%m%d)*"` 로 순번 결정.

## 릴리스 구조

| 역할 | 서비스 |
|------|--------|
| 빌드 파일 (DMG/zip) | GitHub Releases (무료, 무제한) |
| 릴리스 메타데이터 | Firestore (`releases` 컬렉션) |
| 릴리스 웹페이지 | Firebase Hosting (`release.html`) |

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
태그 push → GitHub Actions `release.yml` 자동 실행 → Windows CI 빌드 + GitHub Release 생성

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

### 5. Mac DMG 업로드 + Firestore 등록
```bash
# macOS DMG → GitHub Release 업로드 + Firestore 메타데이터 등록
node scripts/firebase_release.mjs \
  --version "vM.m.p.YYYYMMDD_NN" \
  --macos "frontend/dist-electron/Imagine-M.m.p-arm64.dmg" \
  --notes "릴리스 노트"

# dry-run으로 먼저 확인 가능
node scripts/firebase_release.mjs --version "vM.m.p.YYYYMMDD_NN" --macos "..." --dry-run

# Firestore 메타데이터만 업데이트 (파일 업로드 건너뛰기)
node scripts/firebase_release.mjs --version "vM.m.p.YYYYMMDD_NN" --skip-upload --notes "..."
```

릴리스 확인: https://imagine-b1e9c.web.app/release.html

### 6. Firebase Hosting 갱신 (필요 시)
```bash
cd website && firebase deploy --only hosting && cd ..
```

## 롤백
```bash
git tag -d vTAG && git push origin :refs/tags/vTAG
gh release delete vTAG --yes
git revert HEAD
# Firestore에서 릴리스 문서 수동 삭제 (Firebase Console)
```
