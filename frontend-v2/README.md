# Imagine v2 (frontend-v2)

기존 `frontend/`와 **완전 별개**의 새 앱. 구 앱은 참조용으로만 두고 건드리지 않는다.

- **스펙**: `../mockups/ui-rebuild-2026-06-11/` (v26) + `../docs/ui_rebuild_plan_2026-06-11.md`
- 스택: Vite + React + react-router(hash) + TanStack Query (폴링 통합 지점)
- 원칙: 용어 0(일반 표면에 MC/VV/MV 금지 — 관리 화면만 예외), 폴더 지정형,
  작업 등록 의미론, 화면당 질문 하나, 뒤로 = 이력 스택

## 구조
```
src/shell/      셸 (역할 게이팅, +추가, 서버 칩)
src/screens/    시작 / 검색 / 폴더 / 분석 / 관리 / 설정
src/flows/      AddFlow (분석 작업 등록)
src/state/      AuthContext (Firebase + JWT) · AppContext (역할·서버 — AuthContext에서 파생, 데모 토글 없음)
src/api/        서버 클라이언트 (TanStack Query)
src/components/ SearchPanel 등 공용 컴포넌트
src/hooks/      공용 훅
src/lib/        순수 로직 (roleGuard, paging)
src/i18n/       로케일 (en-US ↔ ko-KR 키 패리티 유지)
src/services/   브리지·업데이트 등 실행 표면 서비스
src/styles/     목업에서 추출한 토큰·컴포넌트 CSS
electron/       main.cjs · preload.cjs · PACKAGING.md
```

## 현재 상태
U1~U4 완료 — 백엔드 연동(TanStack Query), 검색면 이식, 실인증(Firebase + 초대) 모두 배선됨.
남은 것: 구 `frontend/` 표면 제거, 실서버 수동 검증.

```sh
npm run dev        # 개발
npm run build      # 빌드
npm test           # Vitest (페이징·역할가드·i18n 패리티 불변식) — CI 게이트
npm run lint       # ESLint
node smoke.mjs     # 렌더 스모크 (playwright, preview 서버 필요)
```
