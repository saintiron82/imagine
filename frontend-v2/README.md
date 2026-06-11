# Imagine v2 (frontend-v2)

기존 `frontend/`와 **완전 별개**의 새 앱. 구 앱은 참조용으로만 두고 건드리지 않는다.

- **스펙**: `../mockups/ui-rebuild-2026-06-11/` (v26) + `../docs/ui_rebuild_plan_2026-06-11.md`
- 스택: Vite + React + react-router(hash) + TanStack Query (폴링 통합 지점)
- 원칙: 용어 0(일반 표면에 MC/VV/MV 금지 — 관리 화면만 예외), 폴더 지정형,
  작업 등록 의미론, 화면당 질문 하나, 뒤로 = 이력 스택

## 구조
```
src/shell/      셸 (역할 게이팅, +추가, 서버 칩)
src/screens/    시작 / 검색(이식 자리) / 폴더 / 분석 / 관리 / 설정
src/flows/      AddFlow (분석 작업 등록)
src/state/      AppContext (역할·서버 — 실인증 전 데모 토글)
src/styles/     목업에서 추출한 토큰·컴포넌트 CSS
```

## 현재 상태
정적 이식 완료(목업 v26 기준). 다음: 백엔드 연동(TanStack Query로 잡/분석기/폴더),
검색면 이식(U4 — 구 SearchPanel/FileGrid, 재작성 금지), 실인증(Firebase+이메일 초대).

```sh
npm run dev        # 개발
npm run build      # 빌드
node smoke.mjs     # 렌더 스모크 (playwright, preview 서버 필요)
```
