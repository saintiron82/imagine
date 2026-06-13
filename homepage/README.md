# Imagine 홈페이지 (계정의 집)

상거래·계정 표면. **앱(`frontend-v2`)과 완전 별개** — 앱은 키만 있으면 홈페이지 없이
동작하고, 홈페이지는 라이선스의 판매·관리 창구다. (스펙: 메모리 `project_licensing_model`)

## 스택
- **Cloudflare Pages** (SPA: Vite + React + react-router) + **Pages Functions** (`functions/api/*`)
- **Stripe** 결제(Checkout + Webhook)
- **Firebase** 신원(Auth) + **라이선스 원장**(Firestore `groups/{key}`) — 앱이 `lookupGroup`으로 읽는 바로 그 컬렉션

## 6개 표면
| 경로 | 역할 |
|------|------|
| `/` | 소개 |
| `/buy` | 플랜 → 로그인 → Stripe 결제 / 14일 체험 즉시 발급 |
| `/account` | 내 라이선스 포털(키·만료·좌석·갱신/업그레이드) |
| `/download` | 앱 받기 + 서버 만들기 안내 |
| `/join/:code` | 초대 랜딩(딥링크 + 설치 유도) |
| (내부) | 판매자 사용현황 대시보드 — 추후 |

## 라이선스 원장 계약 (Firestore `groups/{group_name}`)
결제/체험이 기록하고 앱이 읽는 필드:
`plan_id`, `status`, `owner_uid`, `owner_email`, `seat_limit`, `analyzer_limit`,
`expires_at`(ISO), `stripe_customer`. 앱의 `LicenseManager`/`lookupGroup`이 소비.

## 로컬 개발
```sh
npm install
npm run dev        # http://localhost:9280 (SPA만 — Functions 는 wrangler 필요)
# Functions 까지 로컬 실행: npx wrangler pages dev dist  (빌드 후)
```

## 환경변수 (`.env.example` 참고)
- 빌드: `VITE_STRIPE_PUBLISHABLE_KEY`
- Functions(CF 대시보드 또는 `.dev.vars`): `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`,
  `STRIPE_PRICE_STUDIO`, `STRIPE_PRICE_TEAM`, `FIREBASE_SERVICE_ACCOUNT`(서비스계정 JSON 한 줄),
  `FIREBASE_PROJECT_ID`, `PUBLIC_BASE_URL`

## Stripe 설정
1. 상품/가격 생성: 스튜디오·팀 (연 구독) → Price ID를 `STRIPE_PRICE_*`에.
2. Webhook 등록: `https://<배포도메인>/api/stripe-webhook`, 이벤트 `checkout.session.completed`
   → 서명 시크릿을 `STRIPE_WEBHOOK_SECRET`에.
3. 키 없이 빌드/렌더는 되지만 결제·발급은 키 설정 후 동작(미설정 시 501 + 안내).

## 배포 (Cloudflare Pages)
- 빌드 명령 `npm run build`, 출력 `dist`, Functions 디렉터리 `functions/` 자동 인식.
- SPA fallback 은 `public/_redirects`.

## 현재 상태
SPA 6페이지 + 결제/웹훅 Functions 구현 완료(키 주입 시 동작). 결제 E2E·실 발급은
Stripe/Firebase Admin/CF 배포가 필요 — 코드는 그 전제로 작성됨.
