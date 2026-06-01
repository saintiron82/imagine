# Control Plane MVP — 의사결정 트래커

> 부모 문서: `docs/imagine_operations_control_plane_2026-05-31.md`
> 작성일: 2026-05-31 (decisions still open)
> 목적: P0~P10 빌드 시작 전에 풀어야 할 3개의 전략 결정을 명시적으로 기록·추적한다. 결정이 내려질 때마다 본 문서에 "Decision" 행을 채워 넣고, 그 결정이 어느 phase에 영향을 주는지 cross-link한다.

---

## 배경

`docs/imagine_operations_control_plane_2026-05-31.md`가 P0~P10 로드맵을 정의했지만, **빌드 시퀀스와 호스팅·Build/Buy 결정은 비어 있다**. 이 결정들은 §15의 데이터 모델, §13의 업데이트 흐름, §12의 Cloudflare 모델 모두에 연쇄적으로 영향을 준다. 따라서 어떤 phase도 시작하기 전에 본 문서 3개 항목이 먼저 잠겨야 한다.

기존 결정 사항 (이미 합의된 것):
- BYO Cloudflare 기본 (모델 A) — managed는 유료 option
- Firebase auth 유지 (개인 신원 계층)
- 결제는 자체 풀빌드 비추 — 외부 결제 인프라 활용

남은 결정 3가지가 본 문서 대상.

---

## 결정 1 — MVP scope: P0~P10 중 어디서 끊는가?

### 옵션

| 옵션 | 포함 phase | 의미 |
|------|-----------|------|
| **A. 최소 MVP** | P0 + P2 + P4 + P5 + P6 | 책임 경계 + 토큰 + 로컬 검증 + Thin CP + 자동 업데이트. 검색 제품 자체는 그대로 두고 라이선스·업데이트·node 등록만 추가. |
| **B. Team-ready MVP** | A + P1 + P3 + P7 | 위 + Route Registry + Signup Gate + Usage Rollup. Team 플랜 출시 가능. |
| **C. Full Commercial v1** | A + B + P8 + P9 | 위 + BYO Cloudflare Wizard + Managed Services. 모든 플랜 동시 출시. |
| **D. Spatial 먼저** | P10 단독 | Control Plane 전부 보류하고 검색 성능 도약(Spatial 백필)부터. |

### 트레이드오프

- **A**: 6~8주 추정. Personal Local 플랜만 출시 가능. 가장 빠른 첫 매출.
- **B**: 12~16주 추정. Team 플랜까지. 대부분의 SaaS의 첫 출시 형태.
- **C**: 6~9개월. 한 번에 다 출시 → 출시 자체가 늦어짐. 비추.
- **D**: 검색 성능 우선. 다만 상용화 없이 데이터 백필만 하면 매출 없음. **검색 천장 0.673에서 더 가져갈 수 있는 마지막 측정 가능 카드**.

### 권장

**A → B → C 점진**. C까지 한 번에 묶지 않는다. 단, **D(Spatial 백필)는 A/B와 병렬 가능**하다 — backfill은 분석 파이프라인 작업이라 Control Plane과 코드 표면이 겹치지 않음.

### Decision

| 항목 | 값 |
|------|-----|
| 선택 | **(pending)** |
| 결정자 | 사용자 |
| 결정일 | — |
| 근거 메모 | — |

---

## 결정 2 — Control Plane 호스팅

### 옵션

| 옵션 | 구성 | 운영비 | 주요 트레이드오프 |
|------|------|-------|----|
| **A. Cloudflare Workers + D1** | Workers + D1(SQLite) + KV | 매우 낮음 (~$5/월 시작) | D1의 schema migration이 아직 약함. row 수 제한이 있고 대용량 usage_rollup 보관에는 부적합. |
| **B. 자체 호스팅 (작은 VPS + Postgres)** | Fly.io / Railway / Hetzner + managed Postgres | ~$10~30/월 | 표준 PG, 마이그레이션 도구 풍부. Cloudflare는 edge layer로만 사용. |
| **C. zeroechodaily.com과 공유** | 사용자가 운영 중인 기존 사이트의 백엔드 + DB 확장 | 거의 0 (이미 운영 중) | 운영 단순화 + 비용 0. 단 사용자가 사이트와 Imagine을 같은 인프라 위에 두는 것에 동의해야 함. |
| **D. Supabase / Firebase** | managed BaaS | $0~25/월 시작 | auth가 이미 Firebase면 자연스러움. 단 entitlement 로직과 webhook 관리가 vendor-specific. |

### 영향 받는 phase

- **모든** P1~P9. 특히 P5 Thin Control Plane의 13개 테이블 (§15.1).
- 자동 업데이트(P6)의 release server 호스팅도 같이 결정.

### 권장

- **C가 운영비/단순성 측면에서 최강**. 단, 사이트와 SaaS Control Plane을 같이 두는 것이 보안적으로 문제 없는지 사용자 판단 필요.
- 만약 분리하고 싶으면 **B (Fly.io + Postgres)**가 가장 표준적.
- A는 D1의 schema migration 한계로 인해 비추. KV만 보조 사용하는 hybrid는 가능.

### Decision

| 항목 | 값 |
|------|-----|
| 선택 | **(pending)** |
| 결정자 | 사용자 |
| 결정일 | — |
| 근거 메모 | — |

---

## 결정 3 — Build vs Buy per plane

각 plane을 어디까지 직접 빌드할지 결정.

### Operations Plane (Cloudflare access)

| 영역 | 선택지 | 권장 |
|------|-------|------|
| Tunnel 자체 | BYO (Cloudflare 무료 plan) / 자체 reverse-tunnel 빌드 | **BYO**. cloudflared 사용 안내만 제공. |
| Tunnel wizard | Build / Skip | **Build** (P8). 사용자 편의 위해. |
| Managed Cloudflare option | Build / Defer | **Defer**. C까지 가지 않으면 불필요. |

### Lifecycle Plane (자동 업데이트)

| 영역 | 선택지 | 권장 |
|------|-------|------|
| Electron auto-updater | Built-in Electron updater / Sparkle / 자체 | **Electron built-in updater** + signed manifest. 표준 패턴. |
| Release CDN | Cloudflare R2 / GitHub Releases / 자체 | **R2** (BYO Cloudflare 결정과 일관). 트래픽 비용 거의 0. |
| Migration runner | Build 필수 | DB schema migration은 자체 빌드 외 답 없음. |

### Commercial Plane (license/seat/billing)

| 영역 | 선택지 | 권장 |
|------|-------|------|
| 결제 | Stripe / Paddle / Lemon Squeezy / 자체 | **Stripe + 한국 지원 시 토스/카카오페이 보조**. 빌드 비추. |
| Auth (개인 신원) | Firebase (현재) / Clerk / Auth0 | **Firebase 유지** — 이미 운영 중. |
| Entitlement engine | Build / Stripe Customer Portal에 위임 | **Build (얇게)**. 핵심 IP. Stripe webhook → 자체 entitlement table 동기화 패턴. |
| License token 서명 | Build 필수 | RSA/Ed25519 키 페어로 자체 서명. 외부 vendor 의존 비추. |
| Seat management UI | Build 필수 | 조직 admin 화면 (§19.1). |

### Decision

| 항목 | 값 |
|------|-----|
| 선택 | **(pending — 위 권장안 그대로 채택 시 별도 결정 불필요)** |
| 결정자 | 사용자 |
| 결정일 | — |
| 근거 메모 | — |

---

## 다음 행동

세 결정이 모두 잠기면 다음을 진행한다:

1. P0 (책임 경계 문서화) — 결정 1·2·3의 결과를 본문에 반영
2. P2 (Token Authority) 스펙 작성 — 결정 2의 호스팅에 맞춘 토큰 서명/검증 도구 결정
3. P5 (Thin Control Plane) 스키마 작성 — 결정 2의 DB 위에 §15.1 13개 테이블 마이그레이션 정의
4. (병렬) P10 Spatial Backfill 1차 시도 — 분석 팀 트랙

---

## Cross-link 인덱스

- 부모 문서: `docs/imagine_operations_control_plane_2026-05-31.md`
- 검색 시스템 상태: `docs/state_report_2026-05-31.md`
- 검색 천장 도달 결과: `docs/superpowers/plans/2026-05-28-perceived-search-quality.md`
- 기존 11-phase Secure External Worker Access: `docs/superpowers/plans/2026-05-25-secure-external-worker-access.md` (P2의 PoC로 활용)
