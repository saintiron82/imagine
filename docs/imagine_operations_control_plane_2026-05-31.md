# Imagine 운영·배포·상용 제어 모델 정리

> 작성일: 2026-05-31  
> 기준 문서: `state_report_2026-05-31.md`  
> 목적: Imagine의 현재 검색 성능 상태를 기준으로, Cloudflare 기반 운영성, 자동 업데이트, 배포, seat 관리, 사용 실태 제어, 개별 서버 접근 라우팅, 접속용 토큰 관리, 최소비용 운영 원칙을 하나의 제품/운영 아키텍처로 정리한다.

---

## 0. 핵심 결론

Imagine은 이미 **개인 PC / 로컬 DB / 저비용 분석 파이프라인 기반 검색 시스템**으로는 최고 수준의 성능에 도달한 상태로 볼 수 있다. 최신 상태 보고서 기준으로 Parse / MC / VV / MV / FTS 메인 경로는 코드와 데이터가 모두 실사용 가능한 상태이며, Triaxis 검색은 SLM-judge 기준 P@5=0.673 수준까지 도달했다.

이제 추가해야 할 것은 검색 알고리즘이 아니라 다음 세 가지다.

| 축 | 목적 |
|---|---|
| **Operations Plane** | Cloudflare 기반 원격 접근, route 관리, 접속 토큰, 최소 health check |
| **Lifecycle Plane** | 자동 업데이트, 배포, 릴리즈, migration, rollback |
| **Commercial Control Plane** | 조직, 회원 가입, seat, 요금제, node, entitlement, license, 사용 실태 집계 |

최종 제품 철학은 다음 문장으로 고정한다.

> **Imagine은 로컬에서 강력하게 돌고, 클라우드는 제품 운영과 상용 제어만 최소한으로 보조한다.**

더 짧게 표현하면:

> **Content local. Routing controlled. Token active. License central. Usage aggregated. Cost minimal.**

---

## 1. 현재 검색 시스템 상태

### 1.1 성능 판정

현재 Imagine은 검색 성능 측면에서 이미 높은 완성도를 갖는다.

| 영역 | 현재 상태 | 판정 |
|---|---:|---|
| Parse / MC / VV / MV / FTS | 코드·데이터 모두 완성권 | 정상 |
| Triaxis 검색 | VV / MV / FTS / Spatial 축 구조 | 정상 |
| Cross-encoder rerank | 구현 완료, 효과 확인 | 정상 |
| AND verification | 구현 완료, 효과 확인 | 정상 |
| Benchmark | frozen queryset + SLM judge 확보 | 정상 |
| Spatial 검색 | 코드만 있고 데이터 거의 없음 | 병목 |
| Feedback 기반 개선 | 기능은 있으나 데이터 없음 | 운영 대기 |
| Confidence calibration | raw score 기반 재실행 필요 | 보류 |

### 1.2 현재 강점

Imagine은 단순 로컬 이미지 검색기가 아니라 다음 조합을 갖춘 검색 엔진이다.

```text
원본 파일: WebDAV/NAS
로컬 캐시: 썸네일 + 메타데이터
로컬 DB: SQLite + sqlite-vec + FTS5
분석: MC caption + AI tags + visual vector + meaning vector
검색: Triaxis retrieval + RRF + rerank + AND verification
운영 UX: confidence label + feedback button + admin dashboard
```

이 구조는 개인 PC 기반 저비용 검색 시스템으로는 매우 높은 수준이다.

### 1.3 남은 병목

가장 큰 병목은 검색 코드가 아니라 **Spatial 데이터 부재**다.

| Spatial 데이터 | 현재 상태 |
|---|---:|
| `file_objects` | 12 / 17,726 |
| `file_depth_layers` | 0 / 17,726 |
| `file_spatial_relations` | 0 / 17,726 |

따라서 다음 도약은 검색팀의 weight 조정이 아니라 **Spatial phase 대량 백필**이다.

---

## 2. 제품 정의

### 2.1 Imagine의 정확한 제품 포지션

Imagine은 다음으로 정의한다.

> **사용자의 PC/NAS에서 검색과 분석을 수행하는 local-first AI search appliance.**

우리는 검색 서비스를 호스팅하는 회사가 아니다. 우리는 다음을 제공한다.

```text
소프트웨어 릴리즈
자동 업데이트
라이선스
seat / node / entitlement 제어
route token
회원 가입 제어
최소 usage rollup
Cloudflare 연동 템플릿 또는 managed option
```

사용자는 다음을 보유한다.

```text
원본 파일
로컬 DB
썸네일
벡터
검색 실행 환경
분석 실행 리소스
로컬 서버
NAS / WebDAV
```

### 2.2 핵심 원칙

| 원칙 | 설명 |
|---|---|
| **Local-first** | 검색, 분석, DB, 썸네일, 벡터는 사용자 서버에 둔다. |
| **Cloud-assisted, not cloud-hosted** | 클라우드는 운영 제어만 보조한다. |
| **Thin Control Plane** | 우리 서버는 권한, route, update, seat, billing metadata만 관리한다. |
| **No content by default** | 원본, 썸네일, 벡터, caption, 파일명, 검색어를 기본 수집하지 않는다. |
| **Token-active control** | route, node, seat, license는 활성 토큰으로 제어한다. |
| **Aggregated usage only** | 사용 실태는 집계값으로만 받는다. |
| **Cost minimal** | 운영비가 검색량에 비례하지 않게 한다. |
| **Optional managed services** | 비용이 발생하는 운영 기능은 기본값이 아니라 선택형이다. |

---

## 3. 전체 아키텍처

```text
[User Browser]
   ↓
Cloudflare / Custom Domain / Local Network
   ↓
Access Routing Layer
   ↓
[User's Imagine Local Node]
   ├─ Search UI
   ├─ Search API
   ├─ SQLite DB
   ├─ thumbnails
   ├─ vectors
   ├─ local sessions
   ├─ seat lease cache
   ├─ route token validator
   ├─ update agent
   ├─ license agent
   └─ usage rollup generator

        ↑
        │ heartbeat / token refresh / membership sync / usage rollup
        ↓

[Imagine Control Plane: 우리가 관리]
   ├─ Organizations
   ├─ Users / Members
   ├─ Seats
   ├─ Nodes
   ├─ Route Registry
   ├─ Active Tokens
   ├─ Invite Tokens
   ├─ License Tokens
   ├─ Entitlements
   ├─ Releases
   ├─ Billing State
   └─ Usage Rollups
```

---

## 4. 책임 경계

### 4.1 우리가 관리하는 부분

| 영역 | 책임 |
|---|---|
| 제품 코드 | Core backend, frontend, worker, update agent |
| 릴리즈 | signed release, manifest, artifact, channel |
| 자동 업데이트 | update check, download, verification, migration, rollback logic |
| 라이선스 | license token, entitlement token, revocation |
| seat | 구매 seat, 할당 seat, active seat, 초과 사용 판단 |
| node | node activation, node limit, heartbeat |
| route | route registry, route token, remote access 활성 상태 |
| 회원 가입 | invite token, signup gate, domain allowlist, admin approval |
| billing | plan, subscription, payment status, entitlement sync |
| 최소 usage rollup | 상용 준수 확인용 집계값 |
| Cloudflare 템플릿 | tunnel/access setup guide, optional managed config |
| 보안 패치 | release signing, dependency fix, forced security update |

### 4.2 사용자가 관리하는 부분

| 영역 | 책임 |
|---|---|
| 로컬 서버 | PC, NAS, mini server, VM, 전원, OS |
| 원본 파일 | PSD, PNG, JPG 등 원본 자산 |
| 로컬 저장소 | SQLite DB, thumbnails, cache |
| 분석 리소스 | CPU, GPU, 로컬 워커 |
| 네트워크 | LAN, 라우터, 방화벽, 자기 Cloudflare 계정 |
| 백업 위치 | local disk, NAS, WebDAV, BYO S3/R2 |
| 조직 운영 | 사용자 초대, role 지정, 비활성화 |
| 환경 장애 | 디스크 장애, NAS 장애, OS 문제 |

### 4.3 optional managed 영역

| 영역 | 설명 |
|---|---|
| Managed Cloudflare | 우리가 customer subdomain, tunnel, Access policy를 관리 |
| Managed Backup | 우리가 R2/S3 등에 snapshot을 보관 |
| Cloud Worker | 우리가 고성능 분석 worker 제공 |
| Enterprise SSO | SAML/OIDC, SCIM, audit retention |
| Priority Support | 운영 지원, 전담 migration, 복구 지원 |

일반 사용자의 기본 플랜에는 optional managed 기능을 넣지 않는다. 비용 발생 기능은 명확히 별도 add-on으로 분리한다.

---

## 5. 비용 원칙

### 5.1 최우선 원칙

> **일반 사용자는 운영 자체에 추가 비용을 지불하지 않아야 한다.**

사용자는 제품 사용권 또는 seat 비용만 낸다. Cloudflare, 백업 저장소, 실시간 호스팅, 중앙 검색 인프라 비용을 일반 사용자에게 기본 전가하지 않는다.

### 5.2 비용이 폭증하는 잘못된 구조

아래는 금지한다.

```text
검색 요청마다 우리 API로 프록시
검색 요청마다 license check
썸네일을 우리 서버에서 서빙
central vector DB 운영
사용자 DB를 우리 cloud에 동기화
검색어/결과 전체 로그 업로드
managed cloud backup 기본 제공
cloud VLM 분석을 기본 제공
모든 사용자 tunnel을 우리 Cloudflare 계정에서 관리
high-cardinality telemetry 기본 수집
```

### 5.3 저비용 구조

권장 구조는 다음이다.

```text
검색은 100% 로컬
분석은 기본적으로 로컬
license / route / seat token은 로컬 캐시
Control Plane sync는 주기적
usage는 daily rollup
Cloudflare는 기본 BYO 계정
backup은 기본 BYO storage
managed 기능은 선택형
```

### 5.4 권장 cloud call 빈도

| 항목 | 빈도 |
|---|---:|
| license refresh | 1일 1회 |
| route token refresh | 1일 1회 또는 앱 시작 시 |
| node heartbeat | 6~24시간 1회 |
| usage rollup | 1일 1회 |
| member sync | 변경 시 + 1일 1회 |
| update check | 1일 1회 또는 앱 시작 시 |
| search-time cloud check | 하지 않음 |

이 구조라면 운영비는 검색량이 아니라 조직 수, node 수, seat 수에만 완만하게 비례한다.

---

## 6. Access Routing Plane

### 6.1 목적

Access Routing Plane의 역할은 다음이다.

```text
어느 조직의 어느 로컬 서버가 어떤 route로 접근 가능한가
해당 route token이 활성인가
해당 node가 plan 범위 안에 있는가
원격 접속을 허용할 것인가
```

중요한 점:

> **우리는 사용자의 검색 행위를 추적하는 것이 아니라, 사용자의 개별 서버로 가는 접근 경로와 활성 토큰을 제어한다.**

### 6.2 Route Registry

Control Plane은 route registry를 갖는다.

```text
org_id
node_id
route_id
route_mode
public_hostname
custom_domain
cloudflare_tunnel_id
route_status
active_token_status
last_seen_at
health_status
```

예시:

```text
acme-studio.imagine.app
  → org_acme
  → node_main_macstudio
  → route_token active
  → last_seen 2026-05-31 14:20
  → health: healthy
```

### 6.3 Route mode

| 모드 | 설명 | 기본 여부 |
|---|---|---:|
| `local_only` | LAN 또는 localhost에서만 접근 | 기본 |
| `byo_cloudflare` | 사용자 Cloudflare 계정으로 tunnel/access 구성 | 개인/팀 기본 원격 모델 |
| `managed_cloudflare` | 우리가 Cloudflare route를 관리 | 유료 optional |
| `custom_domain` | 사용자 도메인 연결 | 선택 |
| `enterprise_private` | VPN, private network, enterprise connector | 기업용 |

---

## 7. Token Authority

### 7.1 기본 개념

Imagine의 상용 제어는 **활성 토큰** 중심으로 한다.

```text
활성 route token이 있으면 remote access 허용
활성 license token이 있으면 제품 사용 허용
활성 seat lease가 있으면 사용자 접근 허용
활성 node token이 있으면 서버 등록 허용
```

검색 요청마다 cloud에 확인하지 않는다. Local Node가 서명된 토큰을 검증하고 집행한다.

### 7.2 토큰 종류

| 토큰 | 대상 | 용도 | 제어권 |
|---|---|---|---:|
| **License Token** | 조직 / 구독 | 유료 권한 증명 | 우리 |
| **Node Activation Token** | 서버 | 새 서버 등록 | 우리 |
| **Route Token** | 서버 route | 원격 접근 활성화 | 우리 |
| **Invite Token** | 신규 회원 | 회원 가입 제어 | 우리 |
| **Seat Lease Token** | 사용자 | seat 사용권 | 우리 |
| **Session Token** | 로컬 로그인 | UI 세션 유지 | 로컬 중심 |
| **Service Token** | worker / machine | 비인간 접근 | 필요 시 우리 |
| **Tunnel Token** | Cloudflare connector | tunnel 연결 | 모델별 |

### 7.3 Route Token

```text
route_token
  - org_id
  - node_id
  - route_id
  - allowed_hostname
  - issued_at
  - expires_at
  - status: active | expired | revoked | suspended | rotated
  - signature
```

정책:

| 상태 | 처리 |
|---|---|
| `active` | remote route 정상 허용 |
| `expiring` | 허용 + 갱신 시도 |
| `expired` | 신규 remote session 제한 |
| `revoked` | remote route 차단 |
| `suspended` | admin-only 또는 local-only |
| `rotated` | 이전 토큰 무효화 |

### 7.4 Node Activation Token

새 서버 설치 시 필요하다.

```text
node_activation_token
  - one-time use
  - org_id
  - max_nodes 검사
  - machine_fingerprint_hash에 bind
  - node_id 생성
```

흐름:

```text
1. 사용자가 Imagine 설치
2. activation code 입력
3. Control Plane이 org / plan / node limit 검사
4. node_id 발급
5. Local Node 등록
6. route token 발급 가능 상태가 됨
```

### 7.5 Invite Token

회원 가입은 반드시 invite token 기반으로 한다.

```text
invite_token
  - org_id
  - email
  - role
  - invited_by
  - expires_at
  - status
  - signature or token_hash
```

기본 정책:

```text
public_signup = false
self_signup = false
join_policy = invite_only
seat_required = true
```

---

## 8. 회원 가입과 Seat 제어

### 8.1 우리가 반드시 가져야 할 제어권

| 항목 | 설명 |
|---|---|
| 구매 seat 수 | 몇 명 요금제인지 |
| 할당 seat 수 | 현재 몇 명에게 seat가 배정됐는지 |
| active member 수 | 최근 7/30일 활성 사용자 수 |
| 초대 가능 여부 | 남은 seat가 있는지 |
| 신규 가입 허용 여부 | invite token / domain / approval 기준 |
| node 수 | 몇 대의 서버가 활성화됐는지 |
| feature entitlement | Spatial, worker, backup, remote 등 기능 권한 |
| 초과 사용 여부 | plan compliance 검사 |

핵심 관계:

```text
Control Plane = 진짜 권한 원장
Local Node    = 캐시된 권한 집행자
```

### 8.2 Seat lease

Seat는 단순 user row가 아니라 lease로 다룬다.

```text
seat_lease
  - org_id
  - member_id
  - seat_type
  - issued_at
  - expires_at
  - status
  - signed_by_control_plane
```

### 8.3 Seat 규칙

| 상황 | 처리 |
|---|---|
| 초대만 받고 미수락 | seat 미소비 |
| 초대 수락 후 최초 로그인 | seat 소비 |
| 동일 사용자의 여러 기기 | seat 1개 소비 |
| 동일 사용자의 여러 세션 | seat 1개 소비 |
| 비활성화 | seat 반환 |
| service account | user seat이 아니라 service entitlement |
| worker | user seat이 아니라 worker/node entitlement |
| seat 초과 | 신규 초대/신규 lease 차단 |
| 미납 | grace 후 신규 기능 제한 |

### 8.4 가입 흐름

```text
1. 조직 admin이 사용자 초대
2. Control Plane이 plan.max_seats 확인
3. seat 여유 있음
4. invite_token 발급
5. 사용자가 초대 수락
6. membership 생성
7. seat lease 발급
8. Local Node가 membership sync
9. 사용자는 해당 조직의 Imagine 서버에 접근 가능
```

Seat가 부족한 경우:

```text
초대 생성 차단
또는 pending invite로 두고 upgrade 요구
```

---

## 9. 사용 실태 수집 범위

### 9.1 우리가 봐야 하는 것

상용 운영과 요금제 준수를 위해 다음은 수집한다.

```text
org_id
plan_id
purchased_seats
assigned_seats
active_members_7d
active_members_30d
node_count
active_nodes
route_token_status
license_status
app_version
db_schema_version
indexed_file_count 또는 bucket
worker_count
analysis_credit_used
search_count_daily 총량
error_count_daily 총량
health_status
```

### 9.2 기본적으로 보지 않을 것

다음은 기본 수집하지 않는다.

```text
검색어 원문
검색 결과 목록
클릭한 파일
파일명
파일 경로
썸네일
원본 이미지
MC caption 전체
AI tag 전체
embedding vector
사용자별 검색 이력
개별 클릭 로그
```

핵심 원칙:

> **우리가 가져야 할 것은 상용 통제권이지, 콘텐츠 관찰권이 아니다.**

### 9.3 Daily usage rollup 예시

```json
{
  "org_id": "org_123",
  "node_id": "node_abc",
  "date": "2026-05-31",
  "app_version": "1.4.2",
  "db_schema_version": 18,

  "seats": {
    "purchased": 5,
    "assigned": 5,
    "active_7d": 4,
    "active_30d": 5
  },

  "nodes": {
    "active_nodes": 1,
    "active_workers": 1
  },

  "usage": {
    "search_count": 842,
    "indexing_jobs_completed": 133,
    "analysis_jobs_completed": 71,
    "analysis_jobs_failed": 3,
    "indexed_file_count": 17726
  },

  "route": {
    "status": "active",
    "token_status": "active"
  },

  "health": {
    "status": "healthy",
    "last_error_code": null
  }
}
```

---

## 10. Plan compliance

### 10.1 검사 항목

Control Plane은 다음을 검사한다.

```text
assigned_seats <= plan.max_seats
active_nodes <= plan.max_nodes
indexed_file_count <= plan.max_indexed_files
active_workers <= plan.max_worker_slots
feature_usage ⊆ plan.entitlements
analysis_credit_used <= plan.monthly_analysis_credits
route_count <= plan.max_routes
```

### 10.2 상태 분류

| 상태 | 의미 | 처리 |
|---|---|---|
| `compliant` | 요금제 내 사용 | 정상 |
| `near_limit` | 80~90% 도달 | admin 경고 |
| `over_limit_soft` | 일시 초과 | grace + 신규 추가 차단 |
| `over_limit_hard` | 장기 초과 | 신규 로그인/분석/route 제한 |
| `past_due` | 결제 문제 | grace 후 제한 |
| `suspended` | 장기 미납/남용 | admin-only 또는 local-only |
| `revoked` | 라이선스 취소 | 사용 차단 |

### 10.3 Seat 초과 예시

```text
Plan: 5 seats
Assigned users: 7
Active users 30d: 6
Status: over_limit_soft
```

처리:

```text
1. admin dashboard에 초과 표시
2. 신규 사용자 초대 차단
3. 신규 seat lease 발급 중단
4. 기존 사용자는 grace 기간 유지
5. admin이 2명을 비활성화하거나 플랜 업그레이드
6. grace 종료 후 제한 적용
```

기존 데이터와 검색을 갑자기 잠그는 것은 피한다. 먼저 신규 초대, 신규 node, 신규 분석, 신규 route부터 제한한다.

---

## 11. Offline grace

Local-first 제품이므로 인터넷 장애 시에도 일정 기간은 동작해야 한다. 단, 무기한 오프라인은 seat 통제를 무력화한다.

### 11.1 권장 grace

| 플랜 | Offline grace |
|---|---:|
| Personal | 14~30일 |
| Team | 7~14일 |
| Business | 14~30일 |
| Enterprise offline | 계약별 30~180일 |
| Trial | 1~3일 |

### 11.2 오프라인 중 허용

```text
기존 활성 사용자 검색
기존 로컬 DB 사용
기존 기능 사용
로컬 관리자 접근
로컬 백업
```

### 11.3 오프라인 중 제한

```text
신규 사용자 초대
신규 seat 발급
신규 node activation
신규 route 활성화
고급 managed feature 활성화
plan 변경 반영
```

---

## 12. Cloudflare 운영 모델

### 12.1 Cloudflare의 역할

Cloudflare는 검색 품질을 높이는 도구가 아니다. 역할은 운영성이다.

| 구성 | 역할 |
|---|---|
| Tunnel | 사용자 로컬 서버를 외부에서 접근 가능하게 함 |
| Access | 접근 제어, identity gate |
| DNS / TLS | 도메인, HTTPS |
| WAF / rate limit | edge 보호 |
| R2 | optional backup, artifact, snapshot 보관 후보 |
| Workers | 일부 control API 또는 lightweight routing 후보 |

### 12.2 모델 A — 사용자 소유 Cloudflare

일반 사용자 / 저비용 원칙에 가장 적합하다.

```text
사용자 Cloudflare 계정
사용자 domain 또는 tunnel
우리 setup wizard / guide 제공
검색 트래픽은 우리 서버를 통과하지 않음
```

장점:

```text
우리 운영비 최소
사용자 데이터 경로 분리
일반 사용자에게 추가 운영비 청구 불필요
```

단점:

```text
사용자 설정 복잡도 존재
우리가 Cloudflare 자체를 완전히 통제하지는 않음
```

이 모델에서는 우리가 Cloudflare 계정 자체보다 **Imagine route token**을 통제한다.

### 12.3 모델 B — 우리가 관리하는 Cloudflare

팀 / 기업 / managed option용이다.

```text
customer.imagine.app
우리 Cloudflare 계정
우리가 tunnel / access / route 관리
사용자는 단순 접속
```

장점:

```text
사용자 경험 단순
route revoke 쉬움
지원 쉬움
```

단점:

```text
우리 운영 책임 증가
비용 증가
보안 책임 증가
```

따라서 일반 기본값이 아니라 **유료 managed remote access**로 둔다.

---

## 13. 자동 업데이트와 배포

### 13.1 자동 업데이트의 역할

상용 제품에서 자동 업데이트는 필수다. 단순 파일 교체가 아니라 상태 보존형 업데이트여야 한다.

필요 구성:

| 구성 | 역할 |
|---|---|
| Release Server | 최신 버전, 채널, manifest 제공 |
| Update Agent | 로컬에서 확인·다운로드·검증·설치 |
| Signed Manifest | 파일 hash와 signature 검증 |
| Migration Runner | DB schema / index / config migration |
| Pre-update Snapshot | DB / config 백업 |
| Health Check | 업데이트 후 정상성 확인 |
| Rollback Manager | 실패 시 이전 버전 복구 |
| Release Channels | stable / beta / nightly / enterprise pinned |

### 13.2 업데이트 흐름

```text
Imagine Node
  ↓
/updates/check
  - current_version
  - os / arch
  - db_schema_version
  - release_channel
  - license_token
  - node_id

Control Plane
  ↓
update_manifest.json
  - target_version
  - artifact_url
  - sha256
  - signature
  - migration_plan
  - min_supported_version
  - rollback_policy

Imagine Node
  ↓
1. artifact download
2. signature verification
3. DB/config snapshot
4. service stop
5. binary update
6. migration 실행
7. service start
8. health check
9. success report
10. 실패 시 rollback
```

### 13.3 배포 형태

| 배포 형태 | 대상 |
|---|---|
| Desktop installer | 개인 사용자 |
| Docker Compose | NAS / homelab / prosumer |
| System service | Windows Service / systemd / launchd |
| Headless node | 사무실 PC / NAS / mini server |
| Worker package | 외부 분석 워커 |
| Enterprise package | 오프라인 / 고정 버전 / 프록시 환경 |

### 13.4 첫 설치 흐름

```text
1. Imagine 설치
2. Local Node 생성
3. 브라우저에서 onboarding
4. 계정 로그인
5. 조직 선택 또는 생성
6. 라이선스/플랜 확인
7. Node activation
8. NAS/WebDAV 연결
9. Cloudflare 연결 여부 선택
10. 초기 인덱싱 시작
```

---

## 14. Commercial Control Plane

### 14.1 역할

Commercial Control Plane은 다음을 담당한다.

```text
조직
사용자
회원 가입
seat
role
plan
subscription
entitlement
license
node
route
usage rollup
revocation
```

검색 실행은 담당하지 않는다.

### 14.2 요금 단위

처음부터 복잡한 usage-based billing을 넣기보다 다음 구조가 현실적이다.

```text
요금 = 기본 플랜 + seat 수 + node 수 + 선택적 analysis credit / managed add-on
```

가능한 과금 단위:

| 과금 단위 | 설명 |
|---|---|
| Seat | 로그인 가능한 사용자 수 |
| Node | 설치된 Imagine 서버 수 |
| Indexed files | 인덱싱 가능한 파일 수 |
| Worker slots | 외부 분석 워커 수 |
| Analysis credits | MC / Spatial / VLM 분석량 |
| Backup storage | managed backup 용량 |
| Managed remote access | 우리가 관리하는 Cloudflare routing |
| Enterprise features | SSO, audit, SLA, offline license |

### 14.3 플랜 초안

| 플랜 | 대상 | 구성 |
|---|---|---|
| Personal Local | 개인 | 1 user, 1 node, local only |
| Personal Remote | 개인 고급 | BYO Cloudflare, 자동 업데이트, local backup |
| Team | 소규모 팀 | per-seat, role, shared server, route token |
| Studio / Business | 제작팀 | 다중 node, worker, backup option, admin dashboard |
| Enterprise | 기업 | SSO, pinned version, offline license, SLA, managed option |

### 14.4 Entitlement key 예시

```text
max_users
max_nodes
max_routes
max_indexed_files
max_worker_slots
remote_access_enabled
managed_remote_access_enabled
cloud_backup_enabled
backup_retention_days
spatial_backfill_enabled
monthly_analysis_credits
audit_log_enabled
sso_enabled
api_access_enabled
offline_license_days
priority_support
```

---

## 15. 데이터 모델 초안

### 15.1 Control Plane DB

```text
organizations
  id
  name
  status
  plan_id
  created_at

users
  id
  email
  name
  created_at

memberships
  id
  org_id
  user_id
  role
  status
  joined_at

plans
  id
  name
  max_seats
  max_nodes
  max_routes
  max_indexed_files
  max_worker_slots
  features_json

subscriptions
  id
  org_id
  provider
  provider_subscription_id
  status
  purchased_seats
  current_period_end

seat_allocations
  id
  org_id
  user_id
  seat_type
  status
  assigned_at
  released_at

licenses
  id
  org_id
  status
  issued_at
  expires_at
  offline_grace_days

nodes
  id
  org_id
  machine_fingerprint_hash
  hostname_hash
  app_version
  db_schema_version
  status
  last_seen_at

routes
  id
  org_id
  node_id
  hostname
  route_mode
  status
  last_seen_at

route_tokens
  id
  org_id
  node_id
  route_id
  token_hash
  status
  issued_at
  expires_at
  revoked_at

node_activation_tokens
  id
  org_id
  token_hash
  status
  expires_at
  used_at

invite_tokens
  id
  org_id
  email
  role
  token_hash
  status
  expires_at
  accepted_at

seat_leases
  id
  org_id
  member_id
  token_hash
  status
  issued_at
  expires_at

entitlements
  id
  org_id
  key
  value
  source
  valid_until

usage_rollups
  id
  org_id
  node_id
  date
  assigned_seats
  active_members_7d
  active_members_30d
  indexed_file_count
  search_count
  analysis_jobs_completed
  health_status

revocations
  id
  org_id
  subject_type
  subject_id
  reason
  revoked_at
```

### 15.2 Local Node DB

```text
local_license_cache
  signed_license_token
  valid_until
  offline_until

local_route_state
  route_id
  signed_route_token
  status
  last_sync_at

local_membership_cache
  member_id
  email_hash
  role
  status
  seat_valid_until

local_seat_leases
  member_id
  seat_type
  signed_lease
  valid_until

local_sessions
  session_id
  member_id
  expires_at

local_entitlement_cache
  org_id
  signed_entitlement_token
  fetched_at
  valid_until

local_usage_daily
  date
  active_member_count
  search_count
  indexing_jobs
  analysis_jobs
  error_count

local_revocation_cache
  subject_type
  subject_id
  revoked_at

local_sync_outbox
  event_type
  payload
  created_at
  sent_at
```

---

## 16. Enforcement 정책

### 16.1 Hard enforcement

즉시 막아도 되는 것:

| 항목 | 처리 |
|---|---|
| 신규 seat 초과 | 차단 |
| 신규 node 초과 | 차단 |
| 신규 route 초과 | 차단 |
| trial 만료 후 신규 activation | 차단 |
| 권한 없는 feature 사용 | 차단 |
| revoked user 로그인 | 차단 |
| revoked node sync | 차단 |
| 위조 token | 차단 |

### 16.2 Soft enforcement

부드럽게 처리해야 하는 것:

| 항목 | 처리 |
|---|---|
| 일시적 seat 초과 | grace + admin 경고 |
| 결제 실패 직후 | grace |
| indexed file 초과 | 신규 인덱싱 제한, 기존 검색 유지 |
| analysis credit 초과 | 추가 분석 제한, 기존 검색 유지 |
| route token 만료 임박 | 경고 + 갱신 시도 |
| old version | 경고 후 업데이트 요구 |

### 16.3 제한 순서

고객 경험을 깨지 않기 위해 제한은 다음 순서로 적용한다.

```text
1. 신규 초대 차단
2. 신규 seat lease 차단
3. 신규 node activation 차단
4. 신규 route activation 차단
5. 신규 분석 job 제한
6. remote access 제한
7. admin-only mode
8. 최종 suspension
```

기존 로컬 데이터와 기존 검색을 갑자기 잠그는 것은 최후 수단으로 둔다.

---

## 17. 보안과 남용 방지

사용자 서버에서 실행되는 제품이므로 완전한 DRM은 불가능하다. 목표는 합리적인 상용 라이선스 통제다.

필수 장치:

| 장치 | 역할 |
|---|---|
| signed license token | 라이선스 위조 방지 |
| signed entitlement token | 기능 권한 위조 방지 |
| signed route token | remote route 임의 활성화 방지 |
| node activation | 서버 수 제한 |
| machine fingerprint hash | node 공유/남용 탐지 |
| seat lease signature | 로컬 seat 조작 방지 |
| periodic refresh | 장기 오프라인 남용 방지 |
| revocation list | 차단된 user/node/route 반영 |
| audit event hash chain | 로컬 이벤트 조작 탐지 후보 |
| grace period limit | 무한 오프라인 사용 방지 |
| activation rate limit | 라이선스 공유 방지 |

---

## 18. 우리가 저장해도 되는 데이터와 안 되는 데이터

### 18.1 저장 가능

```text
org_id
user_id
email
role
seat_status
plan
subscription_status
license_id
node_id
app_version
db_schema_version
last_seen_at
health_summary
release_channel
route_status
feature_flags
billing_customer_id
usage_rollup 집계값
```

### 18.2 기본 저장 금지

```text
원본 파일
썸네일
파일 전체 경로
파일명 전체
검색 쿼리 원문
검색 결과 목록
MC caption 전체
AI tags 전체
embedding vector
SQLite DB
사용자 NAS credential
Cloudflare token 원문
개별 클릭 로그
개별 사용자 검색 이력
```

### 18.3 opt-in 예외

```text
diagnostic bundle
crash log
redacted config
anonymized benchmark result
support용 일부 로그
```

---

## 19. 관리자 화면

### 19.1 사용자 조직 관리자 화면

```text
Organization Admin
  ├─ Plan
  ├─ Billing
  ├─ Seats
  ├─ Members
  ├─ Roles
  ├─ Invites
  ├─ Nodes
  ├─ Routes
  ├─ Route Tokens
  ├─ Backups
  ├─ Usage Summary
  ├─ Health
  └─ API / Service Tokens
```

조직 admin이 할 수 있는 것:

```text
사용자 초대
사용자 비활성화
seat 할당/회수
role 변경
node 이름 변경
route 활성/비활성 요청
usage 확인
billing portal 이동
upgrade 요청
```

### 19.2 우리 공급자 콘솔

```text
Imagine Operator Console
  ├─ Customers
  ├─ Organizations
  ├─ Plans
  ├─ Subscriptions
  ├─ Purchased seats
  ├─ Assigned seats
  ├─ Active members 7d / 30d
  ├─ Active nodes
  ├─ Routes
  ├─ Token status
  ├─ Version distribution
  ├─ Failed updates
  ├─ Over-limit status
  ├─ License status
  ├─ Entitlements
  ├─ Revocations
  ├─ Usage rollups
  └─ Support tools
```

우리가 직접 할 수 있어야 하는 조치:

```text
seat limit 변경
plan 변경
entitlement 부여/회수
node 비활성화
route token revoke
user revoke
license 재발급
forced update 요구
grace period 연장
trial 연장
suspension
billing 상태 동기화
```

---

## 20. SLA와 책임 문구

### 20.1 우리가 보증하는 것

```text
릴리즈 무결성
업데이트 서명
라이선스 서버 가용성
seat / entitlement 정확성
route token 발급과 회수
보안 패치 제공
문서화된 백업/복구 도구
Control Plane 장애 시 grace period
```

### 20.2 우리가 보증하지 않는 것

```text
사용자 PC 전원 상태
사용자 NAS 장애
사용자 디스크 손상
사용자 네트워크 장애
사용자 Cloudflare 계정 제한
사용자 OS 문제
사용자가 직접 수정한 DB
사용자 하드웨어 성능
```

### 20.3 우리가 제공해야 하는 복구 도구

```text
health check
diagnostic export
restore guide
safe mode
repair command
DB integrity check
thumbnail cache rebuild
index rebuild
support bundle 생성
```

---

## 21. 로드맵

### P0 — 책임 경계 문서화

```text
What we manage
What customer manages
What is optional managed
What data we collect
What data we never collect
What happens offline
What happens when PC is off
What is included in base plan
What costs extra
```

### P1 — Route Registry

```text
organizations
nodes
routes
route_status
last_seen
hostname
route_mode
```

### P2 — Token Authority

```text
license_token
node_activation_token
route_token
invite_token
seat_lease_token
revocation_list
```

### P3 — Membership / Signup Gate

```text
invite_only
seat check
admin approval
domain allowlist optional
public signup disabled
```

### P4 — Local Token Validator

```text
route token 검증
seat lease 검증
license token 검증
offline grace
revocation cache
```

### P5 — Thin Control Plane

```text
organizations
users
memberships
seats
plans
subscriptions
entitlements
licenses
nodes
routes
usage_rollups
```

### P6 — Automatic Update Agent

```text
signed manifest
artifact download
hash verification
DB snapshot
migration
health check
rollback
update report
```

### P7 — Minimal Usage Rollup

```text
assigned seats
active members
node status
route status
search count total
error count total
indexed file count bucket
```

### P8 — BYO Cloudflare Wizard

```text
사용자 Cloudflare 계정 연결
Tunnel 생성 가이드
Access policy 템플릿
local endpoint 연결
health check
```

### P9 — Optional Managed Services

```text
managed Cloudflare
managed backup
cloud worker
enterprise SSO
long-retention audit
priority support
```

### P10 — Spatial Backfill Sprint

검색 시스템 자체의 다음 성능 도약은 Spatial 데이터 백필이다.

```text
spatial-only backfill job
500~1,000개 샘플 실행
spatial queryset 생성
S3.2 boost ablation
전체 17,000+ 파일 백필
```

---

## 22. 최종 책임 매트릭스

| 영역 | 우리 | 사용자 | Optional managed |
|---|---:|---:|---:|
| 검색 엔진 코드 | 책임 | 사용 | - |
| 로컬 실행 환경 | 지원 | 책임 | 가능 |
| 원본 파일 | 없음 | 책임 | 없음 |
| SQLite DB | schema 책임 | 파일 보관 책임 | 백업 지원 |
| 썸네일 | 생성 코드 책임 | 저장 책임 | 백업 가능 |
| 분석 파이프라인 | 코드 책임 | 실행 리소스 책임 | cloud worker 가능 |
| 자동 업데이트 | 릴리즈/서명 책임 | 적용 환경 책임 | - |
| DB migration | 코드 책임 | 실행 환경 책임 | - |
| License | 책임 | 캐시 사용 | - |
| Seat | 책임 | 조직 운영 | - |
| Billing | 책임 | 결제 정보 | - |
| Route Registry | 책임 | 연결 사용 | managed 가능 |
| Route Token | 책임 | 로컬 검증 | managed 가능 |
| 회원 가입 | 정책/토큰 책임 | 초대 운영 | - |
| Cloudflare Tunnel | 템플릿 제공 | 기본 책임 | 대행 가능 |
| Cloudflare Access | 정책 템플릿 | 기본 책임 | 대행 가능 |
| Backup | 도구 제공 | 기본 책임 | managed 가능 |
| Monitoring | 최소 node health | 로컬 장애 대응 | enterprise 확장 |
| Support | 제품 지원 | 환경 정보 제공 | premium 가능 |
| SLA | Control Plane 한정 | 로컬 서버 제외 | managed 범위 한정 |

---

## 23. 최종 제품 문구

### 23.1 내부 정의

> **Imagine은 사용자의 로컬 서버에서 검색과 분석을 수행하는 local-first AI search appliance이며, 우리는 그 서버로 가는 route, 활성 토큰, 회원 가입, seat, node, entitlement, update를 중앙 Control Plane에서 최소 비용으로 제어한다.**

### 23.2 외부 설명용 문구

> **Imagine keeps your content and search on your own machine, while providing secure access control, automatic updates, licensing, and team seat management through a lightweight control plane.**

### 23.3 운영 원칙 문구

> **사용자 서버는 검색을 수행하고, 우리는 그 서버로 가는 route와 활성 토큰, 회원 가입, seat 한도를 통제한다. 접속자는 관리하지만, 사용자의 검색 행위는 기본적으로 추적하지 않는다.**

### 23.4 비용 원칙 문구

> **운영비가 늘어나는 기능은 기본값이 아니라 선택형이다. 일반 사용자는 로컬 검색 제품 사용을 위해 별도의 운영비를 추가로 지불하지 않는다.**

---

## 24. 한 줄 결론

> **Imagine의 다음 단계는 검색 성능 개선이 아니라, local-first 검색 엔진을 상용 제품으로 만들기 위한 Access Routing Plane, Token Authority, Lifecycle Plane, Commercial Control Plane의 구축이다.**
