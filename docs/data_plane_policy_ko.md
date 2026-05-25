# Imagine Data Plane Policy

> **상태:** Phase 7. AWS relay 비용 폭주와 개인정보 위험을 막기 위해
> 어떤 데이터가 어디로 흐르는지를 명시한다. 본 문서는 코드 변경의
> 검토 기준이자 외부 워커 안내 자료의 단일 출처다.

## 핵심 원칙

1. **Relay = control plane only.** 메시지 봉투는 16KB로 제한되며,
   파일·썸네일·모델 가중치·DB 파일·대용량 벡터 bulk는 봉투 안에 들어가서는
   안 된다. 위반은 `FORBIDDEN_TYPE` 에러로 즉시 거절된다(`infra/aws-relay/
   src/auth.py`의 `FORBIDDEN_BODY_KEYS`).
2. **Original assets stay on the owner side.** 사용자의 원본 이미지와
   썸네일은 그 사용자 PC 또는 사용자 본인이 제어하는 스토리지에만 존재한다.
   외부 워커가 필요로 하는 경우, **Imagine 서버가 직접 발급한 short-lived
   storage URL(또는 LAN/manual_external 경로)** 을 통해 가져가도록 한다.
3. **Workers only see their assigned tasks.** 워커는 자기에게 할당된
   `file_tasks` 행에 한해서만 다운로드/업로드 권한을 갖는다.
4. **Quarantine on revoke.** 워커가 차단되면(`worker_sessions.status=
   'blocked'`) 즉시 그 워커가 들고 있던 task는 reclaim되고, 추가 claim도
   거부된다.

## Relay를 통과해도 되는 것

| 종류           | 비고 |
|----------------|------|
| task metadata  | task_id, phase, status (작은 JSON 필드) |
| heartbeat      | online 여부, 간이 stats (workers_online 등) |
| phase status   | mc/vv/mv 단계별 진행률 (정수/짧은 문자열) |
| 작은 JSON 결과 | 1KB 이하의 요약(에러 메시지, 소요 ms 등) |
| 오류 요약      | 스택 트레이스가 아니라 사람이 읽을 수 있는 1줄 |

## Relay를 통과해서는 안 되는 것

| 종류                | 거절 메커니즘 |
|---------------------|---------------|
| 원본 이미지 byte    | `FORBIDDEN_BODY_KEYS = {file_bytes, image_bytes, raw_bytes, ...}` |
| 썸네일 bulk         | `thumbnail_bytes` 키 거절 |
| DB 파일             | `db_bytes` 키 거절 |
| 모델 가중치         | `weights`, `model_weights` 키 거절 |
| 큰 벡터 bulk        | 16KB 제한으로 자연 차단 |
| 서버 패스워드(해시) | `server_password`, `server_password_hash` 키 거절 |
| 관리자 refresh token| `admin_refresh_token` 키 거절 |
| Firebase ID token   | `firebase_id_token` 키 거절 |

## 워커 접근 권한 모델

`/api/v1/workers/*` 엔드포인트와 `analysis_jobs.file_tasks` 행 사이의
관계는 다음 4가지 규칙으로 단언한다.

1. **자기 task만 다운로드 가능.** 워커가 다른 사용자의 파일 또는
   자기 자신에게 할당되지 않은 task의 파일을 다운로드하려고 하면 403.
2. **자기 task만 업데이트 가능.** 다른 worker_session에 할당된 task의
   상태를 update/complete 하려고 하면 403.
3. **차단된 워커는 claim 불가.** `worker_sessions.status='blocked'` 인
   세션은 `/api/v1/workers/claim` 호출을 거부 (409).
4. **소유자 일치.** worker_session 의 `user_id` 와 호출자 토큰의 `user_id`
   가 다르면 어떤 동작도 거부.

이 4가지는 `tests/test_worker_data_access_scope.py` 에서 단위 테스트로
강제된다. 정책이 바뀌면 테스트도 함께 바뀌어야 한다.

## 외부 워커가 자산을 가져가는 방법

권장 경로 우선순위:

1. **direct_lan / manual_external.** 워커가 같은 LAN/터널을 통해 직접
   `GET /api/v1/files/{file_id}/thumbnail` 로 가져간다. 가장 단순하고 가장
   안전하다.
2. **Pre-signed storage URL.** 사용자가 자기 클라우드 스토리지(S3/GCS 등)
   사용 시 서버가 그 스토리지의 1회용 GET URL을 생성해 워커에게 전달.
3. **Relay를 통한 메타데이터 + LAN/터널 fallback.** Relay는 task 배정과
   완료 보고에만 쓰고, 실제 파일 전송은 1번/2번 경로를 사용.

**Relay를 통한 직접 파일 전송은 어떤 경우에도 허용하지 않는다.** 향후 cost
폭주를 막기 위한 가장 강한 단일 규칙이다.

## 위반 시 동작

| 위반                                  | 동작 |
|---------------------------------------|------|
| `FORBIDDEN_BODY_KEYS` 포함 envelope   | Lambda가 즉시 `FORBIDDEN_TYPE` 반환, 메시지 폐기 |
| 16KB 초과 envelope                    | Lambda가 즉시 `PAYLOAD_TOO_BIG` 반환 |
| 워커가 자기 task가 아닌 파일 다운로드 | `/api/v1/files/...` 가 403 |
| 차단된 워커의 claim                   | `/api/v1/workers/claim` 가 409 |

## 변경 시 체크리스트

- `FORBIDDEN_BODY_KEYS` 추가/삭제 → `tests/test_relay_protocol_contract.py`
  의 `forbidden_key` 파라미터와 본 문서의 표를 동시 수정.
- 워커 접근 권한 변경 → `tests/test_worker_data_access_scope.py` 동시 수정.
- 권한이 추가될 때마다 Audit log(Phase 8)에 이벤트 종류를 추가.
