# CAS 파이프라인 정밀 설계 — 콘텐츠 주소 파생물 캐시

작성: 2026-06-11 (재범위: 같은 날 냉정 평가 후)
상태: 설계 확정 대기 (구현 전)

> **재범위 (2026-06-11 실측 후)**: 해시 보유 표본 3,568건의 중복률 실측 = **0.6%**
> (설계 초안의 "10~30% 예상"은 기각). 따라서 이 설계의 **1차 가치는 모델 버전닝
> 파도(§5)와 이동·개명 내성**이며, 중복 제거는 M2 백필 후 전수 재측정을 통과해야
> 부활하는 **조건부 효익**으로 강등한다. §4.3 다운로드 dedup과 trust_size_mtime
> 모드는 그 게이트 통과 전까지 구현하지 않는다.
> **스프린트 어시스트(§9)는 설계 동결** — 첫 실제 GPU 임대 수요가 발생할 때 해동.
선행 문서: `docs/nas_processing_flow.md`, `docs/worker_runtime_contract_ko.md`
요구사항 출처: 2026-06-11 합의 15개 항목 (CAS 본체, 압도적 GPU 6건, 대형 버퍼 4건, 스프린트 4건, 자동/수동 경계 1건)

---

## 1. 목표와 비목표

### 목표

1. **모델 교체 = 자연스러운 파도**: 새 모델 버전 키의 부재가 곧 pending이고, 구 결과는 교체 완료까지 검색을 계속 서빙한다. ← **1차 가치 (재범위)**
2. **이동·개명 내성**: 파일을 옮겨도 파생물이 따라온다 (relink 도구의 점진적 불필요화).
3. **중복 내용 1회 계산** *(조건부 — 실측 0.6%로 강등)*: M2 백필 후 전수 재측정에서 유의미한 중복률이 나와야 dedup 경로(§4.3)를 구현한다.
4. **고비용 GPU의 유휴 0 운용**: 사전 적재 → 돌격 → 회수의 스프린트 패턴을 1급 운용으로.
5. **대형 버퍼(수백 GB 스테이징) 운용 안전성**: 바이트 회계, 재시작 생존, 다운로드 생략.

### 비목표 (명시적으로 하지 않는 것)

- **조정 계층 교체 없음**: 원장(SQLite) + 풀 기반 워커 + 폴링 핸드오프는 유지한다. 이벤트 로그/워크플로 엔진/푸시 배정은 검토 후 기각됨(중단 내성·로컬 우선 제약에서 열위).
- **검색 평면 재설계 없음**: 검색 인덱스(FTS, vec_files, vec_structure)는 파일 단위 키를 유지한다. §2 참조.
- **자동 비용 결정 없음**: GPU 임대/투입/분리는 운영자 결정. 시스템은 신호만 계산한다.

---

## 2. 핵심 원리: 계산 평면과 검색 평면의 분리

이 설계의 가장 중요한 결정이다.

```text
계산 평면 (CAS)                          검색 평면 (파일 구체화)
─────────────────────                    ─────────────────────
derivations 테이블                        files / files_fts / vec_files / vec_structure
키: (content_hash, phase,                키: file_id (현행 유지)
     model_version)
"이 내용을 이 모델로 분석한 결과"          "이 파일이 검색에서 어떻게 보이는가"
                  │
                  └── 구체화(materialize) ──▶
                      캐시 히트 시 결과를 파일 행으로 복사
```

**왜 검색 평면을 hash 키로 바꾸지 않는가**: 검색 결과의 단위는 파일이다(사용자는 파일을 찾는다). 벡터 인덱스를 hash 단위로 통합하면 검색 코어(sqlite_search, scoring, RRF)를 전부 수정해야 하고, P@K 벤치마크 체계가 무효화된다. 반면 파일 단위 구체화는:

- 검색 코드 **무변경** — 벤치마크 연속성 유지
- 저장 중복 비용: 벡터 1개 ≈ 4.6KB (FLOAT[1152]) — 중복 파일 N개면 N배 저장하지만, **계산이 비싸고 저장이 싼** 시스템이므로 올바른 트레이드
- FTS/메타데이터는 어차피 파일 단위가 필요(파일명·경로가 팩트 축 입력)

캐시 히트의 의미: "계산을 생략하고 구체화만 수행한다."

---

## 3. 데이터 모델

### 3.1 derivations (신규)

```sql
CREATE TABLE IF NOT EXISTS derivations (
    content_hash  TEXT NOT NULL,
    phase         TEXT NOT NULL CHECK (phase IN ('mc','vv','mv')),
    model_version TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'done' CHECK (status IN ('done','failed')),
    result_json   TEXT,            -- mc: {caption, ai_tags, image_type, ...}
                                   -- vv/mv: NULL (벡터는 BLOB 컬럼)
    vector_blob   BLOB,            -- vv/mv 원본 벡터 (구체화 시 vec_* 테이블로 복사)
    error_message TEXT,            -- failed일 때
    created_at    TEXT DEFAULT (datetime('now')),
    created_by    TEXT,            -- worker_name (감사 추적)
    PRIMARY KEY (content_hash, phase, model_version)
);
CREATE INDEX IF NOT EXISTS idx_deriv_phase_ver ON derivations(phase, model_version);
```

설계 노트:

- **parse는 캐시 대상에서 제외**한다. 파싱 산출물(썸네일·레이어 메타)은 이미 files 테이블에 영속이고, content_hash 자체가 파싱(다운로드 후) 시점에 확정되므로 캐시 조회의 입장권이지 캐시 항목이 아니다. 단, 썸네일은 hash 기반 파일명(`{hash[:16]}_thumb.png`)으로 저장해 중복 파일의 썸네일 재생성을 생략한다.
- **failed도 기록**한다 — 같은 내용을 같은 모델로 재시도하는 낭비 방지. 단 재시도 정책(모델 업그레이드 시 자동 무효)은 model_version이 해결한다.
- `vector_blob`: sqlite-vec 가상 테이블은 임의 키 조회가 불편하므로 원본 벡터를 BLOB으로 보관하고, 구체화 시 `vec_files(file_id, embedding)`에 복사한다.

### 3.2 model_registry (신규)

```sql
CREATE TABLE IF NOT EXISTS model_registry (
    phase          TEXT NOT NULL CHECK (phase IN ('mc','vv','mv')),
    model_version  TEXT NOT NULL,   -- 예: "qwen2.5-vl-7b/prompt-v3/domain-game-v2"
    is_active      INTEGER NOT NULL DEFAULT 0,
    activated_at   TEXT,
    PRIMARY KEY (phase, model_version)
);
```

- **model_version의 구성**: 모델 ID만이 아니라 **출력에 영향을 주는 모든 설정**을 포함한다 — MC는 `{모델ID}/{프롬프트 버전}/{도메인 프로파일 버전}`, VV/MV는 `{모델ID}`. 버전 문자열 생성은 단일 함수(`get_model_version(phase)`)로 일원화하고, 프롬프트/도메인 YAML 변경 시 버전을 올리는 규율을 강제한다.
- **MC 입력의 내용 한정 원칙**: MC 프롬프트 입력은 내용 유래(픽셀, 파일 내부 메타: 레이어명·텍스트 레이어)로 한정한다. 폴더 경로·사용자 태그를 MC 입력에 넣으면 내용 주소화가 깨진다 — 경로·사용자 신호는 검색 시점(quality rerank)에서만 쓴다. 향후 컨텍스트 주입이 꼭 필요해지면 키에 `context_hash`를 추가하는 탈출구를 남긴다(현 설계에서는 미사용).
- phase당 active는 1개. 활성 버전 변경이 곧 재처리 파도의 선언이다(§5).

### 3.3 staging_manifest (신규 — 대형 버퍼)

```sql
CREATE TABLE IF NOT EXISTS staging_manifest (
    file_id      INTEGER PRIMARY KEY,
    temp_path    TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    content_hash TEXT,              -- 다운로드 직후 계산 (재시작 검증용)
    verified_at  TEXT,
    created_at   TEXT DEFAULT (datetime('now'))
);
```

재시작 시: `parse_status='pending' AND download_status='done'`인 작업의 temp 파일을 매니페스트로 검증(존재+크기, hash는 선택적 정밀 검증)하고 살아 있으면 재사용한다. 검증 실패분만 `download_status='pending'`으로 되돌린다.

### 3.4 기존 테이블 변경 (최소)

```sql
-- files: 변경 없음 (content_hash 컬럼 기존재. 백필 필요: 현재 3,568/17,726)
-- file_tasks: 캐시 히트 추적 컬럼만 추가
ALTER TABLE file_tasks ADD COLUMN mc_cache_hit INTEGER DEFAULT 0;
ALTER TABLE file_tasks ADD COLUMN vv_cache_hit INTEGER DEFAULT 0;
ALTER TABLE file_tasks ADD COLUMN mv_cache_hit INTEGER DEFAULT 0;
```

**file_tasks는 그대로 잡 단위 운영 원장으로 남는다.** derivations는 영속 캐시, file_tasks는 "이번 잡에서 이 파일이 어디까지 갔나"다. 역할이 다르다.

---

## 4. 흐름 변화

### 4.1 캐시 조회 지점 — 파싱 완료 직후 (단일 지점)

content_hash는 실물이 있어야 계산되므로, 조회는 **파싱 완료 트랜잭션 안에서** 한 번 수행한다:

```python
# FileTaskParsePool._process_task 말미 (의사코드)
hash = compute_content_hash(file)          # 기존 단계
for phase in ("mc", "vv", "mv"):
    ver = get_model_version(phase)
    d = SELECT * FROM derivations WHERE (hash, phase, ver)
    if d and d.status == 'done':
        materialize(file_id, phase, d)     # files 행 갱신 + vec_* 복사 + FTS 갱신
        UPDATE file_tasks SET {phase}_status='done', {phase}_cache_hit=1
# 전 phase 히트면 이 파일은 워커에 한 번도 가지 않고 완료된다
```

- 워커·스케줄러·claim 경로는 **무변경** — 캐시 히트분은 애초에 pending으로 나타나지 않는다. 조정 계층 불변 원칙의 실현.
- `materialize()`는 기존 `update_vision_fields`/`save_vv`/`save_mv` 저장 경로를 재사용한다(새 저장 코드 작성 금지 — 동작 검증된 경로 하나 유지).

### 4.2 결과 수집 = 캐시 적립 (shadow write)

워커 결과 저장 엔드포인트(`POST /files/{id}/mc|vv|mv`)가 파일 행 갱신과 **동시에 derivations에 기록**한다:

```python
# 한 트랜잭션
UPDATE files ... (기존)
INSERT OR REPLACE INTO derivations (hash, phase, active_ver, 'done', result, vector, worker)
COMMIT
```

files.content_hash가 NULL인 레거시 행은 캐시 적립을 건너뛴다(백필 후 자연 해소).

### 4.3 다운로드 단계 dedup (대형 버퍼 §8과 연동)

원격 파일은 받기 전엔 hash를 모른다. 2단계 프리필터:

1. **잡 생성 시**: NAS 목록의 `(파일명, size)`가 기존 files의 `(file_name, file_size)`와 일치하고 그 파일의 전 phase 파생물이 active 버전으로 존재하면 `dedup_candidate=1` 마킹. **다운로드는 수행하되 우선순위를 낮춘다** (보수적 — 추정만으로 생략하지 않음).
2. **다운로드 직후**: hash 계산 → 전 phase 캐시 히트면 파싱에서 §4.1 경로로 즉시 완료. 원본은 즉시 버퍼에서 해제.

설정 `dedup.trust_size_mtime: true`(기본 false)일 때만 1단계에서 다운로드 자체를 생략한다(NAS 재인덱싱 가속 모드 — 운영자 옵트인).

---

## 5. 모델 교체 파도

```text
운영자: model_registry에서 vv의 active를 v2로 변경 (관리 UI/CLI)
   ↓ (시스템)
재처리 잡 자동 생성: "VV 재임베딩 (v1→v2)"
   대상 = files 중 (content_hash, 'vv', 'v2') 파생물이 없는 전부
   file_tasks 생성: parse_status='done', vv_status='pending'만 (mc/mv는 'n/a')
   ↓
일반 파이프라인이 처리 (스케줄러는 그냥 vv pending이 많아진 것으로 인식)
   ↓
검색: 구체화된 최신 결과를 서빙 — 파일별로 v2 구체화가 끝나는 순간 교체됨.
      전역 스위치/rebuild 배지 불필요 (점진 교체)
```

- 파도 잡은 일반 분석 잡과 동일한 단위(일시정지/취소/진행률) — 새 운영 개념을 만들지 않는다.
- **스프린트와의 결합**: 파도 잡 생성 → 스프린트 어시스트(§9)에 "백로그 N시간분" 표시 → 임대 GPU 투입. 이것이 스프린트의 대표 유스케이스다.
- 기존 rebuild_needed/FTS 버전 배지 체계는 M5에서 model_registry 조회로 단순화한다.

---

## 6. 스케줄러 개정

### 6.1 압력 공식: time-to-drain (요구 3)

```python
# 현행: pressure = (pending / (workers_on + 1)) × PHASE_TIME × speed_factor
# 개정: 머릿수가 아니라 용량으로 나눈다
def pressure(phase, me):
    assigned_capacity = SUM(speed(w) for w in online_workers
                            if w.assigned_mode == phase)   # 실측 EMA, 없으면 baseline
    drain_min = pending[phase] / (assigned_capacity + speed(me, phase))
    p = drain_min × PHASE_TIME[phase]      # 시간 가중은 유지 (MC 우선 정책의 표현)
    if phase == 'mv':
        p += min(pending, 50) × 10 / NORM  # 완료 보너스 — 새 스케일에 맞춰 재정규화
    return p
```

- 의미: "내가 이 phase에 합류하면 남은 일이 몇 분치인가 × 그 일의 무게".
- unserved boost(×1.5)는 자연 흡수된다 — 아무도 없으면 분모가 내 속도뿐이라 drain이 커진다.
- 안정성 계수 2.0, MC 능력 게이팅, throttle 감산, 핀(2026-06-11 구현 완료)은 유지.
- MV 보너스의 `NORM`은 구현 시 캘리브레이션: "MC 잔량 < 31×(클러스터 MC 용량/baseline)일 때 MV가 이긴다"는 현행 동작점을 보존하도록 정한다.

### 6.2 배치 공식: claim 왕복 암모타이즈로 재정의 (요구 2)

phase가 sticky한 워커는 배치 사이에도 모델이 상주한다. 배치가 갚아야 할 비용은 모델 로드(분 단위)가 아니라 claim 왕복(밀리초)이다:

```python
TARGET_BATCH_SECONDS = {"mc": 120, "vv": 60, "mv": 60}   # 현행 600/120/120에서 인하
MAX_BATCH           = {"mc": 50,  "vv": 200, "mv": 200}  # 현행 200/500/500에서 인하
```

- 효과: 80장/분 괴물의 MC 배치 = 120초×80/60 = **160→캡 50**. 죽었을 때 블라스트 반경 50개(현행 최대 200), 처리량 손실 없음(claim 왕복 ≪ 배치 시간).
- 단 **phase 전환 직후 첫 배치**는 모델 로드를 갚아야 하므로 예외: 전환 첫 claim에 한해 target×3 허용(스위칭 직후 미니 배치 방지).

### 6.3 file-major(combo) 모드 — 조건부 보류 (요구 4)

VRAM이 3모델 동시 상주 가능한 노드에 "한 파일 전 phase" 모드를 배정하는 안. **이번 구현 범위에서 제외**하고, 전제 데이터를 먼저 확보한다: phase_elapsed_s 축적분으로 모델 스위칭 비용이 전체의 5%를 넘는지 측정 → 넘을 때만 설계 착수. (6.2가 시행되면 스위칭 빈도 자체가 줄어 필요성이 더 낮아질 것으로 예상.)

### 6.4 복수 고성능 워커 (N대 동시 연결)

설계 전체가 워커별 특수 케이스 없이 **합산 기반**이므로 N대로 자연 확장된다:

- **배정**: time-to-drain의 분모가 Σ(실측 속도)라, 괴물 N대는 MC 잔량이 클 때 전원 MC에 붙고(MC=연산의 86%), 꼬리에서 안정성 계수에 의해 시차를 두고 VV/MV로 이탈한다. N대 전용 로직 없음.
- **claim**: BEGIN IMMEDIATE 직렬화 + 작은 배치(§6.2)로 N대 동시 claim 경합은 밀리초 비용. 이중 배정 구조적 불가.
- **수집**: 결과 배칭(§10)이 합산 처리량을 흡수 (N×600건/분 → 배치 커밋 N×30회/분).

**유일한 N-제약은 공급 천장**: 파싱은 서버 한 대의 CPU에 갇혀 있어
(`clamp(…, cpu_count−2)`), Σ수요가 서버 파싱 한계를 넘으면 충전 중 버퍼가
실시간 고갈된다. 대응 순서:

1. **사전 적재 깊이가 1차 답** — §9.2의 ready 조건은 의도적으로 "충전 중
   공급 기여 0" 가정의 최악 케이스 공식이다. 백로그가 임대 시간 전체를
   덮으면 공급 천장은 무관해진다.
2. **다운로드 dedup(§4.3)이 수요 자체를 줄인다** — 재인덱싱·파도에서 특히.
3. **(미래 탈출구) shared_fs 파싱 분산**: NAS가 워커에도 마운트된 환경
   (storage_mode=shared_fs)에서는 파싱을 워커로 내릴 수 있다 — 원본 이동
   없이 분산 가능한 유일한 케이스. 본 설계 범위 밖, 천장이 실측으로
   확인될 때만 착수.

---

## 7. 공급 자동 스케일 (요구 1)

> **실측 (2026-06-11, M5 / 실 PSD 12개·126MB·3~44레이어)**: 파싱 처리량
> 1스레드 103장/분, 3스레드 149장/분(1.44×), 6스레드 149장/분(추가 이득 0 — GIL).
> 시사점: ① 일반적 PSD 구성에서 파싱은 괴물 MC(80장/분)도 단일 스레드로 먹일 수
> 있다 — "파싱=클러스터 천장" 진단은 **대형 파일 꼬리(수백 MB PSB)와 다중 괴물
> 동시 투입에 한정**되는 문제다. ② parse_workers 기본 3은 타당, 그 이상은 낭비.
> ③ 병렬화의 실가치는 처리량 +44%보다 **head-of-line blocking 제거**(거대 파일
> 하나가 뒤의 작은 파일들을 막지 않음)다. ④ 자동 스케일(아래)의 상한은 GIL상
> 3~4가 실효 한계 — cpu_count 기반 상한은 과대하므로 구현 시 4로 캡.


```python
# 서버 백그라운드, 30초 주기
demand_fpm  = SUM(mc_speed of online workers assigned/capable to mc)   # 클러스터 MC 소비력
supply_fpm  = parse pool 실측 처리량 (EMA, parse_elapsed_s 기반)
target = clamp(ceil(demand_fpm × 1.5 / per_thread_fpm), 1, cpu_count - 2)
parse_pool.resize(target)        # ThreadPoolExecutor max_workers 조정
```

- ×1.5 헤드룸: 공급이 소비를 앞서야 버퍼가 차고, 버퍼가 차야 스프린트가 가능하다.
- 다운로드 워커도 동일 패턴이되 상한을 보수적으로(NAS 대역폭이 본질 한계): `download_workers = clamp(demand 기반, 1, 6)`.
- resize는 executor 재생성이 아니라 세마포어 기반 동시 실행 상한으로 구현(in-flight 작업 무중단).

---

## 8. 대형 버퍼 (요구 7~10)

### 8.1 바이트 회계

```python
# 현행: Semaphore(max_files=350) — 개수 기준
# 개정: 바이트 예산
buffer_budget = min(
    config("download.buffer_bytes", 30 * GB),
    disk_free(temp_dir) - HEADROOM(20 * GB),
)
# 다운로드 시작 전 예약: reserve(expected_size)  (NAS PROPFIND가 크기 제공)
# 실패/완료 시 해제. 8KB 아이콘과 2GB PSB가 공정하게 회계된다.
```

조건변수 기반(`threading.Condition`)으로 예약 대기. `disk_free`는 60초 캐시(매 파일 statvfs 금지).

### 8.2 재시작 생존

§3.3 staging_manifest. 서버 기동 시퀀스에 추가: 매니페스트 검증 → 생존분 `download_status='done'` 유지 → 사망분만 pending 복귀. `_cleanup_stale_temp_files`는 매니페스트에 없는 파일만 제거하도록 개정.

### 8.3 원장 규모 점검 (요구 10)

수십만 file_tasks 대비: WorkersPanel 진행률 폴링 쿼리(잡별 SUM 집계)를 50만 행 합성 데이터로 측정. 50ms 초과 시 analysis_jobs에 카운터 비정규화(`done_mc, done_vv, ...` — complete_task_phase에서 증분 갱신). **측정 먼저, 비정규화는 조건부.**

---

## 9. 스프린트 어시스트 (요구 11~15)

### 9.1 상태 기계

```text
idle ──(운영자: 의도 선언)──▶ staging ──(백로그 ≥ 목표)──▶ ready
                                                            │ (운영자: GPU 투입)
        done ◀──(운영자: 분리)── draining ◀──(소진 임박)── charging
```

```sql
CREATE TABLE IF NOT EXISTS sprint_plan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL DEFAULT 'staging'
        CHECK (state IN ('staging','ready','charging','draining','done','cancelled')),
    target_phase TEXT NOT NULL DEFAULT 'mc',
    rental_minutes INTEGER NOT NULL,         -- 운영자 입력
    expected_fpm REAL NOT NULL,              -- 투입 예정 GPU **함대 합산** 속도
                                             -- (N대면 Σ — 운영자 입력 or 과거 벤치마크 합)
    created_at TEXT, updated_at TEXT
);
```

### 9.2 신호 계산

```python
backlog       = COUNT(file_tasks: parse done, mc pending)      # MC-ready
backlog_min   = backlog / expected_fpm                          # "괴물 N분치"
staging_eta   = (목표 backlog - 현재) / parse 실측 fpm
drain_eta     = backlog / (현재 charging 워커들의 mc_speed 합)   # charging 중
ready 조건    = backlog_min ≥ rental_minutes × 1.1              # 10% 여유
# 주: 의도적 최악 케이스 — 충전 중 파싱 공급 기여를 0으로 가정한다.
# 복수 괴물(Σ수요 > 서버 파싱 천장)에서도 이 조건이면 유휴 0이 보장된다.
draining 전이 = drain_eta ≤ 15분 → 알림 (StatusBar/토스트 + 로그)
```

### 9.3 API와 UI

```text
POST   /api/v1/admin/sprint          {rental_minutes, expected_fpm, target_phase}
GET    /api/v1/admin/sprint/status   → 상태 + 신호 일체 (5초 폴링)
DELETE /api/v1/admin/sprint          (취소)
```

UI는 WorkersPanel 상단 카드 1장: 상태 배지 + 백로그 게이지("2.6h / 4.0h분") + 적재 ETA + (charging 중) 소진 ETA. **이 카드가 곧 선행 마감 과제였던 "스케줄러 계기판"의 1차 구현이다** — 압력·실측 속도 표시를 같은 카드에 싣는다.

### 9.4 자동/수동 경계 (요구 15 — 불변 규칙)

- 시스템이 **하는** 것: 적재 운전, 환산·ETA 계산, 상태 전이 알림, charging 중 배정 최적화.
- 시스템이 **하지 않는** 것: GPU 투입/분리, phase 토글 자동 발동, 핀 자동 설정, 비용이 걸린 모든 결정.
- staging 중 VV/MV 일시정지 여부도 운영자 선택(스프린트 카드에 토글 노출 — 기존 paused_phases 재사용).

---

## 10. 결과 인제스트 배칭 (요구 6)

```text
POST /api/v1/tasks/complete-batch
{ "results": [ {task_id, phase, success, elapsed_s, error?}, ... ] }   # ≤ 50건
```

- 워커 IO 스레드: result_queue를 **2초 윈도 또는 20건** 단위로 모아 전송. 벡터 저장(`/files/{id}/vv|mv`)도 멀티 레코드 변형 추가.
- 서버: 한 트랜잭션으로 N건 처리(file_tasks 갱신 + derivations 적립 + 잡 완료 체크 1회).
- 단건 엔드포인트는 호환용 유지. 효과: 괴물 VV 500/분 = 분당 커밋 500→25회.

---

## 11. 동시성·일관성 규칙

1. **모든 DML 후 무조건 commit/rollback** (CLAUDE.md SQLite 규칙) — derivations 적립은 결과 저장과 같은 트랜잭션.
2. 캐시 적립은 `INSERT OR REPLACE` — 동시 적립 경합은 마지막 승자로 무해(같은 입력=같은 결과).
3. materialize는 파싱 워커 스레드(thread-local 커넥션)에서 수행 — 기존 병렬 파싱과 동일 패턴.
4. content_hash 계산은 다운로드/파싱 단계에서만 (워커는 hash를 모름 — 서버가 결과를 적립할 때 files.content_hash를 참조).
5. 백필·파도 잡은 일반 잡과 같은 원장을 쓰므로 추가 동시성 개념 없음.

---

## 12. 마이그레이션 단계 (단계별 커밋, 각각 독립 검증)

| 단계 | 내용 | 위험 | 검증 |
|---|---|---|---|
| **M0** | ✅ **완료 (2026-06-11, a7e21d6)** — 잡 공정성(3개 큐 전부 잡 인터리브), 썸네일 잔여물, '분석 대기' 배지(점진 가용성 가시화) | 낮음 | 테스트 4건 추가, 전체 447 passed |
| **M1** | ✅ **완료 (2026-06-11, b8810a6)** — DDL(양 경로), get_model_version 단일 소스(도메인 내용 해시 자동 포함), 3개 저장 엔드포인트 shadow write | 낮음 | 테스트 7건, 전체 454 passed |
| **M2** | content_hash 백필 잡 (14,158건 잔여) — 일반 잡으로 생성, 파싱 풀이 hash만 계산 | 낮음 | 카운트 100% |
| **M3** | 파싱 완료 시 캐시 조회 + materialize 활성화 (**중복 제거 발효**) | 중간 | 중복 파일 잡: 계산 1회·구체화 N회 확인 |
| **M4** | 모델 레지스트리 활성 버전 + 파도 잡 생성기 | 중간 | 소형 파도 E2E |
| **M5** | rebuild 배지를 registry 조회로 단순화, 구버전 파생물 GC 정책(보존 1세대) | 낮음 | 기존 배지와 결과 동치 확인 |
| **병행 A** | 스케줄러: time-to-drain + 배치 공식 (CAS 무관, 독립 배포 가능) | 중간 | 시뮬레이션 단위 테스트(워커 프로필 조합별 assign 분포) |
| **병행 B** | 공급 자동 스케일 + 바이트 회계 + staging_manifest | 중간 | 재시작 생존 E2E, 대량 합성 파일 |
| **병행 C** | 스프린트 어시스트 (sprint_plan + API + 카드) + 결과 배칭 | 낮음 | 신호 계산 단위 테스트 + 모의 시나리오 |

순서 원칙: M1~M2는 무위험 선행 가능. M3가 본 발효 지점. 병행 트랙은 CAS와 의존성이 없어 별도 브랜치로 진행한다.

---

## 13. 측정 계획 (전후 비교)

| 지표 | 측정법 | 기대 |
|---|---|---|
| 중복 절감률 | M2 후 `SELECT COUNT(*)-COUNT(DISTINCT content_hash) FROM files` | 에셋 폴더 특성상 10~30% 예상 |
| 캐시 히트율 | file_tasks.{phase}_cache_hit 집계 | 재인덱싱 시 >90% |
| 괴물 duty cycle | charging 중 (배치 처리 시간 합)/(세션 시간) | ≥95% (사전 적재 시) |
| 파싱 공급 충족 | supply_fpm / demand_fpm | ≥1.0 유지 |
| 블라스트 반경 | 워커 사망 시 잠긴 task 수 | ≤50 (현행 최대 200) |
| 파도 소요 | 17k 파일 VV 재임베딩 벽시계 | 측정 후 기준선 수립 |

---

## 14. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| model_version 규율 미준수(프롬프트 바꾸고 버전 안 올림) → 오염된 캐시 | get_model_version() 단일 소스 + 프롬프트/도메인 YAML의 해시를 버전 문자열에 자동 포함 |
| hash 충돌/부분 hash | 기존 compute_content_hash 알고리즘 검토(전체 내용 기반인지) — 부분 샘플링이면 전체 hash로 교체 후 백필 |
| 구체화 누락(캐시는 있는데 파일 행 비어 있음) | 정합성 감사(audit)에 "hash 파생물 존재 but 파일 미구체화" 항목 추가 — 기존 audit 인프라 재사용 |
| 대형 버퍼 디스크 고갈 | HEADROOM 강제 + 예약 실패 시 다운로드 일시정지(기존 network_paused 패턴 재사용) |
| 파도 중 검색 품질 혼재(v1/v2 벡터 공존) | 같은 vec 테이블 내 공존은 어차피 파일 단위 교체라 일관 — 벤치마크를 파도 전/중/후 3회 실행해 확인 |

---

## 15. 요구사항 추적표

| # | 요구 | 반영 위치 |
|---|---|---|
| CAS 본체 | 중복 제거·파도·이동 내성 | §2~5 |
| 1 | 공급 자동 스케일 | §7 |
| 2 | 작은 배치+잦은 claim | §6.2 |
| 3 | 용량 가중 압력 | §6.1 |
| 4 | file-major 모드 | §6.3 (조건부 보류) |
| 5 | 임대 GPU 파도 | §5+§9 |
| 6 | 결과 배칭 | §10 |
| 7 | 바이트 회계 동적 버퍼 | §8.1 |
| 8 | 재시작 생존 버퍼 | §8.2 |
| 9 | 다운로드 dedup | §4.3 |
| 10 | 원장 규모 점검 | §8.3 |
| 11~14 | 스프린트 4단계 | §9 |
| 15 | 자동/수동 경계 | §9.4 |
| 선행 마감 | 잡 공정성·썸네일 잔여물·계기판 | M0 + §9.3 |
