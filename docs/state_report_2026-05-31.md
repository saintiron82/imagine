# Imagine 시스템 상태 보고서 — 2026-05-31

> 본 보고서는 main 브랜치 HEAD `20bedba` (Sprint 3 머지 직후) 기준의
> 정밀 스냅샷이다. 코드 완성도 · 개념 구조 · 개선 이력 · 측정 수준 ·
> 저장 공간 · 남은 한계를 한 문서로 묶는다.

## TL;DR

- **검색 시스템과 분석 파이프라인의 메인 경로(Parse/MC/VV/MV/FTS)는
  코드·데이터 모두 완성.** 92.9% 파일에 실제 분석이 적용돼 있고
  Triaxis 검색의 체감 품질은 SLM-judge 기준 P@5=0.673에서 천장에
  도달했다.
- **Spatial 검색 경로(객체 위치/관계/깊이)는 코드만 완성, 데이터는
  사실상 비어 있음(0.07%).** 인프라가 있지만 측정도, 서비스도 불가.
- 다음 도약 지점은 검색팀이 아니라 **분석 파이프라인의 spatial phase
  대량 백필**이다.
- 1장당 저장 공간은 **약 405 KB** (썸네일 390 KB + DB 15 KB).
  원본 파일은 WebDAV/NAS에 둠.

---

## 1. 개념적 상태 (Architecture)

```
원본 파일 (WebDAV/NAS)
  -> Parse        : 구조/메타/썸네일 (로컬)
  -> Vision (MC)  : 비전 모델 해석 → MC caption + AI tags
  -> VV Encode    : SigLIP2 등 visual vector
  -> MV Encode    : Qwen3-Embed 등 meaning vector
  -> SQLite 저장  : files + vec_files/vec_text/vec_structure + FTS
  -> Triaxis 검색 : Query → Decompose → 4축 후보 → RRF → Rerank
```

검색 시스템은 단일 점수 함수가 아니라 **여러 독립 축을 만들고 RRF로
합치는 구조**다.

- **VV**: 픽셀 유사성 (구도/색감/스타일)
- **MV**: MC 캡션을 다시 임베딩한 의미 공간
- **FTS**: SQLite FTS5 기반 키워드/팩트 매칭
- **Spatial**: 객체 위치/관계/깊이 기반 (코드만 완성, 데이터 결핍)
- **Cross-encoder rerank** (α): BGE-reranker-v2-m3로 상위 30개 재정렬
- **AND verification** (β1): 다중 요소 시너님(KO|EN) AND 매칭 후 패널티

```
Query
  -> Decompose                (LLM: Codex/MLX/규칙 기반)
  -> Per-axis retrieval       (VV/MV/FTS [/spatial])
  -> RRF merge (다축)
  -> Cross-encoder rerank (α) (Sprint 1)
  -> Spatial intent boost     (Sprint 3 S3.2)
  -> AND verification (β1)    (Sprint 1)
  -> Negative filter
  -> Feedback demotion        (Sprint 2 γ4)
  -> Folder filter (Phase B)
  -> Top-K + confidence label (Phase A)
```

---

## 2. 코드 완성도

| 영역 | 상태 | 비고 |
|------|------|------|
| Parse (PSD/PNG/JPG) | ✅ 완성 | 92.9% 적용됨 |
| Vision/MC | ✅ 완성 | 멀티 백엔드(MLX/Ollama/vLLM) |
| VV/MV/Structure 벡터 인코더 | ✅ 완성 | sqlite-vec v0.1.6 |
| FTS5 인덱스 | ✅ 완성 | 100% 적용 |
| Triaxis 검색 (RRF n축) | ✅ 완성 | scoring.rrf_merge_multi |
| Query Decomposer (Codex/MLX/규칙) | ✅ 완성 | agentcli LLMClient 세션 재사용 |
| Confidence levels (Phase A) | ✅ 완성 | 임계 0.20/0.35/0.55 |
| Folder filter (Phase B) | ✅ 완성 | 경로 segment 매칭 |
| Cross-encoder rerank (α) | ✅ 완성 | env 토글 가능 |
| AND verification (β1, KO\|EN) | ✅ 완성 | env 토글 가능 |
| Spatial intent boost (S3.2) | ✅ 완성 | env 토글, no-op 가능 |
| Dominant axis badge (S3.4) | ✅ 완성 | 프론트 UX |
| `search_feedback` 수집 | ✅ 완성 | "관련 없음" 버튼 상시 노출 |
| Auto user_tags (γ4) | ✅ 완성 | 3회 이상 시 'low-relevance' |
| Admin feedback dashboard (γ3) | ✅ 완성 | 30일 집계 + top 파일/쿼리 |
| Bench 인프라 (frozen queryset + LLM rejudge) | ✅ 완성 | 결정적 A/B |
| Analysis Jobs (file_tasks 상태기계) | ✅ 완성 | DownloadAhead → Parse → AI phases |
| Pressure-based scheduler | ✅ 완성 | MC penalty + phase stability + MV bonus |
| 외부 워커 보안 access (AWS/Cloudflare relay) | ✅ 완성 | 11-phase 구현 끝, 배포는 사용자 몫 |
| Confidence calibration tool | ⚠️ 부분 | rank-proxy degenerate, raw score 저장 후 재실행 필요 |
| **Spatial 데이터 백필 워크플로우** | ❌ 부재 | phase는 있으나 대량 백필 잡 없음 |

---

## 3. 데이터 보유 현황 (17,726 파일 모집단)

```
MC caption       : 16,470 / 17,726  (92.9%)  ✅
AI tags          : 16,470 / 17,726  (92.9%)  ✅
FTS              : 17,726 / 17,726  (100.0%) ✅
VV/MV/Structure  :  ~93%                     ✅ (file_tasks done 기준)
search_logs      : 161 entries               (운영 흔적)
search_feedback  : 0 entries                 (γ4 입력 아직 없음)
─────────────────────────────────────────────
file_objects     : 12 / 17,726  ( 0.07%)     ❌ Spatial 결핍
file_depth_layers:  0 / 17,726  ( 0.00%)     ❌
file_spatial_relations: 0 / 17,726 (0.00%)   ❌
```

- `file_tasks` 총 28,850 entries — 같은 파일이 여러 job에서 재처리된
  흔적 포함.
- `analysis_jobs` 15개 중 1개만 "completed". 나머지는 진행/중단 상태로
  남아 있음.
- **MC가 채워진 16,470개는 검색 측정 모집단**으로 충분히 통계적이다.
- **Spatial 데이터는 사실상 미존재** — 2026-05-16에 12개 파일에만 한
  번 백필된 흔적이 남아 있다.

---

## 4. 측정 수준

### 측정된 항목 (frozen_30_v1 + SLM-judge Qwen3.5-9B-MLX-4bit)

| 검색 모드 | P@5 keyword | P@5 SLM-judge | 비고 |
|----------|-----------:|--------------:|------|
| 폴더 + 2 요소 (visual 타입) | 0.380~0.393 | **0.673** | 천장 도달 |
| Keyword-only/Semantic-only | 별도 | 별도 | 측정 완료 |
| Cross-encoder rerank (α) on/off | — | +0.12p | 효과 확인 |
| AND verification (β1) on/off | — | +0.12p | 효과 확인, α와 stack 안 함 |
| **Spatial 경로** | — | — | **미측정** (모집단 12개) |

### Sprint 3 ablation 요약 (전부 frozen_30_v1)

| 설정 | P@5 keyword | P@5 SLM-judge |
|------|------------:|--------------:|
| Sprint 2 baseline | 0.380 | **0.673** |
| Sprint 3 full | 0.393 | 0.593 (−0.080) |
| Sprint 3 — spatial OFF | 0.393 | 0.593 (불변) |
| Sprint 3 — expansion OFF | 0.380 | **0.673** (회복) |

→ S3.3 query expansion이 단독 회귀 원인. 롤백.

### 측정 인프라 (확보됨)

- `tools/bench_precision.py` — `--queryset` / `--save-queryset` 지원
- `tools/bench_llm_rejudge.py` — `--backend auto|agentcli|mlx`
- 결정적 A/B (변동성 ±0p), env 토글로 feature 분리 측정
- 환경변수: `IMAGINE_BENCH_DISABLE_RERANK`, `_DISABLE_AND`, `_DISABLE_SPATIAL`

### 측정 공백

- **Spatial 검색 경로**: 데이터 0.07%로 의미있는 P@K 측정 불가.
- **장기 운영 효과 (γ track)**: `search_feedback`이 비어 있어
  자동 user_tags가 ranking에 실질적으로 들어가지 않음. 시간 누적
  필요.
- **30 쿼리 frozen set**: 100~200 쿼리로 확장하면 잔여 변동성을 줄여
  미세 최적화 신호를 잡을 수 있을 가능성. 현재는 천장 근처에서 ±0.02p
  변동성 잔존.

---

## 5. 개선 이력 (Sprint 1 → 3)

| Sprint | 항목 | 효과 (P@5 SLM) | 상태 |
|--------|------|--------------:|------|
| 1 α | Cross-encoder rerank (BGE-reranker-v2-m3) | +0.12p | shipped |
| 1 β1 | Multi-element AND verification (KO\|EN) | +0.12p | shipped |
| 2 β2 | Confidence threshold calibration | (rank-proxy degenerate) | tool 보관, 재실행 대기 |
| 2 γ2 | Admin feedback dashboard | — | shipped (UX) |
| 2 γ3 | Frontend feedback button 상시 노출 | — | shipped (UX) |
| 2 γ4 | Auto user_tags from feedback | (데이터 0) | shipped, 효과 보류 |
| 3 S3.1 | Bench per-result raw axis scores | (calibration enabler) | shipped |
| 3 S3.2 | Spatial intent boost | (no-op, 측정 불가) | shipped, 토글 대기 |
| 3 S3.3 | Query expansion for MV | −0.080p | **reverted** |
| 3 S3.4 | Dominant axis badge | — | shipped (UX) |

Sprint 1 + 2의 핵심 효과로 keyword P@5 0.333 → SLM-judge P@5 0.673 도달.
Sprint 3는 천장 확인 + 회귀 검출 + UX 마무리 + 미래 측정 인프라
(raw scores 저장).

---

## 6. 저장 공간

### 절대량 (현재, 17,726 파일 / 17,037 썸네일)

| 구성 | 크기 |
|------|----:|
| 썸네일 (`thumbnails/`) | 6.3 GB |
| DB (`imageparser.db`) | 267 MB |
| DB 백업들 (`*.bak_*`) | ~785 MB (정리 후보) |
| **합계 (현역만)** | **약 6.6 GB** |

### 1장당 평균

| 구성 | 평균 |
|------|---:|
| 썸네일 | 약 390 KB |
| DB 점유분 (MC + 태그 + 벡터 4종 + FTS index) | 약 15 KB |
| **합계** | **약 405 KB / 장** |

> 원본 파일은 로컬에 없음. WebDAV/NAS에 그대로 두고 시스템은 썸네일 +
> 메타데이터만 보유한다.

### 스케일별 예상

| 규모 | 썸네일 | DB | 합계 |
|------|----:|---:|---:|
| 17,726 (현재) | 6.3 GB | 267 MB | 6.6 GB |
| 50,000 | 17.8 GB | 753 MB | 18.6 GB |
| 100,000 | 35.6 GB | 1.5 GB | 37.1 GB |
| 500,000 | 178 GB | 7.5 GB | 186 GB |
| 1,000,000 | 356 GB | 15 GB | 371 GB |

### 시사점

- **DB 비중 4% 정도** — 벡터 4종이 다 들어있는데도 작음.
  sqlite-vec chunk 압축 효과.
- **96%가 썸네일** — 저장 비용의 본체는 결국 이미지. WebP/AVIF 같은
  포맷 변경이나 사이즈 축소가 가장 직접적인 회수 수단.
- Spatial 데이터를 17,726 풀로 채워도 DB 추가 증가는 **수십 MB 수준**
  (객체 평균 2개 × ~300 bytes/row).
- 백업 4종(~785 MB) 정리하면 즉시 회수 가능.

---

## 7. 남은 한계와 다음 도약 후보

### 검색팀 입장에서 끝난 항목

- Triaxis + α + β1 + UX (badge, confidence label, feedback button)
- Frozen queryset 결정적 A/B 측정 인프라
- Sprint 1–3 ablation 끝, 천장 0.673 식별
- 보안 외부 워커 access 11-phase 구현

### 검색 외부에 있는 다음 도약

1. **Spatial 데이터 백필**
   - 분석 파이프라인의 spatial phase를 17,000+ 파일에 실행.
   - 비용: VLM 호출량 + 수 시간~수일.
   - 효과: spatial 검색 경로가 사실상 처음으로 가동되며, S3.2 boost와
     RRF spatial weight 0.50이 의미를 가짐.
2. **MC 캡션 품질 자체 개선**
   - 분석 영역. 사용자가 "분석은 다른 방법으로 해결한다고 본다"고
     명시한 영역과 일치.
3. **사용자 피드백 누적**
   - 운영 시간이 쌓이면서 `search_feedback`이 채워지면 γ4 자동
     `low-relevance` user_tag가 실제 ranking에 들어감.
4. **Frozen queryset 100~200개 확장**
   - 잔여 변동성을 줄여 미세 최적화 시도 가능.
5. **Confidence calibration 재실행**
   - S3.1로 raw axis scores가 저장되기 시작했으므로, 충분한 데이터가
     쌓이면 `tools/calibrate_confidence.py` 재실행 가능.

### 명시적으로 reject된 방향

- 모델 학습/파인튜닝 — 본 프로젝트의 길이 아님.

---

## 부록: 핵심 파일 인덱스

- 검색 진입: `backend/search/sqlite_search.py:triaxis_search`
- 점수 결합: `backend/search/scoring.py:rrf_merge_multi`, `quality_rerank`
- 질의 분해: `backend/search/query_decomposer.py`
- Cross-encoder: `backend/search/cross_encoder.py`
- 비전 인터페이스: `backend/vision/base.py`, `backend/vision/vision_factory.py`
- 모델 lifecycle: `backend/pipeline/model_manager.py`
- 파이프라인 phase 실행: `backend/pipeline/phase_runner.py`
- 스케줄러: `backend/server/queue/scheduler.py`
- Bench: `tools/bench_precision.py`, `tools/bench_llm_rejudge.py`
- Sprint 3 plan + 결과: `docs/superpowers/plans/2026-05-28-perceived-search-quality.md`
