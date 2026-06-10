# Decomposer Scope 정확도 개선 (2026-06-10)

## 문제 설정

벤치마크 메모(2026-04-05): "Scope filter matched 0 files 빈발 — Triaxis가
MV 단독보다 낮아지는 주요 원인. 개선하면 Triaxis 점수 대폭 향상 예상."

## 진단 (tools/diagnose_scope_extraction.py)

frozen_30_v1 30쿼리 전수 추적 결과, 통념과 다른 사실 3가지:

1. **"0건 매칭"은 이미 해소돼 있었다.** relax + query-hint 체인이 30쿼리 전부에서
   scope를 복구함 (zero_scope = 0/30).
2. **진짜 문제는 라벨이었다.** frozen_30_v1의 GT는 `scope_ground_truth=False`로
   생성돼 폴더 스코프를 무시한다. 예: "#02에서 집과 거리" GT 86개 중 경로에
   #02가 있는 건 1개. 이 라벨 기준에서는 scope를 정확히 잡을수록 점수가
   떨어진다 — "Triaxis < MV 단독" 현상의 실제 메커니즘.
3. **decomposer 결함 2종 발견:**
   - 환각: "도시 낮에서 거울과 거리" → `image_type='ui_element'` 날조
     → scope 9파일, GT 0개 포함
   - 요소 흡수: "#08에서 캐릭터과 밤" → 검색 요소 "캐릭터"를
     `image_type='character'` 필터로 흡수 → 폴더∩타입 = 0건인데
     relax가 format만 풀어서 **빈 결과 페이지** 반환

## 수정

1. `validate_scope_evidence()` (query_decomposer.py) — 쿼리 텍스트에 해당
   타입의 ko/en 단어가 있을 때만 image_type/format scope 채택. 폴더는 불변.
2. `_relax_unmatched_scope()` (sqlite_search.py) — 폴더+타입 조합이 0건이면
   image_type도 relax. 폴더는 유지되므로 strict-folder 정책 보존.
3. 측정 기반 정비 — `frozen_30_scoped_v1.json`: `scope_ground_truth=True`로
   생성한 scope 정합 쿼리셋 (제품 동작과 라벨 일치).

## 결과 (frozen_30_scoped_v1, keyword judge, seed 42)

| | P@3 | P@5 | P@10 | found | 빈 결과 |
|---|---:|---:|---:|---:|---:|
| baseline (수정 전) | 0.767 | 0.707 | 0.573 | 27/30 | 1 |
| fix 1 (환각 차단) | 0.767 | 0.707 | 0.573 | 27/30 | 1 |
| fix 1+2 (relax 확장) | **0.800** | **0.740** | **0.607** | **28/30** | **0** |

- "#08에서 캐릭터과 밤": 0건(빈 페이지) → 20건 검색, found
- 전 지표 +0.033. 잔여 missed 2건은 scope가 완벽(GT 전부 scope 안)한
  상태에서의 랭킹 문제 — scope 작업 범위 밖.

## frozen_30_v1 (구 라벨) 회귀 체크 — 회귀 없음

| | P@3 | P@5 | P@10 | found |
|---|---:|---:|---:|---:|
| 2026-06-03 기준치 | 0.400 | 0.367 | 0.287 | 24/30 |
| 수정 후 (2026-06-10) | **0.444** | 0.367 | **0.297** | **26/30** |

구 라벨(폴더 무시 GT)에서도 P@5 동일, P@3/found는 소폭 개선.
결과: `benchmarks/results/precision_20260610_frozen30v1_regression.json`

## SLM Judge 재측정 (Codex judge, top-5)

| | keyword P@5 | SLM P@5 (30쿼리 정규화) |
|---|---:|---:|
| baseline | 0.707 | 0.747 |
| fix 1+2 | 0.740 | **0.767** |

- "#08에서 캐릭터과 밤": 빈 페이지(0건) → SLM 판정 **5/5 만점 top5**.
  복구된 결과가 양적으로만이 아니라 질적으로도 정답이었음을 확인.
- keyword→SLM lift가 +0.03 수준으로 작음 — scope 정합 라벨
  (scope_ground_truth=True)이 구 라벨 대비 훨씬 정확하다는 방증
  (Sprint 3 구 라벨은 lift +0.3이었음).
- 주의: 동일 top5에 judge 판정이 갈리는 노이즈 케이스 존재
  (크랑베르무 캐릭터+밤, 두 런 검색 결과 동일한데 판정 상이).
  SLM delta는 ±0.03 노이즈 대역을 가짐. keyword delta(+0.033)는 결정적.

## 남은 것

- 측정 표준을 frozen_30_scoped_v1으로 이전할지 결정 필요 — 구 frozen_30_v1
  라벨은 폴더 무시 GT라서 scope 동작을 측정할 수 없음.
- 요소 흡수의 근본 해결은 decomposer 프롬프트 개선(요소 vs 타입 필터 구분
  지시)이지만, relax 안전망이 빈 결과를 막아주므로 우선순위 낮음.
