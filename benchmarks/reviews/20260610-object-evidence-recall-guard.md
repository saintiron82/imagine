# Object-Evidence Recall Guard (2026-06-10)

## 문제 설정

s19 진단: 다조건 쿼리에서 GT 5건이 `object_evidence_present_but_not_top10`.
evidence rerank는 RRF 후보 풀 안에서만 동작하므로, 어떤 축의 top-N에도 안
잡힌 full-evidence 파일은 구조적으로 구제 불가 — 라는 가설로 recall guard를
구현했다.

## 구현 (유지됨)

- `_object_evidence_guard_ids()`: file_objects에서 모든 조건 그룹을 만족하는
  파일 직접 조회 (evidence 텍스트 매칭과 동일한 substring 시맨틱)
- pre-trim 단계에서 풀에 없는 가드 파일을 주입 (`rrf_score=0`, evidence
  boost로만 상승)
- 안전장치: 부정 쿼리 시 스킵 / scope 내로 제한 / 상한
  `search.rerank.object_evidence_guard_max`(기본 50, 0=비활성)
- A/B 토글: `IMAGINE_BENCH_DISABLE_EVIDENCE_GUARD=1`
- 러너: `tools/run_multicondition_bench.py` (s19 결과 포맷 호환)

## 측정 결과 — 이 쿼리셋에서 delta 0

frozen_multicondition_pairs_s15_sample100 (24쿼리), current vs no_guard:

| variant | P@5 | P@10 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| current (guard ON) | 0.3333 | 0.2042 | 22 | 23 |
| no_guard | 0.3333 | 0.2042 | 22 | 23 |

**가드가 발동하지 않았다** — 가설이 틀렸기 때문: "벽, 커튼" 쿼리에서 조건
만족 파일 19개가 **이미 전부 후보 풀(118개) 안에 있었다.**

## 실제 원인 — 라벨 천장 (rank trace)

miss였던 31639의 top-50 추적:

| rank | id | evidence | obj | quality_score | GT? |
|---:|---|---:|---|---:|---|
| 1 | 31477 | 6.0 | 2/2 | 0.888 | 라벨엔 없음 |
| 2 | 31578 | 6.0 | 2/2 | 0.869 | 라벨엔 없음 |
| 3 | 31144 | 6.0 | 2/2 | 0.824 | GT |
| ... | | | | | |
| 19 | 31639 | 6.0 | 2/2 | 0.491 | GT (miss) |

**top-19 전원이 evidence 만점.** 31639 위의 "비-GT"들도 실제로 벽+커튼을
가진 동등한 정답 후보다 (S17 감사: 저득점 13쿼리 중 11쿼리의 top5에
라벨 누락 full-evidence 후보 존재 — 같은 현상). GT가 100파일 샘플 내
co-occurrence로만 생성됐기 때문에 시스템이 찾아낸 진짜 정답을 오답으로
센다. 여기서 31639를 더 올리는 튜닝은 라벨 과적합이다.

## 결론

1. **다조건 검색의 회수는 사실상 포화** — full-evidence 후보는 풀에 다
   들어오고 있고, micro Recall@10 0.92의 잔여 갭은 라벨 한계.
2. **가드는 안전망으로 유지** — 이 쿼리셋에선 발동 조건이 없었지만,
   풀이 작거나 축이 약한 환경(예: 인덱싱 초기, 희귀 객체)에서 풀 누락을
   막는 방어선. 비용은 풀에 가드 파일이 없을 때만 발생.
3. **다조건 품질의 다음 단계는 라벨** — keyword GT가 아니라 SLM judge
   또는 전수 라벨로 측정해야 0.33이라는 숫자의 실제 의미가 나온다.
   (S17 감사 기준으로는 체감 P@5가 측정치보다 상당히 높다.)

## 참고: s19 대비 절대치 차이

s19(06-05) current P@5 0.4083 vs 본 런 0.3333 — 코드 상태(scope 수정 포함)와
decomposer 비결정성이 섞인 run-to-run 차이. 유효한 비교는 동일 프로세스 내
current vs no_guard이며, 그 delta가 0이다.
