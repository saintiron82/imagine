---
description: Imagine Triaxis 검색 엔진 벤치마크 실행, 추적, 비교 도구
---
# Benchmark Skill

## Overview

Imagine-Bench를 사용하여 Triaxis 검색 엔진의 성능을 측정하고 추적합니다.
변경사항이 검색 품질에 미치는 영향을 정량적으로 평가합니다.

## Commands

| Command | Description |
|---------|-------------|
| `/bench` | 벤치마크 실행 + 추적 문서 업데이트 |
| `/bench compare A B` | 두 실행의 점수 비교 |
| `/bench history` | 엔진별 점수 이력 표시 |
| `/bench ablation` | 축별 ablation 실행 (triaxis, vv_only, mv_only, fts_only) |
| `/bench baseline` | 현재 최신 실행을 기준선으로 저장 |

## Paths

| File | Purpose |
|------|---------|
| `/Users/saintiron/Projects/Imagine-Bench/` | 벤치마크 프로젝트 루트 |
| `/Users/saintiron/Projects/Imagine-Bench/src/run.py` | CLI 엔트리포인트 |
| `/Users/saintiron/Projects/Imagine-Bench/docs/benchmark_tracker.md` | 점수 추적 마스터 문서 |
| `/Users/saintiron/Projects/Imagine-Bench/config/benchmark.yaml` | 엔진/메트릭/게이트 설정 |
| `/Users/saintiron/Projects/Imagine-Bench/runs/` | 실행 결과 JSON |
| `/Users/saintiron/Projects/Imagine-Bench/reports/` | 리포트 (MD + JSON) |
| `/Users/saintiron/Projects/Imagine-Bench/baselines/` | 저장된 기준선 |

## Workflow: `/bench`

### Step 1: Tag & Context
사용자에게 확인:
- **Tag**: 이 실행의 이름 (예: `rrf_weight_tuning_v2`, `siglip2_pro_upgrade`)
- **Change**: 무엇을 변경했는지 한 줄 요약

### Step 2: Run Benchmark
```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine triaxis --tag "{tag}"
```

### Step 3: Compare with Baseline
마지막 기준선이 있으면 자동 비교:
```bash
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --compare "{baseline_tag},{new_tag}"
```

### Step 4: Update Tracker
`docs/benchmark_tracker.md`의 Score History 테이블에 새 행 추가:
- 날짜, tag, 엔진, 각 메트릭 점수, Notes (변경 내용)
- 점수가 기준선보다 올랐으면 baseline 테이블도 갱신

### Step 5: Report
사용자에게 결과 요약:
```
## Benchmark Result: {tag}
| Metric    | Score  | Delta   |
|-----------|--------|---------|
| nDCG@10   | 0.7500 | +0.0226 |
| Recall@50 | 0.9100 | +0.0149 |
| MRR@10    | 0.9700 | +0.0033 |
```

### Step 6: Baseline Update
점수가 개선되었으면 새 기준선 저장 여부 확인:
```bash
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --save-baseline "{tag}"
```

## Workflow: `/bench ablation`

4개 엔진을 순차 실행하여 각 축의 기여도를 측정:

```bash
cd /Users/saintiron/Projects/Imagine-Bench

IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine triaxis --tag "{tag}_triaxis"
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine vv_only --tag "{tag}_vv_only"
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine mv_only --tag "{tag}_mv_only"
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine fts_only --tag "{tag}_fts_only"
```

결과를 비교 테이블로 출력하고 `docs/benchmark_tracker.md`의 Ablation History에 기록.

## Workflow: `/bench compare A B`

```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --compare "{tag_a},{tag_b}"
```

## Workflow: `/bench history`

```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --history triaxis
```

또는 `docs/benchmark_tracker.md`를 읽어서 표시.

## Workflow: `/bench baseline`

```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --save-baseline "{tag}"
```

## Metrics Quick Reference

| Metric | Good | Great | Meaning |
|--------|------|-------|---------|
| nDCG@10 | > 0.7 | > 0.85 | 상위 10개 순위 품질 |
| Recall@50 | > 0.8 | > 0.95 | 관련 이미지 포괄성 |
| MRR@10 | > 0.9 | > 0.95 | 첫 관련 결과 순위 |
| P@10 | > 0.6 | > 0.8 | 상위 10개 정확도 |
| p95 Latency | < 1000ms | < 500ms | 응답 속도 |

## Rules

1. **변경 전후 벤치마크 필수** — 검색 관련 코드 수정 시 반드시 before/after 비교
2. **추적 문서 항상 업데이트** — 벤치마크 실행할 때마다 tracker.md에 기록
3. **기준선 갱신 신중히** — 확실한 개선인 경우에만 기준선 교체
4. **Ablation 주기적 실행** — 대규모 변경 후 축별 기여도 확인
5. **회귀 차단 활용** — CI/CD에서 `--gate` 사용하여 성능 하락 방지
