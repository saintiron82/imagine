---
description: Triaxis 검색 벤치마크 실행 및 추적
---
# /bench — 벤치마크 워크플로우

## 사용법

```
/bench                    # 벤치마크 실행 + 추적 업데이트
/bench compare A B        # 두 실행 비교
/bench history            # 점수 이력
/bench ablation           # 축별 ablation (VV, MV, FTS)
/bench baseline           # 기준선 저장
```

## 실행 절차

### `/bench` (기본)

1. **Tag 확인** — 사용자에게 실행 이름과 변경 내용을 질문
2. **벤치마크 실행**
```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine triaxis --tag "{tag}"
```
3. **기준선 비교** — baselines/ 에 기준선이 있으면 자동 비교
```bash
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --compare "{baseline},{tag}"
```
4. **추적 문서 업데이트** — `docs/benchmark_tracker.md`의 Score History에 새 행 추가
5. **결과 보고** — 점수와 delta를 사용자에게 요약
6. **기준선 갱신 여부** — 개선되었으면 저장 여부 확인

### `/bench ablation`

1. 4개 엔진 순차 실행:
```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine triaxis --tag "{tag}_triaxis"
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine vv_only --tag "{tag}_vv"
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine mv_only --tag "{tag}_mv"
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --engine fts_only --tag "{tag}_fts"
```
2. 비교 테이블 출력
3. `docs/benchmark_tracker.md`의 Ablation History에 기록

### `/bench compare A B`

```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --compare "{A},{B}"
```

### `/bench history`

```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --history triaxis
```

### `/bench baseline`

```bash
cd /Users/saintiron/Projects/Imagine-Bench
IMAGINE_ROOT=/Users/saintiron/Projects/Imagine python3 src/run.py --save-baseline "{tag}"
```

## 참조

- 스킬 정의: `.agent/skills/benchmark/SKILL.md`
- 추적 문서: `/Users/saintiron/Projects/Imagine-Bench/docs/benchmark_tracker.md`
- 설정: `/Users/saintiron/Projects/Imagine-Bench/config/benchmark.yaml`
