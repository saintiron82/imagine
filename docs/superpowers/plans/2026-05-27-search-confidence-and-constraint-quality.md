# Search Confidence and Constraint Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "있는데 못 찾음 vs 없는데 답함"을 시스템과 벤치마크가 모두 구분하게 만들고, 그 위에서 Decomposer가 사용자 의도(folder scope · AND elements · negative)를 hard constraint로 추출하게 한다.

**Architecture:**
검색 응답을 `{results, confidence, empty_mode}` 구조로 확장해서 절대 점수 기반 임계로 "낮은 확신/없음"을 구분한다. 동일한 절대 점수를 사용해 Decomposer 출력은 JSON Schema로 강제하고 folder/AND를 SQL hard filter로 강등한다. 벤치마크는 found/missed/falsely-answered를 분리 집계한다. UI는 confidence별 톤으로 표시하고 사용자가 "관련 없음" 한 번에 라벨을 보정할 수 있게 한다.

**Tech Stack:** Python (FastAPI), SQLite, React, jsonschema, 기존 SqliteVectorSearch / QueryDecomposer / scoring.py 재사용.

---

## 핵심 정의

본 문서가 도입하는 용어는 코드와 벤치마크에서 동일하게 쓴다.

- **confidence**: 검색 응답의 절대 확신 레벨. `high` | `medium` | `low` | `empty`.
- **empty mode**: top-1 절대 점수가 임계 τ_low 미만이라 시스템이 정직하게 "관련 결과 없음"이라고 답하는 상태. `confidence=empty` 와 동의.
- **constraint plan**: Decomposer가 내놓는 구조화된 query plan. `folder`, `elements[]`, `negatives[]`, `vector_query`, `query_type` 의 6필드 JSON.
- **hard filter**: 검색 후보를 만들기 전 단계에서 SQL `WHERE`로 적용되는 제약. RRF 점수 가중치가 아님.
- **found**: P@K=1.0 — top-K 안에 최소 1건 관련.
- **missed**: GT > 0 이고 top-K∩GT = 0 — 있는데 못 찾음.
- **honest_empty**: 시스템이 `confidence=empty` 로 반환했고 GT=0 — 정직하게 비었다고 답함.
- **false_answer**: 시스템이 `confidence` ≥ low 로 답했는데 GT=0 — 없는데 답함.

---

## 현재 스펙 (As-Is)

### 검색 응답
- `backend/server/routers/search.py:150-159` — 응답 = `{success, results, count, elapsed_ms}` + `diagnostic?`. 절대 확신 채널 없음.
- `backend/api_search.py:158-200` — 각 결과는 `vector_score`, `text_vec_score`, `text_score`, `combined_score(rrf_score)` 를 들고 있음. 절대 점수가 있지만 **응답 레벨의 "낮은 확신"·"없음" 모드는 없음.**
- 결과가 0건이거나 무관해도 항상 top-K 반환. 빈 결과 모드 없음.

### Decomposer
- `backend/search/query_decomposer.py:85-163` — `decompose(query) -> Dict` 가 자유 텍스트 LLM 답을 파싱.
- 출력 키: `vector_query`, `negative_query`, `fts_keywords`, `exclude_keywords`, `filters{}`, `query_type`, `decomposed`, `_decomp_backend`.
- `filters{}` 안에 folder 스코프가 들어갈 자리는 있지만 **JSON Schema 강제 없음 · 검증 루프 없음 · "matched 0 files" 빈발**(메모리 `project_benchmark_findings_20260405.md` 참조).
- multi-element AND 검증 없음. 결과 단계에서 element 등장 여부를 확인하지 않음.

### 벤치마크
- `benchmarks/results/scoped_weak_v3_clean_tags3_20260502.md` — P@K 단일 지표.
- `missed`(있는데 못 찾음)와 `false_answer`(없는데 답함)가 동일하게 P@K=0 으로 묶임.
- 시스템이 항상 top-K 반환하므로 `honest_empty` 라는 상태가 측정 가능한 형태로 존재하지 않음.

### 사용자 정정 경로
- 잘못된 분석을 사용자가 1-클릭으로 보정하는 UI 없음.
- `users` 의 `user_tags` / `user_note` 컬럼은 있으나 "이 검색 결과는 관련 없음" 같은 부정 라벨링 경로 없음.

---

## 목표 스펙 (To-Be)

### 검색 응답
- 응답 shape: `{success, results, count, elapsed_ms, confidence, top1_raw_score, empty_reason?}`.
- `confidence` 결정 규칙:
  - `empty`: top-1의 `vector_score` 와 `text_vec_score` 모두 τ_low 미만, FTS 매칭도 0건.
  - `low`: top-1 raw 점수가 τ_low 이상 τ_mid 미만.
  - `medium`: τ_mid 이상 τ_high 미만.
  - `high`: τ_high 이상.
- τ 값은 `config.search.confidence_thresholds` 에서 설정. 기본값: `{low: 0.20, mid: 0.35, high: 0.55}` (cosine 유사도 기준, 캘리브레이션 가능).
- `confidence=empty` 일 때 `results=[]`, `empty_reason` 에 사람이 읽을 수 있는 한 줄.

### Decomposer
- `decompose(query) -> ConstraintPlan` 가 JSON Schema 로 검증된 결과를 반환.
- ConstraintPlan 필드: `folder`, `elements[]`, `negatives[]`, `vector_query`, `query_type`, `confidence` (Decomposer 자체의 확신도).
- LLM 출력이 schema fail 시: 1회 재시도 → 실패하면 rule-based fallback.
- `folder` 가 비어있지 않으면 SQL `WHERE folder_path LIKE ?` 로 후보를 먼저 자름. 매칭이 0이면 fallback으로 전체 풀고 `empty_reason="folder_scope_missing"` 표시.

### 벤치마크
- `bench_precision.py` 가 per-query에 카테고리를 부여: `found` | `missed` | `honest_empty` | `false_answer`.
- 집계 지표 4종: `precision_when_answered`, `recall_when_present`, `missed_rate`, `false_answer_rate`.
- 기존 P@K 도 계속 출력하되 4종 분리를 1차 지표로 표기.

### 사용자 정정 경로
- 검색 결과 카드에 "관련 없음" 버튼 (`POST /api/v1/search/feedback`).
- `search_feedback` 테이블에 `{query, file_id, label='irrelevant', user_id, ts}` 누적.
- Triaxis 검색이 동일 쿼리에서 자주 irrelevant 라벨된 file_id를 결과에서 제외(soft demotion, 점수 패널티).

---

## 갭 (As-Is → To-Be)

1. 응답에 confidence 채널 없음 → **추가.**
2. 임계 기반 empty 모드 없음 → **추가.**
3. Decomposer 출력이 schema-validated가 아님 → **JSON schema + validation loop 추가.**
4. folder가 hard filter가 아님 → **SQL pre-filter로 강등.**
5. multi-element AND 검증 없음 → **post-retrieval verify 단계 추가.**
6. 벤치마크가 4-way 카테고리 분리 안 함 → **bench_precision.py 갱신.**
7. 사용자 정정 경로 없음 → **/feedback endpoint + 테이블 + soft demotion.**

---

## File Structure

생성 또는 수정되는 파일 목록을 책임별로 정리한다.

- Create: `backend/search/confidence.py` — 절대 점수 → confidence 레벨 변환, 임계 로딩.
- Create: `backend/search/constraint_plan.py` — ConstraintPlan dataclass + JSON Schema + validate.
- Modify: `backend/search/query_decomposer.py` — schema 강제 + folder/elements/negatives 추출.
- Modify: `backend/search/sqlite_search.py` — ConstraintPlan 받아서 hard filter 적용, raw 점수 보존.
- Modify: `backend/server/routers/search.py:106-159` — 응답에 confidence 필드 추가, empty mode 분기.
- Modify: `backend/api_search.py:158-200` — `format_result` 에 raw_score 들 보존.
- Create: `backend/server/routers/search_feedback.py` — `/api/v1/search/feedback` endpoint.
- Modify: `backend/db/sqlite_schema_auth.sql` — `search_feedback` 테이블.
- Modify: `backend/db/sqlite_migrations.py` — `migrate_search_feedback`.
- Modify: `benchmarks/scripts/bench_precision.py` (또는 동등 경로) — 4-way 카테고리.
- Modify: `frontend/src/components/SearchPanel.jsx` — confidence 톤 표시 + 결과 카드 "관련 없음" 버튼.
- Modify: `frontend/src/api/search.js` — confidence 필드 + feedback API.
- Modify: `frontend/src/i18n/locales/{ko-KR,en-US}.json` — confidence/empty 라벨.
- Test: `tests/test_search_confidence.py` — confidence 변환 단위 테스트.
- Test: `tests/test_constraint_plan.py` — schema validation.
- Test: `tests/test_search_response_shape.py` — API 응답 shape 테스트.
- Test: `tests/test_decomposer_folder_extraction.py` — folder hard filter 동작.
- Test: `tests/test_search_feedback.py` — feedback endpoint 동작.

---

## Phase A — Confidence-Aware Search Response

**목표:** 검색 응답이 절대 점수 기반 confidence를 가지고, top-1 raw 점수가 낮으면 정직하게 "empty"를 반환한다.

### Task A1: Confidence 변환 모듈

**Files:**
- Create: `backend/search/confidence.py`
- Test: `tests/test_search_confidence.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_search_confidence.py`:

```python
"""Phase A: confidence level mapping from raw scores."""
from __future__ import annotations

import pytest

from backend.search.confidence import (
    ConfidenceLevel,
    ConfidenceThresholds,
    classify,
)


DEFAULT = ConfidenceThresholds(low=0.20, mid=0.35, high=0.55)


@pytest.mark.parametrize("score, expected", [
    (0.05, ConfidenceLevel.EMPTY),
    (0.19, ConfidenceLevel.EMPTY),
    (0.20, ConfidenceLevel.LOW),
    (0.34, ConfidenceLevel.LOW),
    (0.35, ConfidenceLevel.MEDIUM),
    (0.54, ConfidenceLevel.MEDIUM),
    (0.55, ConfidenceLevel.HIGH),
    (0.99, ConfidenceLevel.HIGH),
])
def test_classify_uses_inclusive_lower_bounds(score, expected):
    assert classify(score, DEFAULT) is expected


def test_classify_uses_max_of_vector_and_text_scores():
    """The relevant scalar is the BEST raw score across axes, not a sum."""
    from backend.search.confidence import classify_topk

    # vector strong, text weak — should be MEDIUM
    assert classify_topk(
        vector_score=0.40, text_vec_score=0.05, fts_hit=False,
        thresholds=DEFAULT,
    ) is ConfidenceLevel.MEDIUM


def test_classify_returns_low_when_only_fts_hits():
    """FTS keyword hit alone keeps us out of EMPTY even if vector low."""
    from backend.search.confidence import classify_topk

    assert classify_topk(
        vector_score=0.05, text_vec_score=0.05, fts_hit=True,
        thresholds=DEFAULT,
    ) is ConfidenceLevel.LOW


def test_thresholds_from_config_uses_defaults_when_missing():
    from backend.search.confidence import thresholds_from_config

    cfg = {}
    t = thresholds_from_config(cfg)
    assert t.low == 0.20 and t.mid == 0.35 and t.high == 0.55


def test_thresholds_from_config_reads_overrides():
    from backend.search.confidence import thresholds_from_config

    cfg = {"search.confidence_thresholds.low": 0.10,
           "search.confidence_thresholds.mid": 0.30,
           "search.confidence_thresholds.high": 0.50}
    t = thresholds_from_config(cfg)
    assert t.low == 0.10 and t.mid == 0.30 and t.high == 0.50
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_search_confidence.py -q`
Expected: `ImportError` (module doesn't exist)

- [ ] **Step 3: 모듈 구현**

`backend/search/confidence.py`:

```python
"""Phase A: absolute-score confidence classification."""
from __future__ import annotations

import enum
from dataclasses import dataclass


class ConfidenceLevel(str, enum.Enum):
    EMPTY = "empty"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ConfidenceThresholds:
    low: float = 0.20
    mid: float = 0.35
    high: float = 0.55


def classify(score: float, thresholds: ConfidenceThresholds) -> ConfidenceLevel:
    if score >= thresholds.high:
        return ConfidenceLevel.HIGH
    if score >= thresholds.mid:
        return ConfidenceLevel.MEDIUM
    if score >= thresholds.low:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.EMPTY


def classify_topk(
    *,
    vector_score: float,
    text_vec_score: float,
    fts_hit: bool,
    thresholds: ConfidenceThresholds,
) -> ConfidenceLevel:
    """Pick the most generous signal for the top-1 result."""
    best = max(vector_score or 0.0, text_vec_score or 0.0)
    level = classify(best, thresholds)
    if level is ConfidenceLevel.EMPTY and fts_hit:
        return ConfidenceLevel.LOW
    return level


def thresholds_from_config(cfg) -> ConfidenceThresholds:
    def _read(key: str, default: float) -> float:
        if isinstance(cfg, dict):
            return float(cfg.get(key, default))
        return float(cfg.get(key, default))

    return ConfidenceThresholds(
        low=_read("search.confidence_thresholds.low", 0.20),
        mid=_read("search.confidence_thresholds.mid", 0.35),
        high=_read("search.confidence_thresholds.high", 0.55),
    )
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_search_confidence.py -q`
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/search/confidence.py tests/test_search_confidence.py
git commit -m "feat: add absolute-score confidence classification (phase A)"
```

### Task A2: Search 응답에 confidence 결합

**Files:**
- Modify: `backend/server/routers/search.py:106-159`
- Modify: `backend/api_search.py:158-200`
- Test: `tests/test_search_response_shape.py`

- [ ] **Step 1: 응답 shape 실패 테스트 작성**

`tests/test_search_response_shape.py`:

```python
"""Phase A: search endpoint returns confidence + empty mode."""
from __future__ import annotations

import sqlite3
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeSearcher:
    """Stand-in that lets us steer top-1 raw scores."""
    def __init__(self, results):
        self._results = results

    def search(self, *args, **kwargs):
        return self._results


def _client(results, threshold_cfg=None):
    from backend.server.routers import search as search_router
    from backend.server import deps

    monkey = types.SimpleNamespace()

    fake = _FakeSearcher(results)

    def _get_searcher():
        return fake

    search_router._get_searcher = _get_searcher

    if threshold_cfg is not None:
        from backend.search import confidence
        confidence._test_override = threshold_cfg  # noqa - test hook

    app = FastAPI()
    app.include_router(search_router.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": 1, "username": "u"}
    return TestClient(app)


def test_search_returns_confidence_field():
    results = [{
        "id": 1, "file_path": "a.png",
        "vector_score": 0.7, "text_vec_score": 0.4,
        "rrf_score": 0.5,
    }]
    client = _client(results)
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "high"
    assert body["top1_raw_score"] == pytest.approx(0.7)


def test_search_returns_empty_mode_when_top1_below_low_threshold():
    results = [{
        "id": 1, "file_path": "a.png",
        "vector_score": 0.05, "text_vec_score": 0.05,
        "rrf_score": 0.01,
    }]
    client = _client(results)
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    body = resp.json()
    assert body["confidence"] == "empty"
    assert body["results"] == []
    assert "empty_reason" in body


def test_search_returns_low_when_only_fts_hits():
    results = [{
        "id": 1, "file_path": "a.png",
        "vector_score": 0.05, "text_vec_score": 0.05,
        "text_score": 4.5,
        "rrf_score": 0.01,
    }]
    client = _client(results)
    resp = client.post("/api/v1/search/triaxis", json={"query": "x", "limit": 5})
    body = resp.json()
    assert body["confidence"] == "low"
    assert len(body["results"]) == 1
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_search_response_shape.py -q`
Expected: assertion fails — `confidence` 키가 응답에 없음.

- [ ] **Step 3: format_result 가 raw 점수 보존하는지 확인 + search router 갱신**

`backend/server/routers/search.py` 의 `_do_search` 끝부분 `response = {...}` 직전에 confidence 계산을 추가:

```python
# (search.py 상단 import 영역에 추가)
from backend.search.confidence import (
    ConfidenceLevel,
    classify_topk,
    thresholds_from_config,
)
```

`_do_search` 함수 안 `response = {` 블록을 다음으로 교체:

```python
        # Phase A: confidence + empty-mode envelope.
        try:
            from backend.utils.config import get_config
            thresholds = thresholds_from_config(get_config())
        except Exception:
            from backend.search.confidence import ConfidenceThresholds
            thresholds = ConfidenceThresholds()

        top1 = formatted[0] if formatted else None
        if top1 is None:
            confidence = ConfidenceLevel.EMPTY
            top1_score = 0.0
        else:
            vec = float(top1.get("vector_score") or 0.0)
            txt_vec = float(top1.get("text_vec_score") or 0.0)
            fts_hit = bool(top1.get("text_score"))
            confidence = classify_topk(
                vector_score=vec,
                text_vec_score=txt_vec,
                fts_hit=fts_hit,
                thresholds=thresholds,
            )
            top1_score = max(vec, txt_vec)

        if confidence is ConfidenceLevel.EMPTY:
            response = {
                "success": True,
                "results": [],
                "count": 0,
                "elapsed_ms": elapsed_ms,
                "confidence": confidence.value,
                "top1_raw_score": top1_score,
                "empty_reason": "no_result_above_confidence_threshold",
            }
        else:
            response = {
                "success": True,
                "results": formatted,
                "count": len(formatted),
                "elapsed_ms": elapsed_ms,
                "confidence": confidence.value,
                "top1_raw_score": top1_score,
            }
        if diag is not None:
            response["diagnostic"] = diag

        return response
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_search_response_shape.py -q`
Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/server/routers/search.py tests/test_search_response_shape.py
git commit -m "feat: search response carries confidence and empty mode (phase A)"
```

### Task A3: Frontend confidence 표시

**Files:**
- Modify: `frontend/src/api/search.js`
- Modify: `frontend/src/components/SearchPanel.jsx`
- Modify: `frontend/src/i18n/locales/ko-KR.json`
- Modify: `frontend/src/i18n/locales/en-US.json`

- [ ] **Step 1: api/search.js 가 confidence 필드를 그대로 패스스루**

이미 응답 전체를 그대로 받는 구조라면 변경 불필요. 확인:

Run: `grep -n "results\|confidence" frontend/src/api/search.js | head -10`

대응 코드가 `results` 만 추출하면 confidence/top1_raw_score/empty_reason 도 같이 반환하도록 수정. 변경이 필요한 함수 한 곳만 패치.

- [ ] **Step 2: SearchPanel.jsx 에 confidence 라벨 표시**

검색 응답 표시 영역(`{count}건` 같은 표시) 옆에 confidence 배지 추가:

```jsx
{response?.confidence && (
  <span className={confidenceBadgeClass(response.confidence)}>
    {t(`search.confidence_${response.confidence}`)}
  </span>
)}
{response?.confidence === 'empty' && (
  <p className="text-sm text-gray-400 mt-2">
    {t('search.empty_explainer')}
  </p>
)}
```

`confidenceBadgeClass` 헬퍼:

```jsx
function confidenceBadgeClass(level) {
  return {
    high:   'bg-emerald-900/30 text-emerald-300 border border-emerald-700/40',
    medium: 'bg-blue-900/30 text-blue-300 border border-blue-700/40',
    low:    'bg-amber-900/30 text-amber-300 border border-amber-700/40',
    empty:  'bg-gray-900/40 text-gray-400 border border-gray-700/40',
  }[level] || 'text-gray-400';
}
```

- [ ] **Step 3: i18n 추가 (ko-KR.json / en-US.json)**

```
"search.confidence_high": "확실함" / "Confident"
"search.confidence_medium": "보통" / "Likely"
"search.confidence_low": "낮은 확신" / "Low confidence"
"search.confidence_empty": "관련 결과 없음" / "No relevant match"
"search.empty_explainer": "검색 조건에 충분히 부합하는 이미지가 없습니다. 다른 표현으로 시도해 보세요." /
                          "No image meets the search criteria with enough confidence. Try a different phrasing."
```

- [ ] **Step 4: build 검증**

Run: `cd frontend && npm run build`
Expected: `built in` 정상 출력.

- [ ] **Step 5: 커밋**

```bash
git add frontend/src/api/search.js frontend/src/components/SearchPanel.jsx \
        frontend/src/i18n/locales/ko-KR.json frontend/src/i18n/locales/en-US.json
git commit -m "feat: frontend shows search confidence and empty mode (phase A)"
```

---

## Phase B — Decomposer 구조화 + Hard Filter

**목표:** Decomposer 출력을 JSON Schema로 검증해 `folder/elements[]/negatives[]` 를 항상 가지게 하고, folder는 SQL hard filter로 강등한다.

### Task B1: ConstraintPlan dataclass + schema

**Files:**
- Create: `backend/search/constraint_plan.py`
- Test: `tests/test_constraint_plan.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_constraint_plan.py`:

```python
"""Phase B: ConstraintPlan validation."""
from __future__ import annotations

import pytest

from backend.search.constraint_plan import (
    ConstraintPlan,
    ConstraintPlanError,
    from_decomposer_output,
)


def test_valid_payload_roundtrips():
    raw = {
        "folder": "#07",
        "elements": ["캐릭터", "방"],
        "negatives": [],
        "vector_query": "character in a room",
        "query_type": "balanced",
    }
    plan = from_decomposer_output(raw)
    assert isinstance(plan, ConstraintPlan)
    assert plan.folder == "#07"
    assert plan.elements == ["캐릭터", "방"]
    assert plan.negatives == []


def test_missing_required_field_raises():
    raw = {"folder": "#07"}  # no vector_query
    with pytest.raises(ConstraintPlanError):
        from_decomposer_output(raw)


def test_unknown_query_type_rejected():
    raw = {
        "folder": "",
        "elements": [],
        "negatives": [],
        "vector_query": "x",
        "query_type": "nonsense",
    }
    with pytest.raises(ConstraintPlanError):
        from_decomposer_output(raw)


def test_elements_strip_empty_strings():
    raw = {
        "folder": "",
        "elements": ["a", "", " "],
        "negatives": ["", "x"],
        "vector_query": "x",
        "query_type": "visual",
    }
    plan = from_decomposer_output(raw)
    assert plan.elements == ["a"]
    assert plan.negatives == ["x"]


def test_to_dict_is_stable_for_logging():
    plan = ConstraintPlan(
        folder="", elements=["a"], negatives=[],
        vector_query="x", query_type="visual", confidence=0.7,
    )
    d = plan.to_dict()
    assert set(d.keys()) == {
        "folder", "elements", "negatives", "vector_query", "query_type", "confidence",
    }
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_constraint_plan.py -q`
Expected: `ImportError`

- [ ] **Step 3: 모듈 구현**

`backend/search/constraint_plan.py`:

```python
"""Phase B: Decomposer 출력의 구조화된 표현.

ConstraintPlan 은 검색 파이프라인 전체에서 단일 출처로 쓰인다.
LLM의 자유 텍스트 응답을 from_decomposer_output()이 이 객체로 정규화
하고, 정규화 실패는 ConstraintPlanError 로 명시.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping


ALLOWED_QUERY_TYPES = frozenset({"visual", "keyword", "semantic", "balanced"})


class ConstraintPlanError(ValueError):
    """LLM payload did not satisfy the ConstraintPlan schema."""


@dataclass(frozen=True)
class ConstraintPlan:
    folder: str
    elements: tuple[str, ...]
    negatives: tuple[str, ...]
    vector_query: str
    query_type: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "folder": self.folder,
            "elements": list(self.elements),
            "negatives": list(self.negatives),
            "vector_query": self.vector_query,
            "query_type": self.query_type,
            "confidence": self.confidence,
        }


def _clean_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ConstraintPlanError(f"expected list, got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return tuple(out)


def from_decomposer_output(payload: Mapping[str, Any]) -> ConstraintPlan:
    """Validate a decomposer payload and return a ConstraintPlan.

    Raises ConstraintPlanError on schema violation. Callers should catch
    and either retry (one shot) or fall back to a rule-based plan.
    """
    if not isinstance(payload, Mapping):
        raise ConstraintPlanError("payload must be a mapping")
    for required in ("vector_query", "query_type"):
        if required not in payload:
            raise ConstraintPlanError(f"missing required field: {required}")

    query_type = payload["query_type"]
    if not isinstance(query_type, str) or query_type not in ALLOWED_QUERY_TYPES:
        raise ConstraintPlanError(
            f"query_type must be one of {sorted(ALLOWED_QUERY_TYPES)}; got {query_type!r}"
        )

    vector_query = payload["vector_query"]
    if not isinstance(vector_query, str) or not vector_query.strip():
        raise ConstraintPlanError("vector_query must be a non-empty string")

    folder = payload.get("folder", "") or ""
    if not isinstance(folder, str):
        raise ConstraintPlanError("folder must be a string")

    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return ConstraintPlan(
        folder=folder.strip(),
        elements=_clean_list(payload.get("elements", [])),
        negatives=_clean_list(payload.get("negatives", [])),
        vector_query=vector_query.strip(),
        query_type=query_type,
        confidence=confidence,
    )
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_constraint_plan.py -q`
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/search/constraint_plan.py tests/test_constraint_plan.py
git commit -m "feat: introduce ConstraintPlan with schema validation (phase B)"
```

### Task B2: Decomposer에 schema 강제 + 재시도

**Files:**
- Modify: `backend/search/query_decomposer.py:109-163`
- Test: `tests/test_decomposer_folder_extraction.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_decomposer_folder_extraction.py`:

```python
"""Phase B: Decomposer normalises to ConstraintPlan."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _decomposer(monkeypatch, llm_raw: str):
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)
    monkeypatch.setattr(decomp, "_generate_llm", lambda q: llm_raw)
    return decomp


def test_folder_prefix_in_korean_query_is_extracted_into_constraint_plan(monkeypatch):
    llm_raw = json.dumps({
        "folder": "#07",
        "elements": ["캐릭터", "방"],
        "negatives": [],
        "vector_query": "character in a room",
        "query_type": "balanced",
    })
    decomp = _decomposer(monkeypatch, llm_raw)

    plan = decomp.decompose_plan("#07에서 캐릭터과 방 있는 이미지")
    assert plan.folder == "#07"
    assert set(plan.elements) == {"캐릭터", "방"}


def test_decomposer_retries_once_on_schema_failure(monkeypatch):
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)

    bad = "not-json"
    good = json.dumps({
        "folder": "",
        "elements": ["x"],
        "negatives": [],
        "vector_query": "x",
        "query_type": "visual",
    })
    calls = {"n": 0}

    def fake(_q):
        calls["n"] += 1
        return good if calls["n"] > 1 else bad

    monkeypatch.setattr(decomp, "_generate_llm", fake)

    plan = decomp.decompose_plan("x")
    assert calls["n"] == 2
    assert plan.vector_query == "x"


def test_decomposer_falls_back_when_retry_also_fails(monkeypatch):
    from backend.search import query_decomposer as qd

    decomp = qd.QueryDecomposer(use_codex=False)
    monkeypatch.setattr(decomp, "_generate_llm", lambda q: "still-bad")

    plan = decomp.decompose_plan("아무 쿼리")
    # fallback never raises — it returns a degraded but valid ConstraintPlan
    assert plan.vector_query  # non-empty
    assert plan.query_type in {"visual", "keyword", "semantic", "balanced"}
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_decomposer_folder_extraction.py -q`
Expected: `AttributeError: 'QueryDecomposer' object has no attribute 'decompose_plan'`

- [ ] **Step 3: Decomposer 에 `decompose_plan` 추가**

`backend/search/query_decomposer.py` 의 `QueryDecomposer` 클래스 끝에 메서드 추가 (기존 `decompose()` 는 그대로 유지 — 호환성):

```python
    # ── Phase B: structured ConstraintPlan output ─────────────────

    def decompose_plan(self, query: str):
        """Return a ConstraintPlan with one retry on schema violation."""
        from backend.search.constraint_plan import (
            ConstraintPlan,
            ConstraintPlanError,
            from_decomposer_output,
        )
        import json as _json

        def _try_parse(raw_text: str):
            if raw_text is None:
                raise ConstraintPlanError("LLM returned no text")
            try:
                payload = _json.loads(raw_text)
            except Exception as exc:
                raise ConstraintPlanError(f"LLM output is not JSON: {exc}") from exc
            return from_decomposer_output(payload)

        for attempt in (1, 2):
            try:
                raw = self._generate_llm(query)
                return _try_parse(raw)
            except ConstraintPlanError as exc:
                logger.warning(
                    "[DECOMP] schema attempt %d failed: %s", attempt, exc
                )

        # Both attempts failed — degrade gracefully via rule-based fallback.
        fb = self._fallback(query)
        return ConstraintPlan(
            folder="",
            elements=tuple(fb.get("fts_keywords") or [])[:5],
            negatives=tuple(fb.get("exclude_keywords") or [])[:5],
            vector_query=fb.get("vector_query") or query,
            query_type=fb.get("query_type") or "balanced",
            confidence=0.0,
        )
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_decomposer_folder_extraction.py -q`
Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/search/query_decomposer.py tests/test_decomposer_folder_extraction.py
git commit -m "feat: decomposer emits ConstraintPlan with retry (phase B)"
```

### Task B3: Hard folder filter

**Files:**
- Modify: `backend/search/sqlite_search.py`
- Test: `tests/test_decomposer_folder_extraction.py` (확장)

- [ ] **Step 1: 테스트 확장 — folder가 SQL pre-filter로 강등되는지**

`tests/test_decomposer_folder_extraction.py` 끝에 추가:

```python
def test_folder_scope_applies_as_sql_prefilter(tmp_path):
    """When ConstraintPlan.folder is set, candidates are restricted before RRF."""
    from backend.search.sqlite_search import apply_folder_filter

    # 5 files across two folders
    rows = [
        {"id": 1, "folder_path": "/lib/#07/a.png"},
        {"id": 2, "folder_path": "/lib/#07/b.png"},
        {"id": 3, "folder_path": "/lib/#08/c.png"},
        {"id": 4, "folder_path": "/lib/other/d.png"},
        {"id": 5, "folder_path": "/lib/#07/e.png"},
    ]
    filtered = apply_folder_filter(rows, folder="#07")
    assert {r["id"] for r in filtered} == {1, 2, 5}


def test_folder_scope_empty_returns_input_unchanged():
    from backend.search.sqlite_search import apply_folder_filter

    rows = [{"id": 1, "folder_path": "/x/y.png"}]
    assert apply_folder_filter(rows, folder="") is rows
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_decomposer_folder_extraction.py -q`
Expected: `ImportError: cannot import name 'apply_folder_filter'`

- [ ] **Step 3: `apply_folder_filter` 추가**

`backend/search/sqlite_search.py` 끝에 추가 (모듈 함수):

```python
# ── Phase B: hard folder constraint ───────────────────────────────

def apply_folder_filter(rows, folder: str):
    """Drop rows whose folder_path doesn't contain the requested folder.

    Substring match for now — the same string the Decomposer extracts
    ("#07", "studio_bg" etc.) is matched against folder_path. Exact
    boundary handling is upgraded once we have evidence it matters.
    """
    if not folder:
        return rows
    needle = folder.strip()
    if not needle:
        return rows
    return [r for r in rows if needle in (r.get("folder_path") or "")]
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_decomposer_folder_extraction.py -q`
Expected: `5 passed` (앞 3건 + 신규 2건)

- [ ] **Step 5: 커밋**

```bash
git add backend/search/sqlite_search.py tests/test_decomposer_folder_extraction.py
git commit -m "feat: apply folder scope as hard pre-filter (phase B)"
```

---

## Phase C — Benchmark 4-Way 카테고리

**목표:** `bench_precision.py` 가 쿼리별로 `found / missed / honest_empty / false_answer` 를 분류해서 집계한다.

### Task C1: 분류 헬퍼 + 단위 테스트

**Files:**
- Create: `benchmarks/scripts/classify_outcome.py` (또는 동등 모듈)
- Test: `tests/test_bench_outcome_classifier.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_bench_outcome_classifier.py`:

```python
"""Phase C: per-query benchmark outcome classification."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "scripts"))

from classify_outcome import classify, Outcome  # noqa: E402


def test_top1_in_gt_is_found():
    assert classify(top_k=[1, 2, 3], gt={1, 4}, system_confidence="medium") is Outcome.FOUND


def test_no_overlap_but_gt_present_is_missed():
    assert classify(top_k=[5, 6, 7], gt={1, 2}, system_confidence="medium") is Outcome.MISSED


def test_empty_response_and_no_gt_is_honest_empty():
    assert classify(top_k=[], gt=set(), system_confidence="empty") is Outcome.HONEST_EMPTY


def test_answer_when_gt_empty_is_false_answer():
    assert classify(top_k=[5, 6], gt=set(), system_confidence="medium") is Outcome.FALSE_ANSWER


def test_empty_confidence_with_present_gt_is_still_missed():
    # System said "no" but there were relevant items — that's a miss.
    assert classify(top_k=[], gt={1, 2}, system_confidence="empty") is Outcome.MISSED
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_bench_outcome_classifier.py -q`
Expected: `ImportError`

- [ ] **Step 3: 모듈 구현**

`benchmarks/scripts/classify_outcome.py`:

```python
"""Phase C: classify each benchmark query into 4 buckets.

found          top_k ∩ gt ≠ ∅
missed         gt ≠ ∅ and top_k ∩ gt = ∅
honest_empty   gt = ∅ and system_confidence = empty
false_answer   gt = ∅ and system_confidence ≠ empty
"""
from __future__ import annotations

import enum
from typing import Iterable


class Outcome(str, enum.Enum):
    FOUND = "found"
    MISSED = "missed"
    HONEST_EMPTY = "honest_empty"
    FALSE_ANSWER = "false_answer"


def classify(*, top_k: Iterable[int], gt: set[int], system_confidence: str) -> Outcome:
    top_set = set(top_k or [])
    has_gt = bool(gt)
    if has_gt:
        if top_set & gt:
            return Outcome.FOUND
        return Outcome.MISSED
    # gt is empty
    if system_confidence == "empty":
        return Outcome.HONEST_EMPTY
    return Outcome.FALSE_ANSWER
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_bench_outcome_classifier.py -q`
Expected: `5 passed`

- [ ] **Step 5: 커밋**

```bash
git add benchmarks/scripts/classify_outcome.py tests/test_bench_outcome_classifier.py
git commit -m "feat: bench outcome classifier — found/missed/honest_empty/false_answer (phase C)"
```

### Task C2: bench_precision.py 가 4종을 집계

**Files:**
- Modify: 기존 precision 벤치 스크립트 (위치는 `benchmarks/scripts/` 또는 `benchmarks/bench_precision.py` 중 존재하는 곳)

- [ ] **Step 1: 위치 확인**

Run: `find /Users/saintiron/Projects/Imagine/benchmarks -name 'bench_precision*' -o -name '*precision*.py' | head -5`

위치를 확인한 뒤, 본 스크립트가 결과를 집계하는 함수를 찾아 다음 두 곳을 수정:

1. 결과 dict에 `outcome` 필드 추가.
2. 최종 요약 출력에 `found / missed / honest_empty / false_answer` 4종 비율 추가.

- [ ] **Step 2: 집계 코드 패치**

기존 per-query 집계 루프를 다음 형태로:

```python
from benchmarks.scripts.classify_outcome import classify, Outcome

outcomes = {o: 0 for o in Outcome}
for q in queries:
    outcome = classify(
        top_k=[r["id"] for r in q["results"][: top_k]],
        gt=set(q["gt_ids"]),
        system_confidence=q.get("confidence", "medium"),
    )
    q["outcome"] = outcome.value
    outcomes[outcome] += 1

n = max(1, sum(outcomes.values()))
print(f"found        : {outcomes[Outcome.FOUND]}/{n} ({outcomes[Outcome.FOUND]/n:.1%})")
print(f"missed       : {outcomes[Outcome.MISSED]}/{n} ({outcomes[Outcome.MISSED]/n:.1%})")
print(f"honest_empty : {outcomes[Outcome.HONEST_EMPTY]}/{n} ({outcomes[Outcome.HONEST_EMPTY]/n:.1%})")
print(f"false_answer : {outcomes[Outcome.FALSE_ANSWER]}/{n} ({outcomes[Outcome.FALSE_ANSWER]/n:.1%})")
```

- [ ] **Step 3: 실행 (dry-run 모드가 있으면 그쪽으로) 확인**

Run: `.venv/bin/python benchmarks/scripts/bench_precision.py --help`
Expected: usage 출력 (실행 옵션 확인용; 실제 데이터셋 돌릴 필요 없음).

- [ ] **Step 4: 커밋**

```bash
git add benchmarks/scripts/bench_precision.py  # 또는 실제 수정한 경로
git commit -m "feat: bench_precision reports 4-way outcome split (phase C)"
```

---

## Phase D — User Correction Loop

**목표:** 사용자가 검색 결과에 "관련 없음" 라벨을 1-클릭으로 남기면 그 정보가 향후 검색에서 약한 demotion으로 반영된다.

### Task D1: DB 스키마 + 마이그레이션

**Files:**
- Modify: `backend/db/sqlite_schema_auth.sql`
- Modify: `backend/db/sqlite_migrations.py`
- Test: `tests/test_search_feedback_migration.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_search_feedback_migration.py`:

```python
"""Phase D: search_feedback table schema."""
from __future__ import annotations

import sqlite3
import types

import pytest


def _db():
    conn = sqlite3.connect(":memory:")
    return types.SimpleNamespace(
        conn=conn,
        _table_exists=lambda name: conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None,
    )


def test_migration_creates_search_feedback_table():
    from backend.db.sqlite_migrations import migrate_search_feedback

    db = _db()
    assert db._table_exists("search_feedback") is False
    migrate_search_feedback(db)
    assert db._table_exists("search_feedback") is True

    cols = {row[1] for row in db.conn.execute(
        "PRAGMA table_info(search_feedback)"
    ).fetchall()}
    assert {"id", "query", "file_id", "label", "user_id", "created_at"}.issubset(cols)


def test_migration_is_idempotent():
    from backend.db.sqlite_migrations import migrate_search_feedback

    db = _db()
    migrate_search_feedback(db)
    migrate_search_feedback(db)  # second call must not raise
    assert db._table_exists("search_feedback")
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_search_feedback_migration.py -q`
Expected: `ImportError: cannot import name 'migrate_search_feedback'`

- [ ] **Step 3: 스키마 + 마이그레이션 추가**

`backend/db/sqlite_schema_auth.sql` 의 search_logs 섹션 뒤에 추가:

```sql
-- ═══════════════════════════════════════════════════════════════
-- Phase D — Search Feedback (irrelevant labels)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS search_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    file_id INTEGER NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('irrelevant')),
    user_id INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_search_feedback_query ON search_feedback(query);
CREATE INDEX IF NOT EXISTS idx_search_feedback_file ON search_feedback(file_id);
```

`backend/db/sqlite_migrations.py` 의 `migrate_audit_log` 뒤에 추가:

```python
def migrate_search_feedback(db):
    """Phase D: store user 'irrelevant' feedback for search results."""
    if db._table_exists('search_feedback'):
        return
    logger.info("Migrating: creating search_feedback table...")
    db.conn.executescript(
        """
        CREATE TABLE search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            label TEXT NOT NULL CHECK (label IN ('irrelevant')),
            user_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_search_feedback_query ON search_feedback(query);
        CREATE INDEX idx_search_feedback_file ON search_feedback(file_id);
        """
    )
    db.conn.commit()
    logger.info("search_feedback table created")
```

그리고 migrations 리스트 + fresh-install 경로 양쪽에 한 줄씩 추가:

```python
("search_feedback", lambda: migrate_search_feedback(db)),
```

```python
migrate_search_feedback(db)
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_search_feedback_migration.py -q`
Expected: `2 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/db/sqlite_schema_auth.sql backend/db/sqlite_migrations.py \
        tests/test_search_feedback_migration.py
git commit -m "feat: add search_feedback table (phase D)"
```

### Task D2: /search/feedback endpoint

**Files:**
- Create: `backend/server/routers/search_feedback.py`
- Modify: `backend/server/app.py` (라우터 등록)
- Test: `tests/test_search_feedback_endpoint.py`

- [ ] **Step 1: 실패 테스트 작성**

`tests/test_search_feedback_endpoint.py`:

```python
"""Phase D: POST /api/v1/search/feedback persists irrelevant labels."""
from __future__ import annotations

import sqlite3
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.server import deps


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        """CREATE TABLE search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            label TEXT NOT NULL CHECK (label IN ('irrelevant')),
            user_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    return types.SimpleNamespace(conn=conn)


def _client(db):
    from backend.server.routers import search_feedback

    app = FastAPI()
    app.include_router(search_feedback.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": 7, "username": "u"}
    return TestClient(app)


def test_feedback_persists_irrelevant_row():
    db = _db()
    client = _client(db)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "x", "file_id": 42, "label": "irrelevant"},
    )
    assert resp.status_code == 200
    row = db.conn.execute(
        "SELECT query, file_id, label, user_id FROM search_feedback"
    ).fetchone()
    assert row == ("x", 42, "irrelevant", 7)


def test_feedback_rejects_unknown_label():
    db = _db()
    client = _client(db)
    resp = client.post(
        "/api/v1/search/feedback",
        json={"query": "x", "file_id": 1, "label": "love-it"},
    )
    assert resp.status_code in (400, 422)
```

- [ ] **Step 2: 테스트 FAIL 확인**

Run: `.venv/bin/python -m pytest tests/test_search_feedback_endpoint.py -q`
Expected: `ImportError` (search_feedback 라우터 없음)

- [ ] **Step 3: 라우터 구현**

`backend/server/routers/search_feedback.py`:

```python
"""Phase D: persist user 'irrelevant' labels for search results."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_current_user, get_db_safe

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search-feedback"])


class FeedbackRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    file_id: int
    label: Literal["irrelevant"] = "irrelevant"


@router.post("/search/feedback")
def submit_feedback(
    req: FeedbackRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    db.conn.execute(
        """INSERT INTO search_feedback (query, file_id, label, user_id)
           VALUES (?, ?, ?, ?)""",
        (req.query, req.file_id, req.label, user["id"]),
    )
    db.conn.commit()
    return {"success": True}
```

- [ ] **Step 4: 테스트 PASS 확인**

Run: `.venv/bin/python -m pytest tests/test_search_feedback_endpoint.py -q`
Expected: `2 passed`

- [ ] **Step 5: 라우터 등록**

`backend/server/app.py` 의 라우터 import 블록에 추가:

```python
from backend.server.routers.search_feedback import router as search_feedback_router
```

include 블록:

```python
app.include_router(search_feedback_router, prefix="/api/v1")
```

- [ ] **Step 6: 커밋**

```bash
git add backend/server/routers/search_feedback.py backend/server/app.py \
        tests/test_search_feedback_endpoint.py
git commit -m "feat: POST /search/feedback persists irrelevant labels (phase D)"
```

### Task D3: Frontend "관련 없음" 버튼 + soft demotion

**Files:**
- Modify: `frontend/src/components/SearchPanel.jsx` (또는 결과 카드 컴포넌트)
- Modify: `frontend/src/api/search.js` — `postFeedback(query, fileId)` 추가
- Modify: `frontend/src/i18n/locales/ko-KR.json` / `en-US.json`
- Modify: `backend/search/sqlite_search.py` (또는 `scoring.py`) — RRF 결과에서 irrelevant 라벨된 file_id에 작은 패널티 적용

- [ ] **Step 1: api/search.js 에 postFeedback 추가**

```js
export async function postFeedback(query, fileId) {
  return apiClient.post('/api/v1/search/feedback', {
    query,
    file_id: fileId,
    label: 'irrelevant',
  });
}
```

- [ ] **Step 2: 결과 카드에 버튼 추가**

각 결과 카드에 작은 X 버튼:

```jsx
<button
  type="button"
  onClick={() => postFeedback(currentQuery, result.id).then(() => onMarkedIrrelevant?.(result.id))}
  title={t('search.mark_irrelevant')}
  className="text-xs text-gray-500 hover:text-amber-400"
>
  ✕ {t('search.mark_irrelevant_short')}
</button>
```

- [ ] **Step 3: i18n 추가**

```
"search.mark_irrelevant": "이 결과는 관련 없음" / "Mark as irrelevant"
"search.mark_irrelevant_short": "관련 없음" / "Irrelevant"
```

- [ ] **Step 4: Backend soft demotion**

`backend/search/sqlite_search.py` 안 RRF 결과 정리 직전 또는 enrich 단계에서:

```python
# Phase D: demote items the user has marked irrelevant for similar queries.
try:
    rows = db.conn.execute(
        """SELECT file_id, COUNT(*) AS n FROM search_feedback
           WHERE query = ? AND label = 'irrelevant'
           GROUP BY file_id""",
        (raw_query,),
    ).fetchall()
    penalty_map = {fid: 0.05 * n for fid, n in rows}
    for r in results:
        p = penalty_map.get(r.get("id"))
        if p:
            r["rrf_score"] = (r.get("rrf_score") or 0.0) - p
    results.sort(key=lambda r: r.get("rrf_score", 0.0), reverse=True)
except Exception:
    pass
```

- [ ] **Step 5: build 검증**

Run: `cd frontend && npm run build`
Expected: `built in` 정상 출력.

- [ ] **Step 6: 커밋**

```bash
git add frontend/src/api/search.js frontend/src/components/SearchPanel.jsx \
        frontend/src/i18n/locales/ko-KR.json frontend/src/i18n/locales/en-US.json \
        backend/search/sqlite_search.py
git commit -m "feat: user can mark search results irrelevant; soft demotion (phase D)"
```

---

## 완료 조건 (Done Criteria)

- 검색 응답이 `confidence ∈ {high, medium, low, empty}` 와 `top1_raw_score` 를 항상 가진다.
- `confidence=empty` 인 응답은 `results=[]` 이고 사용자에게 정직하게 "없음"으로 표시된다.
- Decomposer 는 항상 ConstraintPlan 을 반환하고, LLM 실패 시 1회 재시도 후 rule-based fallback.
- ConstraintPlan.folder 가 있으면 SQL pre-filter 로 후보가 폴더 안으로 잘린다.
- `bench_precision.py` 가 `found / missed / honest_empty / false_answer` 4종 비율을 출력.
- 사용자가 결과 카드의 "관련 없음" 버튼을 누르면 `search_feedback` 에 기록되고, 같은 쿼리의 후속 검색에서 그 file_id 점수가 낮아진다.
- 본 plan 의 신규 테스트가 모두 통과 (`pytest tests/test_search_confidence.py tests/test_search_response_shape.py tests/test_constraint_plan.py tests/test_decomposer_folder_extraction.py tests/test_bench_outcome_classifier.py tests/test_search_feedback_migration.py tests/test_search_feedback_endpoint.py`).
- 기존 회귀 통과 + `cd frontend && npm run build` 통과.

## 후속 (out of scope, 별도 plan)

- Multi-element AND 검증 단계(현재는 Decomposer 출력에 `elements[]` 만 추출, 결과 단계 enforcement 는 다음 plan).
- Cross-encoder re-ranker.
- 사용자 정정이 일정 임계 누적되면 해당 file 의 `user_tags` 를 자동 갱신.
- Audit log 와의 통합 (search_feedback 도 audit_log 에 미러링).
- Spatial processing 효과의 confidence 임계 캘리브레이션.
