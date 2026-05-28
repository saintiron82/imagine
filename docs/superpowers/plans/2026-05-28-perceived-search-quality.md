# Search Ranking Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the *ranking* of Imagine's search results so that (1) genuinely-relevant items are not missed from top-K, and (2) the most relevant items end up at the top of top-K. Latency is already acceptable; this plan deliberately ignores response-time work.

**Architecture:**
Three focused phases, each improving "what surfaces on top" via a different lever.
- **Phase α (Cross-encoder rerank)** plugs a small multilingual reranker over top-30 candidates so the final order reflects query↔document semantic similarity, not just RRF rank fusion.
- **Phase β (Constraint enforcement + calibration)** verifies that multi-element queries (e.g. "캐릭터과 방") actually match all elements, and recalibrates confidence thresholds from the existing LLM-judge dataset so "low confidence" really means low precision.
- **Phase γ (Feedback-driven demotion)** activates the user-feedback loop end-to-end — discoverable button + admin visibility + automatic user_tags update — so the system improves over time without explicit retraining.

**Tech Stack:** Python (FastAPI, sqlite-vec, transformers for cross-encoder), React, sentence-transformers/cross-encoder model (CPU-friendly `BAAI/bge-reranker-v2-m3`), agentcli (already integrated).

---

## 핵심 정의

- **AND-verified result** = `ConstraintPlan.elements` 의 모든 요소가 결과의 `mc_caption + ai_tags + spatial_objects` 텍스트에 등장하는 결과.
- **calibration set** = `benchmarks/results/precision_20260528_rejudge_input_llm_rejudge_k5.json` 의 per-query `judged` 필드. 각 file_id가 yes/no/skip 인지가 라벨.
- **rerank pool** = RRF 머지 직후 top-30 후보. cross-encoder가 이 풀만 재정렬.
- **discoverable button** = hover-only가 아니라 카드의 정적 영역에 늘 표시되는 버튼. 클릭 가능 영역 ≥ 32×32 px.

---

## 현재 스펙 (As-Is)

### Ranking

- `backend/search/sqlite_search.py:1949` `triaxis_search` 가 VV/MV/FTS 각 축의 후보를 RRF 머지 → 그 점수가 곧 최종 순위. Cross-encoder 등 dense rerank 단계 없음.
- 결과 사이의 거리가 작을 때 (RRF 점수 0.05 이하 차이) 어느 게 먼저 올라가는지가 axis 점수 가중치에만 의존. 의미적 fine grained 정렬은 없음.
- **Multi-element AND**: `ConstraintPlan.elements` 가 추출되지만 **결과 검증에 안 씀.** "캐릭터과 방" 쿼리에서 캐릭터만 있는 이미지가 top-K에 살아남음.

### Honesty (confidence)

- `backend/search/confidence.py:24-29` `ConfidenceThresholds(low=0.20, mid=0.35, high=0.55)` — 측정 없이 추측한 임계.
- 결과로 `confidence` 가 응답에 실리지만 ranking 자체에는 영향 안 줌. 그리고 "low" 가 실제로 얼마나 신뢰할 수 없는지 (precision-at-confidence) 미측정.

### Feedback Loop

- Phase D로 `search_feedback` 테이블 + `POST /api/v1/search/feedback` 가 존재. SearchPanel.jsx에 "관련 없음" 버튼이 있지만 **hover-only**.
- soft demotion 코드(`sqlite_search.py:` Phase D 섹션)가 깔려 있지만 누적 feedback 자체가 0이라 영향 없음.
- admin 가시성 없고, 일정 누적 도달 시 자동 보정 경로도 없음.

---

## 목표 스펙 (To-Be)

### Ranking

- Top-30 후보가 cross-encoder (`BAAI/bge-reranker-v2-m3`)로 query↔document semantic similarity 기반 재정렬. 결과 dict 에 `cross_encoder_score` 가 enrich 됨.
- Multi-element AND 검증: ConstraintPlan.elements 가 비어있지 않으면, 모든 element가 결과 텍스트에 등장하는지 점검. 누락 element 수 만큼 score 패널티 (per element 0.10), 또는 strict 모드로 제외.
- 결과: 동일 30 쿼리 keyword GT P@5 가 (계측된) baseline 대비 측정 가능한 lift.

### Honesty

- `ConfidenceThresholds` 가 LLM-judge calibration set 으로 fit된 값. 기준: precision-at-confidence ≥ 0.5 (low) / 0.7 (mid) / 0.85 (high). 미달 데이터 시 기본값 보존.
- 결과 카드에 dominant-axis 작은 배지 ("VV 강함", "MV 의미 일치", "FTS 키워드 일치"). 이미 응답에 있는 axis 점수를 표시만.

### Feedback Loop

- "관련 없음" 버튼이 카드의 항상 보이는 footer, 32×32 px 이상. 클릭으로 즉시 demotion 영향이 다음 검색에 반영됨.
- `/api/v1/admin/search-feedback/summary` 가 (1) 최근 30일 피드백 수, (2) top 20 flagged file, (3) top 10 flagged query를 반환. AdminPage에 새 "정정 피드백" 탭.
- N=3 (configurable) 이상 누적된 file_id는 백그라운드 job이 `user_tags`에 `low-relevance` 추가 → ranking에 직접 반영.

---

## 갭 (As-Is → To-Be)

1. RRF 만으로 최종 순위 → top-30 후보를 cross-encoder로 재정렬.
2. ConstraintPlan.elements 가 추출되지만 결과 검증에 안 씀 → 모든 element 등장 여부 점검 후 점수 패널티.
3. 추측한 threshold → 실측 데이터로 fit.
4. 카드가 axis 정보 안 보임 → dominant axis badge.
5. Hover-only feedback → 항상 보이는 footer 버튼.
6. Admin 가시성 없음 → dashboard endpoint + 패널.
7. Feedback 누적이 자동 보정으로 안 이어짐 → auto user_tags updater.

---

## File Structure

**Phase α — Cross-encoder rerank**
- Create: `backend/search/cross_encoder.py`
- Modify: `backend/search/sqlite_search.py` — rerank 단계 추가
- Test: `tests/test_cross_encoder.py`

**Phase β — Constraint + Calibration**
- Modify: `backend/search/sqlite_search.py` — AND verification + element penalty
- Create: `tools/calibrate_confidence.py` — LLM-judge 데이터에서 threshold fit
- Modify: `backend/search/confidence.py` — calibrated 기본값
- Modify: `frontend/src/components/SearchPanel.jsx` — axis badge
- Modify: `frontend/src/i18n/locales/{ko-KR,en-US}.json` — badge labels
- Test: `tests/test_threshold_calibration.py`
- Test: `tests/test_element_and_verification.py`

**Phase γ — Feedback loop end-to-end**
- Modify: `frontend/src/components/SearchPanel.jsx` — feedback 버튼을 항상 표시
- Create: `backend/server/routers/feedback_dashboard.py` — `/admin/search-feedback/summary`
- Modify: `backend/server/app.py` — 라우터 등록 + startup hook
- Create: `frontend/src/components/admin/SearchFeedbackPanel.jsx` — admin 표시
- Modify: `frontend/src/pages/AdminPage.jsx` — 새 탭 등록
- Modify: `frontend/src/api/admin.js` — getSearchFeedbackSummary
- Modify: `frontend/src/i18n/locales/{ko-KR,en-US}.json` — admin 라벨
- Create: `backend/server/jobs/auto_user_tags.py` — 누적 정정 → user_tags 갱신
- Test: `tests/test_feedback_dashboard.py`
- Test: `tests/test_auto_user_tags.py`

---

## Phase α — Cross-Encoder Rerank

### Task α1: Reranker module

**Files:**
- Create: `backend/search/cross_encoder.py`
- Test: `tests/test_cross_encoder.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cross_encoder.py`:

```python
"""Phase α: cross-encoder rerank."""
from __future__ import annotations

import pytest

from backend.search.cross_encoder import rerank


def test_rerank_no_op_when_reranker_is_none():
    rows = [{"id": 1}, {"id": 2}]
    out = rerank(query="x", rows=rows, reranker=None)
    assert out is rows


def test_rerank_reorders_by_score_descending():
    class _Stub:
        def score_pairs(self, pairs):
            mapping = {"a": 0.9, "b": 0.2, "c": 0.5}
            return [mapping.get(doc, 0.0) for _, doc in pairs]

    rows = [
        {"id": 1, "mc_caption": "b"},
        {"id": 2, "mc_caption": "a"},
        {"id": 3, "mc_caption": "c"},
    ]
    out = rerank(query="q", rows=rows, reranker=_Stub())
    assert [r["id"] for r in out] == [2, 3, 1]


def test_rerank_stable_for_ties():
    class _Equal:
        def score_pairs(self, pairs):
            return [0.5] * len(pairs)

    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    out = rerank(query="q", rows=rows, reranker=_Equal())
    assert [r["id"] for r in out] == [1, 2, 3]


def test_rerank_uses_caption_then_tags_then_empty():
    class _Recorder:
        def __init__(self):
            self.seen = []

        def score_pairs(self, pairs):
            self.seen = list(pairs)
            return [0.0] * len(pairs)

    rec = _Recorder()
    rows = [
        {"id": 1, "mc_caption": "cap-1"},
        {"id": 2, "ai_tags": "tag-2"},
        {"id": 3},
    ]
    rerank(query="q", rows=rows, reranker=rec)
    assert rec.seen[0] == ("q", "cap-1")
    assert rec.seen[1] == ("q", "tag-2")
    assert rec.seen[2] == ("q", "")


def test_rerank_attaches_cross_encoder_score_to_rows():
    class _Score:
        def score_pairs(self, pairs):
            return [0.3, 0.7]

    rows = [{"id": 1}, {"id": 2}]
    out = rerank(query="q", rows=rows, reranker=_Score())
    # Order: id=2 (0.7) first, id=1 (0.3) second
    assert out[0]["id"] == 2
    assert out[0]["cross_encoder_score"] == 0.7
    assert out[1]["cross_encoder_score"] == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cross_encoder.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Write the module**

`backend/search/cross_encoder.py`:

```python
"""Phase α: cross-encoder rerank over a small candidate pool.

The reranker is a Protocol-shaped duck type so unit tests inject a
stub. The production implementation lazy-loads
`BAAI/bge-reranker-v2-m3` via transformers — `load_default_reranker()`.
"""
from __future__ import annotations

from typing import Protocol


class CrossEncoderReranker(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        ...


def rerank(*, query: str, rows: list[dict], reranker: object) -> list[dict]:
    """Return `rows` reordered by reranker.score_pairs((query, doc))
    descending. Stable for ties. When `reranker` is falsy, returns the
    input list unchanged. Mutates each kept row to add
    `cross_encoder_score`.
    """
    if reranker is None or not rows:
        return rows

    def _doc_text(row: dict) -> str:
        return row.get("mc_caption") or row.get("ai_tags") or ""

    pairs = [(query, _doc_text(r)) for r in rows]
    scores = reranker.score_pairs(pairs)
    for r, s in zip(rows, scores):
        r["cross_encoder_score"] = float(s)

    indexed = list(enumerate(rows))
    indexed.sort(key=lambda iv: -scores[iv[0]])
    return [r for _, r in indexed]


_default_reranker = None


def load_default_reranker():
    """Lazy-load BAAI/bge-reranker-v2-m3 via transformers (CPU OK).

    Returns None if transformers / the model is unavailable so callers
    gracefully skip reranking.
    """
    global _default_reranker
    if _default_reranker is not None:
        return _default_reranker
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception:
        return None
    try:
        model_id = "BAAI/bge-reranker-v2-m3"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        mdl.eval()

        class _BGEReranker:
            def score_pairs(self, pairs):
                if not pairs:
                    return []
                with torch.no_grad():
                    inputs = tok(
                        [p[0] for p in pairs],
                        [p[1] for p in pairs],
                        padding=True, truncation=True,
                        max_length=384, return_tensors="pt",
                    )
                    return mdl(**inputs).logits.view(-1).float().tolist()

        _default_reranker = _BGEReranker()
    except Exception:
        _default_reranker = None
    return _default_reranker
```

- [ ] **Step 4: Verify tests pass**

Run: `.venv/bin/python -m pytest tests/test_cross_encoder.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/search/cross_encoder.py tests/test_cross_encoder.py
git commit -m "feat: add cross-encoder rerank module (phase α)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task α2: Wire rerank into triaxis_search

**Files:**
- Modify: `backend/search/sqlite_search.py`
- Test: `tests/test_cross_encoder.py` (extend)

- [ ] **Step 1: Extend the test**

Append to `tests/test_cross_encoder.py`:

```python
def test_rerank_keeps_tail_beyond_pool_intact():
    """Wire pattern: rerank head pool, append untouched tail."""
    from backend.search.cross_encoder import rerank

    class _Identity:
        def score_pairs(self, pairs):
            # Higher index → higher score so order REVERSES within pool
            return list(range(len(pairs)))

    pool = [{"id": i} for i in range(5)]
    tail = [{"id": i} for i in range(5, 10)]
    out_pool = rerank(query="q", rows=pool, reranker=_Identity())
    out_pool_ids = [r["id"] for r in out_pool]
    full = out_pool + tail
    assert out_pool_ids == [4, 3, 2, 1, 0]
    assert [r["id"] for r in full[5:]] == [5, 6, 7, 8, 9]
```

- [ ] **Step 2: Add the rerank block to triaxis_search**

In `backend/search/sqlite_search.py`, locate the existing Phase D / Phase B late-stage hooks (search for `# Phase D: soft demotion` or `# Phase B: hard folder substring filter`). The new block goes **between RRF merge and the existing late hooks** — i.e., as the first quality pass. Add:

```python
        # Phase α: cross-encoder rerank over top-30 candidates.
        try:
            from backend.search.cross_encoder import rerank as _ce_rerank, load_default_reranker
            reranker = load_default_reranker()
            if reranker is not None and len(merged) > 1:
                pool = merged[: min(30, len(merged))]
                reranked = _ce_rerank(query=query, rows=pool, reranker=reranker)
                merged = reranked + merged[len(pool):]
                diag["cross_encoder_rerank"] = {"pool_size": len(pool)}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Cross-encoder rerank skipped: {exc}")
```

- [ ] **Step 3: Verify**

Run:
```bash
.venv/bin/python -m pytest tests/test_cross_encoder.py -q
.venv/bin/python -m py_compile backend/search/sqlite_search.py
```
Expected: `6 passed`, py_compile OK.

- [ ] **Step 4: Commit**

```bash
git add backend/search/sqlite_search.py tests/test_cross_encoder.py
git commit -m "feat: rerank top-30 via cross-encoder in triaxis_search (phase α)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase β — Constraint Enforcement + Calibration

### Task β1: Multi-element AND verification

**Files:**
- Modify: `backend/search/sqlite_search.py` (decompose call site + post-RRF check)
- Test: `tests/test_element_and_verification.py`

- [ ] **Step 1: Write the failing test**

`tests/test_element_and_verification.py`:

```python
"""Phase β: every element of the ConstraintPlan must appear in each result."""
from __future__ import annotations

import pytest

from backend.search.sqlite_search import apply_element_verification


def test_no_op_when_elements_empty():
    rows = [{"id": 1, "mc_caption": "anything"}]
    out = apply_element_verification(rows, elements=[])
    assert out == rows


def test_keeps_rows_with_all_elements_in_caption():
    rows = [
        {"id": 1, "mc_caption": "character in a room with window"},
        {"id": 2, "mc_caption": "only character here"},
    ]
    out = apply_element_verification(rows, elements=["character", "room"])
    assert [r["id"] for r in out] == [1, 2]   # both kept, but 2 penalised
    # Penalty must reduce rrf_score for id=2
    assert (out[0].get("rrf_score") or 0.0) >= (out[1].get("rrf_score") or 0.0)


def test_uses_tags_and_spatial_objects_as_fallback_text():
    rows = [
        {"id": 1, "mc_caption": "", "ai_tags": "room, window",
         "spatial_objects": ["character", "lamp"]},
    ]
    out = apply_element_verification(rows, elements=["character", "room"])
    # Both elements found across the three text fields
    assert out[0].get("element_match_count") == 2
    assert out[0].get("element_miss_count") == 0


def test_applies_per_missing_element_penalty():
    rows = [
        {"id": 1, "mc_caption": "character", "rrf_score": 1.0},  # misses 'room'
    ]
    out = apply_element_verification(rows, elements=["character", "room"],
                                     penalty=0.10)
    assert out[0]["rrf_score"] == pytest.approx(0.90)
    assert out[0]["element_miss_count"] == 1


def test_korean_and_english_element_both_match():
    """Element strings can be Korean OR English; either form is acceptable."""
    rows = [
        {"id": 1, "mc_caption": "character in a room",
         "ai_tags": "캐릭터, 방"},
    ]
    out = apply_element_verification(rows, elements=["캐릭터", "방"])
    assert out[0]["element_match_count"] == 2
    assert out[0]["element_miss_count"] == 0


def test_resort_after_penalty_so_full_matches_rise():
    rows = [
        {"id": 1, "mc_caption": "character only",
         "rrf_score": 0.80},
        {"id": 2, "mc_caption": "character in a room",
         "rrf_score": 0.75},
    ]
    out = apply_element_verification(rows, elements=["character", "room"],
                                     penalty=0.10)
    # After penalty: id=1 -> 0.70, id=2 -> 0.75 → id=2 first
    assert out[0]["id"] == 2
    assert out[1]["id"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_element_and_verification.py -q`
Expected: `ImportError: cannot import name 'apply_element_verification'`.

- [ ] **Step 3: Add the helper**

In `backend/search/sqlite_search.py`, append at the bottom near `apply_folder_filter`:

```python
def apply_element_verification(
    rows,
    *,
    elements,
    penalty: float = 0.10,
):
    """Penalise rows missing any of the requested elements.

    For each row, count which elements appear (substring match,
    case-insensitive) in the combined text of mc_caption + ai_tags +
    spatial_objects. Apply `penalty * (n_missing)` to rrf_score and
    re-sort. Rows that match every element are unchanged. Elements
    can be in either Korean or English; the substring match works on
    whichever language the result text uses.
    """
    if not elements:
        return rows

    needles = [e.strip().lower() for e in elements if e and e.strip()]
    if not needles:
        return rows

    for r in rows:
        text_parts = [r.get("mc_caption") or "", r.get("ai_tags") or ""]
        sp = r.get("spatial_objects") or []
        if isinstance(sp, list):
            text_parts.extend(str(x) for x in sp)
        haystack = " ".join(text_parts).lower()

        present = sum(1 for n in needles if n in haystack)
        missing = len(needles) - present
        r["element_match_count"] = present
        r["element_miss_count"] = missing
        if missing:
            r["rrf_score"] = float(r.get("rrf_score") or 0.0) - penalty * missing

    rows = list(rows)
    rows.sort(key=lambda r: r.get("rrf_score", 0.0), reverse=True)
    return rows
```

- [ ] **Step 4: Wire into triaxis_search**

In `triaxis_search`, find the Phase D / Phase B late-stage section. Just AFTER the cross-encoder rerank block (Task α2) and BEFORE the existing Phase D feedback demotion, insert:

```python
        # Phase β: enforce ConstraintPlan elements via post-hoc check.
        try:
            from backend.search.query_decomposer import QueryDecomposer
            from backend.search.constraint_plan import (
                ConstraintPlan, from_decomposer_output,
            )
            # Reuse the same decomposed plan that scope/find/exclude
            # already consumed earlier in this function.
            elements_for_check = []
            if isinstance(unified, dict):
                # Unified schema: scope -> find.elements
                find = unified.get("find") or {}
                if isinstance(find, dict):
                    raw_elements = find.get("keywords") or []
                    if isinstance(raw_elements, list):
                        elements_for_check = [str(x) for x in raw_elements][:6]
            if elements_for_check and len(merged) > 1:
                pre_count = len(merged)
                merged = apply_element_verification(
                    merged, elements=elements_for_check, penalty=0.10,
                )
                diag["element_verification"] = {
                    "elements": elements_for_check,
                    "pre_count": pre_count,
                }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Element verification skipped: {exc}")
```

- [ ] **Step 5: Verify**

Run:
```bash
.venv/bin/python -m pytest tests/test_element_and_verification.py -q
.venv/bin/python -m py_compile backend/search/sqlite_search.py
```
Expected: `6 passed`, py_compile OK.

- [ ] **Step 6: Commit**

```bash
git add backend/search/sqlite_search.py tests/test_element_and_verification.py
git commit -m "feat: post-hoc multi-element AND verification with penalty (phase β)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task β2: Calibrate ConfidenceThresholds from LLM-judge data

**Files:**
- Create: `tools/calibrate_confidence.py`
- Modify: `backend/search/confidence.py`
- Modify: `tests/test_search_confidence.py` (boundary cases)
- Test: `tests/test_threshold_calibration.py`

- [ ] **Step 1: Write the failing test**

`tests/test_threshold_calibration.py`:

```python
"""Phase β: confidence threshold calibration."""
from __future__ import annotations

import pytest


def test_calibration_returns_documented_keys():
    from tools.calibrate_confidence import calibrate

    out = calibrate([(0.5, True), (0.5, False)])
    assert set(out.keys()) == {"low", "mid", "high", "n_samples"}


def test_calibration_monotonic():
    from tools.calibrate_confidence import calibrate

    samples = [
        (0.10, False), (0.12, False), (0.15, False),
        (0.22, False), (0.25, True), (0.28, False),
        (0.40, True), (0.42, True), (0.45, False),
        (0.60, True), (0.65, True), (0.70, True),
    ]
    t = calibrate(samples)
    assert t["low"] <= t["mid"] <= t["high"]


def test_calibration_uses_precision_targets():
    """Targets: precision-at-confidence ≥ 0.5 / 0.7 / 0.85."""
    from tools.calibrate_confidence import calibrate, _precision_at

    samples = (
        [(0.1 + i * 0.01, False) for i in range(20)] +
        [(0.30 + i * 0.005, i % 2 == 0) for i in range(20)] +
        [(0.50 + i * 0.005, True) for i in range(40)]
    )
    t = calibrate(samples)
    assert _precision_at(samples, t["low"]) >= 0.45
    assert _precision_at(samples, t["mid"]) >= 0.6
    assert _precision_at(samples, t["high"]) >= 0.8


def test_calibration_falls_back_to_defaults_on_empty_input():
    from tools.calibrate_confidence import calibrate

    t = calibrate([])
    assert t == {"low": 0.20, "mid": 0.35, "high": 0.55, "n_samples": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_threshold_calibration.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Write the calibration tool**

`tools/calibrate_confidence.py`:

```python
#!/usr/bin/env python3
"""Phase β: calibrate ConfidenceThresholds from LLM-judge data.

Reads an LLM-judge precision report (output of bench_llm_rejudge.py)
plus the source precision report (which carries per-result raw axis
scores), aligns them, and emits low/mid/high cuts where the empirical
precision-at-confidence first crosses 0.5 / 0.7 / 0.85.

Usage:
    .venv/bin/python tools/calibrate_confidence.py \\
        --judged benchmarks/results/precision_20260528_rejudge_input_llm_rejudge_k5.json \\
        --raw    benchmarks/results/precision_20260528_rejudge_input.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _precision_at(samples, threshold: float) -> float:
    above = [r for s, r in samples if s >= threshold]
    if not above:
        return 0.0
    return sum(1 for r in above if r) / len(above)


def calibrate(samples):
    """Fit low/mid/high to precision targets 0.5/0.7/0.85."""
    if not samples:
        return {"low": 0.20, "mid": 0.35, "high": 0.55, "n_samples": 0}

    sorted_scores = sorted({round(s, 3) for s, _ in samples})
    targets = {"low": 0.5, "mid": 0.7, "high": 0.85}
    defaults = {"low": 0.20, "mid": 0.35, "high": 0.55}

    result = {"n_samples": len(samples)}
    for level, target in targets.items():
        chosen = defaults[level]
        for cand in sorted_scores:
            if _precision_at(samples, cand) >= target:
                chosen = cand
                break
        result[level] = round(float(chosen), 3)

    # Monotonicity guard.
    result["mid"] = max(result["mid"], result["low"])
    result["high"] = max(result["high"], result["mid"])
    return result


def _load_samples(judged_path: Path, raw_path: Path):
    judged_report = json.loads(judged_path.read_text(encoding="utf-8"))
    raw_report = json.loads(raw_path.read_text(encoding="utf-8"))

    per_query_raw = raw_report.get("axes", {}).get("triaxis", {}).get("per_query", [])
    judged_by_query = {}
    for q in judged_report.get("per_query", []):
        key = (q.get("query") or "").strip()
        judged_by_query[key] = {str(k): str(v) for k, v in (q.get("judged") or {}).items()}

    samples = []
    for q in per_query_raw:
        key = (q.get("query") or "").strip()
        judged = judged_by_query.get(key, {})
        ranked = q.get("ranked_ids") or []
        for rank, fid in enumerate(ranked[:5]):
            verdict = judged.get(str(fid))
            if verdict not in ("yes", "no"):
                continue
            # Use rank-position proxy until per-result raw scores
            # land in the bench JSON (follow-up).
            score = max(0.0, 1.0 - rank / 5.0)
            samples.append((score, verdict == "yes"))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    samples = _load_samples(args.judged, args.raw)
    thresholds = calibrate(samples)

    print(f"Samples aligned: {thresholds['n_samples']}")
    print(f"  low  = {thresholds['low']}  (target precision 0.5)")
    print(f"  mid  = {thresholds['mid']}  (target precision 0.7)")
    print(f"  high = {thresholds['high']}  (target precision 0.85)")

    if args.output:
        args.output.write_text(
            json.dumps(thresholds, indent=2), encoding="utf-8",
        )
        print(f"  written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify the tool runs on existing data**

Run:
```bash
.venv/bin/python -m pytest tests/test_threshold_calibration.py -q
.venv/bin/python tools/calibrate_confidence.py \
    --judged benchmarks/results/precision_20260528_rejudge_input_llm_rejudge_k5.json \
    --raw    benchmarks/results/precision_20260528_rejudge_input.json \
    --output benchmarks/results/confidence_thresholds_20260528.json
```
Capture the printed `low/mid/high` values.

- [ ] **Step 5: Update ConfidenceThresholds defaults**

In `backend/search/confidence.py`, replace the literal defaults with the calibrated values from step 4 (the values from the printed output). Add a comment line above the dataclass that records the calibration date and source file. Use the exact numbers from the tool — if the tool prints `low=0.6, mid=0.6, high=0.6` (e.g., all queries match at rank 1 with score 1.0 → precision 100% → first cand 0.6 hits every target), use those numbers verbatim.

Then update `tests/test_search_confidence.py` boundary values to match the new defaults: in `test_classify_uses_inclusive_lower_bounds`, the parametrized score values should bracket the new thresholds. If the new low is 0.6, use 0.59 → EMPTY, 0.60 → LOW; if new high is 0.6, use 0.59 → MEDIUM, 0.60 → HIGH, etc. Keep the structural test (inclusive lower bound) but use the new numeric markers.

- [ ] **Step 6: Verify regressions**

Run:
```bash
.venv/bin/python -m pytest tests/test_search_confidence.py \
                            tests/test_search_response_shape.py \
                            tests/test_threshold_calibration.py -q
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tools/calibrate_confidence.py tests/test_threshold_calibration.py \
        backend/search/confidence.py tests/test_search_confidence.py \
        benchmarks/results/confidence_thresholds_20260528.json
git commit -m "feat: calibrate ConfidenceThresholds from LLM-judge data (phase β)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task β3: Dominant-axis badge on result cards

**Files:**
- Modify: `frontend/src/components/SearchPanel.jsx`
- Modify: `frontend/src/i18n/locales/ko-KR.json`
- Modify: `frontend/src/i18n/locales/en-US.json`

- [ ] **Step 1: Add helpers**

In `SearchPanel.jsx`, near the existing `confidenceBadgeClass` (added in Phase A), add:

```jsx
function dominantAxis(result) {
  const v = Number(result.vector_score ?? 0);
  const m = Number(result.text_vec_score ?? 0);
  const f = Number(result.text_score ?? 0) > 0 ? 0.4 : 0;
  const scores = [['vv', v], ['mv', m], ['fts', f]];
  scores.sort((a, b) => b[1] - a[1]);
  if (scores[0][1] <= 0) return null;
  return scores[0][0];
}

function axisBadgeClass(axis) {
  return {
    vv:  'bg-blue-900/30 text-blue-300 border border-blue-700/40 px-1.5 py-0.5 rounded text-[10px]',
    mv:  'bg-purple-900/30 text-purple-300 border border-purple-700/40 px-1.5 py-0.5 rounded text-[10px]',
    fts: 'bg-amber-900/30 text-amber-300 border border-amber-700/40 px-1.5 py-0.5 rounded text-[10px]',
  }[axis] || '';
}
```

- [ ] **Step 2: Render the badge on each card**

In `SearchResultCard` (search for `SearchResultCard` in the same file), add the badge near where score/file metadata is shown:

```jsx
{dominantAxis(result) && (
  <span className={axisBadgeClass(dominantAxis(result))}>
    {t(`search.axis_${dominantAxis(result)}`)}
  </span>
)}
```

- [ ] **Step 3: i18n keys**

`ko-KR.json` (append before closing brace):

```
"search.axis_vv": "VV 강함",
"search.axis_mv": "의미 일치",
"search.axis_fts": "키워드 일치",
```

`en-US.json`:

```
"search.axis_vv": "Strong visual match",
"search.axis_mv": "Strong meaning match",
"search.axis_fts": "Keyword match",
```

- [ ] **Step 4: Build + verify**

Run: `cd frontend && npm run build`
Expected: `built in <time>`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SearchPanel.jsx \
        frontend/src/i18n/locales/ko-KR.json \
        frontend/src/i18n/locales/en-US.json
git commit -m "feat: dominant-axis badge on search result cards (phase β)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase γ — Feedback Loop End-to-End

### Task γ1: Discoverable "관련 없음" button

**Files:**
- Modify: `frontend/src/components/SearchPanel.jsx`

- [ ] **Step 1: Locate the hover-only button**

```bash
grep -n "mark_irrelevant\|opacity-0 group-hover" frontend/src/components/SearchPanel.jsx
```

- [ ] **Step 2: Strip hover-only classes**

Find the feedback button JSX (uses `t('search.mark_irrelevant_short')`). Replace its className:

```jsx
// Before
className="opacity-0 group-hover:opacity-100 ... text-xs text-gray-400 hover:text-red-300"

// After
className="text-xs text-gray-400 hover:text-red-300 px-2 py-1 rounded border border-gray-700/40"
```

The button now occupies a stable footer slot on every card.

- [ ] **Step 3: Build + verify**

Run: `cd frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/SearchPanel.jsx
git commit -m "feat: make 'mark irrelevant' button always visible (phase γ)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task γ2: Admin search-feedback summary endpoint

**Files:**
- Create: `backend/server/routers/feedback_dashboard.py`
- Modify: `backend/server/app.py`
- Test: `tests/test_feedback_dashboard.py`

- [ ] **Step 1: Write the failing test**

`tests/test_feedback_dashboard.py`:

```python
"""Phase γ: admin search-feedback summary."""
from __future__ import annotations

import sqlite3
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.modules.setdefault(
    "jwt",
    types.SimpleNamespace(
        ExpiredSignatureError=Exception,
        InvalidTokenError=Exception,
        decode=lambda *a, **k: {},
        encode=lambda *a, **k: "",
    ),
)

from backend.server import deps  # noqa: E402


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
    from backend.server.routers import feedback_dashboard

    app = FastAPI()
    app.include_router(feedback_dashboard.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db
    app.dependency_overrides[deps.require_admin] = lambda: {
        "id": 1, "username": "admin", "role": "admin",
    }
    return TestClient(app)


def test_summary_empty_when_no_feedback():
    db = _db()
    client = _client(db)
    resp = client.get("/api/v1/admin/search-feedback/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_30d"] == 0
    assert body["top_files"] == []
    assert body["top_queries"] == []


def test_summary_groups_top_files_and_queries():
    db = _db()
    rows = [
        ("cat",  10, "irrelevant"),
        ("cat",  10, "irrelevant"),
        ("cat",  11, "irrelevant"),
        ("dog",  10, "irrelevant"),
        ("bird", 22, "irrelevant"),
    ]
    db.conn.executemany(
        "INSERT INTO search_feedback (query, file_id, label, user_id) VALUES (?, ?, ?, 1)",
        rows,
    )
    db.conn.commit()
    client = _client(db)
    body = client.get("/api/v1/admin/search-feedback/summary").json()
    assert body["total_30d"] == 5
    assert body["top_files"][0] == {"file_id": 10, "count": 3}
    assert body["top_queries"][0] == {"query": "cat", "count": 3}


def test_summary_requires_admin():
    from backend.server.routers import feedback_dashboard
    from fastapi import HTTPException

    db = _db()
    app = FastAPI()
    app.include_router(feedback_dashboard.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db_safe] = lambda: db

    def deny():
        raise HTTPException(status_code=403, detail="admin only")

    app.dependency_overrides[deps.require_admin] = deny
    test_client = TestClient(app)
    resp = test_client.get("/api/v1/admin/search-feedback/summary")
    assert resp.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_feedback_dashboard.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement the router**

`backend/server/routers/feedback_dashboard.py`:

```python
"""Phase γ: admin-visible aggregation of search_feedback."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db_safe, require_admin

router = APIRouter(tags=["search-feedback-admin"])


@router.get("/admin/search-feedback/summary")
def search_feedback_summary(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    cur = db.conn.cursor()

    total_30d = cur.execute(
        """SELECT COUNT(*) FROM search_feedback
           WHERE datetime(created_at) >= datetime('now', '-30 days')"""
    ).fetchone()[0]

    top_files_rows = cur.execute(
        """SELECT file_id, COUNT(*) AS n FROM search_feedback
           WHERE datetime(created_at) >= datetime('now', '-30 days')
           GROUP BY file_id
           ORDER BY n DESC, file_id ASC
           LIMIT 20"""
    ).fetchall()
    top_files = [{"file_id": r[0], "count": r[1]} for r in top_files_rows]

    top_queries_rows = cur.execute(
        """SELECT query, COUNT(*) AS n FROM search_feedback
           WHERE datetime(created_at) >= datetime('now', '-30 days')
           GROUP BY query
           ORDER BY n DESC, query ASC
           LIMIT 10"""
    ).fetchall()
    top_queries = [{"query": r[0], "count": r[1]} for r in top_queries_rows]

    return {
        "total_30d": int(total_30d or 0),
        "top_files": top_files,
        "top_queries": top_queries,
    }
```

- [ ] **Step 4: Register in app.py**

In `backend/server/app.py`, alongside other router imports:

```python
from backend.server.routers.feedback_dashboard import router as feedback_dashboard_router
```

In the include block:

```python
app.include_router(feedback_dashboard_router, prefix="/api/v1")
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -m pytest tests/test_feedback_dashboard.py -q`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/server/routers/feedback_dashboard.py backend/server/app.py \
        tests/test_feedback_dashboard.py
git commit -m "feat: admin endpoint for search_feedback summary (phase γ)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task γ3: Admin SearchFeedbackPanel + tab

**Files:**
- Create: `frontend/src/components/admin/SearchFeedbackPanel.jsx`
- Modify: `frontend/src/pages/AdminPage.jsx`
- Modify: `frontend/src/api/admin.js`
- Modify: `frontend/src/i18n/locales/ko-KR.json`
- Modify: `frontend/src/i18n/locales/en-US.json`

- [ ] **Step 1: Add API client function**

In `frontend/src/api/admin.js`, append:

```js
export async function getSearchFeedbackSummary() {
  return apiClient.get('/api/v1/admin/search-feedback/summary');
}
```

- [ ] **Step 2: Create the panel**

`frontend/src/components/admin/SearchFeedbackPanel.jsx`:

```jsx
/**
 * SearchFeedbackPanel — Phase γ of perceived search quality.
 */

import { useCallback, useEffect, useState } from 'react';
import { useLocale } from '../../i18n';
import { getSearchFeedbackSummary } from '../../api/admin';

export default function SearchFeedbackPanel() {
  const { t } = useLocale();
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const resp = await getSearchFeedbackSummary();
      setSummary(resp.data ?? resp);
      setError(null);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'load failed');
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-white">{t('admin.search_feedback_title')}</h2>
      {error && <p className="text-xs text-red-400">{error}</p>}
      {summary && (
        <>
          <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <p className="text-xs text-gray-400">{t('admin.search_feedback_total_30d')}</p>
            <p className="text-2xl text-white font-mono">{summary.total_30d}</p>
          </section>

          <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-white mb-2">
              {t('admin.search_feedback_top_files')}
            </h3>
            {summary.top_files.length === 0 ? (
              <p className="text-xs text-gray-500">{t('admin.search_feedback_empty')}</p>
            ) : (
              <table className="w-full text-xs text-gray-300">
                <thead>
                  <tr className="text-gray-500">
                    <th className="text-left py-1">file_id</th>
                    <th className="text-right py-1">{t('admin.search_feedback_count')}</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.top_files.map(row => (
                    <tr key={row.file_id} className="border-t border-gray-700/40">
                      <td className="py-1 font-mono">{row.file_id}</td>
                      <td className="py-1 text-right font-mono">{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-white mb-2">
              {t('admin.search_feedback_top_queries')}
            </h3>
            {summary.top_queries.length === 0 ? (
              <p className="text-xs text-gray-500">{t('admin.search_feedback_empty')}</p>
            ) : (
              <ul className="text-xs text-gray-300 space-y-1">
                {summary.top_queries.map(q => (
                  <li key={q.query} className="flex justify-between border-b border-gray-700/40 py-1">
                    <span className="truncate">{q.query}</span>
                    <span className="font-mono text-gray-400 ml-2">{q.count}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Register as a tab in AdminPage**

In `frontend/src/pages/AdminPage.jsx`, add the import alongside the others:

```jsx
import { UserCheck, Tag, Clock, Wrench, Network, MessageSquare } from 'lucide-react';
import SearchFeedbackPanel from '../components/admin/SearchFeedbackPanel';
```

Add to the `tabs` array (after `connection`):

```jsx
{ id: 'feedback', label: t('admin.tab_search_feedback'), icon: MessageSquare },
```

Add a render branch in the content area:

```jsx
{activeTab === 'feedback' && <SearchFeedbackPanel />}
```

- [ ] **Step 4: i18n**

In `frontend/src/i18n/locales/ko-KR.json`, append before closing brace:

```
"admin.tab_search_feedback": "정정 피드백",
"admin.search_feedback_title": "검색 정정 누적",
"admin.search_feedback_total_30d": "최근 30일 '관련 없음' 합계",
"admin.search_feedback_top_files": "가장 많이 정정된 파일 (top 20)",
"admin.search_feedback_top_queries": "가장 많이 정정된 쿼리 (top 10)",
"admin.search_feedback_count": "건수",
"admin.search_feedback_empty": "아직 누적된 피드백이 없습니다.",
```

In `frontend/src/i18n/locales/en-US.json`:

```
"admin.tab_search_feedback": "Feedback",
"admin.search_feedback_title": "Search-result corrections",
"admin.search_feedback_total_30d": "'Irrelevant' labels in last 30 days",
"admin.search_feedback_top_files": "Most-flagged files (top 20)",
"admin.search_feedback_top_queries": "Most-flagged queries (top 10)",
"admin.search_feedback_count": "Count",
"admin.search_feedback_empty": "No feedback accumulated yet.",
```

- [ ] **Step 5: Build + verify**

Run: `cd frontend && npm run build`
Expected: `built in <time>`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/admin/SearchFeedbackPanel.jsx \
        frontend/src/pages/AdminPage.jsx frontend/src/api/admin.js \
        frontend/src/i18n/locales/ko-KR.json frontend/src/i18n/locales/en-US.json
git commit -m "feat: admin panel for search feedback summary (phase γ)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task γ4: Auto user_tags from accumulated feedback

**Files:**
- Create: `backend/server/jobs/auto_user_tags.py`
- Modify: `backend/server/app.py`
- Test: `tests/test_auto_user_tags.py`

- [ ] **Step 1: Write the failing test**

`tests/test_auto_user_tags.py`:

```python
"""Phase γ: auto-update user_tags when a file is repeatedly flagged."""
from __future__ import annotations

import sqlite3
import types


def _db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            user_tags TEXT
        );
        CREATE TABLE search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            user_id INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """
    )
    conn.executemany(
        "INSERT INTO files (id, user_tags) VALUES (?, ?)",
        [(1, ""), (2, "existing"), (3, "")],
    )
    conn.commit()
    return types.SimpleNamespace(conn=conn)


def _seed(db, file_id, n):
    for _ in range(n):
        db.conn.execute(
            "INSERT INTO search_feedback (query, file_id, label, user_id) "
            "VALUES ('q', ?, 'irrelevant', 1)",
            (file_id,),
        )
    db.conn.commit()


def test_no_op_below_threshold():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=1, n=2)
    n_updated = apply_feedback_to_user_tags(db, threshold=3)
    assert n_updated == 0
    assert db.conn.execute("SELECT user_tags FROM files WHERE id=1").fetchone()[0] == ""


def test_adds_low_relevance_tag_when_threshold_reached():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=1, n=3)
    n_updated = apply_feedback_to_user_tags(db, threshold=3)
    assert n_updated == 1
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=1").fetchone()[0]
    assert "low-relevance" in row


def test_does_not_duplicate_tag_on_re_run():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=1, n=3)
    apply_feedback_to_user_tags(db, threshold=3)
    second = apply_feedback_to_user_tags(db, threshold=3)
    assert second == 0
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=1").fetchone()[0]
    assert row.count("low-relevance") == 1


def test_preserves_existing_user_tags():
    from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags

    db = _db()
    _seed(db, file_id=2, n=4)
    apply_feedback_to_user_tags(db, threshold=3)
    row = db.conn.execute("SELECT user_tags FROM files WHERE id=2").fetchone()[0]
    assert "existing" in row
    assert "low-relevance" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_auto_user_tags.py -q`
Expected: `ImportError`.

- [ ] **Step 3: Implement the job**

`backend/server/jobs/auto_user_tags.py`:

```python
"""Phase γ: when a file accumulates N 'irrelevant' labels, add a
low-relevance user_tag. Idempotent. Demotion is handled downstream
(soft demotion from Phase D plus the user_tags now visible to the
ranking layer).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LOW_RELEVANCE_TAG = "low-relevance"


def _split_tags(text):
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def _join_tags(tags):
    return ", ".join(tags)


def apply_feedback_to_user_tags(db, *, threshold: int = 3) -> int:
    cur = db.conn.cursor()
    rows = cur.execute(
        """SELECT file_id, COUNT(*) FROM search_feedback
           WHERE label = 'irrelevant'
           GROUP BY file_id
           HAVING COUNT(*) >= ?""",
        (int(threshold),),
    ).fetchall()

    updated = 0
    for file_id, _count in rows:
        existing = cur.execute(
            "SELECT user_tags FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        if existing is None:
            continue
        tags = _split_tags(existing[0])
        if LOW_RELEVANCE_TAG in tags:
            continue
        tags.append(LOW_RELEVANCE_TAG)
        cur.execute(
            "UPDATE files SET user_tags = ? WHERE id = ?",
            (_join_tags(tags), file_id),
        )
        updated += 1
    db.conn.commit()
    if updated:
        logger.info("auto_user_tags: tagged %d file(s) as low-relevance", updated)
    return updated
```

- [ ] **Step 4: Schedule via startup hook**

In `backend/server/app.py`, locate `_activate_server`. At the end of that function (after the existing relay connector hook from Phase 5), append:

```python
    try:
        from backend.server.jobs.auto_user_tags import apply_feedback_to_user_tags
        n_tagged = apply_feedback_to_user_tags(db, threshold=3)
        logger.info(f"auto_user_tags applied: {n_tagged} file(s) tagged")
    except Exception as e:
        logger.warning(f"auto_user_tags failed: {e}")
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -m pytest tests/test_auto_user_tags.py -q`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add backend/server/jobs/auto_user_tags.py backend/server/app.py \
        tests/test_auto_user_tags.py
git commit -m "feat: auto-tag low-relevance files from accumulated feedback (phase γ)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Done Criteria

- Cross-encoder rerank fires on every triaxis search where `len(merged) > 1` and the reranker model is loaded; observable via `diag.cross_encoder_rerank`.
- Multi-element AND verification applies a per-missing-element penalty; observable via `diag.element_verification` and `element_match_count` / `element_miss_count` on each result.
- `ConfidenceThresholds` defaults match values from `tools/calibrate_confidence.py`; the file lists the calibration date in a comment.
- Frontend result cards show a dominant-axis badge ("VV 강함" / "MV 의미 일치" / "FTS 키워드 일치").
- "관련 없음" button is always visible (no hover required), and clicks land in `search_feedback`.
- Admin Panel has "정정 피드백" tab showing 30-day total + top files + top queries.
- `apply_feedback_to_user_tags` runs at activation; files with ≥3 'irrelevant' labels get `low-relevance` user_tag.
- New tests pass: `pytest tests/test_cross_encoder.py tests/test_element_and_verification.py tests/test_threshold_calibration.py tests/test_feedback_dashboard.py tests/test_auto_user_tags.py`.
- `cd frontend && npm run build` succeeds.

## Measurement (after merge)

Re-run the LLM-judge bench to quantify ranking lift:

```bash
.venv/bin/python tools/bench_precision.py --count 30 --judge-mode keyword \
    --output benchmarks/results/precision_post_rerank.json
.venv/bin/python tools/bench_llm_rejudge.py \
    benchmarks/results/precision_post_rerank.json --top-k 5 --backend auto
```

Compare `P@5_llm` against the pre-rerank baseline (0.633 from 2026-05-28). Expectation: lift to 0.70+ from cross-encoder rerank + element verification together.

## Out of Scope (Future Plans)

- ANN index (sqlite-vec full-scan handles 17K rows; revisit at 100K+).
- Latency reduction (parallel axis execution, query embedding cache) — current 5.5s acceptable per user statement.
- Per-axis weighting in RRF (separate plan).
- LLM-driven query expansion when `confidence=empty`.
- Bench live-confidence wiring (still uses `"medium"` placeholder in `bench_precision.py`).
