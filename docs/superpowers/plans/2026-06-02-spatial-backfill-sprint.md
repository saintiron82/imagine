# Spatial Backfill Sprint (P10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `file_objects` / `file_spatial_relations` / `file_depth_layers` from current 0.07% / 0% / 0% coverage to a measurable level (≥30% of MC-captioned files), build a spatial-intent frozen queryset, and run an A/B that proves (or disproves) the value of the spatial search axis on real data. Push past the P@5 SLM-judge ceiling of 0.673 if possible.

**Architecture:** No new pipeline code. The 2026-05-17 hardening shipped the full path: `PhaseRunner.run_vision()` → `analyzer.classify_and_analyze()` → `_refresh_fts_row()` → three `_replace_*()` DB writers. CLI: `tools/backfill_spatial_processing.py`. This sprint runs that infrastructure in stages, gates each stage on quality, and measures the effect.

**Tech Stack:** Existing — Python · sqlite-vec · VLM analyzer · `tools/bench_precision.py` · `tools/bench_llm_rejudge.py` · Qwen3.5-9B SLM-judge.

---

## Context

Sprint 3 closed the search-engine side at P@5 SLM-judge = 0.673 on `frozen_30_v1`, a queryset that contains **0 spatial-positioning queries** (no "왼쪽/오른쪽/위에/뒤쪽" tokens). The spatial-intent code path (RRF spatial weight 0.50 when `query_type=="spatial"`, S3.2 boost) is fully wired but never exercised in production search — only 12 of 17,726 files have any spatial data.

`docs/state_report_2026-05-31.md` and `docs/imagine_operations_control_plane_2026-05-31.md` both identify spatial backfill as the **last remaining lever** for perceived-quality lift inside the current architecture. The user explicitly rejected model training and accepts that this is the measurable next step.

The 2026-05-17 hardening plan already built every piece of code needed — audit, backfill candidate selector, normalization, DB writers, 9 test files. P10 just runs it.

## Scope

**In:**
- Baseline audit + budget estimate
- Staged backfill: 50 samples (sanity) → 500 (bench-sized) → 5,000 or more if both prior stages pass
- Spatial-intent frozen queryset construction (`frozen_spatial_30_v1`)
- A/B measurement with `IMAGINE_BENCH_DISABLE_SPATIAL=1` toggle
- Decision document: continue to full backfill, stop, or pivot

**Out (separate plans / later):**
- Pipeline code changes — re-use everything 2026-05-17 shipped
- Spatial query understanding LLM improvements
- New depth-layer / relation extractors
- Re-running on already-processed files (the 12 existing) unless quality audit flags them

## Convention reused from existing code

- `tools/audit_spatial_processing.py` — read-only coverage audit
- `tools/backfill_spatial_processing.py` — candidate selector + `--execute`
- `tools/bench_precision.py` — `--queryset` / `--save-queryset` for frozen A/B
- `tools/bench_llm_rejudge.py` — SLM-judge re-evaluation
- `backend/pipeline/phase_runner.py:run_vision` — pipeline entry
- `backend/db/sqlite_client.py:_replace_file_objects/relations/depth_layers` — DB writers

## Stage gates

Each stage must pass its gate before the next runs. Skipping a gate is a project failure, not a shortcut.

| Stage | Gate |
|-------|------|
| S1 baseline audit | Coverage numbers match prior state-report figures (within 1%) |
| S2 50-sample sanity | ≥80% of processed files yield at least 1 row in `file_objects` |
| S3 500-sample bench | Coverage + processing time + per-file cost are documented |
| S4 spatial queryset | ≥20 of 30 queries have at least 1 known-relevant file from the populated 500 |
| S5 A/B bench | Statistically meaningful delta between spatial-on and spatial-off measured by SLM-judge |
| S6 scale decision | Explicit go/no-go based on S5 lift × per-file VLM cost |

---

## Stage 1: Baseline audit + budget estimate

**Files:**
- Read: `tools/audit_spatial_processing.py`
- Read-only DB query

- [ ] **Step 1: Run the audit**

```bash
.venv/bin/python tools/audit_spatial_processing.py --db imageparser.db
```

Expected output: coverage stats per evidence table. Confirm matches state report (12 files / 0% / 0%).

- [ ] **Step 2: Capture baseline VLM cost estimate**

Pick 1 file the analyzer hasn't seen yet. Time the full vision phase end-to-end:

```bash
.venv/bin/python -c "
import time
from backend.pipeline.phase_runner import PhaseRunner
# select 1 candidate via the backfill planner dry-run
"
```

Concretely: run the backfill planner in dry-run mode for `--limit 1` and execute the resulting plan synchronously, measuring wall time and noting which analyzer adapter was used.

- [ ] **Step 3: Record baseline**

Create `benchmarks/spatial_backfill/baseline.md`:

```markdown
# Spatial backfill — baseline (2026-06-02)

| Table | Files w/ data | Coverage |
|-------|---:|---:|
| file_objects | 12 / 17,726 | 0.07% |
| file_spatial_relations | 0 / 17,726 | 0.00% |
| file_depth_layers | 0 / 17,726 | 0.00% |

VLM cost: <wall-time-seconds> per file (adapter: <name>, model: <name>)

Projected costs:
- 500 files: <estimate> minutes
- 5,000 files: <estimate> hours
- 17,000 files: <estimate>

Source files with existing spatial data (12 file_ids): <list-or-link>
```

- [ ] **Step 4: Commit baseline**

```bash
git add benchmarks/spatial_backfill/baseline.md
git commit -m "spatial: baseline audit + VLM cost estimate (stage 1)"
```

---

## Stage 2: 50-sample sanity backfill

**Files:**
- Create: `benchmarks/spatial_backfill/s2-sanity.md`

- [ ] **Step 1: Pick 50 candidates**

```bash
.venv/bin/python tools/backfill_spatial_processing.py \
    --db imageparser.db \
    --reason missing_objects \
    --limit 50 \
    --dry-run \
    --output benchmarks/spatial_backfill/s2-candidates.json
```

If `--output` flag doesn't exist in the current tool, redirect stdout to the file instead.

- [ ] **Step 2: Execute the 50-sample backfill**

```bash
.venv/bin/python tools/backfill_spatial_processing.py \
    --db imageparser.db \
    --reason missing_objects \
    --limit 50 \
    --execute
```

This may print "run this ingest command separately" — if so, copy that command and run it. Capture full stdout to `benchmarks/spatial_backfill/s2-run.log`.

- [ ] **Step 3: Verify quality gate**

```bash
.venv/bin/python -c "
import sqlite3, json
db = sqlite3.connect('imageparser.db')
cur = db.cursor()
# count files with at least 1 object row now
n = cur.execute('SELECT COUNT(DISTINCT file_id) FROM file_objects').fetchone()[0]
print(f'files_with_objects={n}')
# breakdown by row count
rows = cur.execute('SELECT COUNT(*) FROM (SELECT file_id, COUNT(*) c FROM file_objects GROUP BY file_id HAVING c >= 1)').fetchone()
print(f'files_with_at_least_one={rows[0]}')
"
```

Pass condition: `files_with_at_least_one >= 50 - 10` (≥80% yield from the 50 processed; the original 12 are excluded from the candidate pool).

If fail: stop. Diagnose why the analyzer is returning empty results — adapter config, model availability, image format edge cases.

- [ ] **Step 4: Sample-inspect 5 random rows**

```bash
.venv/bin/python -c "
import sqlite3, random
db = sqlite3.connect('imageparser.db')
ids = [r[0] for r in db.execute('SELECT DISTINCT file_id FROM file_objects ORDER BY id DESC LIMIT 50').fetchall()]
sample = random.sample(ids, k=min(5, len(ids)))
for fid in sample:
    print('=== file_id', fid)
    for r in db.execute('SELECT name, ko_name, primary_location, locations FROM file_objects WHERE file_id=?', (fid,)).fetchall():
        print(' ', r)
"
```

Eyeball: do object names and locations look plausible? Record 5 examples in `benchmarks/spatial_backfill/s2-sanity.md` with a one-line judgment (looks-good / partially-wrong / nonsense).

- [ ] **Step 5: Commit stage 2**

```bash
git add benchmarks/spatial_backfill/s2-*.md benchmarks/spatial_backfill/s2-*.log benchmarks/spatial_backfill/s2-candidates.json
git commit -m "spatial: 50-sample sanity backfill (stage 2)"
```

---

## Stage 3: 500-sample bench-sized backfill

Gate: Stage 2 passed (≥40 of 50 files yielded ≥1 object).

**Files:**
- Create: `benchmarks/spatial_backfill/s3-bench.md`

- [ ] **Step 1: Execute 500-sample backfill**

```bash
.venv/bin/python tools/backfill_spatial_processing.py \
    --db imageparser.db \
    --reason missing_objects \
    --limit 500 \
    --execute \
    2>&1 | tee benchmarks/spatial_backfill/s3-run.log
```

Time this. Expected wall time ≈ 500 × baseline-cost-per-file from Stage 1.

- [ ] **Step 2: Quality summary**

```bash
.venv/bin/python -c "
import sqlite3
db = sqlite3.connect('imageparser.db')
cur = db.cursor()
total = cur.execute('SELECT COUNT(*) FROM files').fetchone()[0]
for tbl in ('file_objects','file_spatial_relations','file_depth_layers'):
    n = cur.execute(f'SELECT COUNT(DISTINCT file_id) FROM {tbl}').fetchone()[0]
    print(f'{tbl}: {n} / {total} ({100*n/total:.2f}%)')
# objects per file distribution
rows = cur.execute('SELECT c, COUNT(*) FROM (SELECT file_id, COUNT(*) c FROM file_objects GROUP BY file_id) GROUP BY c ORDER BY c').fetchall()
print('objects/file:', rows[:10])
"
```

Write to `benchmarks/spatial_backfill/s3-bench.md`: coverage after stage, average objects/file, average relations/file, average depth_layers/file, total wall time.

- [ ] **Step 3: Commit stage 3**

```bash
git add benchmarks/spatial_backfill/s3-*.md benchmarks/spatial_backfill/s3-*.log
git commit -m "spatial: 500-sample bench-sized backfill (stage 3)"
```

---

## Stage 4: Spatial-intent frozen queryset

**Files:**
- Create: `benchmarks/querysets/frozen_spatial_30_v1.json`
- Create: `benchmarks/spatial_backfill/s4-queryset.md`

- [ ] **Step 1: Generate candidate queries from populated data**

Build 30 queries that:
1. Target objects + locations that actually exist in `file_objects` after Stage 3.
2. Use spatial language (왼쪽/오른쪽/위에/아래/뒤쪽/전경/배경 etc.).
3. Have at least 1 known-relevant file_id as ground truth.

Example query types to mix:
- Single-object location: "오른쪽에 달이 있는 이미지"
- Object pair: "왼쪽에 산 오른쪽에 강이 있는 이미지"
- Depth: "전경에 인물이 있는 이미지"
- Relation: "테이블 위에 컵이 있는 이미지"

Generator approach: query DB for files where `primary_location` is set, group by (object × location), pick 30 (object, location) pairs each represented by ≥2 files, generate the Korean phrasing, mark those file_ids as GT.

Save as `frozen_spatial_30_v1.json` in the same shape as `frozen_30_v1.json` (each row: `{"query": str, "gt_file_ids": [int]}`).

- [ ] **Step 2: Sanity-check the queryset**

```bash
.venv/bin/python -c "
import json
q = json.load(open('benchmarks/querysets/frozen_spatial_30_v1.json'))
assert len(q) == 30
assert all('gt_file_ids' in r and len(r['gt_file_ids']) >= 1 for r in q)
spatial_words = ['왼쪽','오른쪽','위에','아래','전경','배경','뒤','앞']
matched = sum(1 for r in q if any(w in r['query'] for w in spatial_words))
print(f'queries with spatial language: {matched}/30')
assert matched >= 25, f'too few spatial-language queries: {matched}'
"
```

Pass condition: ≥25 of 30 queries contain explicit spatial language. If fail: regenerate.

- [ ] **Step 3: Commit queryset**

```bash
git add benchmarks/querysets/frozen_spatial_30_v1.json benchmarks/spatial_backfill/s4-queryset.md
git commit -m "spatial: frozen spatial-intent queryset v1 (stage 4)"
```

---

## Stage 5: A/B measurement

**Files:**
- Create: `benchmarks/spatial_backfill/s5-ab.md`

- [ ] **Step 1: Run bench with spatial axis ON**

```bash
.venv/bin/python tools/bench_precision.py \
    --queryset benchmarks/querysets/frozen_spatial_30_v1.json \
    --output benchmarks/results/precision_20260602_spatial_on.json
```

- [ ] **Step 2: Run bench with spatial axis OFF**

```bash
IMAGINE_BENCH_DISABLE_SPATIAL=1 .venv/bin/python tools/bench_precision.py \
    --queryset benchmarks/querysets/frozen_spatial_30_v1.json \
    --output benchmarks/results/precision_20260602_spatial_off.json
```

- [ ] **Step 3: SLM-judge both**

```bash
.venv/bin/python tools/bench_llm_rejudge.py \
    benchmarks/results/precision_20260602_spatial_on.json --top-k 5 --backend mlx
.venv/bin/python tools/bench_llm_rejudge.py \
    benchmarks/results/precision_20260602_spatial_off.json --top-k 5 --backend mlx
```

- [ ] **Step 4: Compare**

Compute deltas: P@5 keyword, P@5 SLM-judge, outcome distribution (found/missed/false_answer). Write `benchmarks/spatial_backfill/s5-ab.md`:

```markdown
# Spatial axis A/B — frozen_spatial_30_v1 (2026-06-02)

|  | spatial ON | spatial OFF | Δ |
|---|---:|---:|---:|
| P@5 keyword | x | x | x |
| P@5 SLM-judge | x | x | x |
| found / 30 | x | x | x |

Verdict: <"spatial axis lifts perceived quality by X%" or "no measurable effect" or "regressed">.
```

Pass condition: spatial-on **must beat** spatial-off on SLM-judge by ≥+0.05p AND keyword by ≥0p (no regression on either). If fail, the spatial axis as wired does not pay rent at this coverage level — stop and propose what to change.

- [ ] **Step 5: Commit measurement**

```bash
git add benchmarks/spatial_backfill/s5-ab.md \
        benchmarks/results/precision_20260602_spatial_on*.json \
        benchmarks/results/precision_20260602_spatial_off*.json
git commit -m "spatial: A/B measurement of spatial axis on populated 500 (stage 5)"
```

---

## Stage 6: Scale decision

**Files:**
- Create: `docs/state_report_spatial_backfill_2026-06-02.md` (or append to state report)

- [ ] **Step 1: Compute scale economics**

From Stage 1 cost-per-file and Stage 3 actual wall time, project the cost of backfilling all remaining 16,000+ files.

- [ ] **Step 2: Write the decision**

```markdown
# Spatial backfill — scale decision (2026-06-02)

## What 500-sample A/B showed
<copy verdict from s5-ab.md>

## Cost to scale
- Remaining files: ~16,000
- Wall time estimate: <hours>
- Disk: ~<MB> (Spatial rows are ~300 bytes each, ~2 rows/file expected)

## Recommendation
<go: full backfill | partial: <subset> | stop>

## If go
Run:
  .venv/bin/python tools/backfill_spatial_processing.py \
      --db imageparser.db --reason missing_objects \
      --limit 20000 --execute

## If stop
Document why the 500-sample lift did not justify scale.
```

- [ ] **Step 3: Commit decision**

```bash
git add docs/state_report_spatial_backfill_2026-06-02.md
git commit -m "spatial: scale decision after 500-sample A/B (stage 6)"
```

---

## Stage 7 (conditional): Full backfill

Run only if Stage 6 recommends "go".

- [ ] **Step 1: Execute full backfill**

```bash
.venv/bin/python tools/backfill_spatial_processing.py \
    --db imageparser.db --reason missing_objects \
    --limit 20000 --execute \
    2>&1 | tee benchmarks/spatial_backfill/s7-full-run.log
```

This may run for hours. Monitor periodically.

- [ ] **Step 2: Final A/B**

Re-run Stage 5 on the now-fully-populated database. Compare to Stage 5 numbers — coverage should multiply effect, otherwise the marginal returns are flat.

- [ ] **Step 3: Update state report**

Update `docs/state_report_2026-05-31.md` (or create dated successor) with the new spatial coverage numbers and the final A/B verdict. This is the document of record going forward.

- [ ] **Step 4: Commit and merge**

Branch: `feat/spatial-backfill-p10`. Standard merge flow via `superpowers:finishing-a-development-branch`.

---

## Completion criteria

P10 is complete when:

1. Stages 1–6 are committed on the feature branch (Stage 7 conditional).
2. `frozen_spatial_30_v1.json` exists and has 30 queries with ≥1 GT each.
3. `s5-ab.md` records a clear verdict on whether the spatial axis as currently wired lifts perceived quality.
4. `docs/state_report_spatial_backfill_2026-06-02.md` records the go/no-go decision.
5. If go: full backfill is executed and the new state numbers are documented.
6. Branch is merged to `main`.

The spatial code path will either be **validated** (axis lifts SLM-judge ≥+0.05p with positive coverage) or **disproven** (axis is no-op or regression even at 500-sample coverage). Either is a clean result — the goal of P10 is to know which.

## Risks

1. **VLM hallucinates locations** — generated `primary_location` fields don't match the actual image. Stage 2 sample inspection is the first check. Stage 5 SLM-judge is the second.
2. **Spatial queries underspecified** — the decomposer doesn't recognize spatial intent in 한국어 queries, so `query_type=="spatial"` never triggers. Check `_extract_spatial_intent` output during Stage 5 — if it never returns `active=True`, the axis won't activate regardless of data.
3. **Wall-time blowout** — Stage 3 takes longer than expected and blocks the session. Budget: stop after 2 hours of Stage 3 even if incomplete; partial data still permits an A/B.
4. **Stage 4 GT bias** — queries are auto-generated from populated objects, so spatial axis trivially wins. Mitigate by **also** running each query through the spatial-OFF path; SLM-judge is content-blind to which axis surfaced the result.
