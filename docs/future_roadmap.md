# Future Roadmap — Architecture & Optimization Plans

Consolidated from 5 planning documents (2026-02-16 ~ 2026-02-18).
These are **unimplemented designs** for future reference.

---

## 1. Distributed Architecture (Cloud Expansion)

### 1.1 Design Principles

- Common contract: Same `TaskSpec`, `ResultEnvelope`, state model across all deployment modes
- Idempotency: At-least-once delivery + idempotent upsert (no duplicate results)
- Stateless workers: Workers reference server state, not local state
- Storage separation: Inference path and DB commit path decoupled
- Multi-tenancy: `tenant_id` required across all API/queue/DB/storage layers

### 1.2 Data Contracts

**TaskSpec:**
- `schema_version`, `tenant_id`, `task_id`, `job_id`
- `resource_ref`: local_path | object_uri | content_hash
- `task_type`: vision | vv_embed | mv_embed
- `model_spec`: backend, model_id, revision, dtype, quantization
- `dedupe_key`: tenant_id + content_hash + model_revision + task_type
- `priority`, `deadline_at`, `retry_policy`

**ResultEnvelope:**
- `task_id`, `worker_id`, `attempt`
- `resource_fingerprint`: content_hash, mtime, size
- `outputs`: caption/tags/vector/metrics
- `telemetry`: latency, batch_size, gpu_mem
- `status`: success | partial | failed
- `result_checksum`: payload integrity hash

**State transitions:**
```
queued -> leased -> running -> reported -> persisted -> done
         failure: running -> failed -> retry_queued | dead_letter
         timeout: leased/running -> lease_expired -> retry_queued
```

### 1.3 Deployment Modes

| Mode | Topology | Queue | Storage |
|------|----------|-------|---------|
| L1: Single Machine | Server + worker in-process | SQLite queue table | SQLite local |
| L2: LAN Distributed | Server 1 + Worker N | SQLite queue + REST | NAS/shared folder |
| C1: Cloud/Hybrid | API + central queue + autoscale | PostgreSQL + Redis | S3/GCS + edge cache |

**Current status:** L1 and partial L2 implemented (Phase 4.5 worker system).

### 1.4 Model Parity Policy

- Server enforces `model_spec` per task
- Worker validates local manifest vs `model_revision/hash` before execution
- Mismatch → execution blocked + `model_mismatch` status
- Rollout: canary(5%) → ramp(25%) → ramp(50%) → full(100%)

### 1.5 Reliability

- Lease timeout + heartbeat for automatic task recovery
- Error-code-based retry separation (network / resource / model)
- Dead-letter queue for permanently failed tasks
- Observability: queue depth, lease age, task latency p50/p95/p99, worker failure rate
- Backpressure: throttle ingest at queue_depth > 50K, stop low-priority at > 100K

---

## 2. Data Pipeline Optimization

### 2.1 Current Issues (Code-Based Diagnosis)

| Issue | Severity | Status |
|-------|----------|--------|
| DB commit per call (no batching) | High | Unresolved |
| No async DB writer (inference + write coupled) | High | Unresolved |
| sqlite-vec load failure → fail-open | Medium | Unresolved |
| FTS vocabulary limited to metadata | Medium | Unresolved |
| PIL-only image decoding (no TurboJPEG/OpenCV) | Medium | Unresolved |

### 2.2 Target Architecture

```
Parse Queue (CPU) → Vision Queue (VLM) → Embed Queue (VV+MV) → DB Writer Queue
                                                                    ↓
                                                         Batch commit (500-1000)
                                                         Single dedicated writer
```

### 2.3 Priority Backlog

**P0 (Immediate):**
- sqlite-vec health gate: Block vector mode if extension not loaded
- DB writer queue + batch transaction (500-1000 per commit)
- WAL checkpoint control + ANALYZE auto-execution

**P1 (Short-term):**
- Async DB writer (Queue + bulk transaction + retry)
- Image decoding acceleration (TurboJPEG for JPEG, OpenCV/Pillow-SIMD benchmark)

**P2 (Mid-term):**
- FTS caption/tag vocabulary expansion (weight separation to prevent overfitting)
- Query-type dynamic weight learning (offline eval based)

**P3 (Advanced):**
- Query intent classification → automatic Structure axis weight adjustment
- Top-N reranker with cost ceiling

---

## 3. Search Quality Evaluation

### 3.1 Offline Benchmark System (6-week plan)

**Goal:** Automated quality/speed/cost verification comparing 3 engines (commercial, open-source, internal).

**Components:**
1. `dataset_builder` — Auto-generate evaluation query set (JSONL)
2. `label_builder` — Weak labels (click/dwell) + weekly manual review (100 samples)
3. `engine_adapters` — 3 engine adapters with common I/O contract
4. `benchmark_runner` — Same query set across all engines, metric calculation
5. `gate_evaluator` — Pass/fail decision, CI/CD integration
6. `report_generator` — Single Markdown report

**Metrics:**
- Primary: `nDCG@10`
- Secondary: `Recall@50`, `MRR@10`, `Precision@5`
- Guards: `p95 latency`, `error rate`, `cost_per_1k_queries`

**Deployment gates:**
- nDCG@10 drop > 1.0pp → block
- Recall@50 drop > 2.0pp → block
- p95 latency increase > 20% → block
- Error rate increase > 0.5pp → block

### 3.2 A/B Evaluation Framework

**Offline (Required):**
- Gold set: 300-500 queries (keyword 35%, semantic 35%, visual 20%, mixed 10%)
- Labeling: 2+ independent labelers, adjudication for disagreements (target κ ≥ 0.6)
- Statistical testing: Bootstrap 10,000 iterations, 95% CI

**Online (Optional, after offline pass):**
- Traffic split: 10% → 25% → 50%
- Metrics: time to first useful result, reformulation rate, session success rate
- Abort criteria: error spike, p95 threshold breach, significant metric degradation

**Go/No-Go criteria:**
- nDCG@10 improvement ≥ +8% (relative)
- Recall@20 no regression
- p95 latency degradation ≤ 20%
- Error rate no increase

### 3.3 Automation Schedule

| Schedule | Scope | Purpose | On Failure |
|----------|-------|---------|------------|
| PR Smoke | 50 queries | Obvious regression detection | PR check fail |
| Nightly Full | 500 queries | Quality trend tracking | Alert + auto-issue |
| Pre-release Gate | Latest full bench | Release blocking | Pipeline halt |

---

## 4. ECM: Evidence-Centric Model (Design Philosophy)

### 4.1 Problem

Current architecture uses `file_path UNIQUE + ON CONFLICT DO UPDATE` — last-write-wins with no version history. This makes model comparison, regression tracking, and confidence-based ranking impossible.

### 4.2 Proposed Entity Model

```
assets (content identity)
  → asset_versions (file state/path versions)
    → analysis_runs (per-run provenance: model, config, timing)
      → evidence_text (caption/tag/OCR/classification + confidence)
      → evidence_vectors (VV/MV vectors + norm/quantization/checksum)

search_materialized (current best snapshot view for search)
```

### 4.3 Evidence-Based Search

```
Stage 1: Multi-axis candidate generation (VV + MV + FTS) → recall maximization
Stage 2: Weighted RRF merge → initial ranking
Stage 3: Feature extraction per candidate (sim scores, tag overlap, style match, confidence)
Stage 4: Reranking → final = weighted features - penalties (conflicting evidence, spam tags)
```

**Resource budget:** Candidates 200-500 per axis → Rerank top 80-200 → Return top 20-50.

### 4.4 Benefits

- Multiple model versions coexist per asset
- Regression analysis, A/B testing, rollback support
- Confidence-based weighted combination
- Full provenance tracking (why this result?)

---

## 5. Cloud GPU Optimization (H100 Reference)

### 5.1 Checklist Status (as of 2026-02-16)

| Area | Item | Status |
|------|------|--------|
| I/O | CPU prefetching (producer/consumer) | Partial |
| I/O | Image decode acceleration (TurboJPEG) | Not met |
| I/O | S3/GCS network bandwidth | N/A (local-only) |
| vLLM | Continuous batching | Partial |
| vLLM | KV cache tuning (80GB VRAM) | Partial |
| vLLM | FP8/INT4 quantization | Not met |
| SQLite | WAL mode | Met |
| SQLite | Bulk insert (500-1000 batch) | Not met |
| SQLite | Async DB writer | Not met |
| Export | VACUUM + ANALYZE | Partial |
| Export | Vector normalization consistency | Met |
| Export | Model parity (checkpoint hash) | Partial |

### 5.2 Priority Improvements for GPU Environments

1. DB writer separation + batch transactions
2. Image decoding path acceleration (TurboJPEG/OpenCV)
3. vLLM batch path connection to actual pipeline
4. Export/relink model revision/hash enforcement

---

## Open Decision Items

| Decision | Options | Current |
|----------|---------|---------|
| Queue technology | SQLite (initial) vs Redis/RabbitMQ (scale) | SQLite |
| Central vector store | pgvector vs dedicated vector DB | sqlite-vec |
| Resource strategy | Always-sync vs on-demand fetch + cache | Local files |
| Multi-tenancy boundary | Schema separation vs row-level | Row-level |
| Delivery semantics | At-least-once vs exactly-once | At-least-once |
| Content hashing | mtime string vs content_hash + size + mtime_ns | mtime string |
