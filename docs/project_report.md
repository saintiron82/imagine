# Imagine — Project Report

**Version**: v0.6.1
**Date**: 2026-02-26
**Status**: Phase 4.9 Complete (Active Development)

---

## 1. Project Overview

Imagine is a multimodal image data extraction and vectorization system that transforms PSD, PNG, and JPG files into AI-searchable data. It uses the **Triaxis Architecture** (VV + MV + FTS) to enable cross-modal search across visual similarity, semantic meaning, and keyword matching.

### Key Capabilities

- **Multimodal Search**: Search images by text, by image, or by combining both
- **3-Axis Ranking**: VV (visual), MV (semantic), FTS (keyword) scores fused via RRF
- **PSD Deep Parsing**: Layer tree, text content, font extraction from Photoshop files
- **AI Vision Analysis**: Qwen3-VL generates captions, tags, and classifications
- **Desktop + Web Dual Mode**: Electron app (local) and browser (server) with shared React UI
- **Distributed Workers**: Multiple machines can process images in parallel via job queue
- **External Access**: LAN auto-detection, QR codes, Cloudflare Quick Tunnel

---

## 2. Architecture

### 2.1 Data Pipeline (4-Phase)

```
Image File (PSD/PNG/JPG)
    |
Phase P (Parse)    --> Metadata extraction, thumbnail generation
    |
Phase V (Vision)   --> Qwen3-VL: MC caption + tags + classification
    |
Phase E (Embed)    --> SigLIP2 -> VV | Qwen3-Embedding -> MV
    |
Phase S (Summary)  --> Completion confirmation
```

Each phase saves results immediately in sub-batches. Crash recovery via Smart Skip (resumes from last incomplete phase per file).

### 2.2 Triaxis Search

```
User Query
    |
    +---> VV: SigLIP2 text encoder --> cosine similarity (visual)
    +---> MV: Qwen3-Embedding     --> cosine similarity (semantic)
    +---> FTS: FTS5 BM25           --> keyword matching (16 columns)
    |
    +--> RRF Merge (weighted rank fusion) --> Final ranking
```

QueryDecomposer auto-selects weight presets (visual/keyword/semantic/balanced).

### 2.3 Client-Server Architecture

```
+---------------------------+     +---------------------------+
|   Electron Mode (Local)   |     |   Web Mode (Browser)      |
|   - Auth bypass           |     |   - JWT login required    |
|   - IPC -> Python direct  |     |   - HTTP API -> FastAPI   |
|   - Local DB access       |     |   - Server DB access      |
|   - [Server] toggle       |     |   - Multi-user support    |
+---------------------------+     +---------------------------+
            |                                 |
            +--- Shared React 19 Frontend ----+
```

### 2.4 Distributed Worker System

```
Server (FastAPI)                    Worker (Python daemon)
+----------------+                 +----------------------+
| Job Queue      |<-- claim(N) --- | Prefetch Pool        |
| (SQLite)       |--- jobs[] ----> | (capacity x 2)       |
|                |                 |                      |
| worker_        |<-- heartbeat -- | 30s heartbeat        |
| sessions       |--- command ---> | stop/block commands  |
+----------------+                 +----------------------+
```

---

## 3. Technology Stack

### Backend (Python)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Parser | psd-tools, Pillow | PSD/PNG/JPG metadata extraction |
| VLM | Qwen3-VL (2B/4B/8B) | MC caption/tag generation |
| VV Encoder | SigLIP2 so400m-naflex | Visual vector (1152d) |
| MV Encoder | Qwen3-Embedding (0.6B/8B) | Meaning vector (1024d/4096d) |
| Database | SQLite + sqlite-vec | Metadata + vector storage |
| Search | FTS5 BM25 + RRF | Full-text + rank fusion |
| Server | FastAPI + uvicorn | REST API, JWT auth, SPA serving |
| Auth | JWT (access + refresh) | Role-based access (admin/user) |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 19 | UI components |
| Desktop | Electron 40 | Native app wrapper |
| Build | Vite 6.x | Dev server + production build |
| Styling | Tailwind CSS 4 | Utility-first CSS |
| i18n | Custom (LocaleContext) | Korean/English |
| QR Code | qrcode.react | LAN/Tunnel URL sharing |

### AI Model Matrix (Tier System)

| Tier | VRAM | VLM (MC) | VV (SigLIP2) | MV (Embedding) |
|------|------|----------|-------------|----------------|
| Standard | ~6GB | Qwen3-VL-2B | so400m-naflex (1152d) | 0.6B (1024d) |
| Pro | 8-16GB | Qwen3-VL-4B | so400m-naflex (1152d) | 0.6B (1024d) |
| Ultra | 20GB+ | Qwen3-VL-8B | so400m-naflex (1152d) | 8B (4096d) |

VV model is unified across all tiers. Standard <-> Pro transition is fully seamless.

### VLM Fallback Chain

| Tier | macOS | Windows | Linux |
|------|-------|---------|-------|
| Standard | transformers | transformers | transformers |
| Pro | mlx -> transformers | transformers | transformers |
| Ultra | mlx -> transformers | ollama -> transformers | vllm -> ollama -> transformers |

---

## 4. Codebase Statistics

| Metric | Value |
|--------|-------|
| Total commits | ~350 |
| Backend Python | ~75,000 lines |
| Frontend JSX/JS | ~29,000 lines |
| Electron (main/preload) | ~2,100 lines |
| Database | SQLite (single file, no Docker) |
| Supported formats | PSD, PNG, JPG, JPEG |
| FTS index columns | 16 |
| i18n languages | 2 (ko-KR, en-US) |

---

## 5. Completed Phases

### Phase 1: Structural Parsing
- PSD/PNG/JPG parser (BaseParser, PSDParser, ImageParser)
- Data schema (AssetMeta, LayerInfo, ParseResult)
- Layer name cleaner, thumbnail generator
- Ingest pipeline (4-phase)

### Phase 2: Visual Vectorization
- SigLIP2 VV encoder
- SQLite + sqlite-vec migration (removed ChromaDB/PostgreSQL)
- FTS5 full-text search index

### Phase 3: Descriptive Vision
- Qwen3-VL 2-stage caption/tag/classification
- Qwen3-Embedding MV
- Triaxis search (VV + MV + FTS, RRF fusion)
- 3-Tier system (standard/pro/ultra)
- Cross-platform VLM fallback chain (mlx -> transformers -> ollama)

### Phase 4: Electron GUI + Client-Server
- React 19 + Electron 40 frontend
- File browser + virtual scroll grid
- Triaxis search UI + filters
- Metadata modal (AI analysis + user tags/notes/rating)
- Image search (single/multi, AND/OR mode)
- i18n (Korean/English)
- Registered folder auto-scan + resume dialog
- Settings modal
- Electron/Web dual mode (IPC <-> HTTP bridge)
- FastAPI server (JWT auth, SPA serving)
- Role-based access control (admin/user)

### Phase 4.5: Distributed Worker System
- Worker daemon (prefetch pool + heartbeat + command piggyback)
- Worker session management API
- Worker token one-click setup script
- Phased batch processing (process_batch_phased)
- Inter-phase model unloading (VLM -> SigLIP2 -> Qwen3-Embedding)
- VV/MV sub-batch inference
- Per-phase throughput UI (files/min)
- Multi-worker account creation (Admin Panel)
- Admin worker monitoring (aggregate/per-worker)
- Job queue management (create/claim/complete/fail)

### Phase 4.6: Server External Access
- LAN IP auto-detection (os.networkInterfaces)
- ServerInfoPanel (dropdown — Local/LAN/Tunnel URL)
- QR code generation (qrcode.react — LAN/Tunnel)
- CORS relaxation (cors_allow_all option, JWT-protected)
- Cloudflare Quick Tunnel (one-click internet access, auto-download)

### Codebase Cleanup (v10.4)
- Legacy code removal (PostgreSQL, ChromaDB, Ollama parallel — 17 files, 4575 lines)
- Design rationale documentation (9 architectural decisions)
- Troubleshooting documentation (MC data loss, MV import, IPC session, GPU contention)

### Phase 4.7: Domain Classification + DB Management

#### Domain-Aware Classification System
- YAML-based domain presets (`backend/vision/domains/`) with inheritance (`_base.yaml`)
- Built-in presets: `game_asset` (10 image types, 59 tags), `illustration` (7 types, 35 tags), `stock_photo` (6 types, 22 tags)
- `domain_loader.py`: Dynamic prompt/schema builder from YAML — injects domain-specific `image_types` and `tags` into VLM classification prompts
- All VLM adapters updated (analyzer.py, mlx_adapter.py, ollama_adapter.py, vllm_adapter.py) to accept domain-aware prompts
- Classification API (`/api/v1/admin/classification/`) — list domains, get detail, set active, create from YAML
- Admin Classification Panel — domain list, detail view (image types + tags), YAML editor, AI domain YAML generator
- Electron IPC bridge for domain operations (`listDomains`, `setActiveDomain`, `getActiveDomainConfig`)
- Domain selection modal on first launch when no domain configured (`DomainSelectModal.jsx`)
- 40+ i18n keys added (en-US + ko-KR) for classification and domain UI

#### Database Reset Feature
- `reset_file_data()` method in `sqlite_client.py` — clears files, layers, vectors, FTS, job queue while preserving auth tables and thumbnails
- `POST /api/v1/admin/database/reset` endpoint with dual auth: JWT admin check + bcrypt password re-verification
- Header DB dropdown: admin-only "Reset DB" button (red, separated)
- Confirmation modal with password input, loading spinner, error display
- Reset reports count of deleted files/vectors/jobs, refreshes folder stats UI

### Phase 4.8: Auto Processing & Advanced Job Management
- **Server Auto Processing Mode**: Background (`P→V→VV→MV`) pipeline execution when no workers are active. Toggle available in Admin panel.
- **Job Reclamation System**: Auto-recovers jobs from disconnected, blocked, or timed-out workers.
- **Auto batch skipping**: Automatically skips already-parsed records in pending jobs.
- **DINOv2 Integration**: Integrated `vec_structure` generation directly into the `ParseAheadPool` with auto-queue backfill for missing structure vectors.
- **Unified HF Models**: Standardized auto-download system for all HuggingFace models upon first use.

### Phase 4.9: Web Presence & Auto Distribution
- **Landing Site / App**: GitHub Pages landing site with English/Korean i18n support.
- **Community Features**: Board, Q&A, Guide, and Releases pages integrated with Firebase Authentication and Firestore DB. 
- **Auto-Update System**: Integrated `electron-updater` working with GitHub Releases for seamless local desktop app updates.

---

## 6. Remaining Phases

### Phase 5: UI/UX Enhancement (Priority 1)
- [x] Local folder sync (DB ↔ disk reconciliation — detect moved/deleted/new files)
- [x] Original image download (server mode — download from browser)
- [ ] Lightbox image viewer (zoom/pan/navigation)
- [ ] Search history and autocomplete
- [ ] Result sorting (relevance/date/filename/rating)
- [ ] View mode switching (grid/list/compact)
- [ ] Drag and drop file handling
- [ ] Keyboard shortcuts (Ctrl+K search, etc.)
- [ ] Sidebar collapse/expand

### Phase 6: Advanced Search (Priority 2)
- [ ] Advanced filters (date range, resolution, layer count)
- [ ] Filter presets (save/load)
- [ ] Bookmark/collection system (CRUD + JSON export)
- [ ] Smart collections (dynamic filter-based)
- [ ] "Similar images" one-click search
- [ ] Bulk tag editing
- [ ] Tag cloud visualization

### Phase 7: Performance Optimization (Priority 3)
- [ ] Hash-based incremental indexing
- [ ] Multi-process parallel parsing
- [ ] VLM result caching
- [ ] Client-side search response cache
- [ ] WebP thumbnail conversion
- [ ] Model auto-unload (idle timeout)

### Phase 8: Packaging & Distribution (Priority 4)
- [ ] Python embedded runtime bundling
- [ ] Windows NSIS installer
- [ ] macOS DMG build (code signing)
- [x] electron-updater auto-update
- [ ] First-run guide (model download wizard)
- [ ] Portable mode (USB execution)

### Phase 9: Data Management + Collaboration (Priority 5)
- [ ] Automatic DB backup (schedule + retention policy)
- [ ] Manual DB backup/restore UI
- [ ] File matching/relink UI improvements (match visualization, selective apply)
- [ ] DB export/import (metadata + vector packaging)
- [ ] Collection/folder partial export
- [ ] Per-image comment history
- [ ] Read-only sharing mode

---

## 7. Key Files Index

### Core Pipeline

| File | Purpose |
|------|---------|
| `backend/pipeline/ingest_engine.py` | 4-phase processing pipeline (main entry) |
| `backend/parser/schema.py` | Standard data schema (AssetMeta) |
| `backend/db/sqlite_client.py` | SQLite client (metadata + VV + MV) |
| `backend/db/sqlite_schema.sql` | Database schema definition |
| `backend/search/sqlite_search.py` | Triaxis search engine |
| `backend/search/rrf.py` | RRF weight presets |
| `backend/vision/vision_factory.py` | VLM backend auto-selection |
| `backend/vector/siglip2_encoder.py` | SigLIP2 VV encoder |
| `backend/vector/text_embedding.py` | Qwen3-Embedding MV encoder |

### Server & Worker

| File | Purpose |
|------|---------|
| `backend/server/app.py` | FastAPI server entry |
| `backend/server/auth/router.py` | JWT auth API |
| `backend/server/routers/workers.py` | Worker session API |
| `backend/server/routers/classification.py` | Domain classification admin API |
| `backend/server/routers/database.py` | Database reset API (admin password re-verification) |
| `backend/server/queue/manager.py` | Job queue manager |
| `backend/worker/worker_daemon.py` | Worker daemon (prefetch pool) |

### Domain Classification

| File | Purpose |
|------|---------|
| `backend/vision/domain_loader.py` | Domain YAML loader + prompt/schema builder |
| `backend/vision/domains/_base.yaml` | Base domain preset (inherited by all) |
| `backend/vision/domains/game_asset.yaml` | Game asset preset (10 types, 59 tags) |
| `backend/vision/domains/illustration.yaml` | Illustration preset (7 types, 35 tags) |
| `backend/vision/domains/stock_photo.yaml` | Stock photo preset (6 types, 22 tags) |

### Frontend

| File | Purpose |
|------|---------|
| `frontend/src/App.jsx` | Main application component |
| `frontend/src/api/client.js` | API client (JWT, isElectron) |
| `frontend/src/api/admin.js` | Admin API (users, jobs, classification, DB reset) |
| `frontend/src/services/bridge.js` | Electron/Web mode bridge |
| `frontend/src/components/DomainSelectModal.jsx` | First-launch domain selection modal |
| `frontend/electron/main.cjs` | Electron main process |
| `frontend/electron/preload.cjs` | IPC bridge definitions |

### Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Tier/search/batch settings (single source) |
| `backend/utils/tier_config.py` | Tier setting loader |
| `backend/setup/installer.py` | Unified installer |

---

## 8. Documentation Index

| Document | Purpose | Status |
|----------|---------|--------|
| `CLAUDE.md` | Development guide (terminology, rules, architecture) | Active |
| `phase.md` | Development roadmap (Phase 1-9) | Active |
| `docs/project_report.md` | Comprehensive project report | Active |
| `docs/V3.1.md` | Core design spec (VV/MC/MV/FTS pipeline) | Reference |
| `docs/triaxis_search_architecture.md` | 3-axis search system explanation | Reference |
| `docs/quick_start_guide.md` | 5-minute getting started guide | Active |
| `docs/ollama_setup.md` | Ollama installation and MV backend setup | Active |
| `docs/platform_optimization.md` | Platform-specific VLM optimization | Reference |
| `docs/retrospective_v3.1_tier_system.md` | Tier system implementation retrospective | Historical |
| `docs/troubleshooting.md` | Known issues and solutions | Active |
| `docs/future_roadmap.md` | Consolidated future plans (cloud, optimization, benchmark, ECM) | Future |

---

## 9. Design Decisions

### Why SQLite (not PostgreSQL)?
- Personal/small team tool — no Docker, no server process needed
- sqlite-vec provides vector search within the same file
- WAL mode + workers=1 sufficient for current scale
- Single-file database simplifies backup and portability

### Why SigLIP2 (not PE-Core, CLIP)?
- Apache 2.0 license (PE-Core is CC-BY-NC)
- HuggingFace transformers native integration
- MPS (Apple Silicon) verified
- NaFlex: aspect ratio preservation for varied image dimensions
- Performance difference < 0.5% vs PE-Core at same parameter count

### Why Triaxis (not single-axis search)?
- VV captures visual similarity (color, composition, mood)
- MV captures semantic meaning (concepts, use cases, context)
- FTS captures exact keywords (filenames, layer names, tags)
- RRF fusion produces better results than any single axis alone

### Why Cloudflare Tunnel (not ngrok/localtunnel)?
- No account required for Quick Tunnel
- Official binary, auto-download from GitHub Releases
- Stable infrastructure, no rate limits for personal use
- Free tier sufficient for small team access

---

## 10. Development Environment

| Item | Value |
|------|-------|
| Primary Dev | macOS 26.2, Apple M5, 32GB |
| Tier | Pro (8-16GB VRAM) |
| Python | 3.12.12 (.venv) |
| Node | v24.13.0 |
| Vite dev port | 9274 |
| Server port | 8000 |
| Repository | github.com/saintiron82/imagine |
