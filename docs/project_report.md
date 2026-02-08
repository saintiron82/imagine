# ImageParser - Project Analysis Report

**Date**: 2026-02-07
**Version**: Phase 3 Complete, Phase 4 In Progress

---

## 1. Project Overview

ImageParser is a **multimodal image data extraction and vectorization system** that converts PSD, PNG, and JPG files into AI-searchable data. It uses a **3-Axis Architecture** to decompose images from three independent perspectives:

| Axis | Purpose | Technology | Status |
|------|---------|-----------|--------|
| **Structural** | Layer hierarchy, text, metadata | `psd-tools`, `Pillow` | Complete |
| **Latent** | Visual embeddings for similarity search | CLIP ViT-L-14 (768-dim) | Complete |
| **Descriptive** | AI captions, tags, OCR | Ollama (Qwen2.5-VL) / Transformers (BLIP, Qwen2-VL) | In Progress |

**Core Value**: Designers and developers can search their image library by natural language descriptions ("fantasy character with sword"), visual similarity, or structured metadata filters - all in ~20ms.

---

## 2. System Architecture

```
                        +-----------------------+
                        |   Electron Desktop    |
                        |   (React 19 + Vite)   |
                        +----------+------------+
                                   |
                          IPC (JSON over stdin/stdout)
                                   |
              +--------------------+--------------------+
              |                    |                     |
     +--------v--------+  +-------v--------+  +--------v--------+
     | Ingest Pipeline  |  | Search Engine  |  | API Layer       |
     | (ingest_engine)  |  | (sqlite_search)|  | (api_search,    |
     |                  |  |                |  |  api_stats,      |
     | - PSD Parser     |  | - Vector Search|  |  api_metadata)   |
     | - Image Parser   |  | - FTS5 Search  |  |                 |
     | - CLIP Embed     |  | - Hybrid/RRF   |  |                 |
     | - AI Vision      |  | - LLM Decompose|  |                 |
     | - Translation    |  |                |  |                 |
     | - DFS Discovery  |  |                |  |                 |
     +--------+---------+  +-------+--------+  +--------+--------+
              |                     |                     |
              +---------------------+---------------------+
                                    |
                         +----------v----------+
                         |   SQLite Database    |
                         |   + sqlite-vec       |
                         |   + FTS5             |
                         |                      |
                         | - files (metadata)   |
                         | - CLIP embeddings    |
                         | - Full-text index    |
                         +---------------------+
```

---

## 3. AI/ML Logic

### 3.1 CLIP ViT-L-14 (Visual Embedding)

**Location**: `backend/vector/indexer.py`, `backend/search/sqlite_search.py`

**Purpose**: Generate 768-dimensional visual embeddings for images and text queries, enabling semantic similarity search.

**How it works**:
1. **Ingest**: Image thumbnail -> CLIP model -> 768-dim float vector -> stored in SQLite (sqlite-vec)
2. **Search**: Text query -> CLIP model -> 768-dim float vector -> cosine similarity against all stored vectors

**Key details**:
- Model: `clip-ViT-L-14` via `sentence-transformers` library
- Embedding dimension: 768
- Lazy loading: Model loaded on first use, cached as singleton (`_global_indexer`)
- Memory management: `torch.cuda.empty_cache()` every 10 images
- Device: Auto-detect CUDA/CPU

**Code flow** (ingest):
```
Image file -> Pillow open -> thumbnail (512x512 max)
  -> SentenceTransformer.encode(image) -> 768-dim numpy array
  -> serialize as bytes -> SQLite INSERT (embedding column)
```

**Code flow** (search):
```
Text query -> SentenceTransformer.encode(query) -> 768-dim vector
  -> sqlite-vec: vec_distance_cosine() -> ranked results
```

### 3.2 Ollama Vision (AI Description)

**Location**: `backend/vision/ollama_adapter.py`

**Purpose**: Analyze images to extract captions, tags, OCR text, dominant color, and art style using local LLM.

**Model**: `qwen2.5-vl:7b` (default, configurable via `VISION_MODEL` env)

**Key features**:
- `keep_alive=0`: Model unloaded from GPU/RAM immediately after each analysis (memory efficient)
- JSON-structured output: Sends structured prompt, parses JSON response
- Fallback: Text parsing if JSON extraction fails
- Timeout: 60s per image analysis

**Output schema**:
```json
{
  "caption": "A fantasy character holding a glowing sword...",
  "tags": ["fantasy", "character", "sword", "glowing"],
  "ocr": "LEVEL 99",
  "color": "dark blue",
  "style": "digital art, anime-influenced"
}
```

### 3.3 Transformers Vision (Alternative Backend)

**Location**: `backend/vision/analyzer.py`

**Purpose**: Same as Ollama but using HuggingFace Transformers models directly.

**Supported models**:
| Model | Class | Use Case |
|-------|-------|----------|
| BLIP | `BlipForConditionalGeneration` | Simple, reliable captioning |
| BLIP-2 | `Blip2ForConditionalGeneration` | Higher quality captions + VQA for OCR |
| Qwen2-VL-2B | `Qwen2VLForConditionalGeneration` | Best quality, future Mac M5 target |
| Florence-2 | `AutoModelForCausalLM` | Multi-task vision (currently unused due to flash_attn issues) |

**Analysis pipeline**:
```
Image -> _generate_caption() -> caption text
      -> _extract_tags(caption) -> keyword extraction (stop word filtering)
      -> _extract_text() -> OCR via VQA prompting
      -> _analyze_style(caption) -> keyword matching (minimalist, vintage, etc.)
      -> _analyze_color() -> numpy pixel averaging -> hex color
```

### 3.4 Vision Factory (Backend Selector)

**Location**: `backend/vision/vision_factory.py`

**Purpose**: Select between Transformers and Ollama backends based on environment.

```python
VISION_BACKEND=ollama     -> OllamaVisionAdapter (deployment: memory efficient)
VISION_BACKEND=transformers -> VisionAnalyzer (development: higher accuracy)
```

Singleton pattern with `_cached_analyzer` for efficiency.

### 3.5 LLM Query Decomposer

**Location**: `backend/search/query_decomposer.py`

**Purpose**: Convert natural language search queries into structured search parameters using Ollama LLM.

**Input**: User query (any language)
**Output**:
```json
{
  "vector_query": "fantasy character with magical sword",
  "fts_keywords": ["fantasy", "character", "sword", "판타지", "캐릭터"],
  "filters": {"format": "PSD"},
  "decomposed": true
}
```

**Key features**:
- `vector_query`: Always English (CLIP compatibility)
- `fts_keywords`: Bilingual (Korean + English for FTS5)
- `filters`: Structured metadata extraction
- Graceful fallback: If Ollama is unavailable, raw query used as-is
- Model: `qwen3-vl:8b` with `temperature=0.1` (deterministic)
- `keep_alive=0`: Immediate model unload
- `/no_think` prompt: Suppresses reasoning for faster response

### 3.6 Auto-Translation

**Location**: `backend/pipeline/ingest_engine.py`

**Purpose**: Translate extracted text content, tags, and layer trees between languages.

**Technology**: `deep-translator` (Google Translate)

**Strategy**: Batch translation with chunking
1. Collect all translatable text from AssetMeta
2. Join with `" ||| "` separator
3. Split into 4000-char chunks
4. Translate each chunk
5. Split results back to original structure
6. DFS traversal for layer_tree translation (preserving hierarchy)

**Output fields**:
- `translated_tags` (Korean), `translated_tags_en` (English)
- `translated_text` (Korean), `translated_text_en` (English)
- `translated_layer_tree` (Korean), `translated_layer_tree_en` (English)

---

## 4. Data Processing Pipeline

### 4.1 Full Ingest Flow

```
Input File (PSD/PNG/JPG)
  |
  v
ParserFactory.get_parser()
  |-- PSD: psd-tools -> layers, text, fonts, hierarchy
  |-- Image: Pillow -> EXIF, dimensions, basic metadata
  v
AssetMeta (unified schema)
  |
  +--> Thumbnail Generation (256px, saved to output/thumbnails/)
  |
  +--> Layer Name Cleaning (cleaner.py)
  |    - Remove "copy", "Layer 1", numbering
  |    - Generate semantic_tags from meaningful names
  |
  +--> Auto-Translation (deep-translator)
  |    - Tags: EN -> KR, KR -> EN
  |    - Text content: multilingual translation
  |    - Layer tree: structure-preserving DFS translation
  |
  +--> AI Vision Analysis (optional, if Ollama running)
  |    - Caption generation
  |    - Tag extraction
  |    - OCR text extraction
  |    - Color / Style analysis
  |
  +--> CLIP Embedding (768-dim vector)
  |    - Thumbnail -> CLIP ViT-L-14 -> float[768]
  |
  +--> SQLite Storage
  |    - Metadata INSERT (files table)
  |    - CLIP vector INSERT (embedding column, sqlite-vec)
  |    - FTS5 auto-sync (triggers)
  |
  +--> JSON Output (output/json/{filename}.json)
       - Full AssetMeta serialized for Electron frontend
```

### 4.2 Input Modes

| Mode | CLI Flag | Description |
|------|----------|-------------|
| Single File | `--file "path.psd"` | Process one file |
| Batch | `--files '["f1.psd","f2.png"]'` | Process multiple files |
| Watch | `--watch "dir/"` | Monitor directory + initial DFS scan |
| **DFS Discovery** | `--discover "dir/"` | Recursive folder traversal with smart skip |
| No-Skip | `--no-skip` | Force reprocessing (disable smart skip) |

### 4.3 DFS Folder Discovery (New)

```
Root Directory
  |
  v
discover_files() -- DFS recursive traversal
  |-- Skip: .git, __pycache__, node_modules, hidden dirs
  |-- Collect: .psd, .png, .jpg, .jpeg
  |-- Compute: relative folder path, depth, folder tags
  v
[(file_path, folder_path, depth, folder_tags), ...]
  |
  v
Smart Skip Check
  |-- DB has modified_at for this file?
  |-- File's current mtime == stored mtime? -> SKIP
  |-- Otherwise -> PROCESS
  v
process_file() with folder metadata injected
```

**Folder metadata** is stored as searchable fields:
- `folder_path`: `"Characters/Hero"` (prefix-matchable)
- `folder_depth`: `2` (hierarchy level)
- `folder_tags`: `["Characters", "Hero"]` (FTS5 searchable)

### 4.4 AssetMeta Schema

**Location**: `backend/parser/schema.py`

Core fields (all parsers produce):
```
file_name, file_path, file_size, format, resolution,
layer_count, layer_tree, text_content, used_fonts,
semantic_tags, thumbnail_url, dominant_color,
translated_tags, translated_text, translated_layer_tree,
ai_caption, ai_tags, ocr_text, ai_style,
folder_path, folder_depth, folder_tags
```

---

## 5. Database & Search

### 5.1 SQLite + sqlite-vec + FTS5

**Schema**: `backend/db/sqlite_schema.sql`

**files table** (primary storage):
```sql
id INTEGER PRIMARY KEY,
file_path TEXT UNIQUE,      -- absolute path
file_name TEXT,
format TEXT,                -- PSD/PNG/JPG
width INTEGER, height INTEGER,
file_size INTEGER,
metadata TEXT,              -- JSON blob (layer_tree, etc.)
embedding BLOB,             -- 768-dim CLIP vector (sqlite-vec)
thumbnail_url TEXT,
folder_path TEXT,           -- "Characters/Hero"
folder_depth INTEGER,
folder_tags TEXT,           -- '["Characters","Hero"]'
ai_caption TEXT,
ai_tags TEXT,               -- JSON array
ocr_text TEXT,
user_note TEXT,
user_tags TEXT,             -- JSON array
user_category TEXT,
user_rating INTEGER DEFAULT 0,
modified_at TEXT,
created_at DATETIME DEFAULT CURRENT_TIMESTAMP
```

**files_fts** (FTS5 virtual table, auto-synced via triggers):
```sql
file_path, ai_caption, semantic_tags, ocr_text,
user_note, user_tags, folder_tags
```

### 5.2 Search Modes (Triaxis)

**Location**: `backend/search/sqlite_search.py`

| Mode | Method | Description |
|------|--------|-------------|
| `vector` | `vector_search()` | CLIP cosine similarity only |
| `fts` | `fts_search()` | FTS5 full-text keyword match |
| `hybrid` | `hybrid_search()` | Vector + metadata filters |
| `metadata` | `metadata_query()` | Pure structured filters |
| **`triaxis`** | `search()` | Vector + FTS5 + Filters with RRF merge |

**Triaxis Search Flow**:
```
User Query: "fantasy character"
  |
  v
QueryDecomposer (LLM)
  |-- vector_query: "fantasy character illustration"
  |-- fts_keywords: ["fantasy", "character", "판타지", "캐릭터"]
  |-- filters: {}
  |
  +--> Vector Search (CLIP similarity)
  |    -> Top-K results with cosine scores
  |
  +--> FTS5 Search (keyword match)
  |    -> Results with BM25 ranks
  |
  v
RRF Merge (Reciprocal Rank Fusion)
  score = sum(1 / (k + rank)) for each source
  |
  +--> Apply User Filters (format, category, rating, folder)
  |
  v
Final Ranked Results
```

### 5.3 Filter System

Available filters in search:
- `format`: PSD / PNG / JPG
- `user_category`: Characters / Backgrounds / UI Elements / etc.
- `min_rating`: 1-5 star minimum
- `folder_path`: Prefix match ("Characters/" matches all subfolders)
- `folder_tag`: Contains specific folder name tag

---

## 6. Frontend (Electron + React)

### 6.1 Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Electron | 40 | Desktop app framework |
| React | 19 | UI library |
| Vite | 6 | Build tool + HMR |
| Tailwind CSS | 4 | Utility-first styling |
| Lucide React | - | Icon library |

### 6.2 Application Layout

```
+------------------------------------------------------------------+
|  Header Bar (App Name | Search Tab | Archive Tab | Process Btn)  |
+------------------------------------------------------------------+
|          |                                                        |
| Sidebar  |  Main Content Area                                    |
| (Folder  |                                                        |
|  Tree)   |  [Search Mode] -> SearchPanel                         |
| [Archive |  [Archive Mode] -> FileGrid                           |
|  only]   |                                                        |
|          +--------------------------------------------------------+
|          |  StatusBar (logs, progress)                            |
+----------+--------------------------------------------------------+
|  ProcessingIndicator (bottom-right floating)                     |
+------------------------------------------------------------------+
```

### 6.3 Components

#### App.jsx (Root)
- **State**: currentTab (search/archive), currentPath, selectedFiles, logs, processQueue
- **Tab System**: Search mode (full search UI) / Archive mode (folder browser + file grid)
- **Processing Queue**: Files added to queue -> sequential Python pipeline execution
- **IPC Listeners**: Stable listeners on mount, ref-based queue advancement

#### SearchPanel.jsx (Main Search UI)
- **Search bar** with auto-focus, Enter-to-search
- **Triaxis search**: Calls `window.electron.pipeline.searchVector()` with mode='triaxis'
- **Filter bar** (toggle): Format, Category, Min Rating dropdowns
- **Results grid**: Responsive grid (2-6 columns based on viewport)
- **SearchResultCard**: Thumbnail + similarity badge + AI caption + tags + metadata
- **MetadataModal**: Full detail view (same as FileGrid's modal, inline duplicate)
- **SettingsModal**: Environment check + dependency installer
- **DB Stats**: Shows archived image count and format distribution

#### FileGrid.jsx (Archive File Browser)
- **File listing**: Lists supported files (.psd/.png/.jpg/.jpeg) from current folder
- **Thumbnail system**: Priority queue with batched generation (4 at a time)
  - `thumbnailCache`: Global Map, survives across re-renders
  - `prioritizeFolder()`: New folder -> front of queue (instant UX)
  - Batch IPC: `generateThumbnailsBatch()` -> Python subprocess
- **Selection**: Click, Ctrl+Click, Shift+Click, Drag-select (rectangle)
- **Metadata status**: Red dot indicator for unprocessed files (polls every 3s)
- **META button**: Click to open MetadataModal with full file analysis
- **MetadataModal**: Full-screen detail view with:
  - Basic info (format, resolution, layers, size)
  - User metadata (notes, custom tags, category, 5-star rating) - auto-save 500ms debounce
  - AI Vision analysis (caption, tags, color swatch, style, OCR)
  - Semantic tags with language toggle (Original / KR / EN)
  - Fonts used
  - Extracted text with translation
  - Layer tree viewer (JSON pretty-print)

#### Sidebar.jsx (Folder Navigator)
- **TreeNode**: Recursive lazy-loading folder tree
- Starts from Home directory
- "Browse Folder" button -> native OS folder dialog
- Expand/collapse with chevron icons, yellow folder icons
- Selected folder highlighted with blue border

#### StatusBar.jsx
- Collapsible log panel at bottom of main content
- Color-coded log messages (info/error/success)

#### ProcessingIndicator.jsx
- Floating bottom-right indicator during processing
- Shows progress: "Processing [3/10]: filename.psd"

#### SettingsModal.jsx
- Environment status check (Python dependencies)
- Dependency table: torch, chromadb, sentence-transformers, etc.
- One-click "Install All Dependencies" button
- Real-time install log terminal
- Force re-install / verify AI model option

### 6.4 Electron IPC Bridge

**Preload** (`preload.cjs`) exposes 3 namespaces:

| Namespace | APIs | Transport |
|-----------|------|-----------|
| `electron.fs` | `getHomeDir()`, `listDir()`, `exists()`, `pathJoin()` | Direct Node.js |
| `electron.pipeline` | `run()`, `searchVector()`, `readMetadata()`, `generateThumbnail()`, `generateThumbnailsBatch()`, `checkMetadataExists()`, `getDbStats()`, `checkEnv()`, `installEnv()`, `openFolderDialog()` | IPC -> Python subprocess |
| `electron.metadata` | `updateUserData()` | IPC -> Python subprocess |

**Main Process** (`main.cjs`) IPC handlers:

| Handler | Method | Python Script | Protocol |
|---------|--------|---------------|----------|
| `run-pipeline` | `ipcMain.on` | `ingest_engine.py --files` | stdout streaming |
| `search-vector` | `ipcMain.handle` | `api_search.py` | stdin JSON -> stdout JSON |
| `get-db-stats` | `ipcMain.handle` | `api_stats.py` | stdout JSON |
| `read-metadata` | `ipcMain.handle` | (reads JSON file directly) | Node.js fs |
| `check-metadata-exists` | `ipcMain.handle` | (checks JSON file existence) | Node.js fs |
| `generate-thumbnail` | `ipcMain.handle` | `thumbnail_generator.py` | stdout base64 |
| `generate-thumbnails-batch` | `ipcMain.handle` | `thumbnail_generator.py --batch` | stdout JSON |
| `check-env` | `ipcMain.handle` | `installer.py --check` | stdout JSON |
| `install-env` | `ipcMain.on` | `installer.py --install` | stdout streaming |
| `metadata:updateUserData` | `ipcMain.handle` | `api_metadata_update.py` | stdin JSON -> stdout JSON |
| `open-folder-dialog` | `ipcMain.handle` | (native Electron dialog) | Direct |

---

## 7. Data Flow Diagrams

### 7.1 Processing Flow (Archive -> Process)

```
User selects files in FileGrid
  -> Click "Process" button
  -> App.handleProcess() adds to queue
  -> Queue processor dispatches one at a time
  -> IPC: run-pipeline -> main.cjs
  -> spawn: python ingest_engine.py --files [...]
  -> stdout: progress messages -> pipeline-log event
  -> "[OK] Parsed successfully" -> pipeline-progress event
  -> process.close -> pipeline-file-done event
  -> Queue advances to next file
  -> Repeat until queue empty
```

### 7.2 Search Flow

```
User types query in SearchPanel
  -> Press Enter or click Search
  -> searchVector({query, limit:20, mode:'triaxis', filters})
  -> IPC: search-vector -> main.cjs
  -> spawn: python api_search.py (stdin: JSON)
  -> api_search.py -> SqliteVectorSearch.search()
    -> QueryDecomposer.decompose(query) [Ollama LLM]
    -> vector_search() [CLIP + sqlite-vec]
    -> fts_search() [FTS5 keyword match]
    -> RRF merge -> apply filters -> ranked results
  -> stdout: JSON results
  -> Electron parses JSON -> React renders SearchResultCard grid
```

### 7.3 Metadata View Flow

```
User clicks file card / search result
  -> handleShowMeta(filePath)
  -> IPC: read-metadata -> main.cjs
  -> Read output/json/{filename}.json
  -> Return parsed JSON to renderer
  -> MetadataModal renders:
    - Basic info, AI analysis, tags, text, layer tree
    - Language toggle (Original/KR/EN)
    - User edits (notes, tags, category, rating)
      -> Auto-save (500ms debounce)
      -> IPC: metadata:updateUserData -> main.cjs
      -> spawn: python api_metadata_update.py (stdin: JSON)
      -> SQLite UPDATE
```

---

## 8. Module Inventory

### 8.1 Backend Python Modules

| Module | File | Lines | Purpose |
|--------|------|-------|---------|
| **Schema** | `parser/schema.py` | ~150 | AssetMeta, LayerInfo, ParseResult Pydantic models |
| **Base Parser** | `parser/base_parser.py` | ~50 | Abstract parser interface |
| **PSD Parser** | `parser/psd_parser.py` | ~200 | PSD layer extraction (psd-tools) |
| **Image Parser** | `parser/image_parser.py` | ~100 | PNG/JPG metadata extraction (Pillow) |
| **Cleaner** | `parser/cleaner.py` | ~100 | Layer name normalization, semantic tag generation |
| **Ingest Engine** | `pipeline/ingest_engine.py` | ~500 | Main pipeline: parse + translate + embed + store |
| **SQLite Client** | `db/sqlite_client.py` | ~400 | SQLite wrapper: insert, query, migrate |
| **SQLite Schema** | `db/sqlite_schema.sql` | ~100 | DDL: tables, indexes, FTS5, triggers |
| **SQLite Search** | `search/sqlite_search.py` | ~500 | Vector, FTS5, hybrid, triaxis search |
| **Query Decomposer** | `search/query_decomposer.py` | ~150 | LLM query -> structured search params |
| **Vision Analyzer** | `vision/analyzer.py` | ~500 | Transformers-based vision analysis |
| **Ollama Adapter** | `vision/ollama_adapter.py` | ~260 | Ollama API-based vision analysis |
| **Vision Factory** | `vision/vision_factory.py` | ~130 | Backend selector (Transformers/Ollama) |
| **Vector Indexer** | `vector/indexer.py` | ~130 | CLIP embedding + ChromaDB (legacy) |
| **API Search** | `api_search.py` | ~100 | JSON API for Electron search |
| **API Stats** | `api_stats.py` | ~30 | JSON API for DB statistics |
| **API Metadata** | `api_metadata_update.py` | ~50 | JSON API for user metadata updates |
| **Thumbnail Gen** | `utils/thumbnail_generator.py` | ~100 | PSD/image thumbnail generation |
| **Installer** | `setup/installer.py` | ~200 | Dependency check + auto-install |
| **Migrations** | `db/migrations/*.py` | ~100 | Schema migration scripts |
| **PG Client** | `db/pg_client.py` | ~300 | PostgreSQL client (alternative DB) |
| **PG Search** | `search/pg_search.py` | ~400 | pgvector search (alternative) |

### 8.2 Frontend Components

| Component | File | Purpose |
|-----------|------|---------|
| **App** | `App.jsx` | Root: tabs, processing queue, IPC setup |
| **SearchPanel** | `SearchPanel.jsx` | Search UI: bar, filters, results grid, metadata modal |
| **FileGrid** | `FileGrid.jsx` | Archive: file cards, thumbnails, selection, metadata |
| **Sidebar** | `Sidebar.jsx` | Folder tree navigator |
| **StatusBar** | `StatusBar.jsx` | Log panel |
| **ProcessingIndicator** | `ProcessingIndicator.jsx` | Progress overlay |
| **SettingsModal** | `SettingsModal.jsx` | Environment setup UI |

### 8.3 Electron Layer

| File | Purpose |
|------|---------|
| `main.cjs` | Main process: IPC handlers, Python subprocess management |
| `preload.cjs` | Context bridge: expose IPC APIs to renderer |

---

## 9. Key Design Decisions

### 9.1 SQLite over PostgreSQL for Desktop

SQLite chosen for desktop deployment:
- Zero configuration (no Docker/service)
- Single file database (portable)
- sqlite-vec for vector search (comparable to pgvector)
- FTS5 for full-text search (built-in)
- PostgreSQL still supported as alternative (`pg_client.py`, `pg_search.py`)

### 9.2 Python Subprocess per IPC Call

Each Electron -> Python call spawns a new subprocess:
- **Pros**: Process isolation, no memory leaks, clean state
- **Cons**: ~200ms overhead per call, model reload
- **Mitigation**: CLIP model caching in `_global_indexer` singleton, Ollama server always-on

### 9.3 Triaxis Search with RRF

Three independent search signals merged via Reciprocal Rank Fusion:
- Avoids tuning individual weights
- Robust to missing signals (e.g., no Ollama -> only vector + FTS)
- Naturally handles multilingual queries via QueryDecomposer

### 9.4 Thumbnail Priority Queue

Browser-side queue with folder prioritization:
- Current folder files pushed to front of queue
- Batch size: 4 concurrent thumbnails
- Global cache survives folder navigation
- Prevents UI blocking during large folder loads

---

## 10. Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Single file ingest (PSD) | ~3-5s | Parse + translate + CLIP + store |
| Single file ingest (PNG) | ~1-2s | Simpler parsing |
| CLIP model load (first use) | ~5-10s | Lazy loaded, then cached |
| Triaxis search (200 files) | ~20ms | Vector + FTS5 + RRF |
| Triaxis search (10K files) | ~40ms | HNSW + GIN indexes |
| AI Vision (Ollama) | ~5-15s | Per image, model unloaded after |
| Thumbnail generation (PSD) | ~500ms | Per file, batched in groups of 4 |
| DFS Discovery (1000 files) | ~1s | File listing only, no processing |
| Smart skip check | ~1ms | Single SQLite query per file |

---

## 11. File Structure

```
ImageParser/
+-- backend/
|   +-- parser/
|   |   +-- schema.py          # AssetMeta, LayerInfo, ParseResult
|   |   +-- base_parser.py     # Abstract parser interface
|   |   +-- psd_parser.py      # PSD file parser
|   |   +-- image_parser.py    # PNG/JPG parser
|   |   +-- cleaner.py         # Layer name normalization
|   +-- pipeline/
|   |   +-- ingest_engine.py   # Main ingest pipeline + DFS discovery
|   +-- db/
|   |   +-- sqlite_schema.sql  # SQLite DDL
|   |   +-- sqlite_client.py   # SQLite client
|   |   +-- pg_client.py       # PostgreSQL client (alternative)
|   |   +-- migrations/        # Schema migrations
|   +-- search/
|   |   +-- sqlite_search.py   # Triaxis search engine
|   |   +-- pg_search.py       # PostgreSQL search (alternative)
|   |   +-- query_decomposer.py # LLM query -> search params
|   +-- vision/
|   |   +-- analyzer.py        # Transformers vision (BLIP, Qwen2-VL)
|   |   +-- ollama_adapter.py  # Ollama vision adapter
|   |   +-- vision_factory.py  # Backend selector
|   +-- vector/
|   |   +-- indexer.py         # CLIP + ChromaDB (legacy)
|   |   +-- searcher.py        # ChromaDB search (legacy)
|   +-- utils/
|   |   +-- thumbnail_generator.py
|   +-- setup/
|   |   +-- installer.py       # Dependency installer
|   +-- api_search.py          # Search API for Electron
|   +-- api_stats.py           # Stats API for Electron
|   +-- api_metadata_update.py # Metadata update API
|   +-- cli_search.py          # CLI search tool
|   +-- cli_search_pg.py       # CLI search (PostgreSQL)
+-- frontend/
|   +-- electron/
|   |   +-- main.cjs           # Electron main process
|   |   +-- preload.cjs        # Context bridge
|   +-- src/
|   |   +-- App.jsx            # Root component
|   |   +-- main.jsx           # React entry point
|   |   +-- components/
|   |       +-- SearchPanel.jsx
|   |       +-- FileGrid.jsx
|   |       +-- Sidebar.jsx
|   |       +-- StatusBar.jsx
|   |       +-- ProcessingIndicator.jsx
|   |       +-- SettingsModal.jsx
+-- output/
|   +-- json/                  # Metadata JSON files
|   +-- thumbnails/            # Generated thumbnails
+-- docs/
|   +-- troubleshooting.md    # Issue tracking
|   +-- phase_roadmap.md      # Development roadmap
|   +-- project_report.md     # This report
+-- test_assets/              # Test PSD/PNG/JPG files
+-- requirements.txt          # Python dependencies
+-- docker-compose.yml        # PostgreSQL (optional)
+-- CLAUDE.md                 # Project instructions
+-- INSTALLATION.md           # Setup guide
```

---

## 12. Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Structural Parsing (PSD layers, metadata) | Complete |
| **Phase 2** | Latent Vectorization (CLIP embeddings) | Complete |
| **Phase 3** | Vision Data Storage (PostgreSQL/SQLite + pgvector/sqlite-vec) | Complete |
| **Phase 4** | Descriptive Vision (AI captions, OCR, tags) | In Progress |
| **Phase 5** | Optimization (layer-level indexing, packaging) | Planned |

**Recent additions** (Phase 3+):
- DFS folder discovery with smart skip
- Folder metadata (path, depth, tags) as searchable fields
- Triaxis search (Vector + FTS5 + Metadata + RRF)
- LLM query decomposition
- User metadata (notes, tags, category, rating)
- Auto-save with debounce
- Ollama vision integration
- Settings modal with dependency management
