# Spatial Processing Operational Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Imagine spatial processing from MVP to an operationally stable processing layer with raw preservation, provenance, quality status, normalization, backfill, and benchmarkable validation.

**Architecture:** Keep extraction, processing, storage, and usage separate. Store parsed extraction JSON in `files.structured_meta`, preserve raw VLM output separately, normalize searchable evidence into dedicated tables, and expose processing quality so repair/backfill can target only rows that need work.

**Tech Stack:** Python, SQLite/FTS5, project `.venv`, existing `backend.vision`, `backend.db.sqlite_client`, `backend.search.sqlite_search`, FastAPI routers, existing benchmark/review tools.

---

## Current Baseline

Current spatial extraction and processing already handles:

- `structured_meta.objects` for object + 3x3 location evidence.
- `structured_meta.relations` for object-to-object relations.
- `structured_meta.depth_layers` for foreground/midground/background evidence.
- `file_objects`, `file_spatial_relations`, `file_depth_layers` normalized tables.
- `files_fts.spatial` search text.
- API fields: `spatial_objects`, `spatial_relations`, `depth_layers`.

Operational gaps to close:

- Raw VLM responses are not retained.
- Spatial schema/prompt/parser versions are not first-class data.
- Empty extraction cannot be distinguished from extraction failure or partial parse repair.
- Object naming is not normalized beyond basic string cleanup.
- Existing DB rows need a safe spatial processing audit and reprocess/backfill lane.
- Benchmark/e2e checks do not yet gate the new relation/depth processing layer.

## File Map

- Modify `backend/vision/repair.py`
  - Return parse diagnostics with repaired/raw status without breaking current callers.
  - Preserve parsed fields while recording whether fallback repair was used.

- Modify `backend/vision/mlx_adapter.py`
  - Capture raw Stage 1/Stage 2 text and prompt/model metadata.
  - Attach extraction provenance to the structured result.

- Modify `backend/vision/ollama_adapter.py`, `backend/vision/vllm_adapter.py`, `backend/vision/analyzer.py`
  - Apply the same provenance/raw capture contract where these adapters produce VLM output.

- Modify `backend/pipeline/phase_runner.py`, `backend/pipeline/ingest_engine.py`
  - Pass raw/provenance/quality fields into storage.

- Modify `backend/db/sqlite_client.py`
  - Add raw/provenance tables or columns.
  - Add spatial quality normalization.
  - Add object canonicalization helpers.
  - Refresh FTS with quality-aware evidence.

- Modify `backend/db/sqlite_migrations.py`
  - Ensure new tables/columns exist in existing databases.

- Modify `backend/search/sqlite_search.py`, `backend/api_search.py`, `backend/server/routers/files.py`
  - Surface quality/provenance for detail/debug flows, not necessarily every grid card.

- Modify `tools/rebuild_fts_v3.py`
  - Rebuild all normalized spatial evidence and FTS text using the new processors.

- Modify `tools/verify_ingest_e2e.py`
  - Verify objects, relations, depth layers, raw capture, provenance, and quality status.

- Create `tools/audit_spatial_processing.py`
  - Read-only audit for spatial processing health and repair targets.

- Create `tools/backfill_spatial_processing.py`
  - Backfill/reprocess driver that supports dry-run, file-id scope, and reason filters.

- Add tests:
  - `tests/test_vision_raw_provenance.py`
  - `tests/test_spatial_processing_quality.py`
  - `tests/test_spatial_canonicalization.py`
  - Extend `tests/test_sqlite_spatial_objects.py`
  - Extend `tests/test_api_search_spatial_evidence.py`

---

## Task 1: Define Stable Spatial Processing Contract

**Files:**
- Modify: `backend/vision/schemas.py`
- Modify: `backend/vision/prompts.py`
- Test: `tests/test_vision_spatial_objects.py`

- [ ] **Step 1: Write the failing schema test**

Add assertions that every Stage 2 schema includes explicit processing metadata:

```python
def test_stage2_schema_declares_spatial_processing_contract():
    schemas = load_vision_modules()["schemas"]

    schema = schemas.get_schema("background")

    assert "spatial_schema_version" in schema
    assert "extraction_quality" in schema
    assert "objects_status" in schema["extraction_quality"]
    assert "relations_status" in schema["extraction_quality"]
    assert "depth_status" in schema["extraction_quality"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python3 -m pytest -q -s tests/test_vision_spatial_objects.py::test_stage2_schema_declares_spatial_processing_contract
```

Expected:

```text
FAILED ... AssertionError
```

- [ ] **Step 3: Add schema fields**

In `backend/vision/schemas.py`, add:

```python
SPATIAL_SCHEMA_VERSION_FIELD = "integer. Current value must be 2."

EXTRACTION_QUALITY_FIELD = (
    "object with objects_status, relations_status, depth_status "
    "(each one of: ok, empty, failed, partial), confidence "
    "(high, medium, low), and notes (short string)."
)
```

In `get_schema()`, set:

```python
schema.setdefault("spatial_schema_version", SPATIAL_SCHEMA_VERSION_FIELD)
schema.setdefault("extraction_quality", EXTRACTION_QUALITY_FIELD)
```

- [ ] **Step 4: Update prompt contract**

In `backend/vision/prompts.py`, add to the concise JSON example:

```json
"spatial_schema_version": 2,
"extraction_quality": {
  "objects_status": "ok|empty|failed|partial",
  "relations_status": "ok|empty|failed|partial",
  "depth_status": "ok|empty|failed|partial",
  "confidence": "high|medium|low",
  "notes": "short reason when empty or partial"
}
```

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest -q -s tests/test_vision_spatial_objects.py
python3 -m py_compile backend/vision/schemas.py backend/vision/prompts.py
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/vision/schemas.py backend/vision/prompts.py tests/test_vision_spatial_objects.py
git commit -m "feat: define spatial processing contract"
```

---

## Task 2: Preserve Raw VLM Output and Provenance

**Files:**
- Modify: `backend/vision/repair.py`
- Modify: `backend/vision/mlx_adapter.py`
- Modify: `backend/vision/ollama_adapter.py`
- Modify: `backend/vision/vllm_adapter.py`
- Modify: `backend/vision/analyzer.py`
- Create: `tests/test_vision_raw_provenance.py`

- [ ] **Step 1: Write failing tests for parser diagnostics**

Create `tests/test_vision_raw_provenance.py`:

```python
from backend.vision.repair import parse_structured_output
from backend.vision.schemas import get_schema


def test_parse_structured_output_attaches_diagnostics_for_direct_json():
    parsed = parse_structured_output(
        '{"caption":"x","objects":[]}',
        get_schema("background"),
        image_type="background",
        include_diagnostics=True,
    )

    assert parsed["_parse_diagnostics"]["status"] == "direct"
    assert parsed["_parse_diagnostics"]["repaired"] is False


def test_parse_structured_output_attaches_diagnostics_for_fallback_json():
    parsed = parse_structured_output(
        '{"caption":"x","objects":[{"name":"moon","locations":["right"]}],',
        get_schema("background"),
        image_type="background",
        include_diagnostics=True,
    )

    assert parsed["_parse_diagnostics"]["status"] in {"repaired", "fallback"}
    assert parsed["_parse_diagnostics"]["repaired"] is True
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python3 -m pytest -q -s tests/test_vision_raw_provenance.py
```

Expected:

```text
TypeError: parse_structured_output() got an unexpected keyword argument 'include_diagnostics'
```

- [ ] **Step 3: Add optional diagnostics to parser**

Change signature in `backend/vision/repair.py`:

```python
def parse_structured_output(
    raw: str,
    schema: dict,
    image_type: str = "other",
    include_diagnostics: bool = False,
) -> dict:
```

Add helper:

```python
def _with_diagnostics(result: dict, status: str, repaired: bool, include: bool) -> dict:
    if include:
        result = dict(result)
        result["_parse_diagnostics"] = {
            "status": status,
            "repaired": repaired,
        }
    return result
```

Use it in direct, repaired, and fallback branches.

- [ ] **Step 4: Attach raw/provenance in adapters**

In each Stage 2 adapter after `raw = ...` and parsed `result = ...`, add:

```python
result["_vlm_raw"] = raw
result["_vlm_provenance"] = {
    "stage": "stage2",
    "model": getattr(self, "model_id", "") or getattr(self, "model", ""),
    "adapter": self.__class__.__name__,
    "prompt_version": "spatial_v2",
}
```

For Stage 1, either do not store raw or store it as `_stage1_raw` only inside `_vlm_provenance` if the adapter already merges Stage 1 and Stage 2.

- [ ] **Step 5: Verify parser tests and existing vision tests**

Run:

```bash
python3 -m pytest -q -s tests/test_vision_raw_provenance.py tests/test_vision_spatial_objects.py
python3 -m py_compile backend/vision/repair.py backend/vision/mlx_adapter.py backend/vision/ollama_adapter.py backend/vision/vllm_adapter.py backend/vision/analyzer.py
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/vision tests/test_vision_raw_provenance.py
git commit -m "feat: preserve vlm extraction provenance"
```

---

## Task 3: Store Raw Output Separately From Processed Metadata

**Files:**
- Modify: `backend/db/sqlite_client.py`
- Modify: `backend/db/sqlite_migrations.py`
- Modify: `backend/pipeline/phase_runner.py`
- Modify: `backend/pipeline/ingest_engine.py`
- Test: `tests/test_spatial_processing_quality.py`

- [ ] **Step 1: Write failing storage test**

Create `tests/test_spatial_processing_quality.py`:

```python
import json
import sqlite3
import types

from backend.db.sqlite_client import SQLiteDB


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, structured_meta TEXT)")
    db = object.__new__(SQLiteDB)
    db._local = types.SimpleNamespace(conn=conn)
    return db, conn


def test_replace_vlm_raw_outputs_persists_latest_raw_payload():
    db, conn = make_db()
    conn.execute("INSERT INTO files(id, structured_meta) VALUES (1, '{}')")

    db._ensure_vlm_raw_outputs_table()
    db._replace_vlm_raw_output(
        conn.cursor(),
        file_id=1,
        stage="stage2",
        adapter="MLXVisionAnalyzer",
        model="Qwen/Qwen3.5-9B",
        prompt_version="spatial_v2",
        raw_text='{"caption":"x"}',
        parse_status="direct",
    )
    conn.commit()

    row = conn.execute("SELECT file_id, stage, model, raw_text, parse_status FROM vlm_raw_outputs").fetchone()
    assert dict(row) == {
        "file_id": 1,
        "stage": "stage2",
        "model": "Qwen/Qwen3.5-9B",
        "raw_text": '{"caption":"x"}',
        "parse_status": "direct",
    }
```

- [ ] **Step 2: Verify test fails**

Run:

```bash
python3 -m pytest -q -s tests/test_spatial_processing_quality.py::test_replace_vlm_raw_outputs_persists_latest_raw_payload
```

Expected:

```text
AttributeError: 'SQLiteDB' object has no attribute '_ensure_vlm_raw_outputs_table'
```

- [ ] **Step 3: Add raw output table**

In `backend/db/sqlite_client.py`:

```python
def _ensure_vlm_raw_outputs_table(self):
    self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS vlm_raw_outputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            stage TEXT NOT NULL,
            adapter TEXT,
            model TEXT,
            prompt_version TEXT,
            raw_text TEXT NOT NULL,
            parse_status TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_vlm_raw_outputs_file_id ON vlm_raw_outputs(file_id);
        CREATE INDEX IF NOT EXISTS idx_vlm_raw_outputs_stage ON vlm_raw_outputs(stage);
    """)
    self.conn.commit()
```

Add:

```python
def _replace_vlm_raw_output(self, cursor, file_id: int, stage: str, adapter: str,
                            model: str, prompt_version: str, raw_text: str,
                            parse_status: str) -> None:
    cursor.execute(
        "DELETE FROM vlm_raw_outputs WHERE file_id = ? AND stage = ?",
        (file_id, stage),
    )
    cursor.execute(
        """INSERT INTO vlm_raw_outputs
           (file_id, stage, adapter, model, prompt_version, raw_text, parse_status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (file_id, stage, adapter, model, prompt_version, raw_text, parse_status),
    )
```

- [ ] **Step 4: Wire migrations**

In `backend/db/sqlite_migrations.py`, add a migration function that calls `db._ensure_vlm_raw_outputs_table()` and add it to the migration list after file/spatial evidence migrations.

- [ ] **Step 5: Store raw output during vision updates**

In `update_vision_fields()` or the Phase 2 storage path, after resolving `file_id`, detect:

```python
structured = fields.get("structured_meta")
structured_dict = json.loads(structured) if isinstance(structured, str) else structured
raw_text = structured_dict.pop("_vlm_raw", None) if isinstance(structured_dict, dict) else None
provenance = structured_dict.pop("_vlm_provenance", {}) if isinstance(structured_dict, dict) else {}
diagnostics = structured_dict.pop("_parse_diagnostics", {}) if isinstance(structured_dict, dict) else {}
```

Then store raw separately:

```python
if raw_text:
    self._ensure_vlm_raw_outputs_table()
    self._replace_vlm_raw_output(
        cursor,
        file_id=file_id,
        stage=provenance.get("stage") or "stage2",
        adapter=provenance.get("adapter") or "",
        model=provenance.get("model") or "",
        prompt_version=provenance.get("prompt_version") or "",
        raw_text=raw_text,
        parse_status=diagnostics.get("status") or "",
    )
    fields["structured_meta"] = json.dumps(structured_dict, ensure_ascii=False)
```

- [ ] **Step 6: Verify**

Run:

```bash
python3 -m pytest -q -s tests/test_spatial_processing_quality.py
.venv/bin/python -m py_compile backend/db/sqlite_client.py backend/db/sqlite_migrations.py backend/pipeline/phase_runner.py backend/pipeline/ingest_engine.py
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add backend/db/sqlite_client.py backend/db/sqlite_migrations.py backend/pipeline/phase_runner.py backend/pipeline/ingest_engine.py tests/test_spatial_processing_quality.py
git commit -m "feat: store raw vlm extraction output"
```

---

## Task 4: Add Spatial Processing Quality Status

**Files:**
- Modify: `backend/db/sqlite_client.py`
- Modify: `backend/server/routers/files.py`
- Modify: `backend/api_search.py`
- Test: `tests/test_spatial_processing_quality.py`

- [ ] **Step 1: Add failing quality normalization tests**

Append to `tests/test_spatial_processing_quality.py`:

```python
def test_spatial_quality_distinguishes_empty_from_partial_and_ok():
    ok = SQLiteDB._build_spatial_processing_quality(
        structured_meta={
            "objects": [{"name": "moon", "locations": ["right"]}],
            "relations": [],
            "depth_layers": [],
        },
        parse_status="direct",
    )
    empty = SQLiteDB._build_spatial_processing_quality(
        structured_meta={"objects": [], "relations": [], "depth_layers": []},
        parse_status="direct",
    )
    partial = SQLiteDB._build_spatial_processing_quality(
        structured_meta={"objects": ["name", "forks", "locations", "left"]},
        parse_status="fallback",
    )

    assert ok["objects_status"] == "ok"
    assert empty["objects_status"] == "empty"
    assert partial["objects_status"] == "partial"
    assert partial["parse_status"] == "fallback"
```

- [ ] **Step 2: Verify it fails**

Run:

```bash
python3 -m pytest -q -s tests/test_spatial_processing_quality.py::test_spatial_quality_distinguishes_empty_from_partial_and_ok
```

Expected:

```text
AttributeError: type object 'SQLiteDB' has no attribute '_build_spatial_processing_quality'
```

- [ ] **Step 3: Implement quality builder**

In `backend/db/sqlite_client.py`:

```python
@classmethod
def _build_spatial_processing_quality(cls, structured_meta: Any, parse_status: str = "") -> dict:
    if isinstance(structured_meta, str):
        try:
            structured_meta = json.loads(structured_meta)
        except (json.JSONDecodeError, TypeError):
            return {
                "objects_status": "failed",
                "relations_status": "failed",
                "depth_status": "failed",
                "parse_status": parse_status or "invalid_json",
                "confidence": "low",
                "notes": "structured_meta_json_error",
            }
    if not isinstance(structured_meta, dict):
        structured_meta = {}

    raw_objects = structured_meta.get("objects")
    objects = cls._normalize_spatial_objects_from_meta(structured_meta)
    relations = cls._normalize_spatial_relations_from_meta(structured_meta)
    depth_layers = cls._normalize_depth_layers_from_meta(structured_meta)

    return {
        "objects_status": "partial" if raw_objects and not objects else ("ok" if objects else "empty"),
        "relations_status": "ok" if relations else "empty",
        "depth_status": "ok" if depth_layers else "empty",
        "parse_status": parse_status or "",
        "confidence": "medium" if objects else "low",
        "notes": "",
    }
```

- [ ] **Step 4: Store quality inside `structured_meta`**

During `update_vision_fields()`, before writing `structured_meta`, inject:

```python
structured_dict["spatial_processing_quality"] = self._build_spatial_processing_quality(
    structured_dict,
    parse_status=diagnostics.get("status") or "",
)
structured_dict["spatial_schema_version"] = structured_dict.get("spatial_schema_version") or 2
```

- [ ] **Step 5: Expose quality only in detail/debug responses**

In `backend/server/routers/files.py`, detail response already returns `structured_meta`; ensure it includes `spatial_processing_quality`.

In `backend/api_search.py`, add a lightweight top-level field:

```python
"spatial_processing_quality": result.get("spatial_processing_quality", {}),
```

Only add this if `sqlite_search.py` loads it from `structured_meta` during `_parse_json_fields()`.

- [ ] **Step 6: Verify**

Run:

```bash
python3 -m pytest -q -s tests/test_spatial_processing_quality.py tests/test_api_search_spatial_evidence.py
python3 -m py_compile backend/db/sqlite_client.py backend/api_search.py backend/server/routers/files.py
```

Expected:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add backend/db/sqlite_client.py backend/api_search.py backend/server/routers/files.py tests/test_spatial_processing_quality.py tests/test_api_search_spatial_evidence.py
git commit -m "feat: classify spatial processing quality"
```

---

## Task 5: Add Object Canonicalization Layer

**Files:**
- Modify: `backend/db/sqlite_client.py`
- Create: `tests/test_spatial_canonicalization.py`

- [ ] **Step 1: Write failing canonicalization tests**

Create `tests/test_spatial_canonicalization.py`:

```python
from backend.db.sqlite_client import SQLiteDB


def test_canonical_object_name_normalizes_plural_and_common_synonyms():
    assert SQLiteDB._canonical_object_name("shelves") == "shelf"
    assert SQLiteDB._canonical_object_name("cupboard") == "cabinet"
    assert SQLiteDB._canonical_object_name("bottles") == "bottle"


def test_korean_object_name_uses_known_dictionary_when_vlm_translation_is_bad():
    assert SQLiteDB._canonical_ko_name("shelf", "가까이") == "선반"
    assert SQLiteDB._canonical_ko_name("cabinet", "장롱") == "수납장"
```

- [ ] **Step 2: Verify it fails**

Run:

```bash
python3 -m pytest -q -s tests/test_spatial_canonicalization.py
```

Expected:

```text
AttributeError
```

- [ ] **Step 3: Implement conservative dictionaries**

In `backend/db/sqlite_client.py`:

```python
_OBJECT_SYNONYMS = {
    "shelves": "shelf",
    "cupboard": "cabinet",
    "bottles": "bottle",
}

_OBJECT_KO_NAMES = {
    "shelf": "선반",
    "cabinet": "수납장",
    "bottle": "병",
    "cup": "컵",
    "table": "테이블",
    "moon": "달",
    "cloud": "구름",
}
```

Add:

```python
@classmethod
def _canonical_object_name(cls, value: Any) -> str:
    name = str(value or "").strip().lower()
    if name in cls._OBJECT_SYNONYMS:
        return cls._OBJECT_SYNONYMS[name]
    if name.endswith("s") and len(name) > 3:
        return name[:-1]
    return name

@classmethod
def _canonical_ko_name(cls, canonical_name: str, ko_name: Any) -> str:
    known = cls._OBJECT_KO_NAMES.get(canonical_name)
    if known:
        return known
    return str(ko_name or "").strip()
```

- [ ] **Step 4: Apply canonicalization to objects, relations, depth layers**

In `_normalize_spatial_objects_from_meta()`:

```python
name = cls._canonical_object_name(raw.get("name"))
ko_name = cls._canonical_ko_name(name, raw.get("ko_name"))
```

In `_normalize_spatial_relations_from_meta()`:

```python
subject = cls._canonical_object_name(raw.get("subject"))
obj = cls._canonical_object_name(raw.get("object"))
```

In `_normalize_depth_layers_from_meta()`:

```python
name = cls._canonical_object_name(raw.get("name") or raw.get("object"))
ko_name = cls._canonical_ko_name(name, raw.get("ko_name"))
```

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest -q -s tests/test_spatial_canonicalization.py tests/test_sqlite_spatial_objects.py
python3 -m py_compile backend/db/sqlite_client.py
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add backend/db/sqlite_client.py tests/test_spatial_canonicalization.py tests/test_sqlite_spatial_objects.py
git commit -m "feat: canonicalize spatial object names"
```

---

## Task 6: Add Read-Only Spatial Processing Audit

**Files:**
- Create: `tools/audit_spatial_processing.py`
- Test: `tests/test_audit_spatial_processing.py`

- [ ] **Step 1: Write failing audit test**

Create `tests/test_audit_spatial_processing.py`:

```python
import sqlite3

from tools.audit_spatial_processing import collect_spatial_processing_stats


def test_collect_spatial_processing_stats_counts_repair_targets(tmp_path):
    db_path = tmp_path / "audit.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, structured_meta TEXT, mc_caption TEXT)")
    conn.execute("CREATE TABLE file_objects(file_id INTEGER)")
    conn.execute("CREATE TABLE file_spatial_relations(file_id INTEGER)")
    conn.execute("CREATE TABLE file_depth_layers(file_id INTEGER)")
    conn.execute("INSERT INTO files(id, structured_meta, mc_caption) VALUES (1, '{}', 'caption')")
    conn.commit()
    conn.close()

    stats = collect_spatial_processing_stats(db_path)

    assert stats["total_files_with_caption"] == 1
    assert stats["missing_objects"] == 1
    assert stats["missing_relations"] == 1
    assert stats["missing_depth_layers"] == 1
```

- [ ] **Step 2: Verify it fails**

Run:

```bash
python3 -m pytest -q -s tests/test_audit_spatial_processing.py
```

Expected:

```text
ModuleNotFoundError: No module named 'tools.audit_spatial_processing'
```

- [ ] **Step 3: Implement audit tool**

Create `tools/audit_spatial_processing.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


def collect_spatial_processing_stats(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM files WHERE mc_caption IS NOT NULL AND mc_caption != ''"
        ).fetchone()[0]
        missing_objects = conn.execute(
            """SELECT COUNT(*) FROM files f
               WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                 AND NOT EXISTS (SELECT 1 FROM file_objects o WHERE o.file_id = f.id)"""
        ).fetchone()[0]
        missing_relations = conn.execute(
            """SELECT COUNT(*) FROM files f
               WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                 AND NOT EXISTS (SELECT 1 FROM file_spatial_relations r WHERE r.file_id = f.id)"""
        ).fetchone()[0]
        missing_depth_layers = conn.execute(
            """SELECT COUNT(*) FROM files f
               WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                 AND NOT EXISTS (SELECT 1 FROM file_depth_layers d WHERE d.file_id = f.id)"""
        ).fetchone()[0]
        return {
            "total_files_with_caption": total,
            "missing_objects": missing_objects,
            "missing_relations": missing_relations,
            "missing_depth_layers": missing_depth_layers,
        }
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    args = parser.parse_args()
    print(json.dumps(collect_spatial_processing_stats(Path(args.db)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify on test and real DB**

Run:

```bash
python3 -m pytest -q -s tests/test_audit_spatial_processing.py
.venv/bin/python tools/audit_spatial_processing.py --db imageparser.db
```

Expected:

```text
passed
```

The real DB command should print JSON counts.

- [ ] **Step 5: Commit**

```bash
git add tools/audit_spatial_processing.py tests/test_audit_spatial_processing.py
git commit -m "feat: audit spatial processing coverage"
```

---

## Task 7: Add Controlled Backfill/Reprocess Driver

**Files:**
- Create: `tools/backfill_spatial_processing.py`
- Test: `tests/test_backfill_spatial_processing.py`

- [ ] **Step 1: Write failing plan-selection test**

Create `tests/test_backfill_spatial_processing.py`:

```python
import sqlite3

from tools.backfill_spatial_processing import select_reprocess_candidates


def test_select_reprocess_candidates_limits_and_filters_by_missing_relations(tmp_path):
    db_path = tmp_path / "backfill.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files(id INTEGER PRIMARY KEY, file_path TEXT, mc_caption TEXT)")
    conn.execute("CREATE TABLE file_spatial_relations(file_id INTEGER)")
    conn.execute("INSERT INTO files(id, file_path, mc_caption) VALUES (1, '/a.png', 'caption')")
    conn.execute("INSERT INTO files(id, file_path, mc_caption) VALUES (2, '/b.png', 'caption')")
    conn.execute("INSERT INTO file_spatial_relations(file_id) VALUES (2)")
    conn.commit()
    conn.close()

    rows = select_reprocess_candidates(db_path, reason="missing_relations", limit=10)

    assert rows == [{"id": 1, "file_path": "/a.png", "reason": "missing_relations"}]
```

- [ ] **Step 2: Verify it fails**

Run:

```bash
python3 -m pytest -q -s tests/test_backfill_spatial_processing.py
```

Expected:

```text
ModuleNotFoundError
```

- [ ] **Step 3: Implement candidate selection**

Create `tools/backfill_spatial_processing.py` with:

```python
#!/usr/bin/env python3
import argparse
import json
import sqlite3
from pathlib import Path


def select_reprocess_candidates(db_path: Path, reason: str, limit: int) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if reason == "missing_relations":
            rows = conn.execute(
                """SELECT f.id, f.file_path
                   FROM files f
                   WHERE f.mc_caption IS NOT NULL AND f.mc_caption != ''
                     AND NOT EXISTS (
                        SELECT 1 FROM file_spatial_relations r WHERE r.file_id = f.id
                     )
                   ORDER BY f.id
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        else:
            raise ValueError(f"unsupported reason: {reason}")
        return [{"id": row["id"], "file_path": row["file_path"], "reason": reason} for row in rows]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="imageparser.db")
    parser.add_argument("--reason", default="missing_relations")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    rows = select_reprocess_candidates(Path(args.db), args.reason, args.limit)
    print(json.dumps({"dry_run": args.dry_run, "candidates": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify dry-run only**

Run:

```bash
python3 -m pytest -q -s tests/test_backfill_spatial_processing.py
.venv/bin/python tools/backfill_spatial_processing.py --db imageparser.db --reason missing_relations --limit 10 --dry-run
```

Expected:

```text
passed
```

The real DB command should print candidates and must not mutate DB.

- [ ] **Step 5: Extend driver to call ingest engine only after explicit `--execute`**

Add:

```python
parser.add_argument("--execute", action="store_true")
```

When `--execute` is present, print the exact ingest command first and require the caller to run it separately:

```python
print(json.dumps({
    "execute": True,
    "command": [".venv/bin/python", "backend/pipeline/ingest_engine.py", "--files", json.dumps(paths, ensure_ascii=False), "--no-skip"],
}, ensure_ascii=False, indent=2))
```

This keeps the first version safe and auditable.

- [ ] **Step 6: Commit**

```bash
git add tools/backfill_spatial_processing.py tests/test_backfill_spatial_processing.py
git commit -m "feat: plan spatial processing backfill"
```

---

## Task 8: Extend E2E Verification and Benchmark Gate

**Files:**
- Modify: `tools/verify_ingest_e2e.py`
- Modify: `tools/run_search_benchmark.py` only if a spatial evidence field is needed in run artifacts.
- Add or extend tests under `tests/`.

- [ ] **Step 1: Add E2E assertions for new evidence**

In `tools/verify_ingest_e2e.py`, require:

```python
out["objects"]["file_objects_count"] = ...
out["relations"]["file_spatial_relations_count"] = ...
out["depth_layers"]["file_depth_layers_count"] = ...
out["fts"]["spatial_nonempty"] = ...
out["raw"]["vlm_raw_outputs_count"] = ...
```

- [ ] **Step 2: Add CLI summary output**

Print:

```text
✓ file_objects count=N
✓ file_spatial_relations count=N
✓ file_depth_layers count=N
✓ vlm_raw_outputs count=N
✓ files_fts.spatial non-empty
```

- [ ] **Step 3: Run focused verification on one known meaningful file**

Run:

```bash
.venv/bin/python tools/verify_ingest_e2e.py --db imageparser.db --file-id <known_file_id>
```

Expected:

```text
✓ structured_meta
✓ file_objects
✓ file_spatial_relations
✓ file_depth_layers
✓ files_fts.spatial
```

- [ ] **Step 4: Add benchmark labels for spatial questions**

Create or extend a QuerySet with:

```json
{"query_id":"spatial_relation_001","query_text":"컵이 테이블 위에 있는 이미지","query_type":"spatial_relation","locale":"ko-KR"}
{"query_id":"spatial_depth_001","query_text":"전경에 테이블이 있는 이미지","query_type":"spatial_depth","locale":"ko-KR"}
```

Use the existing `benchmarks/querysets/` and `benchmarks/reviews/` layout.

- [ ] **Step 5: Run benchmark with venv**

Run:

```bash
.venv/bin/python tools/run_search_benchmark.py --help
```

Then run the project-standard benchmark command for the selected QuerySet. Use `.venv/bin/python`, not system Python, because `sqlite-vec` is only available in the project environment.

- [ ] **Step 6: Commit**

```bash
git add tools/verify_ingest_e2e.py benchmarks/querysets benchmarks/reviews
git commit -m "test: add spatial processing verification gate"
```

---

## Task 9: Document Processing Contract

**Files:**
- Create: `docs/spatial_processing_contract_ko.md`

- [ ] **Step 1: Write the document**

Create:

```markdown
# 공간 정보 가공 계약

## 단계

1. 원천 추출: VLM raw output
2. 파싱 원본: files.structured_meta
3. 정규화 저장: file_objects, file_spatial_relations, file_depth_layers
4. 검색 가공: files_fts.spatial
5. 활용: API search results and file detail

## 기본 필드

- objects
- relations
- depth_layers
- spatial_processing_quality
- spatial_schema_version
- vlm_raw_outputs

## 운영 규칙

- raw output은 변경하지 않는다.
- structured_meta는 현재 파서 기준의 파싱 결과다.
- 정규화 테이블은 검색과 UI를 위한 파생 데이터다.
- FTS는 재생성 가능한 파생 인덱스다.
- 품질 상태가 failed 또는 partial이면 재처리 후보가 된다.
```

- [ ] **Step 2: Commit**

```bash
git add docs/spatial_processing_contract_ko.md
git commit -m "docs: define spatial processing contract"
```

---

## Validation Checklist

Run after all tasks:

```bash
python3 -m py_compile \
  backend/vision/schemas.py \
  backend/vision/prompts.py \
  backend/vision/repair.py \
  backend/db/sqlite_client.py \
  backend/db/sqlite_migrations.py \
  backend/search/sqlite_search.py \
  backend/api_search.py \
  backend/server/routers/files.py \
  tools/rebuild_fts_v3.py \
  tools/audit_spatial_processing.py \
  tools/backfill_spatial_processing.py

python3 -m pytest -q -s \
  tests/test_vision_spatial_objects.py \
  tests/test_vision_raw_provenance.py \
  tests/test_sqlite_spatial_objects.py \
  tests/test_spatial_processing_quality.py \
  tests/test_spatial_canonicalization.py \
  tests/test_api_search_spatial_evidence.py \
  tests/test_audit_spatial_processing.py \
  tests/test_backfill_spatial_processing.py

.venv/bin/python tools/audit_spatial_processing.py --db imageparser.db
```

Expected:

- Python compile succeeds.
- All focused tests pass.
- Audit prints JSON counts.
- No DB mutation occurs during audit.

## Rollback Rules

- If raw output storage increases DB size too much, keep only latest Stage 2 raw per file and delete older rows with a maintenance script.
- If object canonicalization causes false merges, disable `_OBJECT_SYNONYMS` entries one by one; do not remove raw or structured data.
- If benchmark quality drops, revert FTS expansion changes but keep raw/provenance/quality storage.
- If backfill candidate selection is too broad, keep it dry-run only until a smaller reason filter is added.

## Execution Order

1. Contract version and prompt fields.
2. Raw/provenance capture.
3. Raw storage.
4. Quality status.
5. Canonicalization.
6. Audit tool.
7. Backfill planner.
8. E2E/benchmark gate.
9. Documentation.
