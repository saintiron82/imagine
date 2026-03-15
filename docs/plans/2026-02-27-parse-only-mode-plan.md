# `parse_only` Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `parse_only` global worker mode where server only parses (Phase P) and delegates all GPU work (V+VV+MV) to workers.

**Architecture:** Extend the existing global mode system (`mc_only`/`auto`) with a third option `parse_only`. Server's ParseAheadPool runs Phase P only with zero GPU model loading. Workers receive `"full"` processing_mode and handle V+VV+MV. When no workers are online, server keeps parsing and queues jobs (no auto fallback).

**Tech Stack:** Python (FastAPI backend), React 19 (frontend), i18n (en-US/ko-KR)

---

### Task 1: Backend — Allow `parse_only` in mode validation

**Files:**
- Modify: `backend/server/queue/manager.py:16-32`

**Step 1: Update `get_processing_mode()` to accept `parse_only`**

In `backend/server/queue/manager.py`, change the valid mode set:

```python
def get_processing_mode() -> str:
    """Get effective processing mode from config.

    Returns "mc_only", "parse_only", or "auto" (default).
    - mc_only: Server P+VV+MV, workers do V(MC) only.
    - parse_only: Server P only (zero GPU), workers do V+VV+MV.
    - auto: Server P + gap-fill, workers distribute V/VV/MV by capability.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        mode = cfg.get("server.processing_mode") or "auto"
        # Normalize legacy values
        if mode not in ("mc_only", "parse_only", "auto"):
            mode = "auto"
        return mode
    except Exception:
        return "auto"
```

**Step 2: Commit**

```
feat: allow parse_only in processing mode validation
```

---

### Task 2: Backend — Add `parse_only` to `_recalculate_server_pools`

**Files:**
- Modify: `backend/server/routers/workers.py:92-183`

**Step 1: Update `_recalculate_server_pools()` docstring and logic**

At line 92, update docstring to include parse_only:
```python
def _recalculate_server_pools(app, db: "SQLiteDB") -> None:
    """온라인 워커 + admin 모드(mc_only/parse_only/auto)에 따라 서버 사이드 풀 자동 설정.

    - 워커 없음 + auto → auto 모드 (서버가 P→V→VV→MV 전부)
    - 워커 없음 + parse_only → parse_only 유지 (서버 P만, 워커 대기)
    - mc_only 모드 → ParseAhead(P+VV) + EmbedAhead(MV), 워커는 V(MC)만
    - parse_only 모드 → ParseAhead(P만), 워커가 V+VV+MV 전부
    - auto 모드 → ParseAhead(P + gap-fill V), 워커는 능력별 V/VV/MV 분산
    """
```

At line 120-144, add `parse_only` branch:

```python
        if not has_workers:
            if global_mode == "parse_only":
                # parse_only: keep parsing, queue jobs, wait for workers
                app.state.parse_ahead._processing_mode = "parse_only"
                if old_mode != "parse_only":
                    logger.info("No workers online, parse_only mode — server parsing only, waiting for workers")
            else:
                # auto/mc_only without workers — auto if enabled, else distribute
                from backend.utils.config import get_config
                cfg = get_config()
                auto_enabled = cfg.get("server.auto_processing.enabled", True)
                if auto_enabled:
                    app.state.parse_ahead._processing_mode = "auto"
                    if old_mode != "auto":
                        logger.info("No workers online, auto-processing enabled")
                else:
                    app.state.parse_ahead._processing_mode = "distribute"
        elif global_mode == "mc_only":
            app.state.parse_ahead._processing_mode = "mc_only"
            if old_mode != "mc_only":
                logger.info(f"Workers connected, switching to mc_only mode")
        elif global_mode == "parse_only":
            app.state.parse_ahead._processing_mode = "parse_only"
            if old_mode != "parse_only":
                logger.info(f"Workers connected, switching to parse_only mode")
        else:
            app.state.parse_ahead._processing_mode = "distribute"
            app.state.parse_ahead._has_lightweight_workers = has_lightweight
            if old_mode != "distribute":
                logger.info(
                    f"Workers connected, switching to distribute mode "
                    f"(lightweight={has_lightweight})"
                )
```

At line 155, EmbedAheadPool is NOT needed for parse_only (workers handle MV):
```python
    needs_embed_ahead = has_workers and global_mode == "mc_only"
    # parse_only: no EmbedAhead needed (workers do MV)
    # auto: no EmbedAhead needed (workers do MV)
```
(No change needed here — existing logic already correct since `parse_only != "mc_only"`)

**Step 2: Unload server GPU models when switching to parse_only**

When switching TO parse_only, unload any VV/Structure encoders that might be loaded from a previous mc_only session:

After the parse_only branch (around line 134 area), add:
```python
        elif global_mode == "parse_only":
            app.state.parse_ahead._processing_mode = "parse_only"
            # Unload any GPU models from previous mode (e.g., mc_only → parse_only switch)
            app.state.parse_ahead._unload_models()
            if old_mode != "parse_only":
                logger.info(f"Workers connected, switching to parse_only mode")
```

**Step 3: Commit**

```
feat: add parse_only to _recalculate_server_pools
```

---

### Task 3: Backend — Update global config API and worker responses

**Files:**
- Modify: `backend/server/routers/workers.py:54-55, 246-260, 341-368, 675-727`

**Step 1: Update `GlobalModeUpdate` schema (line 54-55)**

```python
class GlobalModeUpdate(BaseModel):
    processing_mode: str  # "mc_only" | "parse_only" | "auto"
```

**Step 2: Update `admin_update_global_config()` validation (line 690-692)**

```python
    mode = req.processing_mode
    if mode not in ("mc_only", "parse_only", "auto"):
        raise HTTPException(status_code=400, detail="processing_mode must be 'mc_only', 'parse_only', or 'auto'")
```

**Step 3: Add parse_only handling in `admin_update_global_config()` (line 696-707)**

```python
    if mode == "mc_only":
        cursor.execute(
            """UPDATE worker_sessions
               SET processing_mode_override = 'mc_only'
               WHERE status = 'online'""",
        )
    elif mode == "parse_only":
        # parse_only: all workers do full pipeline (V+VV+MV)
        cursor.execute(
            """UPDATE worker_sessions
               SET processing_mode_override = 'full'
               WHERE status = 'online'""",
        )
    else:
        # auto: let workers keep their auto-detected roles (full/embed_only)
        pass
```

**Step 4: Update `worker_connect()` response (line 246-252)**

```python
    global_mode = _get_global_processing_mode()
    if global_mode == "mc_only":
        processing_mode = "mc_only"
    elif global_mode == "parse_only":
        processing_mode = "full"
    else:
        processing_mode = (ov[0] if ov and ov[0] else None) or "full"
```

**Step 5: Update `worker_heartbeat()` response (line 344-348)**

```python
    global_mode = _get_global_processing_mode()
    if global_mode == "mc_only":
        processing_mode = "mc_only"
    elif global_mode == "parse_only":
        processing_mode = "full"
    else:
        processing_mode = mode_override or "full"
```

**Step 6: Commit**

```
feat: update global config API and worker responses for parse_only
```

---

### Task 4: Backend — Add `parse_only` branch in ParseAheadPool main loop

**Files:**
- Modify: `backend/server/queue/parse_ahead.py:10-18, 44-47, 531-590`

**Step 1: Update module docstring (line 10-18)**

Add parse_only to the mode list:
```python
Modes:
- auto: No workers connected — server processes all phases (P→V→VV→MV).
  Models loaded per-phase and unloaded between phases.
- mc_only: Also runs Phase VV (SigLIP2 + DINOv2) on parsed jobs since
  VV/Structure only need the image (independent of MC). Workers handle
  V(MC) only; EmbedAheadPool handles MV.
- parse_only: Server only runs Phase P (zero GPU models loaded). Workers
  handle all GPU phases (V+VV+MV). Jobs queue as pre-parsed, waiting for
  worker claims.
- distribute: Pre-parse + gap-fill V(MC) for lightweight workers. Full
  workers handle V+VV+MV, lightweight workers handle VV+MV. Server fills
  vision gaps so lightweight workers can claim vision-done jobs.
```

**Step 2: Update class docstring (line 44-47)**

```python
    Modes:
    - auto: Full pipeline P→V→VV→MV (no workers connected).
    - mc_only: P + VV (SigLIP2 + DINOv2); workers do V(MC) only.
    - parse_only: P only (zero GPU); workers do V+VV+MV.
    - distribute: P + gap-fill V(MC) for lightweight workers.
```

**Step 3: Update `_loop()` docstring (line 531-538)**

```python
    def _loop(self):
        """Main loop: continuously pre-parse pending jobs to fill the buffer.

        Modes:
        - auto: Server processes all phases (P→V→VV→MV) when no workers connected.
        - mc_only: Pre-parse + VV embedding; workers handle V(MC) only.
        - parse_only: Pre-parse only (zero GPU); workers handle V+VV+MV.
        - distribute: Pre-parse + gap-fill V(MC) for lightweight workers;
          full workers handle V+VV+MV, lightweight workers handle VV+MV.
        """
```

**Step 4: Add `parse_only` branch in main loop (after line 571, before line 573)**

The parse_only mode uses the same `_run_pre_parse_buffer()` as distribute/mc_only, but since `get_processing_mode()` returns `"parse_only"` (not `"mc_only"`), the VV encoding at line 729-734 is automatically skipped. And the gap-fill V at line 577 only runs for `"distribute"` mode.

So parse_only naturally works with the existing non-auto path (line 573-590):
```python
                    # Non-auto modes (mc_only, parse_only, distribute): pre-parse pending jobs
                    self._run_pre_parse_buffer()

                    # Distribute mode: gap-fill V(MC) for lightweight workers
                    if (self._processing_mode == "distribute"
                            and self._has_lightweight_workers):
                        ...
```

Only the comment at line 573 needs updating to mention parse_only. No logic change needed since:
- `_run_pre_parse_buffer()` calls `_parse_single_job()` which checks `get_processing_mode() == "mc_only"` for VV (line 729-730) — parse_only won't match, so VV is skipped.
- Gap-fill only runs for `"distribute"` mode — parse_only won't trigger it.

**Step 5: Commit**

```
feat: add parse_only mode to ParseAheadPool
```

---

### Task 5: Frontend — Add Parse Only button to Admin UI

**Files:**
- Modify: `frontend/src/pages/AdminPage.jsx:352-381`

**Step 1: Add Parse Only button between MC Only and Auto**

Replace the button group (line 352-381):

```jsx
          <div className="flex rounded-lg overflow-hidden border border-gray-600">
            <button
              onClick={() => handleGlobalMode('mc_only')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'mc_only'
                  ? 'bg-amber-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_mc_only')}
            </button>
            <button
              onClick={() => handleGlobalMode('parse_only')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'parse_only'
                  ? 'bg-teal-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_parse_only')}
            </button>
            <button
              onClick={() => handleGlobalMode('auto')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'auto'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_auto')}
            </button>
          </div>
```

And add description for parse_only:
```jsx
        {globalMode === 'mc_only' && (
          <div className="text-xs text-amber-400/70 mt-2">{t('admin.worker_mode_mc_only_desc')}</div>
        )}
        {globalMode === 'parse_only' && (
          <div className="text-xs text-teal-400/70 mt-2">{t('admin.worker_mode_parse_only_desc')}</div>
        )}
        {globalMode === 'auto' && (
          <div className="text-xs text-blue-400/70 mt-2">{t('admin.worker_mode_auto_desc')}</div>
        )}
```

**Step 2: Commit**

```
feat: add Parse Only button to admin worker mode UI
```

---

### Task 6: Frontend — Update ClientWorkerView Phase display

**Files:**
- Modify: `frontend/src/components/ClientWorkerView.jsx:227-241`

**Step 1: Add parse_only mode to serverPhases logic**

```javascript
          const isMcOnly = wp.processingMode === 'mc_only';
          const isEmbedOnly = wp.processingMode === 'embed_only';
          // Note: in parse_only mode, worker receives "full" so no special
          // handling needed here — all phases shown as worker-handled (no SVR pills).
          // This is correct because parse_only workers do V+VV+MV.
```

No actual code change needed — when global mode is `parse_only`, workers get `"full"` processing_mode, which already falls through to `new Set()` (empty set = no SVR pills). This is correct behavior.

**Step 2: Commit (skip if no changes)**

No changes needed — existing logic handles this correctly.

---

### Task 7: Frontend — Add i18n labels

**Files:**
- Modify: `frontend/src/i18n/locales/en-US.json:459-462`
- Modify: `frontend/src/i18n/locales/ko-KR.json:459-462`

**Step 1: Add English labels**

After line 460 in en-US.json:
```json
  "admin.worker_mode_mc_only": "MC Only",
  "admin.worker_mode_mc_only_desc": "Server handles P+VV+MV, delegates MC to worker",
  "admin.worker_mode_parse_only": "Parse Only",
  "admin.worker_mode_parse_only_desc": "Server handles Parse only. V+VV+MV fully delegated to workers.",
  "admin.worker_mode_auto": "Auto",
  "admin.worker_mode_auto_desc": "Distribute V/VV/MV based on worker GPU. Server fills gaps.",
```

**Step 2: Add Korean labels**

After line 460 in ko-KR.json:
```json
  "admin.worker_mode_mc_only": "MC Only",
  "admin.worker_mode_mc_only_desc": "서버가 P+VV+MV 처리, 워커에 MC만 위임",
  "admin.worker_mode_parse_only": "Parse Only",
  "admin.worker_mode_parse_only_desc": "서버는 Parse만 처리. V+VV+MV는 모두 워커에게 위임.",
  "admin.worker_mode_auto": "자동",
  "admin.worker_mode_auto_desc": "워커 GPU에 따라 V/VV/MV 자동 분배, 서버가 빈 자리 채움.",
```

**Step 3: Commit**

```
feat: add i18n labels for parse_only worker mode
```

---

### Task 8: Verify and commit all changes

**Step 1: Run frontend build to verify no errors**

```bash
cd frontend && npm run build
```

**Step 2: Verify backend imports**

```bash
cd /Users/saintiron/Projects/Imagine && python -c "from backend.server.queue.manager import get_processing_mode; print(get_processing_mode())"
```

**Step 3: Final commit if not already committed per-task**

All tasks should already be committed individually.
