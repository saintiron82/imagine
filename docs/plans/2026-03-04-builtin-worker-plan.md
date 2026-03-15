# Builtin Worker Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add "builtin_worker" as a 4th global processing mode so the server processes the full pipeline (P→V→VV→MV) regardless of external worker connections, with UI visibility via a virtual worker session.

**Architecture:** Extend ParseAheadPool's existing `auto` mode processing with a new `builtin_worker` global mode that bypasses worker-presence checks. A virtual `__builtin__` worker session provides UI visibility (progress, phase, speed) in the worker table.

**Tech Stack:** Python/FastAPI (backend), React/Tailwind (frontend), SQLite (worker_sessions)

---

### Task 1: Backend — `get_processing_mode()` allow `builtin_worker`

**Files:**
- Modify: `backend/server/queue/manager.py:16-33`

**Step 1: Add `builtin_worker` to allowed modes**

In `get_processing_mode()`, add `"builtin_worker"` to the validation set:

```python
def get_processing_mode() -> str:
    """Get effective processing mode from config.

    Returns "mc_only", "parse_only", "auto", or "builtin_worker" (default: auto).
    - mc_only: Server P+VV+MV, workers do V(MC) only.
    - parse_only: Server P only (zero GPU), workers do V+VV+MV (full mode).
    - auto: Server P + gap-fill, workers distribute V/VV/MV by capability.
    - builtin_worker: Server processes full P→V→VV→MV always, regardless of workers.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        mode = cfg.get("server.processing_mode") or "auto"
        # Normalize legacy values
        if mode not in ("mc_only", "parse_only", "auto", "builtin_worker"):
            mode = "auto"
        return mode
    except Exception:
        return "auto"
```

**Step 2: Commit**

```
feat: allow builtin_worker in get_processing_mode()
```

---

### Task 2: Backend — Virtual worker session helpers

**Files:**
- Modify: `backend/server/routers/workers.py` (add 2 new functions after `_recalculate_server_pools`)

**Step 1: Add `_ensure_builtin_worker_session()`**

Add after `_recalculate_server_pools()` function (~line 198):

```python
BUILTIN_WORKER_NAME = "__builtin__"


def _ensure_builtin_worker_session(db: "SQLiteDB") -> int:
    """Create or reactivate the virtual builtin worker session.

    Returns the session_id.
    """
    now = _utcnow_sql()
    cursor = db.conn.cursor()

    # Check if already online
    cursor.execute(
        "SELECT id FROM worker_sessions WHERE worker_name = ? AND status = 'online'",
        (BUILTIN_WORKER_NAME,),
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    # Try reactivating existing offline session
    cursor.execute(
        """UPDATE worker_sessions
           SET status = 'online', last_heartbeat = ?, disconnected_at = NULL
           WHERE worker_name = ? AND status = 'offline'""",
        (now, BUILTIN_WORKER_NAME),
    )
    if cursor.rowcount > 0:
        db.conn.commit()
        cursor.execute(
            "SELECT id FROM worker_sessions WHERE worker_name = ? AND status = 'online'",
            (BUILTIN_WORKER_NAME,),
        )
        row = cursor.fetchone()
        logger.info(f"Builtin worker session reactivated (id={row[0]})")
        return row[0]

    # Create new session (user_id=1 = default admin)
    batch_size = 5
    try:
        from backend.utils.config import get_config
        batch_size = get_config().get("server.auto_processing.batch_size", 5)
    except Exception:
        pass

    cursor.execute(
        """INSERT INTO worker_sessions
           (user_id, worker_name, hostname, batch_capacity, status,
            processing_mode_override, connected_at, last_heartbeat)
           VALUES (1, ?, 'server (built-in)', ?, 'online', 'full', ?, ?)""",
        (BUILTIN_WORKER_NAME, batch_size, now, now),
    )
    session_id = cursor.lastrowid
    db.conn.commit()
    logger.info(f"Builtin worker session created (id={session_id})")
    return session_id


def _deactivate_builtin_worker_session(db: "SQLiteDB"):
    """Mark builtin worker session as offline."""
    now = _utcnow_sql()
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions
           SET status = 'offline', disconnected_at = ?,
               current_job_id = NULL, current_file = NULL, current_phase = NULL
           WHERE worker_name = ? AND status = 'online'""",
        (now, BUILTIN_WORKER_NAME),
    )
    if cursor.rowcount > 0:
        db.conn.commit()
        logger.info("Builtin worker session deactivated")
```

**Step 2: Commit**

```
feat: add virtual builtin worker session helpers
```

---

### Task 3: Backend — `_recalculate_server_pools()` handle `builtin_worker`

**Files:**
- Modify: `backend/server/routers/workers.py:92-197` (`_recalculate_server_pools`)

**Step 1: Add builtin_worker handling at the top of the function**

After `global_mode = _get_global_processing_mode()` (line 111), add early return for `builtin_worker`:

```python
    global_mode = _get_global_processing_mode()  # "mc_only" | "parse_only" | "auto" | "builtin_worker"

    # ── builtin_worker: server always processes full pipeline ──
    if global_mode == "builtin_worker":
        if hasattr(app.state, "parse_ahead") and app.state.parse_ahead:
            old_mode = getattr(app.state.parse_ahead, "_processing_mode", None)
            app.state.parse_ahead._processing_mode = "auto"
            if old_mode != "auto":
                logger.info("Builtin worker mode: ParseAheadPool set to auto (full pipeline)")

        # Stop EmbedAheadPool if running (not needed — ParseAhead does P→V→VV→MV)
        if (hasattr(app.state, "embed_ahead") and app.state.embed_ahead
                and getattr(app.state.embed_ahead, "_thread", None)
                and app.state.embed_ahead._thread.is_alive()):
            try:
                app.state.embed_ahead.stop()
                app.state.embed_ahead = None
                logger.info("EmbedAheadPool stopped (builtin_worker mode)")
            except Exception as e:
                logger.warning(f"Failed to stop EmbedAheadPool: {e}")

        # Ensure virtual worker session exists
        _ensure_builtin_worker_session(db)
        return
```

Also: exclude `__builtin__` session from `has_workers` count so it doesn't affect other modes:

Change line 106-109 from:
```python
    cursor.execute(
        "SELECT processing_mode_override FROM worker_sessions WHERE status = 'online'"
    )
    rows = cursor.fetchall()
    has_workers = len(rows) > 0
```
to:
```python
    cursor.execute(
        "SELECT processing_mode_override FROM worker_sessions WHERE status = 'online' AND worker_name != ?",
        (BUILTIN_WORKER_NAME,),
    )
    rows = cursor.fetchall()
    has_workers = len(rows) > 0
```

**Step 2: Commit**

```
feat: handle builtin_worker in _recalculate_server_pools
```

---

### Task 4: Backend — `admin_update_global_config()` allow `builtin_worker`

**Files:**
- Modify: `backend/server/routers/workers.py:696-756`

**Step 1: Update validation and add builtin_worker case**

Change line 713-714 from:
```python
    if mode not in ("mc_only", "parse_only", "auto"):
        raise HTTPException(status_code=400, detail="processing_mode must be 'mc_only', 'parse_only', or 'auto'")
```
to:
```python
    if mode not in ("mc_only", "parse_only", "auto", "builtin_worker"):
        raise HTTPException(status_code=400, detail="processing_mode must be 'mc_only', 'parse_only', 'auto', or 'builtin_worker'")
```

Add `builtin_worker` case after `elif mode == "auto":` block (before `db.conn.commit()`):

```python
    elif mode == "builtin_worker":
        # Builtin worker: server processes full pipeline. External workers unaffected.
        _ensure_builtin_worker_session(db)
```

Also: when switching AWAY from `builtin_worker`, deactivate the virtual session.
Add before `db.conn.commit()` (line 738):

```python
    # Deactivate builtin worker session when switching to other modes
    if mode != "builtin_worker":
        _deactivate_builtin_worker_session(db)
```

**Step 2: Update `GlobalModeUpdate` schema**

Change line 55:
```python
class GlobalModeUpdate(BaseModel):
    processing_mode: str  # "mc_only" | "parse_only" | "auto" | "builtin_worker"
```

**Step 3: Commit**

```
feat: allow builtin_worker in admin global config API
```

---

### Task 5: Backend — Heartbeat watchdog exclude `__builtin__`

**Files:**
- Modify: `backend/server/app.py:283-290`

**Step 1: Exclude builtin worker from stale check**

Change line 283-289 from:
```python
                cursor.execute(
                    """SELECT id, worker_name FROM worker_sessions
                       WHERE status = 'online'
                         AND last_heartbeat IS NOT NULL
                         AND datetime(last_heartbeat, '+' || ? || ' minutes') < datetime('now')""",
                    (TIMEOUT,)
                )
```
to:
```python
                cursor.execute(
                    """SELECT id, worker_name FROM worker_sessions
                       WHERE status = 'online'
                         AND worker_name != '__builtin__'
                         AND last_heartbeat IS NOT NULL
                         AND datetime(last_heartbeat, '+' || ? || ' minutes') < datetime('now')""",
                    (TIMEOUT,)
                )
```

**Step 2: Commit**

```
fix: exclude builtin worker from heartbeat timeout check
```

---

### Task 6: Backend — ParseAheadPool update virtual session progress

**Files:**
- Modify: `backend/server/queue/parse_ahead.py`

**Step 1: Add `_update_builtin_session()` method**

Add after `_process_auto_batch()` method (~line 263):

```python
    def _update_builtin_session(self, phase: str, file_name: str = None,
                                 jobs_done: int = 0):
        """Update virtual builtin worker session for UI visibility.

        Only active when global mode is builtin_worker.
        """
        try:
            from backend.server.queue.manager import get_processing_mode, _utcnow_sql
            if get_processing_mode() != "builtin_worker":
                return

            cursor = self.db.conn.cursor()
            now = _utcnow_sql()

            if jobs_done > 0:
                cursor.execute(
                    """UPDATE worker_sessions
                       SET current_phase = ?, current_file = ?,
                           jobs_completed = jobs_completed + ?,
                           last_heartbeat = ?
                       WHERE worker_name = '__builtin__' AND status = 'online'""",
                    (phase, file_name, jobs_done, now),
                )
            else:
                cursor.execute(
                    """UPDATE worker_sessions
                       SET current_phase = ?, current_file = ?,
                           last_heartbeat = ?
                       WHERE worker_name = '__builtin__' AND status = 'online'""",
                    (phase, file_name, now),
                )
            self.db.conn.commit()
        except Exception as e:
            logger.debug(f"Builtin session update failed: {e}")
```

**Step 2: Add progress calls in `_process_auto_batch()`**

Insert progress calls at each phase boundary in `_process_auto_batch()`:

After `logger.info(f"Auto processing: starting batch of {len(jobs)} files")` (line 125):
```python
        self._update_builtin_session("parse", f"batch({len(jobs)})")
```

After `# ── Phase V: Vision/VLM (MC generation) ──` comment (line 191):
```python
        self._update_builtin_session("vision", f"batch({len(contexts)})")
```

After `# ── Phase VV: SigLIP2 visual embedding ──` comment (line 199):
```python
        self._update_builtin_session("embed_vv", f"batch({len(contexts)})")
```

After `# ── Phase MV: Qwen3-Embedding text embedding ──` comment (line 218):
```python
        self._update_builtin_session("embed_mv", f"batch({len(contexts)})")
```

After `logger.info(f"Auto processing: {completed_count} files completed (P→V→VV→MV)")` (line 262):
```python
        self._update_builtin_session(None, None, jobs_done=completed_count)
```

**Step 3: Commit**

```
feat: update builtin worker session progress during auto processing
```

---

### Task 7: Frontend — i18n translation keys

**Files:**
- Modify: `frontend/src/i18n/locales/ko-KR.json`
- Modify: `frontend/src/i18n/locales/en-US.json`

**Step 1: Add Korean translations**

Add after `"admin.worker_mode_auto_desc"` line in `ko-KR.json`:

```json
"admin.worker_mode_builtin": "내장 워커",
"admin.worker_mode_builtin_desc": "서버가 직접 전체 파이프라인(P→V→VV→MV)을 처리합니다. 외부 워커 없이 독립 운영.",
```

**Step 2: Add English translations**

Add after `"admin.worker_mode_auto_desc"` line in `en-US.json`:

```json
"admin.worker_mode_builtin": "Built-in Worker",
"admin.worker_mode_builtin_desc": "Server processes the full pipeline (P→V→VV→MV) directly. Operates independently without external workers.",
```

**Step 3: Commit**

```
feat: add i18n keys for builtin worker mode
```

---

### Task 8: Frontend — AdminPage UI button + conditional sections

**Files:**
- Modify: `frontend/src/pages/AdminPage.jsx`

**Step 1: Add builtin_worker button in the mode button group**

After the `Parse Only` button (line 372) and before the `자동` button (line 373), add:

```jsx
            <button
              onClick={() => handleGlobalMode('builtin_worker')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'builtin_worker'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_builtin')}
            </button>
```

**Step 2: Add description for builtin_worker mode**

After the `auto` description block (line 393), add:

```jsx
        {globalMode === 'builtin_worker' && (
          <div className="text-xs text-purple-400/70 mt-2">{t('admin.worker_mode_builtin_desc')}</div>
        )}
```

**Step 3: Hide "서버 자동 처리" section when builtin_worker**

Wrap the auto-processing section (lines 396-433) with a condition:

```jsx
      {globalMode !== 'builtin_worker' && (
        {/* existing auto processing section */}
      )}
```

**Step 4: Mark `__builtin__` worker specially in the table**

In the worker table row rendering, check for `worker_name === '__builtin__'` and show a distinct badge/icon (e.g., purple "내장" tag instead of the usual "full" badge).

**Step 5: Commit**

```
feat: add builtin worker mode button and UI handling in AdminPage
```

---

### Task 9: Integration test — verify full flow

**Step 1: Manual test checklist**

1. Start server (`python -m backend.server.app`)
2. Open Admin → Workers tab
3. Click "내장 워커" button → verify:
   - Button turns purple
   - "서버 자동 처리" section disappears
   - Worker table shows "내장 워커" / "server (built-in)" / online
4. Trigger a discover scan → verify:
   - ParseAheadPool processes P→V→VV→MV
   - Worker table shows current_phase updates
   - Jobs complete normally
5. Switch to "자동" → verify:
   - Builtin worker goes offline/disappears
   - "서버 자동 처리" section reappears
6. Switch back to "내장 워커" → verify session reactivates

**Step 2: Commit all and verify build**

```bash
cd frontend && npm run build
```

**Step 3: Final commit**

```
chore: verify builtin worker mode integration
```
