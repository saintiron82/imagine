"""
Worker session management — connect, heartbeat, disconnect, admin control.

Workers register sessions on connect and send periodic heartbeats.
Server piggybacks commands (stop/block) in heartbeat responses.
"""

import logging
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, get_current_user, require_admin
from backend.server.queue.manager import _utcnow_sql, JobQueueManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workers"])


# ── Schemas ──────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    worker_name: str
    hostname: Optional[str] = None
    batch_capacity: int = 5
    resources: Optional[dict] = None  # GPU info for immediate mode detection at connect time


class HeartbeatRequest(BaseModel):
    session_id: int
    jobs_completed: int = 0
    jobs_failed: int = 0
    current_job_id: Optional[int] = None
    current_file: Optional[str] = None
    current_phase: Optional[str] = None
    pool_size: int = 0
    resources: Optional[dict] = None
    throttle_level: Optional[str] = None  # normal/warning/danger/critical
    worker_state: Optional[str] = None    # active/idle/resting
    phase_counts: Optional[dict] = None  # {"mc": N, "vv": N, "mv": N}


class DisconnectRequest(BaseModel):
    session_id: int


class WorkerConfigUpdate(BaseModel):
    batch_capacity: Optional[int] = None     # 1~32 | null (=worker default)


class AutoProcessingUpdate(BaseModel):
    enabled: Optional[bool] = None
    mode: Optional[str] = None  # "full" | "parse_vv" | "parse_only"
    rest_after_batch_s: Optional[int] = None
    batch_size: Optional[int] = None
    verbose_log: Optional[bool] = None


class EmbeddedWorkerUpdate(BaseModel):
    enabled: Optional[bool] = None


def _get_global_processing_mode() -> str:
    """Read global processing_mode from config (cached singleton).

    Checks server.processing_mode first (Admin API), then falls back
    to worker.processing_mode (WorkerPage UI in user-settings.yaml).
    """
    from backend.server.queue.manager import get_processing_mode
    return get_processing_mode()


def _auto_detect_mode_from_resources(resources: dict) -> Optional[str]:
    """워커 resources_json에서 GPU 정보를 읽어 processing_mode 자동 결정.

    서버의 현재 활성 tier를 기준으로 워커가 VLM을 실행할 수 있는지 판단한다.

    Returns:
        "full" or "embed_only", or None if detection fails
    """
    try:
        from backend.utils.gpu_detect import determine_worker_mode
        from backend.utils.tier_config import get_active_tier
        server_tier, _ = get_active_tier()
        return determine_worker_mode(resources, server_tier)
    except Exception as e:
        logger.warning(f"Auto mode detection failed: {e}")
        return None


def _recalculate_server_pools(app, db: "SQLiteDB") -> None:
    """Recalculate server pool state when workers connect/disconnect.

    ParseAheadPool is always parse_only (PSD parse + thumbnail).
    Embedded worker auto-management based on worker availability.
    """
    if not app:
        return

    cursor = db.conn.cursor()

    # Survey online worker modes (exclude builtin embedded worker)
    cursor.execute(
        """SELECT processing_mode_override FROM worker_sessions
           WHERE status = 'online' AND worker_name != ?""",
        (BUILTIN_WORKER_NAME,),
    )
    worker_modes = [r[0] or "full" for r in cursor.fetchall()]
    has_workers = len(worker_modes) > 0

    # Publish mode for stats API
    from backend.server.queue.manager import set_server_pool_mode
    set_server_pool_mode("parse_only")

    # Seed demand for ParseAheadPool when workers are active
    if has_workers:
        try:
            from backend.server.queue.base_ahead_pool import BaseAheadPool
            BaseAheadPool.record_claim(session_id=-1, count=10)
        except Exception:
            pass

    # Embedded worker stays running regardless of external workers.
    # External workers ADD capacity (1+1=2x), not replace embedded.
    from backend.server.embedded_worker import get_status as _ew_get_status
    ew_running = _ew_get_status().get("running", False)

    logger.debug(f"Pool recalculated: mode=parse_only, workers={len(worker_modes)}, ew={ew_running}")


# ── Builtin worker virtual session ──────────────────────────

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
    else:
        db.conn.commit()  # Release WAL write lock from 0-row UPDATE

    # Create new session — find the first admin user_id dynamically
    batch_size = 5
    try:
        from backend.utils.config import get_config
        batch_size = get_config().get("server.auto_processing.batch_size", 5)
    except Exception:
        pass

    # Find an admin user_id (not hardcoded to 1)
    cursor.execute("SELECT id FROM users WHERE role = 'admin' AND is_active = 1 LIMIT 1")
    admin_row = cursor.fetchone()
    admin_user_id = admin_row[0] if admin_row else 1

    cursor.execute(
        """INSERT INTO worker_sessions
           (user_id, worker_name, hostname, batch_capacity, status,
            processing_mode_override, connected_at, last_heartbeat)
           VALUES (?, ?, 'server (built-in)', ?, 'online', 'full', ?, ?)""",
        (admin_user_id, BUILTIN_WORKER_NAME, batch_size, now, now),
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
        logger.info("Builtin worker session deactivated")
    db.conn.commit()  # Always commit to release WAL write lock


# ── Worker → Server endpoints ────────────────────────────────

@router.post("/workers/connect")
def worker_connect(
    req: ConnectRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Register a new worker session."""
    # Reject external workers when builtin_worker mode is active
    global_mode = _get_global_processing_mode()
    if global_mode == "builtin_worker":
        logger.warning(f"Rejected worker connect from {req.worker_name}: builtin_worker mode active")
        raise HTTPException(
            status_code=409,
            detail="Server is in built-in worker mode. External workers are not accepted."
        )

    now = _utcnow_sql()
    cursor = db.conn.cursor()

    # Mark any stale sessions from this user as offline
    cursor.execute(
        """UPDATE worker_sessions SET status = 'offline', disconnected_at = ?
           WHERE user_id = ? AND status = 'online'""",
        (now, user["id"])
    )

    cursor.execute(
        """INSERT INTO worker_sessions
           (user_id, worker_name, hostname, batch_capacity, status, connected_at, last_heartbeat)
           VALUES (?, ?, ?, ?, 'online', ?, ?)""",
        (user["id"], req.worker_name, req.hostname, req.batch_capacity, now, now)
    )
    session_id = cursor.lastrowid

    # Auto-detect processing_mode from resources provided at connect time.
    # Only set if no manual override already exists (admin pre-configuration).
    auto_mode = None
    if req.resources:
        auto_mode = _auto_detect_mode_from_resources(req.resources)
        if auto_mode:
            # Store as initial assigned_mode, NOT override (override = admin manual only)
            cursor.execute(
                """UPDATE worker_sessions
                   SET assigned_mode = ?, resources_json = ?,
                       processing_mode_override = NULL
                   WHERE id = ?""",
                (auto_mode, json.dumps(req.resources), session_id)
            )
            vram_gb = req.resources.get('gpu_memory_total_gb') or 0
            gpu_type = req.resources.get('gpu_type') or 'none'
            logger.info(
                f"Worker {req.worker_name} auto-detected mode: {auto_mode} "
                f"(VRAM={vram_gb:.1f}GB, GPU={gpu_type})"
            )

    db.conn.commit()

    # Recalculate server pools with the new worker included
    _recalculate_server_pools(request.app, db)

    # Determine effective processing mode:
    # - mc_only global → ALL workers get mc_only (regardless of GPU capability)
    # - auto global → workers get their auto-detected mode (full/embed_only)
    cursor.execute(
        "SELECT processing_mode_override, batch_capacity_override FROM worker_sessions WHERE id = ?",
        (session_id,)
    )
    ov = cursor.fetchone()
    effective_batch = (ov[1] if ov and ov[1] else None) or req.batch_capacity

    global_mode = _get_global_processing_mode()
    if global_mode == "mc_only":
        processing_mode = "mc_only"
    elif ov and ov[0]:
        processing_mode = ov[0]  # Admin manual override
    else:
        # Dynamic mode: server decides based on queue state
        from backend.server.queue.manager import JobQueueManager
        try:
            _qm = JobQueueManager(db)
            processing_mode = _qm._decide_worker_mode(session_id)
        except Exception:
            processing_mode = "full"

    logger.info(f"Worker connected: {req.worker_name} (session={session_id}, user={user['username']}, mode={processing_mode})")
    return {
        "session_id": session_id,
        "pool_hint": effective_batch * 2,
        "batch_hint": effective_batch,
        "processing_mode": processing_mode,
    }


@router.post("/workers/heartbeat")
def worker_heartbeat(
    req: HeartbeatRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Periodic heartbeat from worker. Returns pending commands."""
    now = _utcnow_sql()
    cursor = db.conn.cursor()

    # Verify session ownership + read overrides
    cursor.execute(
        """SELECT id, status, pending_command, batch_capacity,
                  processing_mode_override, batch_capacity_override
           FROM worker_sessions WHERE id = ? AND user_id = ?""",
        (req.session_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_status = row[1]
    pending_cmd = row[2]
    batch_capacity = row[3]
    mode_override = row[4]
    batch_override = row[5]

    # Blocked sessions should stop immediately
    if session_status == "blocked":
        return {"ok": True, "command": "block", "pool_hint": 0}

    # Update metrics (merge throttle_level + worker_state into resources JSON)
    resources_data = dict(req.resources) if req.resources else {}
    if req.throttle_level:
        resources_data["throttle_level"] = req.throttle_level
    if req.worker_state:
        resources_data["worker_state"] = req.worker_state
    if req.phase_counts:
        resources_data["phase_counts"] = req.phase_counts
    # Track phase_job_count: increment by delta of jobs_completed since last heartbeat
    cursor.execute(
        "SELECT jobs_completed FROM worker_sessions WHERE id = ?",
        (req.session_id,)
    )
    prev_row = cursor.fetchone()
    prev_completed = prev_row[0] if prev_row else 0
    delta = max(0, req.jobs_completed - prev_completed)

    cursor.execute(
        """UPDATE worker_sessions
           SET last_heartbeat = ?,
               jobs_completed = ?,
               jobs_failed = ?,
               current_job_id = ?,
               current_file = ?,
               current_phase = ?,
               resources_json = ?,
               phase_job_count = COALESCE(phase_job_count, 0) + ?,
               pending_command = NULL
           WHERE id = ?""",
        (now, req.jobs_completed, req.jobs_failed,
         req.current_job_id, req.current_file, req.current_phase,
         json.dumps(resources_data) if resources_data else None,
         delta, req.session_id)
    )
    db.conn.commit()

    # Auto-detect mode on first heartbeat when:
    #   - worker reports resources (GPU info available)
    #   - no manual override is set (NULL = never auto-detected or admin-configured)
    pool_needs_recalc = False
    if req.resources and not mode_override:
        detected_mode = _auto_detect_mode_from_resources(req.resources)
        if detected_mode:
            cursor.execute(
                "UPDATE worker_sessions SET processing_mode_override = ? WHERE id = ?",
                (detected_mode, req.session_id)
            )
            db.conn.commit()
            mode_override = detected_mode
            pool_needs_recalc = True
            logger.info(
                f"Worker session {req.session_id} auto-detected mode: {detected_mode} "
                f"(VRAM={req.resources.get('gpu_memory_total_gb', 0):.1f}GB)"
            )

    if pool_needs_recalc:
        _recalculate_server_pools(request.app, db)

    # Determine effective processing mode:
    # - builtin_worker → stop external workers
    # - mc_only global → ALL workers get mc_only
    # - parse_only global → ALL workers get full (V+VV+MV)
    # - auto global → workers get their auto-detected mode (full/embed_only)
    global_mode = _get_global_processing_mode()

    # In builtin_worker mode, tell external workers to stop
    if global_mode == "builtin_worker":
        # Check if this is an external worker (not the built-in one)
        cursor.execute(
            "SELECT worker_name FROM worker_sessions WHERE id = ?",
            (req.session_id,)
        )
        name_row = cursor.fetchone()
        if name_row and name_row[0] != BUILTIN_WORKER_NAME:
            logger.info(f"Sending stop to external worker session {req.session_id}: builtin_worker mode active")
            return {
                "ok": True,
                "command": "stop",
                "pool_hint": 0,
                "batch_hint": 0,
                "processing_mode": "full",
            }

    if global_mode == "mc_only":
        processing_mode = "mc_only"
    elif mode_override:
        processing_mode = mode_override  # Admin manual override
    else:
        # Dynamic mode: server decides based on queue state + worker's current phase
        from backend.server.queue.manager import JobQueueManager
        try:
            _qm = JobQueueManager(db)
            processing_mode = _qm._decide_worker_mode(req.session_id)
        except Exception:
            processing_mode = "full"
    # For embedded worker: read live batch_size from config (Admin UI changes)
    cursor.execute("SELECT worker_name FROM worker_sessions WHERE id = ?", (req.session_id,))
    wn_row = cursor.fetchone()
    if wn_row and wn_row[0] == BUILTIN_WORKER_NAME:
        try:
            from backend.utils.config import get_config
            effective_batch = get_config().get("server.auto_processing.batch_size", 5)
        except Exception:
            effective_batch = batch_override or batch_capacity
    else:
        effective_batch = batch_override or batch_capacity

    # Resource-aware batch_hint: throttle down based on worker resource pressure
    throttle = resources_data.get("throttle_level", "normal") if resources_data else "normal"
    if throttle == "critical":
        resource_batch_hint = 0
    elif throttle == "danger":
        resource_batch_hint = 1
    elif throttle == "warning":
        resource_batch_hint = max(1, int(effective_batch * 0.5))
    else:
        resource_batch_hint = effective_batch

    return {
        "ok": True,
        "command": pending_cmd,
        "pool_hint": resource_batch_hint * 2,
        "batch_hint": resource_batch_hint,
        "processing_mode": processing_mode,
    }


@router.post("/workers/disconnect")
def worker_disconnect(
    req: DisconnectRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Worker graceful disconnect. Reclaims assigned jobs back to pending."""
    # Reclaim jobs assigned to this worker (phase_completed preserved)
    queue = JobQueueManager(db)
    reclaimed = queue.reclaim_worker_jobs(req.session_id)

    now = _utcnow_sql()
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions SET status = 'offline', disconnected_at = ?
           WHERE id = ? AND user_id = ?""",
        (now, req.session_id, user["id"])
    )
    db.conn.commit()
    logger.info(f"Worker disconnected: session={req.session_id}, reclaimed {reclaimed} jobs")

    # Recalculate pools after worker leaves (may deactivate EmbedAhead if last mc_only)
    _recalculate_server_pools(request.app, db)

    return {"ok": True, "reclaimed": reclaimed}


# ── User self-service endpoints ──────────────────────────────

@router.get("/workers/my")
def list_my_workers(
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List current user's worker sessions."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT id, worker_name, hostname, status, batch_capacity,
                  jobs_completed, jobs_failed, current_file, current_phase,
                  last_heartbeat, connected_at
           FROM worker_sessions
           WHERE user_id = ?
           ORDER BY connected_at DESC
           LIMIT 20""",
        (user["id"],)
    )
    workers = []
    for row in cursor.fetchall():
        workers.append({
            "id": row[0], "worker_name": row[1], "hostname": row[2],
            "status": row[3], "batch_capacity": row[4],
            "jobs_completed": row[5], "jobs_failed": row[6],
            "current_file": row[7], "current_phase": row[8],
            "last_heartbeat": row[9], "connected_at": row[10],
        })
    return {"workers": workers}


@router.post("/workers/{session_id}/stop")
def stop_my_worker(
    session_id: int,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Stop own worker (sets pending_command='stop')."""
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions SET pending_command = 'stop'
           WHERE id = ? AND user_id = ? AND status = 'online'""",
        (session_id, user["id"])
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found or not online")
    db.conn.commit()
    return {"ok": True}


# ── Admin endpoints ──────────────────────────────────────────

@router.get("/admin/workers")
def admin_list_workers(
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """List all worker sessions (admin only), with per-worker throughput."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT ws.id, ws.worker_name, ws.hostname, ws.status,
                  ws.batch_capacity, ws.jobs_completed, ws.jobs_failed,
                  ws.current_file, ws.current_phase,
                  ws.last_heartbeat, ws.connected_at, ws.disconnected_at,
                  ws.pending_command, u.username, ws.user_id,
                  ws.processing_mode_override, ws.batch_capacity_override,
                  ws.resources_json, ws.assigned_mode, ws.phase_job_count
           FROM worker_sessions ws
           JOIN users u ON ws.user_id = u.id
           ORDER BY
               CASE ws.status WHEN 'online' THEN 0 WHEN 'blocked' THEN 1 ELSE 2 END,
               ws.last_heartbeat DESC
           LIMIT 100"""
    )
    rows = cursor.fetchall()

    # Per-worker throughput: count any phase completion in last 5min/1min
    # Uses mc_completed_at (MC phase) as primary — most common worker activity.
    # Falls back to completed_at for VV/MV phases.
    # Per-worker throughput: count any phase completion (mc_completed_at or updated_at)
    cursor.execute(
        """SELECT worker_session_id, COUNT(*) FROM job_queue
           WHERE worker_session_id IS NOT NULL
             AND (
               (mc_completed_at IS NOT NULL AND datetime(mc_completed_at) > datetime('now', '-5 minutes'))
               OR (updated_at IS NOT NULL AND datetime(updated_at) > datetime('now', '-5 minutes')
                   AND json_extract(phase_completed, '$.vv') = 1)
             )
           GROUP BY worker_session_id"""
    )
    session_recent_5m = dict(cursor.fetchall())

    cursor.execute(
        """SELECT worker_session_id, COUNT(*) FROM job_queue
           WHERE worker_session_id IS NOT NULL
             AND (
               (mc_completed_at IS NOT NULL AND datetime(mc_completed_at) > datetime('now', '-1 minute'))
               OR (updated_at IS NOT NULL AND datetime(updated_at) > datetime('now', '-1 minute')
                   AND json_extract(phase_completed, '$.vv') = 1)
             )
           GROUP BY worker_session_id"""
    )
    session_recent_1m = dict(cursor.fetchall())

    # Fallback: per-user throughput
    cursor.execute(
        """SELECT assigned_to, COUNT(*) FROM job_queue
           WHERE worker_session_id IS NULL
             AND (
               (mc_completed_at IS NOT NULL AND datetime(mc_completed_at) > datetime('now', '-5 minutes'))
               OR (updated_at IS NOT NULL AND datetime(updated_at) > datetime('now', '-5 minutes')
                   AND json_extract(phase_completed, '$.vv') = 1)
             )
           GROUP BY assigned_to"""
    )
    user_recent_5m = dict(cursor.fetchall())

    cursor.execute(
        """SELECT assigned_to, COUNT(*) FROM job_queue
           WHERE worker_session_id IS NULL
             AND (
               (mc_completed_at IS NOT NULL AND datetime(mc_completed_at) > datetime('now', '-1 minute'))
               OR (updated_at IS NOT NULL AND datetime(updated_at) > datetime('now', '-1 minute')
                   AND json_extract(phase_completed, '$.vv') = 1)
             )
           GROUP BY assigned_to"""
    )
    user_recent_1m = dict(cursor.fetchall())

    workers = []
    for row in rows:
        session_id = row[0]
        user_id = row[14]

        # Prefer per-session throughput; fall back to per-user for legacy jobs
        r1 = session_recent_1m.get(session_id, 0) or user_recent_1m.get(user_id, 0)
        r5 = session_recent_5m.get(session_id, 0) or user_recent_5m.get(user_id, 0)
        # Use 1-min if active, otherwise 5-min average
        if r1 > 0:
            throughput = float(r1)
        elif r5 > 0:
            throughput = round(r5 / 5.0, 1)
        else:
            throughput = 0.0

        workers.append({
            "id": session_id, "worker_name": row[1], "hostname": row[2],
            "status": row[3], "batch_capacity": row[4],
            "jobs_completed": row[5], "jobs_failed": row[6],
            "current_file": row[7], "current_phase": row[8],
            "last_heartbeat": row[9], "connected_at": row[10],
            "disconnected_at": row[11], "pending_command": row[12],
            "username": row[13],
            "throughput": throughput,
            "processing_mode_override": row[15],
            "batch_capacity_override": row[16],
            "resources": json.loads(row[17]) if row[17] else None,
            "assigned_mode": row[18],
            "phase_job_count": row[19] or 0,
        })
    return {
        "workers": workers,
        "global_processing_mode": _get_global_processing_mode(),
    }


@router.post("/admin/workers/{session_id}/stop")
def admin_stop_worker(
    session_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Send stop command to a worker (admin only)."""
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions SET pending_command = 'stop'
           WHERE id = ? AND status = 'online'""",
        (session_id,)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found or not online")
    db.conn.commit()
    logger.info(f"Admin sent stop to worker session {session_id}")
    return {"ok": True}


@router.post("/admin/workers/{session_id}/block")
def admin_block_worker(
    session_id: int,
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Block a worker — it will be forced to disconnect (admin only).

    Immediately reclaims all jobs assigned to this worker back to pending.
    """
    # Reclaim jobs before blocking (phase_completed preserved)
    queue = JobQueueManager(db)
    reclaimed = queue.reclaim_worker_jobs(session_id)

    now = _utcnow_sql()
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions
           SET status = 'blocked', pending_command = 'block', disconnected_at = ?
           WHERE id = ?""",
        (now, session_id)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    db.conn.commit()
    logger.info(f"Admin blocked worker session {session_id}, reclaimed {reclaimed} jobs")

    # Recalculate pools after blocking
    _recalculate_server_pools(request.app, db)

    return {"ok": True, "reclaimed": reclaimed}


@router.patch("/admin/workers/{session_id}/config")
def admin_update_worker_config(
    session_id: int,
    req: WorkerConfigUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Update per-worker settings (batch_capacity only; processing mode is auto-detected).

    Applied on next heartbeat (~30s).
    """
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions
           SET batch_capacity_override = ?
           WHERE id = ? AND status = 'online'""",
        (req.batch_capacity, session_id)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found or not online")
    db.conn.commit()
    logger.info(f"Admin updated worker config: session={session_id}, batch={req.batch_capacity}")

    return {"ok": True}


@router.get("/admin/workers/auto-processing")
def admin_get_auto_processing(
    admin: dict = Depends(require_admin),
):
    """Get current auto_processing config."""
    from backend.utils.config import get_config
    cfg = get_config()
    return {
        "enabled": cfg.get("server.auto_processing.enabled", True),
        "rest_after_batch_s": cfg.get("server.auto_processing.rest_after_batch_s", 30),
        "batch_size": cfg.get("server.auto_processing.batch_size", 5),
        "verbose_log": cfg.get("worker.verbose_log", False),
    }


@router.patch("/admin/workers/auto-processing")
def admin_update_auto_processing(
    req: AutoProcessingUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Update server auto_processing settings and recalculate pools."""
    from backend.utils.config import get_config
    cfg = get_config()

    if req.enabled is not None:
        cfg.save_user_setting("server.auto_processing.enabled", req.enabled)
    if req.mode is not None and req.mode in ("full", "parse_vv", "parse_only"):
        cfg.save_user_setting("server.auto_processing.mode", req.mode)
    if req.rest_after_batch_s is not None:
        cfg.save_user_setting("server.auto_processing.rest_after_batch_s", req.rest_after_batch_s)
    if req.batch_size is not None:
        cfg.save_user_setting("server.auto_processing.batch_size", req.batch_size)
    if req.verbose_log is not None:
        cfg.save_user_setting("worker.verbose_log", req.verbose_log)
        # Apply to running embedded worker immediately
        try:
            import backend.server.embedded_worker as ew_module
            if ew_module._worker_daemon:
                ew_module._worker_daemon.verbose_log = req.verbose_log
        except Exception:
            pass

    # ParseAheadPool is always parse_only — no mode switching needed
    if req.mode is not None:
        from backend.server.queue.manager import set_server_pool_mode
        set_server_pool_mode(req.mode)

    # Start/stop embedded worker based on auto_processing toggle
    if req.enabled is not None:
        if req.enabled:
            _start_embedded_worker(request.app)
        else:
            _stop_embedded_worker()

    logger.info(f"Admin updated auto_processing: enabled={req.enabled}, mode={req.mode}, rest={req.rest_after_batch_s}s")
    return {"ok": True}


# ── Embedded Worker ──────────────────────────────────────────

def _start_embedded_worker(app):
    """Start the embedded worker — no JWT needed (localhost auto-admin).

    Embedded worker connects to 127.0.0.1 → get_current_user() returns
    localhost admin automatically. No JWT token required.
    """
    from backend.server.embedded_worker import start_worker, get_status

    if get_status()["running"]:
        return

    port = getattr(app.state, "port", 8000)

    # No token needed — localhost requests get auto-admin via get_current_user
    result = start_worker(f"http://127.0.0.1:{port}", access_token="")
    if result.get("success"):
        logger.info(f"Embedded worker started (port={port})")
    else:
        logger.warning(f"Embedded worker start failed: {result.get('error')}")


def _stop_embedded_worker():
    """Stop the embedded worker if running."""
    from backend.server.embedded_worker import stop_worker, get_status

    if get_status()["running"]:
        result = stop_worker()
        logger.info(f"Embedded worker stopped: {result}")


@router.get("/admin/workers/embedded-worker")
def admin_get_embedded_worker(
    admin: dict = Depends(require_admin),
):
    """Get embedded worker status and config."""
    from backend.server.embedded_worker import get_status
    from backend.utils.config import get_config

    cfg = get_config()
    status = get_status()
    return {
        "enabled": cfg.get("server.auto_processing.enabled", False),
        **status,
    }


@router.patch("/admin/workers/embedded-worker")
def admin_update_embedded_worker(
    req: EmbeddedWorkerUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
):
    """Enable or disable the embedded worker."""
    from backend.server.embedded_worker import get_status
    from backend.utils.config import get_config

    cfg = get_config()

    if req.enabled is not None:
        cfg._set_dotted("server.embedded_worker.enabled", req.enabled)
        if req.enabled:
            _start_embedded_worker(request.app)
        else:
            _stop_embedded_worker()

    status = get_status()
    logger.info(f"Admin updated embedded_worker: enabled={req.enabled}, running={status['running']}")
    return {"ok": True, **status}
