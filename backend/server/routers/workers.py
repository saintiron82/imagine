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
from backend.server.deps import get_db, get_current_user, require_admin
from backend.server.queue.manager import _utcnow_sql

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


class DisconnectRequest(BaseModel):
    session_id: int


class WorkerConfigUpdate(BaseModel):
    processing_mode: Optional[str] = None    # "full" | "mc_only" | null (=global)
    batch_capacity: Optional[int] = None     # 1~32 | null (=worker default)


class GlobalModeUpdate(BaseModel):
    processing_mode: str  # "full" | "mc_only"


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
    """온라인 워커 모드를 분석하여 서버 사이드 풀을 자동 활성화/비활성화.

    - mc_only 워커 있음  → ParseAhead(P+VV) + EmbedAhead(MV) 활성화
    - embed_only 워커 있음 → ParseAhead(P) 활성화, EmbedAhead 불필요 (워커가 MV 처리)
    - full 워커만 있음   → 풀 최소화 (ParseAhead pre-parse만)
    - 워커 없음          → 서버 IPC 워커(full)가 전체 처리
    """
    if not app:
        return

    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT processing_mode_override FROM worker_sessions WHERE status = 'online'"
    )
    rows = cursor.fetchall()

    global_mode = _get_global_processing_mode()
    modes = [(row[0] if row[0] else global_mode) for row in rows]

    has_mc_only = any(m == "mc_only" for m in modes)
    has_embed_only = any(m == "embed_only" for m in modes)
    needs_embed_ahead = has_mc_only  # embed_only 워커는 MV를 자체 처리하므로 EmbedAhead 불필요

    # ParseAheadPool 모드 업데이트
    if hasattr(app.state, "parse_ahead") and app.state.parse_ahead:
        if has_mc_only:
            app.state.parse_ahead._processing_mode = "mc_only"
        elif has_embed_only:
            app.state.parse_ahead._processing_mode = "embed_only"
        else:
            app.state.parse_ahead._processing_mode = global_mode

        # mc_only/embed_only 워커가 있으면 ParseAhead에 demand seed
        if has_mc_only or has_embed_only:
            try:
                from backend.server.queue.base_ahead_pool import BaseAheadPool
                BaseAheadPool.record_claim(session_id=-1, count=10)
            except Exception:
                pass

    # EmbedAheadPool 관리
    embed_ahead_running = (
        hasattr(app.state, "embed_ahead")
        and app.state.embed_ahead
        and app.state.embed_ahead._thread
        and app.state.embed_ahead._thread.is_alive()
    )

    if needs_embed_ahead and not embed_ahead_running:
        try:
            from backend.server.queue.embed_ahead import EmbedAheadPool
            app.state.embed_ahead = EmbedAheadPool(db)
            app.state.embed_ahead.start()
            logger.info("EmbedAheadPool started (mc_only workers online)")
        except Exception as e:
            logger.warning(f"Failed to start EmbedAheadPool: {e}")
    elif not needs_embed_ahead and embed_ahead_running:
        try:
            app.state.embed_ahead.stop()
            app.state.embed_ahead = None
            logger.info("EmbedAheadPool stopped (no mc_only workers)")
        except Exception as e:
            logger.warning(f"Failed to stop EmbedAheadPool: {e}")

    logger.debug(
        f"Pool recalculated: modes={modes}, mc_only={has_mc_only}, "
        f"embed_only={has_embed_only}, embed_ahead={needs_embed_ahead}"
    )


# ── Worker → Server endpoints ────────────────────────────────

@router.post("/workers/connect")
def worker_connect(
    req: ConnectRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Register a new worker session."""
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
            cursor.execute(
                """UPDATE worker_sessions
                   SET processing_mode_override = ?, resources_json = ?
                   WHERE id = ?""",
                (auto_mode, json.dumps(req.resources), session_id)
            )
            logger.info(
                f"Worker {req.worker_name} auto-detected mode: {auto_mode} "
                f"(VRAM={req.resources.get('gpu_memory_total_gb', 0):.1f}GB, "
                f"GPU={req.resources.get('gpu_type', 'none')})"
            )

    db.conn.commit()

    # Recalculate server pools with the new worker included
    _recalculate_server_pools(request.app, db)

    # Per-worker override > global config fallback
    cursor.execute(
        "SELECT processing_mode_override, batch_capacity_override FROM worker_sessions WHERE id = ?",
        (session_id,)
    )
    ov = cursor.fetchone()
    processing_mode = (ov[0] if ov and ov[0] else None) or _get_global_processing_mode()
    effective_batch = (ov[1] if ov and ov[1] else None) or req.batch_capacity

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
    db: SQLiteDB = Depends(get_db),
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
    cursor.execute(
        """UPDATE worker_sessions
           SET last_heartbeat = ?,
               jobs_completed = ?,
               jobs_failed = ?,
               current_job_id = ?,
               current_file = ?,
               current_phase = ?,
               resources_json = ?,
               pending_command = NULL
           WHERE id = ?""",
        (now, req.jobs_completed, req.jobs_failed,
         req.current_job_id, req.current_file, req.current_phase,
         json.dumps(resources_data) if resources_data else None,
         req.session_id)
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

    # Per-worker override > global config fallback
    processing_mode = mode_override or _get_global_processing_mode()
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
    db: SQLiteDB = Depends(get_db),
):
    """Worker graceful disconnect."""
    now = _utcnow_sql()
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions SET status = 'offline', disconnected_at = ?
           WHERE id = ? AND user_id = ?""",
        (now, req.session_id, user["id"])
    )
    db.conn.commit()
    logger.info(f"Worker disconnected: session={req.session_id}")

    # Recalculate pools after worker leaves (may deactivate EmbedAhead if last mc_only)
    _recalculate_server_pools(request.app, db)

    return {"ok": True}


# ── User self-service endpoints ──────────────────────────────

@router.get("/workers/my")
def list_my_workers(
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
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
    db: SQLiteDB = Depends(get_db),
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
    db: SQLiteDB = Depends(get_db),
):
    """List all worker sessions (admin only), with per-worker throughput."""
    cursor = db.conn.cursor()
    cursor.execute(
        """SELECT ws.id, ws.worker_name, ws.hostname, ws.status,
                  ws.batch_capacity, ws.jobs_completed, ws.jobs_failed,
                  ws.current_file, ws.current_phase,
                  ws.last_heartbeat, ws.connected_at, ws.disconnected_at,
                  ws.pending_command, u.username, ws.user_id,
                  ws.processing_mode_override, ws.batch_capacity_override,\
                  ws.resources_json
           FROM worker_sessions ws
           JOIN users u ON ws.user_id = u.id
           ORDER BY
               CASE ws.status WHEN 'online' THEN 0 WHEN 'blocked' THEN 1 ELSE 2 END,
               ws.last_heartbeat DESC
           LIMIT 100"""
    )
    rows = cursor.fetchall()

    # Per-worker throughput: mode-aware timestamp selection
    # mc_only: use mc_completed_at (worker MC speed, not EmbedAhead MV speed)
    # full:    use completed_at (full pipeline completion)
    processing_mode = _get_global_processing_mode()

    if processing_mode == "mc_only":
        cursor.execute(
            """SELECT worker_session_id, COUNT(*) FROM job_queue
               WHERE mc_completed_at IS NOT NULL
                 AND datetime(mc_completed_at) > datetime('now', '-5 minutes')
                 AND worker_session_id IS NOT NULL
               GROUP BY worker_session_id"""
        )
        session_recent_5m = dict(cursor.fetchall())

        cursor.execute(
            """SELECT worker_session_id, COUNT(*) FROM job_queue
               WHERE mc_completed_at IS NOT NULL
                 AND datetime(mc_completed_at) > datetime('now', '-1 minute')
                 AND worker_session_id IS NOT NULL
               GROUP BY worker_session_id"""
        )
        session_recent_1m = dict(cursor.fetchall())

        # Fallback: per-user throughput for jobs without worker_session_id
        cursor.execute(
            """SELECT assigned_to, COUNT(*) FROM job_queue
               WHERE mc_completed_at IS NOT NULL
                 AND datetime(mc_completed_at) > datetime('now', '-5 minutes')
                 AND worker_session_id IS NULL
               GROUP BY assigned_to"""
        )
        user_recent_5m = dict(cursor.fetchall())

        cursor.execute(
            """SELECT assigned_to, COUNT(*) FROM job_queue
               WHERE mc_completed_at IS NOT NULL
                 AND datetime(mc_completed_at) > datetime('now', '-1 minute')
                 AND worker_session_id IS NULL
               GROUP BY assigned_to"""
        )
        user_recent_1m = dict(cursor.fetchall())
    else:
        cursor.execute(
            """SELECT worker_session_id, COUNT(*) FROM job_queue
               WHERE status = 'completed'
                 AND completed_at IS NOT NULL
                 AND datetime(completed_at) > datetime('now', '-5 minutes')
                 AND worker_session_id IS NOT NULL
               GROUP BY worker_session_id"""
        )
        session_recent_5m = dict(cursor.fetchall())

        cursor.execute(
            """SELECT worker_session_id, COUNT(*) FROM job_queue
               WHERE status = 'completed'
                 AND completed_at IS NOT NULL
                 AND datetime(completed_at) > datetime('now', '-1 minute')
                 AND worker_session_id IS NOT NULL
               GROUP BY worker_session_id"""
        )
        session_recent_1m = dict(cursor.fetchall())

        # Fallback: per-user throughput for jobs without worker_session_id
        cursor.execute(
            """SELECT assigned_to, COUNT(*) FROM job_queue
               WHERE status = 'completed'
                 AND completed_at IS NOT NULL
                 AND datetime(completed_at) > datetime('now', '-5 minutes')
                 AND worker_session_id IS NULL
               GROUP BY assigned_to"""
        )
        user_recent_5m = dict(cursor.fetchall())

        cursor.execute(
            """SELECT assigned_to, COUNT(*) FROM job_queue
               WHERE status = 'completed'
                 AND completed_at IS NOT NULL
                 AND datetime(completed_at) > datetime('now', '-1 minute')
                 AND worker_session_id IS NULL
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
        })
    return {
        "workers": workers,
        "global_processing_mode": _get_global_processing_mode(),
    }


@router.post("/admin/workers/{session_id}/stop")
def admin_stop_worker(
    session_id: int,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
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
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
):
    """Block a worker — it will be forced to disconnect (admin only)."""
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
    logger.info(f"Admin blocked worker session {session_id}")
    return {"ok": True}


@router.patch("/admin/workers/{session_id}/config")
def admin_update_worker_config(
    session_id: int,
    req: WorkerConfigUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
):
    """Update per-worker settings (applied on next heartbeat, ~30s)."""
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions
           SET processing_mode_override = ?,
               batch_capacity_override = ?
           WHERE id = ? AND status = 'online'""",
        (req.processing_mode, req.batch_capacity, session_id)
    )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found or not online")
    db.conn.commit()
    logger.info(f"Admin updated worker config: session={session_id}, mode={req.processing_mode}, batch={req.batch_capacity}")

    # Recalculate pools with the updated worker mode
    try:
        _recalculate_server_pools(request.app, db)
    except Exception as e:
        logger.warning(f"Pool recalculation after config update failed: {e}")

    return {"ok": True}


@router.patch("/admin/workers/global-config")
def admin_update_global_config(
    req: GlobalModeUpdate,
    request: Request,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
):
    """Change processing mode for ALL online workers.

    Also dynamically starts/stops EmbedAheadPool and updates
    ParseAheadPool's processing_mode based on the new mode.
    """

    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions
           SET processing_mode_override = ?
           WHERE status = 'online'""",
        (req.processing_mode,)
    )
    db.conn.commit()
    affected = cursor.rowcount

    # Persist to config so heartbeat reads the updated global mode
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        cfg._set_dotted("server.processing_mode", req.processing_mode)
    except Exception as e:
        logger.warning(f"Failed to persist global processing_mode: {e}")

    # Recalculate server-side pools based on updated worker modes
    try:
        _recalculate_server_pools(request.app, db)
    except Exception as e:
        logger.warning(f"Pool recalculation on global mode switch failed: {e}")

    logger.info(f"Admin set global processing mode: {req.processing_mode} ({affected} workers)")
    return {"ok": True, "affected": affected}
