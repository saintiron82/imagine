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

from datetime import datetime

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_db_safe, get_current_user, require_admin


def _utcnow_sql() -> str:
    """UTC timestamp string for SQLite."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

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
    batch_throughput: Optional[float] = None  # Worker-measured files/min (actual)


class DisconnectRequest(BaseModel):
    session_id: int


class WorkerConfigUpdate(BaseModel):
    batch_capacity: Optional[int] = None     # 1~32 | null (=worker default)


class AutoProcessingUpdate(BaseModel):
    enabled: Optional[bool] = None
    mode: Optional[str] = None  # "auto" | "mc" | "vv" | "mv"
    rest_after_batch_s: Optional[int] = None
    batch_size: Optional[int] = None
    verbose_log: Optional[bool] = None


class EmbeddedWorkerUpdate(BaseModel):
    enabled: Optional[bool] = None


def _get_global_processing_mode() -> str:
    """Read global processing_mode from config.

    Returns 'auto' by default — scheduler handles assignment.
    """
    try:
        from backend.utils.config import get_config
        cfg = get_config()
        mode = cfg.get("server.processing_mode", None)
        if mode:
            return mode
        mode = cfg.get("worker.processing_mode", None)
        return mode or "auto"
    except Exception:
        return "auto"


def _auto_detect_mode_from_resources(resources: dict) -> Optional[str]:
    """워커 resources_json에서 GPU 정보를 읽어 processing_mode 자동 결정.

    서버의 현재 활성 tier를 기준으로 워커가 VLM을 실행할 수 있는지 판단한다.

    Returns:
        "mc" or "vv", or None if detection fails
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
    """No-op in new architecture. Scheduler handles all assignment.

    Kept as stub for any remaining callers during migration.
    """
    pass


# ── Builtin worker virtual session ──────────────────────────

BUILTIN_WORKER_NAME = "embedded"


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
           VALUES (?, ?, 'server (built-in)', ?, 'online', NULL, ?, ?)""",
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

    # Store capability spec and register with scheduler
    if req.resources:
        cursor.execute(
            "UPDATE worker_sessions SET resources_json = ? WHERE id = ?",
            (json.dumps(req.resources), session_id)
        )
        gpu_name = req.resources.get('gpu_name') or ''
        vram_gb = req.resources.get('vram_gb') or req.resources.get('gpu_memory_total_gb') or 0
        gpu_type = req.resources.get('gpu_type') or 'cpu'
        is_metal = req.resources.get('is_metal', False)
        logger.info(
            f"Worker {req.worker_name} capability: "
            f"GPU={gpu_name}, type={gpu_type}, VRAM={vram_gb:.1f}GB, metal={is_metal}"
        )

        # Register with scheduler — scores-based if available
        scheduler = getattr(request.app.state, 'scheduler', None)
        if scheduler:
            scores = req.resources.get('scores')  # from worker_profile.json
            scheduler.register_worker(
                session_id, gpu_name, vram_gb, is_metal, scores=scores
            )

    db.conn.commit()

    # Determine effective processing mode:
    # - mc global → ALL workers get mc
    # - auto global → workers get their auto-detected mode (mc/vv/mv)
    cursor.execute(
        "SELECT processing_mode_override, batch_capacity_override FROM worker_sessions WHERE id = ?",
        (session_id,)
    )
    ov = cursor.fetchone()
    effective_batch = (ov[1] if ov and ov[1] else None) or req.batch_capacity

    global_mode = _get_global_processing_mode()
    if global_mode == "mc":
        processing_mode = "mc"
    elif ov and ov[0]:
        processing_mode = ov[0]  # Admin manual override
    else:
        # Dynamic mode: decide from file_tasks pending counts
        try:
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN parse_status='done' AND mc_status='pending' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN parse_status='done' AND vv_status='pending' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN mc_status='done' AND mv_status='pending' THEN 1 ELSE 0 END)
                FROM file_tasks ft
                JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                WHERE aj.status = 'active'
            """)
            r = cursor.fetchone()
            mc_p, vv_p, mv_p = (r[0] or 0), (r[1] or 0), (r[2] or 0)
            if mc_p + vv_p + mv_p > 0:
                cands = {"mc": mc_p, "vv": vv_p, "mv": mv_p}
                processing_mode = max(cands, key=cands.get)
            else:
                processing_mode = "mc"
        except Exception:
            processing_mode = "mc"

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
    if req.batch_throughput is not None:
        resources_data["batch_throughput"] = req.batch_throughput
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

    # Dynamic mode is handled by _decide_worker_mode() — no auto-detect override needed.
    # processing_mode_override is ONLY for admin manual pinning.
    pool_needs_recalc = False
    if False and req.resources and not mode_override:  # DISABLED: was overriding dynamic scheduling
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
    # - mc global → ALL workers get mc
    # - auto global → workers get their auto-detected mode (mc/vv/mv)
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
                "processing_mode": "mc",
            }

    mode_reason = None
    queue_snapshot = None

    # Check if this is the embedded worker
    cursor.execute("SELECT worker_name FROM worker_sessions WHERE id = ?", (req.session_id,))
    wn_row = cursor.fetchone()
    is_builtin = wn_row and wn_row[0] == BUILTIN_WORKER_NAME

    if is_builtin:
        # Embedded worker decides its own mode from file_tasks — heartbeat is report-only.
        # Just echo back current_phase from the worker's heartbeat.
        processing_mode = req.current_phase or "mc"
        mode_reason = "self_managed"
        effective_batch = 0  # Not used — embedded worker sets its own batch size
    elif global_mode == "mc":
        processing_mode = "mc"
        mode_reason = "global_mode=mc"
        effective_batch = batch_override or batch_capacity
    elif mode_override:
        processing_mode = mode_override
        mode_reason = f"admin_override={mode_override}"
        effective_batch = batch_override or batch_capacity
    else:
        # Dynamic mode for external workers: decide from file_tasks
        try:
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN parse_status='done' AND mc_status='pending' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN parse_status='done' AND vv_status='pending' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN mc_status='done' AND mv_status='pending' THEN 1 ELSE 0 END)
                FROM file_tasks ft
                JOIN analysis_jobs aj ON ft.analysis_job_id = aj.id
                WHERE aj.status = 'active'
            """)
            r = cursor.fetchone()
            mc_p, vv_p, mv_p = (r[0] or 0), (r[1] or 0), (r[2] or 0)
            if mc_p + vv_p + mv_p > 0:
                cands = {"mc": mc_p, "vv": vv_p, "mv": mv_p}
                processing_mode = max(cands, key=cands.get)
                mode_reason = f"file_tasks({cands})"
            else:
                processing_mode = "mc"
                mode_reason = "idle_default"
        except Exception:
            processing_mode = "mc"
            mode_reason = "fallback (decision error)"
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

    resp = {
        "ok": True,
        "command": pending_cmd,
        "pool_hint": resource_batch_hint * 2,
        "batch_hint": resource_batch_hint,
        "processing_mode": processing_mode,
    }
    if mode_reason:
        resp["mode_reason"] = mode_reason
    if queue_snapshot:
        resp["queue_snapshot"] = queue_snapshot
    return resp


@router.post("/workers/disconnect")
def worker_disconnect(
    req: DisconnectRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db_safe),
):
    """Worker graceful disconnect. Reclaims assigned tasks back to pending."""
    now = _utcnow_sql()
    cursor = db.conn.cursor()
    cursor.execute(
        "SELECT id FROM worker_sessions WHERE id = ? AND user_id = ?",
        (req.session_id, user["id"]),
    )
    if cursor.fetchone() is None:
        raise HTTPException(status_code=403, detail="Worker session is not owned by current user")

    from backend.server.queue.analysis_manager import AnalysisJobManager
    mgr = AnalysisJobManager(db)
    reclaimed = mgr.reclaim_worker_tasks(req.session_id)

    cursor.execute(
        """UPDATE worker_sessions SET status = 'offline', disconnected_at = ?
           WHERE id = ? AND user_id = ?""",
        (now, req.session_id, user["id"])
    )
    db.conn.commit()
    logger.info(f"Worker disconnected: session={req.session_id}, reclaimed {reclaimed} jobs")

    # Recalculate pools after worker leaves
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
           LEFT JOIN users u ON ws.user_id = u.id
           ORDER BY
               CASE ws.status WHEN 'online' THEN 0 WHEN 'blocked' THEN 1 ELSE 2 END,
               ws.last_heartbeat DESC
           LIMIT 100"""
    )
    rows = cursor.fetchall()

    # Per-worker throughput from file_tasks (Analysis Job System v1)
    try:
        cursor.execute(
            """SELECT
                   mc_assigned_to AS worker_id,
                   COUNT(*) FILTER (WHERE mc_status='done'
                                    AND datetime(mc_completed_at) > datetime('now', '-5 minutes')) AS mc_5m,
                   COUNT(*) FILTER (WHERE mc_status='done'
                                    AND datetime(mc_completed_at) > datetime('now', '-1 minute')) AS mc_1m
               FROM file_tasks
               WHERE mc_assigned_to IS NOT NULL
               GROUP BY mc_assigned_to"""
        )
        mc_counts = {r[0]: {"mc_5m": r[1], "mc_1m": r[2]} for r in cursor.fetchall()}

        cursor.execute(
            """SELECT
                   vv_assigned_to AS worker_id,
                   COUNT(*) FILTER (WHERE vv_status='done'
                                    AND datetime(vv_completed_at) > datetime('now', '-5 minutes')) AS vv_5m,
                   COUNT(*) FILTER (WHERE vv_status='done'
                                    AND datetime(vv_completed_at) > datetime('now', '-1 minute')) AS vv_1m
               FROM file_tasks
               WHERE vv_assigned_to IS NOT NULL
               GROUP BY vv_assigned_to"""
        )
        vv_counts = {r[0]: {"vv_5m": r[1], "vv_1m": r[2]} for r in cursor.fetchall()}

        cursor.execute(
            """SELECT
                   mv_assigned_to AS worker_id,
                   COUNT(*) FILTER (WHERE mv_status='done'
                                    AND datetime(mv_completed_at) > datetime('now', '-5 minutes')) AS mv_5m,
                   COUNT(*) FILTER (WHERE mv_status='done'
                                    AND datetime(mv_completed_at) > datetime('now', '-1 minute')) AS mv_1m
               FROM file_tasks
               WHERE mv_assigned_to IS NOT NULL
               GROUP BY mv_assigned_to"""
        )
        mv_counts = {r[0]: {"mv_5m": r[1], "mv_1m": r[2]} for r in cursor.fetchall()}
    except Exception:
        mc_counts, vv_counts, mv_counts = {}, {}, {}

    # Merge per-worker phase counts
    session_phase_counts = {}
    all_worker_ids = set(mc_counts) | set(vv_counts) | set(mv_counts)
    for wid in all_worker_ids:
        session_phase_counts[wid] = {
            **(mc_counts.get(wid, {})),
            **(vv_counts.get(wid, {})),
            **(mv_counts.get(wid, {})),
        }

    workers = []
    for row in rows:
        session_id = row[0]
        assigned_mode = row[18]  # "mc", "vv", "mv", or None (full pipeline)

        # Pick the count matching this worker's assigned mode
        pc = session_phase_counts.get(session_id, {})
        if assigned_mode == "vv":
            r1, r5 = pc.get("vv_1m", 0), pc.get("vv_5m", 0)
        elif assigned_mode == "mv":
            r1, r5 = pc.get("mv_1m", 0), pc.get("mv_5m", 0)
        else:
            # "mc" mode or full pipeline — MC is the bottleneck, use MC count
            r1, r5 = pc.get("mc_1m", 0), pc.get("mc_5m", 0)

        # 1-min if active, otherwise 5-min average
        if r1 > 0:
            throughput = float(r1)
        elif r5 > 0:
            throughput = round(r5 / 5.0, 1)
        else:
            throughput = 0.0

        # Prefer worker-measured throughput (from heartbeat) over SQL inference
        resources = json.loads(row[17]) if row[17] else None
        worker_measured = (resources or {}).get("batch_throughput")
        if worker_measured and worker_measured > 0:
            throughput = worker_measured

        # BUG-002: use session-scoped phase_counts from heartbeat (memory-based)
        phase_counts = (resources or {}).get("phase_counts")

        workers.append({
            "id": session_id, "worker_name": row[1], "hostname": row[2],
            "status": row[3], "batch_capacity": row[4],
            "jobs_completed": row[5], "jobs_failed": row[6],
            "current_file": row[7], "current_phase": row[8],
            "last_heartbeat": row[9], "connected_at": row[10],
            "disconnected_at": row[11], "pending_command": row[12],
            "username": row[13] or "system",
            "throughput": throughput,
            "throughput_mode": assigned_mode or "full",
            "processing_mode_override": row[15],
            "batch_capacity_override": row[16],
            "resources": resources,
            "assigned_mode": row[18],
            "phase_counts": phase_counts,       # session-scoped (resets on restart)
            "phase_job_count": row[19] or 0,    # DB cumulative (reference only)
        })

    # BUG-005: override embedded worker's data from live memory
    # (heartbeat may be blocked during long MC batch)
    try:
        from backend.server.embedded_worker import get_status as _ew_status
        ew = _ew_status()
        if ew.get("running"):
            for w in workers:
                if w["worker_name"] == "embedded":
                    w["status"] = "online"  # force online if running
                    w["current_phase"] = ew.get("current_phase") or w["current_phase"]
                    w["current_file"] = ew.get("current_file") or w["current_file"]
                    if ew.get("batch_capacity"):
                        w["batch_capacity"] = ew["batch_capacity"]
                    if ew.get("phase_counts"):
                        w["phase_counts"] = ew["phase_counts"]
                    if ew.get("phase_throughput"):
                        if not w.get("resources"):
                            w["resources"] = {}
                        w["resources"]["phase_throughput"] = ew["phase_throughput"]
                    if ew.get("throughput") and ew["throughput"] > 0:
                        w["throughput"] = ew["throughput"]
                    break
    except Exception:
        pass

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

    Immediately reclaims all tasks assigned to this worker back to pending.
    """
    from backend.server.queue.analysis_manager import AnalysisJobManager
    mgr = AnalysisJobManager(db)
    reclaimed = mgr.reclaim_worker_tasks(session_id)

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
    if req.mode is not None and req.mode in ("auto", "mc", "vv", "mv"):
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

    # Scheduler handles mode assignment — no global mode switching needed

    # Start/stop embedded worker
    # Note: start is handled by _activate_server Phase 6 (on login).
    # This API only handles STOP (user explicitly disables).
    if req.enabled is not None and not req.enabled:
        _stop_embedded_worker()

    logger.info(f"Admin updated auto_processing: enabled={req.enabled}, mode={req.mode}, rest={req.rest_after_batch_s}s")
    return {"ok": True}


# ── Embedded Worker ──────────────────────────────────────────

def _start_embedded_worker(app):
    """Start the embedded worker using LocalTransport (no HTTP needed)."""
    from backend.server.embedded_worker import start_worker, get_status

    if get_status()["running"]:
        return

    # server_url and access_token are unused by LocalTransport but kept for API compat
    result = start_worker(server_url="", access_token="")
    if result.get("success"):
        logger.info("Embedded worker started (LocalTransport)")
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
        # Start handled by _activate_server. Only stop here.
        if not req.enabled:
            _stop_embedded_worker()

    status = get_status()
    logger.info(f"Admin updated embedded_worker: enabled={req.enabled}, running={status['running']}")
    return {"ok": True, **status}


# ── Benchmark ──────────────────────────────────────────────

@router.get("/admin/workers/benchmark")
def get_benchmark(
    admin: dict = Depends(require_admin),
):
    """Get cached benchmark results (speed profile + scores)."""
    from backend.worker.capability import load_speed_profile, collect_capability, calculate_scores

    cap = collect_capability(include_speed=False)
    profile = load_speed_profile()

    if profile:
        scores = profile.get("scores") or calculate_scores(profile)
    else:
        scores = None

    return {
        "hardware": {
            "gpu_name": cap.get("gpu_name", ""),
            "gpu_type": cap.get("gpu_type", "cpu"),
            "vram_gb": cap.get("vram_gb", 0),
        },
        "profile": profile,
        "scores": scores,
    }


@router.post("/admin/workers/benchmark")
def run_benchmark_endpoint(
    admin: dict = Depends(require_admin),
):
    """Run benchmark on server machine. Returns speed profile + scores.

    This runs MC/VV/MV benchmarks sequentially (~30s total).
    Blocks until complete.
    """
    from backend.worker.capability import run_benchmark, collect_capability

    cap = collect_capability(include_speed=False)
    logger.info(f"Benchmark started: {cap.get('gpu_name', 'unknown')}")

    profile = run_benchmark()

    logger.info(
        f"Benchmark complete: MC={profile.get('mc_speed')}/m "
        f"VV={profile.get('vv_speed')}/m MV={profile.get('mv_speed')}/m "
        f"Grade={profile.get('scores', {}).get('grade', '?')}"
    )

    return {
        "hardware": {
            "gpu_name": cap.get("gpu_name", ""),
            "gpu_type": cap.get("gpu_type", "cpu"),
            "vram_gb": cap.get("vram_gb", 0),
        },
        "profile": profile,
        "scores": profile.get("scores"),
    }
