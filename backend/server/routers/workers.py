"""
Worker session management — connect, heartbeat, disconnect, admin control.

Workers register sessions on connect and send periodic heartbeats.
Server piggybacks commands (stop/block) in heartbeat responses.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["workers"])


# ── Schemas ──────────────────────────────────────────────────

class ConnectRequest(BaseModel):
    worker_name: str
    hostname: Optional[str] = None
    batch_capacity: int = 5


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
    """Read global processing_mode from config (cached singleton)."""
    try:
        from backend.utils.config import get_config
        return get_config().get("server.processing_mode", "full")
    except Exception:
        return "full"


# ── Worker → Server endpoints ────────────────────────────────

@router.post("/workers/connect")
def worker_connect(
    req: ConnectRequest,
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Register a new worker session."""
    now = datetime.now(timezone.utc).isoformat()
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
    db.conn.commit()

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
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Periodic heartbeat from worker. Returns pending commands."""
    now = datetime.now(timezone.utc).isoformat()
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
    user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Worker graceful disconnect."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions SET status = 'offline', disconnected_at = ?
           WHERE id = ? AND user_id = ?""",
        (now, req.session_id, user["id"])
    )
    db.conn.commit()
    logger.info(f"Worker disconnected: session={req.session_id}")
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

    # Per-worker throughput: completed jobs by worker_session_id (preferred),
    # falling back to assigned_to (user_id) for legacy jobs without session tracking
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
    return {"workers": workers}


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
    now = datetime.now(timezone.utc).isoformat()
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
    return {"ok": True}


@router.patch("/admin/workers/global-config")
def admin_update_global_config(
    req: GlobalModeUpdate,
    admin: dict = Depends(require_admin),
    db: SQLiteDB = Depends(get_db),
):
    """Change processing mode for ALL online workers."""
    cursor = db.conn.cursor()
    cursor.execute(
        """UPDATE worker_sessions
           SET processing_mode_override = ?
           WHERE status = 'online'""",
        (req.processing_mode,)
    )
    db.conn.commit()
    affected = cursor.rowcount
    logger.info(f"Admin set global processing mode: {req.processing_mode} ({affected} workers)")
    return {"ok": True, "affected": affected}
