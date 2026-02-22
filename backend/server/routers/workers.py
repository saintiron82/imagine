"""
Worker session management — connect, heartbeat, disconnect, admin control.

Workers register sessions on connect and send periodic heartbeats.
Server piggybacks commands (stop/block) in heartbeat responses.
"""

import logging
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


class DisconnectRequest(BaseModel):
    session_id: int


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

    # Read processing mode from config
    try:
        from backend.utils.config import get_config
        processing_mode = get_config().get("server.processing_mode", "full")
    except Exception:
        processing_mode = "full"

    logger.info(f"Worker connected: {req.worker_name} (session={session_id}, user={user['username']}, mode={processing_mode})")
    return {
        "session_id": session_id,
        "pool_hint": req.batch_capacity * 2,
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

    # Verify session ownership
    cursor.execute(
        "SELECT id, status, pending_command, batch_capacity FROM worker_sessions WHERE id = ? AND user_id = ?",
        (req.session_id, user["id"])
    )
    row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    session_status = row[1]
    pending_cmd = row[2]
    batch_capacity = row[3]

    # Blocked sessions should stop immediately
    if session_status == "blocked":
        return {"ok": True, "command": "block", "pool_hint": 0}

    # Update metrics
    cursor.execute(
        """UPDATE worker_sessions
           SET last_heartbeat = ?,
               jobs_completed = ?,
               jobs_failed = ?,
               current_job_id = ?,
               current_file = ?,
               current_phase = ?,
               pending_command = NULL
           WHERE id = ?""",
        (now, req.jobs_completed, req.jobs_failed,
         req.current_job_id, req.current_file, req.current_phase,
         req.session_id)
    )
    db.conn.commit()

    # Read processing mode (allows runtime switching without restart)
    try:
        from backend.utils.config import get_config
        processing_mode = get_config().get("server.processing_mode", "full")
    except Exception:
        processing_mode = "full"

    return {
        "ok": True,
        "command": pending_cmd,
        "pool_hint": batch_capacity * 2,
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
                  ws.pending_command, u.username, ws.user_id
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
