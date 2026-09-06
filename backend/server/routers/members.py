"""
Member & invite management (IMGV2-14).

  GET    /api/v1/members                  list members (users) — admin
  POST   /api/v1/members/invite           invite by email (reserves seat) — admin
  GET    /api/v1/members/invites          pending invites — admin
  DELETE /api/v1/members/invites/{id}     revoke invite (frees seat) — admin
  DELETE /api/v1/members/{id}             remove member (frees seat) — admin
  POST   /api/v1/members/{id}/deactivate  deactivate (keeps seat) — admin
  GET    /api/v1/members/usage            seats/plan/expiry — admin

Seat model: active users + pending invites ≤ max_users (0 = unlimited).
Invite acceptance happens at /auth/connect (see auth.membership).
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db_safe, require_admin
from backend.server.auth import membership as m
from backend.server import mail

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/members", tags=["members"])


class InviteRequest(BaseModel):
    emails: List[str]
    role: str = "user"


def _invite_link(request: Request, token: str) -> str:
    origin = str(request.base_url).rstrip("/")
    return f"{origin}/#/start?invite={token}"


@router.get("")
def list_members(_admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    rows = db.conn.execute(
        "SELECT id, username, email, role, is_active, last_login_at, firebase_uid "
        "FROM users ORDER BY role DESC, id ASC"
    ).fetchall()
    members = [{
        "id": r[0], "username": r[1], "email": r[2], "role": r[3],
        "is_active": bool(r[4]), "last_login_at": r[5],
        "via": "firebase" if r[6] else "password",
    } for r in rows]
    return {"members": members}


@router.get("/invites")
def list_invites(_admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    m.ensure_invites_table(db)
    rows = db.conn.execute(
        "SELECT id, email, role, created_at FROM member_invites WHERE status='pending' ORDER BY created_at DESC"
    ).fetchall()
    return {"invites": [{"id": r[0], "email": r[1], "role": r[2], "created_at": r[3]} for r in rows]}


@router.post("/invite")
def invite_members(req: InviteRequest, request: Request,
                   admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    emails = [e.strip() for e in req.emails if e.strip()]
    if not emails:
        raise HTTPException(status_code=400, detail="No emails provided")
    results = []
    for email in emails:
        try:
            inv = m.create_invite(db, email, role=req.role, invited_by=admin.get("id"))
        except HTTPException as e:
            results.append({"email": email, "ok": False, "error": e.detail})
            continue
        link = _invite_link(request, inv["token"])
        sent = mail.send_invite_email(email, link, group_name="Imagine")
        results.append({"email": email, "ok": True, "reused": inv["reused"], "sent": sent, "link": link})
    return {"results": results, "smtp_configured": mail.smtp_configured()}


@router.delete("/invites/{invite_id}")
def revoke_invite(invite_id: int, _admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    if not m.revoke_invite(db, invite_id):
        raise HTTPException(status_code=404, detail="Pending invite not found")
    return {"success": True}


@router.delete("/{user_id}")
def remove_member(user_id: int, admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    row = db.conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Member not found")
    if user_id == admin.get("id"):
        raise HTTPException(status_code=400, detail="자기 자신은 제거할 수 없습니다")
    if row[0] == "admin":
        n_admins = db.conn.execute(
            "SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1"
        ).fetchone()[0]
        if n_admins <= 1:
            raise HTTPException(status_code=400, detail="마지막 운영자는 제거할 수 없습니다")
    db.conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.conn.commit()
    return {"success": True}


@router.post("/{user_id}/deactivate")
def deactivate_member(user_id: int, admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    row = db.conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Member not found")
    if user_id == admin.get("id"):
        raise HTTPException(status_code=400, detail="자기 자신은 비활성화할 수 없습니다")
    db.conn.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
    db.conn.commit()
    return {"success": True}


@router.get("/usage")
def usage(_admin: dict = Depends(require_admin), db: SQLiteDB = Depends(get_db_safe)):
    return {
        "seat_limit": m.seat_limit(db),
        "seats_used": m.seats_used(db),
        "members": m.count_active_users(db),
        "pending_invites": m.count_pending_invites(db),
        "expires_at": m.server_expires_at(db),
        "expired": m.is_expired(db),
        "smtp_configured": mail.smtp_configured(),
    }
