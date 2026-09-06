"""
Tests for member/invite/seat/expiry (IMGV2-14).

Covers the testable core without Firebase: membership helpers (seats, invites,
expiry, the connect-gating decision) and the members router endpoints.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.db.sqlite_client import SQLiteDB
from backend.db.sqlite_migrations import migrate_auth_tables
from backend.server.auth import membership as m
from backend.server.routers import members as members_router
from backend.server.deps import get_db_safe, require_admin


@pytest.fixture
def db(tmp_path):
    from backend.server.queue.analysis_manager import AnalysisJobManager
    AnalysisJobManager._initialized = False
    d = SQLiteDB(str(tmp_path / "auth.db"))
    migrate_auth_tables(d)
    d.conn.execute("CREATE TABLE IF NOT EXISTS system_meta (key TEXT PRIMARY KEY, value TEXT)")
    d.conn.execute("INSERT OR REPLACE INTO system_meta(key, value) VALUES('max_users', '2')")
    d.conn.execute(
        "INSERT INTO users(username, email, password_hash, role, is_active, firebase_uid) "
        "VALUES('admin', 'admin@x', '', 'admin', 1, 'uid_admin')"
    )
    d.conn.commit()
    return d


@pytest.fixture
def client(db):
    app = FastAPI()
    app.include_router(members_router.router, prefix="/api/v1")
    admin = {"id": 1, "username": "admin", "role": "admin"}
    app.dependency_overrides[require_admin] = lambda: admin
    app.dependency_overrides[get_db_safe] = lambda: db
    return TestClient(app)


# ── Seat accounting ──────────────────────────────────────────
def test_seat_used_counts_users_and_pending(db):
    assert m.seat_limit(db) == 2
    assert m.seats_used(db) == 1  # admin only
    m.create_invite(db, "bob@x")
    assert m.seats_used(db) == 2  # admin + pending invite


def test_seat_limit_blocks_overbooking(db):
    m.create_invite(db, "bob@x")           # 2/2 (admin + bob)
    with pytest.raises(HTTPException) as e:
        m.create_invite(db, "carol@x")     # would be 3/2
    assert e.value.status_code == 409


# ── Connect-gating decision (the logic /auth/connect uses) ───
def test_uninvited_is_rejected_but_invited_resolves(db):
    # uninvited email → no pending invite
    assert m.find_pending_invite(db, "stranger@x") is None
    # invite bob → pending invite found with role
    m.create_invite(db, "bob@x", role="user")
    inv = m.find_pending_invite(db, "BOB@X")  # case-insensitive
    assert inv is not None and inv[1] == "user"
    # accepting consumes the pending invite (seat converts to the new user)
    m.mark_invite_accepted(db, inv[0], user_id=99)
    assert m.find_pending_invite(db, "bob@x") is None
    assert m.count_pending_invites(db) == 0


# ── Expiry hard block ────────────────────────────────────────
def test_expiry_blocks_nonadmin_only(db):
    db.conn.execute("INSERT OR REPLACE INTO system_meta(key,value) VALUES('server_expires_at','2000-01-01T00:00:00Z')")
    db.conn.commit()
    assert m.is_expired(db) is True
    with pytest.raises(HTTPException) as e:
        m.assert_entry_allowed(db, "user")
    assert e.value.status_code == 403
    m.assert_entry_allowed(db, "admin")  # admin exempt — can still renew


def test_not_expired_when_future_or_unset(db):
    assert m.is_expired(db) is False  # unset
    db.conn.execute("INSERT OR REPLACE INTO system_meta(key,value) VALUES('server_expires_at','2999-01-01T00:00:00Z')")
    db.conn.commit()
    assert m.is_expired(db) is False
    m.assert_entry_allowed(db, "user")  # no raise


# ── Members router ───────────────────────────────────────────
def test_invite_endpoint_reserves_and_lists(client):
    r = client.post("/api/v1/members/invite", json={"emails": ["bob@x"], "role": "user"})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["ok"] and "link" in res and res["sent"] is False  # no SMTP → link returned

    r = client.get("/api/v1/members/invites")
    invites = r.json()["invites"]
    assert len(invites) == 1 and invites[0]["email"] == "bob@x"

    # usage reflects seat reservation
    u = client.get("/api/v1/members/usage").json()
    assert u["seat_limit"] == 2 and u["seats_used"] == 2 and u["pending_invites"] == 1


def test_invite_overbook_returns_error_in_results(client):
    client.post("/api/v1/members/invite", json={"emails": ["bob@x"]})       # 2/2
    r = client.post("/api/v1/members/invite", json={"emails": ["carol@x"]}) # over
    assert r.json()["results"][0]["ok"] is False


def test_revoke_invite_frees_seat(client):
    client.post("/api/v1/members/invite", json={"emails": ["bob@x"]})
    inv_id = client.get("/api/v1/members/invites").json()["invites"][0]["id"]
    assert client.delete(f"/api/v1/members/invites/{inv_id}").status_code == 200
    assert client.get("/api/v1/members/usage").json()["pending_invites"] == 0


def test_cannot_remove_last_admin(client, db):
    admin_id = db.conn.execute("SELECT id FROM users WHERE role='admin'").fetchone()[0]
    # self-removal guarded (admin id is 1 in override); last-admin guarded too
    r = client.delete(f"/api/v1/members/{admin_id}")
    assert r.status_code == 400


def test_remove_and_deactivate_member(client, db):
    db.conn.execute("INSERT INTO users(username,email,password_hash,role,is_active,firebase_uid) VALUES('bob','b@x','','user',1,'uid_bob')")
    db.conn.commit()
    bob_id = db.conn.execute("SELECT id FROM users WHERE username='bob'").fetchone()[0]
    assert client.post(f"/api/v1/members/{bob_id}/deactivate").status_code == 200
    assert db.conn.execute("SELECT is_active FROM users WHERE id=?", (bob_id,)).fetchone()[0] == 0
    assert client.delete(f"/api/v1/members/{bob_id}").status_code == 200
    assert db.conn.execute("SELECT COUNT(*) FROM users WHERE id=?", (bob_id,)).fetchone()[0] == 0
