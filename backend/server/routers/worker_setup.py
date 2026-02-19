"""
Worker setup router — self-service token creation + setup script download.

Any authenticated user can generate a worker token and download
a cross-platform setup script that auto-configures and runs the worker.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from backend.db.sqlite_client import SQLiteDB
from backend.server.deps import get_db, get_current_user
from backend.server.auth.schemas import WorkerTokenResponse

router = APIRouter(prefix="/worker", tags=["worker"])


@router.post("/register", response_model=WorkerTokenResponse)
def register_worker(
    current_user: dict = Depends(get_current_user),
    db: SQLiteDB = Depends(get_db),
):
    """Create a personal worker token (any authenticated user)."""
    token_secret = "WK_" + secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token_secret.encode()).hexdigest()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

    cursor = db.conn.cursor()
    cursor.execute(
        """INSERT INTO worker_tokens (token_hash, name, created_by, expires_at)
           VALUES (?, ?, ?, ?)""",
        (token_hash, f"{current_user['username']}'s worker", current_user["id"], expires_at),
    )
    db.conn.commit()
    token_id = cursor.lastrowid

    return WorkerTokenResponse(
        id=token_id,
        name=f"{current_user['username']}'s worker",
        token=token_secret,
        created_by=current_user["id"],
        expires_at=expires_at,
    )


SETUP_SCRIPT_TEMPLATE = '''\
#!/usr/bin/env python3
"""
Imagine Worker Setup — auto-generated
Run this script from the Imagine project root directory.
"""
import subprocess, sys, os

SERVER_URL = "{server_url}"
WORKER_TOKEN = "{token}"


def main():
    print("=== Imagine Worker Setup ===")
    print()

    # 1. Check project root
    if not os.path.exists("backend/worker/worker_daemon.py"):
        print("ERROR: Run this script from the Imagine project root directory.")
        print("  e.g.  cd /path/to/Imagine && python3 setup_worker.py")
        sys.exit(1)

    # 2. Create venv if needed
    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        print("[1/3] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    else:
        print("[1/3] Virtual environment already exists.")

    # 3. Determine venv python path (cross-platform)
    if os.name == "nt":
        venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_py = os.path.join(venv_dir, "bin", "python")

    # 4. Install dependencies
    print("[2/3] Installing dependencies (this may take a few minutes)...")
    subprocess.run(
        [venv_py, "-m", "pip", "install", "-q", "--upgrade", "pip"],
        check=True,
    )
    subprocess.run(
        [venv_py, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

    # 5. Start worker
    print("[3/3] Starting worker...")
    print(f"  Server: {{SERVER_URL}}")
    print()
    subprocess.run([
        venv_py, "-m", "backend.worker.worker_daemon",
        "--server", SERVER_URL,
        "--token", WORKER_TOKEN,
    ])


if __name__ == "__main__":
    main()
'''


@router.get("/setup-script")
def get_setup_script(
    token: str = Query(..., description="Worker token secret"),
    server_url: str = Query(..., description="Server URL"),
):
    """Generate a downloadable Python setup script with embedded token."""
    script = SETUP_SCRIPT_TEMPLATE.format(
        server_url=server_url,
        token=token,
    )
    return PlainTextResponse(
        content=script,
        media_type="text/x-python",
        headers={"Content-Disposition": "attachment; filename=setup_worker.py"},
    )
