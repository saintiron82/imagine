"""
Static worker runtime contract check.

This check catches drift in the three-runner worker model without requiring
a live server or GPU.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assert_contains(path: str, needle: str) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    if needle not in text:
        raise AssertionError(f"{path} is missing {needle!r}")


def main() -> int:
    assert_contains("backend/server/embedded_worker.py", 'origin="server-local"')
    assert_contains("backend/server/embedded_worker.py", 'launcher="server"')
    assert_contains("backend/worker/worker_ipc.py", 'origin="client-launched"')
    assert_contains("backend/worker/worker_ipc.py", 'launcher="electron"')
    assert_contains("backend/worker/cli.py", 'origin: str = "headless"')
    assert_contains("backend/worker/cli.py", "def main(")
    assert_contains("backend/server/routers/workers.py", "origin")
    assert_contains("backend/server/routers/workers.py", "launcher")
    assert_contains("backend/server/routers/workers.py", "/workers/bootstrap/linux.sh")
    assert_contains("backend/server/routers/workers.py", "/admin/workers/headless-command")
    assert_contains("scripts/cloud_worker_boot.sh", "backend.worker.cli")
    print("worker runtime contract static check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
