"""
Static worker runtime contract check.

Catches drift in the worker-runtime model without needing a live server or
GPU. Every runner is now a SEPARATE process running backend.worker.cli; they
differ only in who launches them and what origin/launcher they register:

  server-local  backend/server/local_worker.py spawns the CLI with
                --launcher server
  client        backend/worker/worker_ipc.py   origin=client-launched,
                launcher=electron
  headless      backend/worker/cli.py          origin=headless (default)

The in-process embedded worker this file used to assert on was removed when
the runners were unified; asserting on the deleted module made this check
die with FileNotFoundError instead of guarding anything.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assert_contains(path: str, needle: str) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    if needle not in text:
        raise AssertionError(f"{path} is missing {needle!r}")


def main() -> int:
    assert_contains("backend/server/local_worker.py", '"--launcher", "server"')
    assert_contains("backend/server/local_worker.py", "backend.worker.cli")
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
