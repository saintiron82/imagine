"""
Headless worker CLI.

Runs WorkerDaemon without Electron. The central server still owns queue,
session state, scheduling, and control.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass


@dataclass
class HeadlessWorkerConfig:
    server_url: str
    access_token: str
    refresh_token: str
    worker_name: str
    origin: str = "headless"
    launcher: str = "cli"
    poll_interval: int = 5


def load_headless_config(argv: list[str] | None = None) -> HeadlessWorkerConfig:
    parser = argparse.ArgumentParser(prog="imagine-worker")
    parser.add_argument("--server-url", default=os.getenv("IMAGINE_SERVER_URL", ""))
    parser.add_argument("--access-token", default=os.getenv("IMAGINE_WORKER_ACCESS_TOKEN", ""))
    parser.add_argument("--refresh-token", default=os.getenv("IMAGINE_WORKER_REFRESH_TOKEN", ""))
    parser.add_argument("--worker-name", default=os.getenv("IMAGINE_WORKER_NAME", ""))
    parser.add_argument(
        "--launcher",
        default=os.getenv("IMAGINE_WORKER_LAUNCHER", "cli"),
        choices=("cli", "service", "cloud"),
    )
    parser.add_argument("--poll-interval", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.server_url:
        parser.error("--server-url or IMAGINE_SERVER_URL is required")
    if not args.access_token:
        parser.error("--access-token or IMAGINE_WORKER_ACCESS_TOKEN is required")

    return HeadlessWorkerConfig(
        server_url=args.server_url.rstrip("/"),
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        worker_name=args.worker_name or f"{socket.gethostname()}-headless",
        launcher=args.launcher,
        poll_interval=args.poll_interval,
    )


def run_headless_worker(cfg: HeadlessWorkerConfig) -> int:
    os.environ["IMAGINE_SERVER_URL"] = cfg.server_url

    from backend.worker.worker_daemon import WorkerDaemon

    daemon = WorkerDaemon(origin=cfg.origin, launcher=cfg.launcher)
    daemon.worker_name = cfg.worker_name
    daemon.set_tokens(cfg.access_token, cfg.refresh_token)

    if not daemon._connect_session():
        print("[worker] session connect failed", file=sys.stderr)
        return 1

    stop_requested = False

    def _handle_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        daemon._stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        from backend.worker.worker_state import WorkerState

        while not stop_requested:
            heartbeat = daemon._heartbeat()
            command = heartbeat.get("command") if heartbeat else None
            if command in ("stop", "block"):
                break

            daemon._state_machine.update(
                is_scheduled_active=True,
                throttle_level=daemon._check_throttle(),
                has_pending_jobs=True,
            )
            if daemon._state_machine.state in (WorkerState.IDLE, WorkerState.RESTING):
                time.sleep(cfg.poll_interval)
                continue

            jobs = daemon.claim_jobs()
            if not jobs:
                daemon._state_machine.update(
                    is_scheduled_active=True,
                    throttle_level=daemon._check_throttle(),
                    has_pending_jobs=False,
                )
                time.sleep(cfg.poll_interval)
                continue

            results = daemon.process_batch_phased(jobs)
            ok = sum(1 for item in results if len(item) > 1 and item[1])
            print(f"[worker] batch complete: {ok}/{len(jobs)} ok", flush=True)
            daemon._state_machine.record_job_activity()
            time.sleep(1)
        return 0
    finally:
        try:
            daemon._on_enter_idle()
        finally:
            daemon._disconnect_session()


def main(argv: list[str] | None = None) -> int:
    cfg = load_headless_config(argv)
    return run_headless_worker(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
