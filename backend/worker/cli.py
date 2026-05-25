"""
Headless worker CLI.

Runs WorkerDaemon without Electron. The central server still owns queue,
session state, scheduling, and control.

Two connection modes (Phase 6):

  direct   — talks to the Imagine server over plain HTTP at
             `IMAGINE_SERVER_URL`. Recommended on LAN and for
             advanced users who run their own tunnel.
  relay    — opens an outbound WebSocket to the AWS control relay
             (`IMAGINE_RELAY_ENDPOINT`) and attaches via worker
             enrollment token. The user PC server never opens a port.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HeadlessWorkerConfig:
    server_url: str
    access_token: str
    refresh_token: str
    worker_name: str
    origin: str = "headless"
    launcher: str = "cli"
    poll_interval: int = 5
    # Phase 6 — relay mode
    connect_mode: str = "direct"
    relay_endpoint: str = ""
    server_id: str = ""
    enrollment_token: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imagine-worker")
    parser.add_argument(
        "--connect-mode",
        default=os.getenv("IMAGINE_CONNECT_MODE", "direct"),
        choices=("direct", "relay"),
        help="direct = HTTP to Imagine server; relay = WebSocket to AWS relay",
    )
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
    # Relay-mode env / args
    parser.add_argument("--relay-endpoint", default=os.getenv("IMAGINE_RELAY_ENDPOINT", ""))
    parser.add_argument("--server-id", default=os.getenv("IMAGINE_SERVER_ID", ""))
    parser.add_argument(
        "--enrollment-token",
        default=os.getenv("IMAGINE_WORKER_ENROLLMENT_TOKEN", ""),
        help="Phase 6 worker enrollment token (relay mode only)",
    )
    return parser


def load_headless_config(argv: list[str] | None = None) -> HeadlessWorkerConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.access_token:
        parser.error("--access-token or IMAGINE_WORKER_ACCESS_TOKEN is required")

    if args.connect_mode == "relay":
        if not args.relay_endpoint:
            parser.error(
                "--relay-endpoint or IMAGINE_RELAY_ENDPOINT is required in relay mode"
            )
        if not args.server_id:
            parser.error(
                "--server-id or IMAGINE_SERVER_ID is required in relay mode"
            )
        if not args.enrollment_token:
            parser.error(
                "--enrollment-token or IMAGINE_WORKER_ENROLLMENT_TOKEN is required in relay mode"
            )
        server_url = args.server_url.rstrip("/") if args.server_url else ""
    else:
        if not args.server_url:
            parser.error("--server-url or IMAGINE_SERVER_URL is required in direct mode")
        server_url = args.server_url.rstrip("/")

    return HeadlessWorkerConfig(
        server_url=server_url,
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        worker_name=args.worker_name or f"{socket.gethostname()}-headless",
        launcher=args.launcher,
        poll_interval=args.poll_interval,
        connect_mode=args.connect_mode,
        relay_endpoint=args.relay_endpoint.strip(),
        server_id=args.server_id.strip(),
        enrollment_token=args.enrollment_token.strip(),
    )


def _build_transport(cfg: HeadlessWorkerConfig):
    """Phase 6 wiring — relay mode swaps in the WS-based transport."""
    if cfg.connect_mode == "relay":
        from backend.worker.relay_transport import RelayTransport
        return RelayTransport(
            endpoint=cfg.relay_endpoint,
            server_id=cfg.server_id,
            enrollment_token=cfg.enrollment_token,
        )
    return None  # direct mode = HTTP fallback in WorkerDaemon


def run_headless_worker(cfg: HeadlessWorkerConfig) -> int:
    if cfg.server_url:
        os.environ["IMAGINE_SERVER_URL"] = cfg.server_url

    from backend.worker.worker_daemon import WorkerDaemon

    transport = _build_transport(cfg)
    if cfg.connect_mode == "relay":
        # Phase 6 MVP: this loop only attaches the worker to the relay
        # and keeps the session alive with heartbeats. Actually claiming
        # and completing tasks over the relay requires the server-side
        # relay router (Phase 7) so the server can address a specific
        # worker connection by msg_id. Until Phase 7 lands, use direct
        # mode for real work and relay mode for connectivity smoke tests.
        transport.connect()
        try:
            print("[worker] relay attached (Phase 6 MVP — control plane only)", flush=True)
            while True:
                transport.heartbeat({"phase": "idle"})
                time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                transport.disconnect()
            except Exception:
                pass
        return 0

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
