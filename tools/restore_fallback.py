"""
restore_fallback.py — master orchestrator for parse_fallback_legacy recovery.

Single entry point that wires the existing scripts into a resumable pipeline:

    Phase 0: preflight   — DB backup + count snapshot + baseline audit
    Phase 1: local       — reingest_fallback.py --scope local --file (sequential)
    Phase 2: webdav-reg  — register_fallback_job.py --scope webdav (file_tasks rows)
    Phase 3: server      — (optional) launch backend.server.app in background
    Phase 4: monitor     — poll analysis_jobs/file_tasks until target completion
    Phase 5: verify      — final audit + before/after diff + failure list

Why this exists: the user requirement is that download and processing must
not be sequential. Phases 1 (local-only, no download) and 2+3+4 (server-mode
WebDAV with DownloadAheadPool + FileTaskParsePool running concurrently) are
both designed around that — local files run as fast as the GPU can keep up,
WebDAV files use the existing 30 GB bounded buffer to overlap downloads
with parsing.

State is persisted in `output/restore_state.json` so any run can be paused
(SIGINT) and resumed (`resume` subcommand). Phases that are already
`completed` are skipped on resume.

Subcommands:
    run        — execute pipeline (with --skip-* / --auto-server / --only flags)
    status     — print state.json + current DB audit (no changes)
    resume     — same as run but uses existing state and skips completed phases
    dry-run    — preflight only, no writes / no child invocations
    verify     — final audit + diff against baseline (if recorded)

Typical usage:

    # Full automated restore (local + webdav register + server + monitor)
    python tools/restore_fallback.py run --auto-server

    # Local only first (fast, no downloads), inspect, then add webdav
    python tools/restore_fallback.py run --only local
    python tools/restore_fallback.py run --only webdav-reg --auto-server

    # Recover after Ctrl-C
    python tools/restore_fallback.py resume

    # See what would happen
    python tools/restore_fallback.py dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT))

import audit_null_ratios as audit  # local helper, stdlib-only

DEFAULT_DB = PROJECT_ROOT / "imageparser.db"
OUTPUT_DIR = PROJECT_ROOT / "output"
STATE_FILE = OUTPUT_DIR / "restore_state.json"
DRYRUN_STATE_FILE = OUTPUT_DIR / "restore_state.dryrun.json"
LOG_FILE = OUTPUT_DIR / "restore_fallback.log"
SERVER_LOG_FILE = OUTPUT_DIR / "restore_server.log"
SERVER_PID_FILE = OUTPUT_DIR / "restore_server.pid"
FAILURES_FILE = OUTPUT_DIR / "restore_failures.txt"

PHASES = ["preflight", "local", "webdav_register", "server", "monitor", "verify"]

# Module-global mutable state-file path so dry-run can divert writes
_state_path: Path = STATE_FILE

logger = logging.getLogger("restore")


def _setup_logging(verbose: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ---------- state I/O ----------

def _empty_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
        "phases": {p: {"status": "pending"} for p in PHASES},
        "baseline": None,
        "final": None,
    }


def load_state() -> Dict[str, Any]:
    if not _state_path.exists():
        return _empty_state()
    try:
        with _state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"state file unreadable ({e}); starting fresh")
        return _empty_state()


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _state_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(_state_path)


def _set_phase(state: Dict[str, Any], phase: str, status: str,
               **extra: Any) -> None:
    entry = state["phases"].get(phase, {})
    entry["status"] = status
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    entry.update(extra)
    state["phases"][phase] = entry
    save_state(state)


def _phase_done(state: Dict[str, Any], phase: str) -> bool:
    return state["phases"].get(phase, {}).get("status") == "completed"


# ---------- phase 0: preflight ----------

def phase_preflight(state: Dict[str, Any], db_path: Path,
                    skip_backup: bool, dry: bool) -> bool:
    logger.info("[phase 0] preflight")
    _set_phase(state, "preflight", "in_progress")

    if not db_path.exists():
        logger.error(f"db not found: {db_path}")
        _set_phase(state, "preflight", "failed", reason="db_missing")
        return False

    snapshot = audit.collect(db_path)
    state["baseline"] = snapshot
    rem = snapshot["fallback_remaining"]
    logger.info(f"  total files: {snapshot['total_files']:,}")
    logger.info(f"  parse_fallback_legacy remaining: "
                f"local={rem['local']:,}, webdav={rem['webdav']:,}, "
                f"total={rem['total']:,}")

    backup_path = None
    if not dry and not skip_backup:
        ts = datetime.now().strftime("%Y_%m_%d_%H%M")
        backup_path = db_path.with_name(f"{db_path.name}.bak_{ts}")
        if backup_path.exists():
            logger.info(f"  backup already exists: {backup_path.name}")
        else:
            shutil.copy2(db_path, backup_path)
            logger.info(f"  db backup -> {backup_path.name} "
                        f"({backup_path.stat().st_size / (1024 ** 2):.1f} MB)")

    if rem["total"] == 0:
        logger.info("  nothing to do — fallback_legacy queue is empty")
        _set_phase(state, "preflight", "completed",
                   backup=str(backup_path) if backup_path else None,
                   nothing_to_do=True)
        return True

    _set_phase(state, "preflight", "completed",
               backup=str(backup_path) if backup_path else None,
               local_pending=rem["local"],
               webdav_pending=rem["webdav"])
    return True


# ---------- phase 1: local re-ingest (subprocess) ----------

def phase_local(state: Dict[str, Any], db_path: Path,
                python_exe: str, dry: bool, limit: int) -> bool:
    logger.info("[phase 1] local re-ingest (sequential --file)")
    _set_phase(state, "local", "in_progress")

    pending = state["phases"]["preflight"].get("local_pending", 0)
    if pending == 0:
        logger.info("  no local fallback files")
        _set_phase(state, "local", "completed", processed=0, failed=0,
                   skipped="empty")
        return True

    cmd = [python_exe, str(PROJECT_ROOT / "tools" / "reingest_fallback.py"),
           "--scope", "local", "--db", str(db_path)]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if dry:
        cmd.append("--dry-run")

    logger.info(f"  exec: {' '.join(cmd)}")
    started = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except OSError as e:
        logger.error(f"  failed to launch reingest: {e}")
        _set_phase(state, "local", "failed", reason=str(e))
        return False

    succeeded = 0
    failed = 0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            logger.info(f"  | {line}")
            if " OK " in line:
                succeeded += 1
            elif " FAIL " in line:
                failed += 1
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.warning("  interrupted; phase saved as in_progress for resume")
        _set_phase(state, "local", "in_progress",
                   succeeded=succeeded, failed=failed,
                   reason="interrupted")
        raise

    elapsed = time.time() - started
    logger.info(f"  reingest exit={rc} succeeded={succeeded} failed={failed} "
                f"in {elapsed / 60:.1f} min")

    _set_phase(state, "local",
               "completed" if rc == 0 else "failed",
               succeeded=succeeded, failed=failed, exit_code=rc,
               elapsed_s=round(elapsed, 1))
    return rc == 0


# ---------- phase 2: webdav register ----------

def phase_webdav_register(state: Dict[str, Any], db_path: Path,
                          python_exe: str, dry: bool, limit: int) -> bool:
    logger.info("[phase 2] register WebDAV file_tasks for server processing")
    _set_phase(state, "webdav_register", "in_progress")

    pending = state["phases"]["preflight"].get("webdav_pending", 0)
    if pending == 0:
        logger.info("  no webdav fallback files")
        _set_phase(state, "webdav_register", "completed", registered=0,
                   skipped="empty")
        return True

    if dry:
        logger.info(f"  dry-run: would register {pending:,} webdav files")
        _set_phase(state, "webdav_register", "completed",
                   would_register=pending, skipped="dry_run")
        return True

    existing_job = _find_existing_webdav_job(db_path)
    if existing_job:
        jid, name = existing_job
        logger.info(f"  reusing existing job #{jid}: {name}")
        _set_phase(state, "webdav_register", "completed",
                   job_id=jid, registered=pending, reused=True)
        return True

    cmd = [python_exe,
           str(PROJECT_ROOT / "tools" / "register_fallback_job.py"),
           "--scope", "webdav", "--db", str(db_path)]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    logger.info(f"  exec: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip():
            logger.info(f"  | {line}")

    if proc.returncode != 0:
        _set_phase(state, "webdav_register", "failed",
                   exit_code=proc.returncode)
        return False

    job = _find_existing_webdav_job(db_path)
    job_id = job[0] if job else None
    _set_phase(state, "webdav_register", "completed",
               job_id=job_id, registered=pending)
    return True


def _find_existing_webdav_job(db_path: Path) -> Optional[tuple[int, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, name FROM analysis_jobs "
            "WHERE source_path = 'fallback_webdav' "
            "  AND status IN ('active','paused') "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return (row[0], row[1]) if row else None
    finally:
        conn.close()


# ---------- phase 3: server (optional) ----------

def phase_server(state: Dict[str, Any], python_exe: str,
                 auto_server: bool, dry: bool) -> bool:
    logger.info("[phase 3] server (auto_server=%s)", auto_server)

    if dry or not auto_server:
        logger.info("  skipping auto server start. To enable processing, run:")
        logger.info(f"    {python_exe} -m backend.server.app")
        _set_phase(state, "server", "skipped",
                   manual_cmd=f"{python_exe} -m backend.server.app")
        return True

    if SERVER_PID_FILE.exists():
        try:
            old_pid = int(SERVER_PID_FILE.read_text().strip())
            os.kill(old_pid, 0)  # signal 0 = check existence
            logger.info(f"  server already running (pid={old_pid})")
            _set_phase(state, "server", "completed", pid=old_pid, reused=True)
            return True
        except (OSError, ValueError):
            SERVER_PID_FILE.unlink(missing_ok=True)

    _set_phase(state, "server", "in_progress")
    log_fh = SERVER_LOG_FILE.open("ab")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    proc = subprocess.Popen(
        [python_exe, "-m", "backend.server.app"],
        cwd=str(PROJECT_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    SERVER_PID_FILE.write_text(str(proc.pid))
    logger.info(f"  server launched (pid={proc.pid}); log={SERVER_LOG_FILE}")

    # quick liveness check
    time.sleep(3)
    if proc.poll() is not None:
        logger.error(f"  server died immediately (exit={proc.returncode}); "
                     f"see {SERVER_LOG_FILE}")
        _set_phase(state, "server", "failed", exit_code=proc.returncode)
        return False

    _set_phase(state, "server", "completed", pid=proc.pid,
               log=str(SERVER_LOG_FILE))
    return True


# ---------- phase 4: monitor ----------

def phase_monitor(state: Dict[str, Any], db_path: Path,
                  poll_interval: int, idle_timeout_min: int,
                  dry: bool) -> bool:
    logger.info("[phase 4] monitor (poll=%ss, idle_timeout=%smin)",
                poll_interval, idle_timeout_min)
    if dry:
        logger.info("  dry-run: skipping monitor loop")
        _set_phase(state, "monitor", "skipped", reason="dry_run")
        return True

    job_id = state["phases"]["webdav_register"].get("job_id")
    if not job_id:
        logger.info("  no webdav job to monitor")
        _set_phase(state, "monitor", "completed", reason="no_job")
        return True

    _set_phase(state, "monitor", "in_progress", job_id=job_id)

    last_done = -1
    last_progress_ts = time.time()
    snapshots: List[Dict[str, Any]] = []

    try:
        while True:
            snap = _job_snapshot(db_path, job_id)
            done = snap["done"]
            total = snap["total"]
            pct = (done / total * 100) if total else 0
            logger.info(
                f"  job#{job_id}: {done:,}/{total:,} ({pct:.1f}%) "
                f"DL:{snap['dl']} P:{snap['parse']} "
                f"MC:{snap['mc']} VV:{snap['vv']} MV:{snap['mv']} "
                f"err:{snap['errors']}"
            )
            snapshots.append({"ts": time.time(), **snap})

            # save rolling snapshot every poll
            _set_phase(state, "monitor", "in_progress",
                       job_id=job_id, last_snapshot=snap,
                       samples=len(snapshots))

            if total and done >= total:
                logger.info(f"  job#{job_id} reached completion ({done}/{total})")
                _set_phase(state, "monitor", "completed",
                           job_id=job_id, last_snapshot=snap,
                           samples=len(snapshots))
                return True

            if done != last_done:
                last_progress_ts = time.time()
                last_done = done
            elif (time.time() - last_progress_ts) > idle_timeout_min * 60:
                logger.warning(f"  no progress for {idle_timeout_min} min — "
                               f"saving state and exiting monitor")
                _set_phase(state, "monitor", "stalled",
                           job_id=job_id, last_snapshot=snap,
                           samples=len(snapshots),
                           idle_minutes=idle_timeout_min)
                return False

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.warning("  monitor interrupted; state preserved for resume")
        _set_phase(state, "monitor", "in_progress",
                   job_id=job_id, samples=len(snapshots),
                   reason="interrupted")
        raise


def _job_snapshot(db_path: Path, job_id: int) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        total, done = conn.execute(
            "SELECT total_files, COALESCE(completed_files, 0) "
            "FROM analysis_jobs WHERE id = ?",
            (job_id,),
        ).fetchone() or (0, 0)
        row = conn.execute(
            "SELECT "
            "  SUM(CASE WHEN download_status='done' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN parse_status='done' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN mc_status='done' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN vv_status='done' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN mv_status='done' THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN parse_status='error' OR mc_status='error' "
            "            OR vv_status='error' OR mv_status='error' "
            "       THEN 1 ELSE 0 END), "
            "  COUNT(*) "
            "FROM file_tasks WHERE analysis_job_id = ?",
            (job_id,),
        ).fetchone()
        dl, parse, mc, vv, mv, errs, n = row or (0, 0, 0, 0, 0, 0, 0)
        return {
            "total": total, "done": done,
            "dl": dl or 0, "parse": parse or 0,
            "mc": mc or 0, "vv": vv or 0, "mv": mv or 0,
            "errors": errs or 0, "task_rows": n or 0,
        }
    finally:
        conn.close()


# ---------- phase 5: verify ----------

def phase_verify(state: Dict[str, Any], db_path: Path,
                 dry: bool = False) -> bool:
    logger.info("[phase 5] verify (audit + diff)")
    final = audit.collect(db_path)
    state["final"] = final
    save_state(state)

    print("\n" + "=" * 60)
    print(audit.render(final))
    print("=" * 60)
    if state.get("baseline"):
        print()
        print(audit.diff(state["baseline"], final))

    _record_failures(db_path, state, dry=dry)
    _set_phase(state, "verify", "completed",
               remaining=final["fallback_remaining"]["total"])
    return True


def _record_failures(db_path: Path, state: Dict[str, Any],
                     dry: bool = False) -> None:
    if dry:
        return
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT file_path FROM files "
            "WHERE processing_status = 'parse_fallback_legacy' "
            "ORDER BY id LIMIT 5000"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        if FAILURES_FILE.exists():
            FAILURES_FILE.unlink()
        return
    FAILURES_FILE.write_text("\n".join(r[0] for r in rows) + "\n",
                             encoding="utf-8")
    logger.info(f"  {len(rows):,} files still in parse_fallback_legacy "
                f"-> {FAILURES_FILE.name}")


# ---------- pipeline driver ----------

PHASE_ORDER_KEYS = ["local", "webdav-reg", "server", "monitor", "verify"]


def run_pipeline(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    python_exe = args.python or sys.executable

    state = load_state()

    only = set(args.only or [])
    skip = set(args.skip or [])

    def _wanted(name: str) -> bool:
        if only and name not in only:
            return False
        if name in skip:
            return False
        return True

    if not phase_preflight(state, db_path, args.skip_backup, args.dry_run):
        return 2
    if state["phases"]["preflight"].get("nothing_to_do"):
        phase_verify(state, db_path)
        return 0

    if _wanted("local"):
        if _phase_done(state, "local") and not args.force:
            logger.info("[phase 1] local: already completed (use --force to re-run)")
        else:
            ok = phase_local(state, db_path, python_exe,
                             args.dry_run, args.local_limit)
            if not ok and not args.continue_on_error:
                return 3
    else:
        logger.info("[phase 1] local: skipped by --only/--skip")

    if _wanted("webdav-reg"):
        if _phase_done(state, "webdav_register") and not args.force:
            logger.info("[phase 2] webdav-register: already completed")
        else:
            ok = phase_webdav_register(state, db_path, python_exe,
                                       args.dry_run, args.webdav_limit)
            if not ok and not args.continue_on_error:
                return 4
    else:
        logger.info("[phase 2] webdav-register: skipped")

    if _wanted("server"):
        ok = phase_server(state, python_exe, args.auto_server, args.dry_run)
        if not ok and args.auto_server and not args.continue_on_error:
            return 5
    else:
        logger.info("[phase 3] server: skipped")

    if _wanted("monitor"):
        ok = phase_monitor(state, db_path,
                           args.poll_interval, args.idle_timeout, args.dry_run)
        if not ok and not args.continue_on_error:
            phase_verify(state, db_path)
            return 6
    else:
        logger.info("[phase 4] monitor: skipped")

    if _wanted("verify"):
        phase_verify(state, db_path, dry=args.dry_run)

    return 0


# ---------- subcommand handlers ----------

def cmd_status(args: argparse.Namespace) -> int:
    state = load_state()
    print("State file:", STATE_FILE)
    print("Updated:", state.get("updated_at") or "(never)")
    print()
    print("Phases:")
    for p in PHASES:
        info = state["phases"].get(p, {})
        st = info.get("status", "pending")
        ts = info.get("ts", "")
        extra = ""
        if p == "local" and "succeeded" in info:
            extra = f"  succeeded={info['succeeded']} failed={info['failed']}"
        elif p == "webdav_register" and "job_id" in info:
            extra = f"  job_id={info['job_id']} registered={info.get('registered')}"
        elif p == "monitor" and "last_snapshot" in info:
            s = info["last_snapshot"]
            extra = f"  done={s.get('done')}/{s.get('total')}"
        print(f"  {p:<18} {st:<12} {ts}{extra}")
    print()
    snap = audit.collect(Path(args.db))
    print(audit.render(snap))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    state = load_state()
    return 0 if phase_verify(state, Path(args.db)) else 1


def cmd_dry_run(args: argparse.Namespace) -> int:
    global _state_path
    _state_path = DRYRUN_STATE_FILE  # divert writes away from real state
    args.dry_run = True
    args.auto_server = False
    args.skip_backup = True
    args.only = []
    args.skip = []
    args.force = True  # always re-check, dry-run is idempotent
    args.continue_on_error = False
    args.local_limit = 0
    args.webdav_limit = 0
    args.poll_interval = 30
    args.idle_timeout = 30
    args.python = None
    rc = run_pipeline(args)
    logger.info(f"dry-run state at {DRYRUN_STATE_FILE.name} "
                f"(real state {STATE_FILE.name} untouched)")
    return rc


def cmd_reset(args: argparse.Namespace) -> int:
    removed = []
    for path in (STATE_FILE, DRYRUN_STATE_FILE):
        if path.exists():
            path.unlink()
            removed.append(path.name)
    if removed:
        print("removed:", ", ".join(removed))
    else:
        print("nothing to remove")
    return 0


# ---------- argparse ----------

def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--python", default=None,
                   help="Python interpreter for child scripts (default: this one)")
    p.add_argument("--dry-run", action="store_true",
                   help="Counts only; no DB writes / no child invocations")
    p.add_argument("--skip-backup", action="store_true")
    p.add_argument("--auto-server", action="store_true",
                   help="Launch backend.server.app in background for phase 4")
    p.add_argument("--only", action="append", choices=PHASE_ORDER_KEYS,
                   help="Run only the named phase(s). Repeat to select multiple")
    p.add_argument("--skip", action="append", choices=PHASE_ORDER_KEYS,
                   help="Skip the named phase(s)")
    p.add_argument("--force", action="store_true",
                   help="Re-run completed phases instead of skipping them")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Don't abort the pipeline when a phase reports failure")
    p.add_argument("--local-limit", type=int, default=0,
                   help="0 = all local fallback files")
    p.add_argument("--webdav-limit", type=int, default=0,
                   help="0 = all webdav fallback files")
    p.add_argument("--poll-interval", type=int, default=30,
                   help="Monitor poll interval (seconds)")
    p.add_argument("--idle-timeout", type=int, default=30,
                   help="Stop monitor if no progress for N minutes")
    p.add_argument("-v", "--verbose", action="store_true")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="restore_fallback")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Execute the restore pipeline")
    _add_run_flags(p_run)

    p_resume = sub.add_parser("resume",
                              help="Resume from last incomplete phase")
    _add_run_flags(p_resume)

    p_status = sub.add_parser("status", help="Print current state + audit")
    p_status.add_argument("--db", default=str(DEFAULT_DB))
    p_status.add_argument("-v", "--verbose", action="store_true")

    p_verify = sub.add_parser("verify", help="Final audit + diff vs baseline")
    p_verify.add_argument("--db", default=str(DEFAULT_DB))
    p_verify.add_argument("-v", "--verbose", action="store_true")

    p_dry = sub.add_parser("dry-run",
                           help="Preflight + count phases, no writes")
    p_dry.add_argument("--db", default=str(DEFAULT_DB))
    p_dry.add_argument("-v", "--verbose", action="store_true")

    p_reset = sub.add_parser("reset",
                             help="Delete state files (run + dry-run)")
    p_reset.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))

    def _on_sigterm(signum, _frame):
        logger.warning(f"received signal {signum}; preserving state")
        sys.exit(130)
    signal.signal(signal.SIGTERM, _on_sigterm)

    if args.cmd == "run":
        return run_pipeline(args)
    if args.cmd == "resume":
        return run_pipeline(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "dry-run":
        return cmd_dry_run(args)
    if args.cmd == "reset":
        return cmd_reset(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # phase handlers already saved state via _set_phase before re-raising;
        # exit cleanly so the user doesn't see a scary traceback on Ctrl-C
        logger.info("interrupted (state preserved — use `resume` to continue)")
        sys.exit(130)
