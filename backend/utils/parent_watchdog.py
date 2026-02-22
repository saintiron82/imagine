"""
Parent process watchdog — auto-exit when the parent process (Electron) dies.

When a Python subprocess is spawned by Electron with piped stdio, the stdin
pipe acts as a "lifeline".  If Electron crashes or is force-killed, the pipe
handle becomes invalid.  This module monitors stdin in a background thread
and triggers process exit when the pipe breaks.

This is critical for processes that do NOT read stdin themselves:
  - FastAPI server (uvicorn)
  - Pipeline / Discover (ingest_engine.py)

Processes that already read stdin (search daemon, worker IPC) do NOT need
this watchdog — they detect pipe closure through their own read loops.

Usage::

    from backend.utils.parent_watchdog import start_parent_watchdog
    start_parent_watchdog()  # Call early in process startup
"""

import logging
import os
import signal
import sys
import threading

logger = logging.getLogger(__name__)


def start_parent_watchdog(check_interval: float = 2.0, exit_code: int = 0):
    """Start a background thread that exits when the parent process dies.

    Detection method (dual approach for robustness):
    1. **stdin pipe monitoring**: Read stdin — when the pipe breaks
       (parent died), read() returns EOF immediately.
    2. **Parent PID polling (fallback)**: If stdin is not piped (e.g.
       running from terminal), poll parent PID every ``check_interval``
       seconds.

    Args:
        check_interval: Seconds between parent PID checks (fallback only).
        exit_code: Exit code when parent death is detected.
    """
    if sys.stdin.isatty():
        # Not a piped subprocess — use PID polling as fallback
        _start_pid_watchdog(check_interval, exit_code)
    else:
        # Piped subprocess — monitor stdin (more responsive)
        _start_stdin_watchdog(exit_code)


def _start_stdin_watchdog(exit_code: int):
    """Monitor stdin pipe — exit when it closes (parent died)."""
    def _watch():
        try:
            # On Windows, sys.stdin.read() may be wrapped in TextIOWrapper.
            # Read from the underlying buffer for reliability.
            if hasattr(sys.stdin, 'buffer'):
                sys.stdin.buffer.read()
            else:
                sys.stdin.read()
        except Exception:
            pass
        logger.info("Parent process died (stdin pipe closed) — exiting")
        os._exit(exit_code)

    t = threading.Thread(target=_watch, name="parent-watchdog", daemon=True)
    t.start()
    logger.debug("Parent watchdog started (stdin pipe mode)")


def _start_pid_watchdog(check_interval: float, exit_code: int):
    """Poll parent PID — exit when parent is no longer alive."""
    import time

    parent_pid = os.getppid()
    if parent_pid <= 1:
        # Parent is init/launchd — we're already orphaned or running standalone
        logger.debug("Parent PID is 1 (init) — watchdog not started")
        return

    def _watch():
        while True:
            time.sleep(check_interval)
            if not _is_pid_alive(parent_pid):
                logger.info(f"Parent process (PID {parent_pid}) died — exiting")
                os._exit(exit_code)

    t = threading.Thread(target=_watch, name="parent-watchdog", daemon=True)
    t.start()
    logger.debug(f"Parent watchdog started (PID poll mode, parent={parent_pid})")


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle == 0:
            return False
        try:
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)  # Signal 0 = existence check only
            return True
        except OSError:
            return False
