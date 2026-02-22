"""
Parent process watchdog — auto-exit when the parent process (Electron) dies.

3-layer defense system for cross-platform reliability:

  Layer 1 (kernel, Linux only):
      prctl(PR_SET_PDEATHSIG, SIGKILL) — instant, 100% reliable.
      Kernel delivers SIGKILL directly when parent exits.

  Layer 2 (stdin pipe monitoring, primary):
      When spawned with piped stdin, reads stdin with os.read().
      Parent death closes the pipe → read returns EOF → exit.
      Response time: ~milliseconds.

  Layer 3 (PID polling, fallback):
      When stdin is a tty (running from terminal), polls parent PID.
      Response time: ~check_interval seconds (default 2s).

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
    """Start parent death detection with 3-layer defense.

    Args:
        check_interval: Seconds between parent PID checks (layer 3 only).
        exit_code: Exit code when parent death is detected.
    """
    # Layer 1: Linux kernel-level death signal (instant, most reliable)
    _try_prctl()

    # Layer 2 or 3: userspace detection
    if sys.stdin and not sys.stdin.closed and not sys.stdin.isatty():
        # Piped subprocess — monitor stdin pipe (layer 2, fast)
        _start_stdin_watchdog(exit_code)
    else:
        # Terminal or no stdin — poll parent PID (layer 3, fallback)
        _start_pid_watchdog(check_interval, exit_code)


def _try_prctl():
    """Layer 1: Linux prctl(PR_SET_PDEATHSIG, SIGKILL).

    Asks the kernel to deliver SIGKILL when the parent process exits.
    This is the fastest and most reliable mechanism — kernel-level,
    no polling, no pipe, instant.

    Only available on Linux. No-op on Windows/macOS.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        result = libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if result == 0:
            logger.debug("Layer 1: prctl(PR_SET_PDEATHSIG, SIGKILL) set")
        else:
            errno = ctypes.get_errno()
            logger.debug(f"prctl failed with errno={errno}")
    except Exception as e:
        logger.debug(f"prctl not available: {e}")


def _start_stdin_watchdog(exit_code: int):
    """Layer 2: Monitor stdin pipe — exit when it closes (parent died).

    Uses os.read() directly instead of sys.stdin.buffer.read() to avoid
    holding Python's BufferedReader internal lock, which could contend
    with C-extension threads (numpy, torch, psd-tools).
    """
    def _watch():
        try:
            fd = sys.stdin.fileno()
        except (ValueError, OSError):
            # stdin already closed or no fd available
            logger.info("stdin has no valid fd — falling back to PID polling")
            _start_pid_watchdog(2.0, exit_code)
            return

        try:
            while True:
                data = os.read(fd, 1024)
                if not data:  # EOF — pipe closed (parent died)
                    break
        except OSError:
            pass  # Broken pipe or invalid fd — same as parent death

        logger.info("Parent process died (stdin pipe closed) — exiting")
        os._exit(exit_code)

    t = threading.Thread(target=_watch, name="parent-watchdog-stdin", daemon=True)
    t.start()
    logger.debug("Layer 2: Parent watchdog started (stdin pipe mode)")


def _start_pid_watchdog(check_interval: float, exit_code: int):
    """Layer 3: Poll parent PID — exit when parent is no longer alive."""
    import time

    parent_pid = os.getppid()
    if parent_pid <= 1:
        # Parent is init/launchd — we're already detached or running standalone
        logger.debug("Parent PID is 1 (init) — watchdog not started")
        return

    def _watch():
        while True:
            time.sleep(check_interval)
            if not _is_pid_alive(parent_pid):
                logger.info(f"Parent process (PID {parent_pid}) died — exiting")
                os._exit(exit_code)

    t = threading.Thread(target=_watch, name="parent-watchdog-pid", daemon=True)
    t.start()
    logger.debug(f"Layer 3: Parent watchdog started (PID poll mode, parent={parent_pid})")


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
