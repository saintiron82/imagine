"""
Platform-aware non-blocking stdin reader for piped subprocess communication.

On Windows, blocking reads on piped stdin hold a CRT/kernel I/O lock that
prevents background threads from completing numpy/C-extension work
(psd.composite(), torch CUDA ops, etc.).  This module provides a unified
``make_stdin_reader()`` factory that returns a callable reading one line at
a time — using Win32 PeekNamedPipe on Windows (non-blocking, no lock held)
and a simple ``iter(sys.stdin)`` wrapper on Unix (safe by design).

Usage::

    from backend.utils.win32_stdin import make_stdin_reader, STDIN_CLOSED

    read_line = make_stdin_reader()
    while True:
        line = read_line()
        if line is STDIN_CLOSED:
            break
        if line is None:
            continue
        process(line)
"""

import sys
import time

# Sentinel object indicating the stdin pipe has been closed.
STDIN_CLOSED = object()


def _make_win32_reader():
    """Create a non-blocking stdin reader for Windows using PeekNamedPipe.

    Returns a callable that yields one stripped line per call:
        str:           A line of text (stripped).
        None:          No data available yet (caller should retry).
        STDIN_CLOSED:  The pipe was closed / broken.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    buf = bytearray()

    def readline():
        nonlocal buf
        # Check for a buffered complete line first
        if b"\n" in buf:
            idx = buf.index(b"\n")
            line = bytes(buf[:idx])
            del buf[:idx + 1]
            return line.decode("utf-8", errors="replace").strip() or None

        # Poll for new data without blocking
        while True:
            available = wintypes.DWORD(0)
            ok = kernel32.PeekNamedPipe(
                handle, None, 0, None, ctypes.byref(available), None
            )
            if not ok:
                return STDIN_CLOSED

            if available.value > 0:
                read_buf = ctypes.create_string_buffer(available.value)
                bytes_read = wintypes.DWORD(0)
                kernel32.ReadFile(
                    handle, read_buf, available.value,
                    ctypes.byref(bytes_read), None,
                )
                buf.extend(read_buf.raw[:bytes_read.value])

                if b"\n" in buf:
                    idx = buf.index(b"\n")
                    line = bytes(buf[:idx])
                    del buf[:idx + 1]
                    return line.decode("utf-8", errors="replace").strip() or None
            else:
                time.sleep(0.05)  # 50ms poll interval

    return readline


def _make_unix_reader():
    """Create a blocking stdin line reader for Unix (POSIX).

    On Unix, blocking stdin reads do not hold a global I/O lock that
    interferes with background threads, so a simple iterator is safe.
    """
    line_iter = iter(sys.stdin)

    def readline():
        try:
            return next(line_iter).strip() or None
        except StopIteration:
            return STDIN_CLOSED

    return readline


def make_stdin_reader():
    """Return a platform-appropriate stdin line reader.

    On Windows: Win32 PeekNamedPipe non-blocking reader.
    On Unix:    Standard blocking ``iter(sys.stdin)`` wrapper.

    The returned callable has the same contract on both platforms:
        str:           A stripped line of text.
        None:          No data available (Windows) or empty line (Unix).
        STDIN_CLOSED:  The pipe was closed.
    """
    if sys.platform == "win32":
        return _make_win32_reader()
    else:
        return _make_unix_reader()
