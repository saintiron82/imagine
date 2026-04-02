"""
JSON Log Formatter for structured stderr output.

When Electron spawns Python processes, stderr is used for logging.
This formatter outputs JSON lines so main.cjs can parse the log level
directly instead of guessing with regex (BUG-001).

Usage:
    from backend.utils.json_log_formatter import setup_json_logging
    setup_json_logging()  # Call before any logging

Output format:
    {"level":"INFO","name":"ImagineWorker","message":"[MC] Batch complete","ts":"2026-04-03 01:23:45"}
"""

import json
import logging
import sys


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured stderr output."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_json_logging(level: int = logging.INFO):
    """Replace default logging with JSON formatter on stderr.

    Safe to call multiple times — removes existing handlers first.
    """
    root = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Suppress noisy libraries
    for name in ("httpx", "httpcore", "urllib3", "requests"):
        logging.getLogger(name).setLevel(logging.WARNING)
