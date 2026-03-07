"""
Simple in-memory rate limiter for auth endpoints.

No external dependencies — uses a dict of timestamps per IP.
Designed for single-process uvicorn (SQLite constraint).
"""

import time
from collections import defaultdict

from fastapi import Request, HTTPException

# Per-endpoint stores: IP -> list of request timestamps
_login_calls: dict[str, list[float]] = defaultdict(list)
_register_calls: dict[str, list[float]] = defaultdict(list)
_refresh_calls: dict[str, list[float]] = defaultdict(list)
_worker_token_calls: dict[str, list[float]] = defaultdict(list)

LOGIN_LIMIT = 5       # max login attempts per window
REGISTER_LIMIT = 3    # max register attempts per window
REFRESH_LIMIT = 10    # max refresh attempts per window
WORKER_TOKEN_LIMIT = 5  # max worker-token exchanges per window
WINDOW_SECONDS = 60   # 1-minute sliding window


def _get_client_ip(request: Request) -> str:
    """Get client IP, considering X-Forwarded-For (Cloudflare Tunnel)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_rate(
    store: dict[str, list[float]],
    key: str,
    max_calls: int,
    window: int,
) -> None:
    """Raise 429 if rate limit exceeded."""
    now = time.time()
    cutoff = now - window
    # Prune expired entries
    store[key] = [t for t in store[key] if t > cutoff]
    if len(store[key]) >= max_calls:
        raise HTTPException(
            status_code=429,
            detail=f"Too many attempts. Try again in {window} seconds.",
        )
    store[key].append(now)


async def check_login_rate(request: Request) -> None:
    """FastAPI dependency: rate-limit login attempts (5/min per IP)."""
    ip = _get_client_ip(request)
    _check_rate(_login_calls, ip, LOGIN_LIMIT, WINDOW_SECONDS)


async def check_register_rate(request: Request) -> None:
    """FastAPI dependency: rate-limit registration attempts (3/min per IP)."""
    ip = _get_client_ip(request)
    _check_rate(_register_calls, ip, REGISTER_LIMIT, WINDOW_SECONDS)


async def check_refresh_rate(request: Request) -> None:
    """FastAPI dependency: rate-limit token refresh attempts (10/min per IP)."""
    ip = _get_client_ip(request)
    _check_rate(_refresh_calls, ip, REFRESH_LIMIT, WINDOW_SECONDS)


async def check_worker_token_rate(request: Request) -> None:
    """FastAPI dependency: rate-limit worker token exchanges (5/min per IP)."""
    ip = _get_client_ip(request)
    _check_rate(_worker_token_calls, ip, WORKER_TOKEN_LIMIT, WINDOW_SECONDS)
