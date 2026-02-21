#!/bin/bash
# stop-all.sh — Kill all Imagine-related processes
# Usage: bash scripts/stop-all.sh

echo "=== Imagine — Stop All ==="

# Vite dev server (port 9274)
lsof -ti :9274 2>/dev/null | xargs kill 2>/dev/null && echo "  Killed Vite on :9274" || true

# FastAPI server (port 8000)
lsof -ti :8000 2>/dev/null | xargs kill 2>/dev/null && echo "  Killed Server on :8000" || true

# Electron
pgrep -f "electron \." 2>/dev/null | while read pid; do
    cmdline=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if echo "$cmdline" | grep -qi "imagine"; then
        kill "$pid" 2>/dev/null && echo "  Killed Electron PID $pid"
    fi
done

# Python backend processes
pkill -f "backend.server.app" 2>/dev/null && echo "  Killed FastAPI" || true
pkill -f "api_search.py" 2>/dev/null && echo "  Killed Search Daemon" || true
pkill -f "ingest_engine.py" 2>/dev/null && echo "  Killed Ingest Engine" || true
pkill -f "worker_daemon.py" 2>/dev/null && echo "  Killed Worker Daemon" || true

# npm/concurrently leftovers
pgrep -f "wait-on tcp:9274" 2>/dev/null | xargs kill 2>/dev/null || true
pgrep -f "npm run electron:dev" 2>/dev/null | xargs kill 2>/dev/null || true

sleep 1

# Force kill if ports still occupied
for port in 9274 8000; do
    if lsof -ti :$port >/dev/null 2>&1; then
        echo "  Force killing remaining on :${port}..."
        lsof -ti :$port | xargs kill -9 2>/dev/null || true
    fi
done

echo "Done."
