#!/bin/bash
# server.sh — FastAPI server (serves SPA from frontend/dist/)
# Usage: bash server.sh [port]

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8000}"

# Activate venv
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Check frontend build
if [ ! -d "$PROJECT_ROOT/frontend/dist" ]; then
    echo "[!] frontend/dist/ not found. Building..."
    cd "$PROJECT_ROOT/frontend" && npm run build
    if [ $? -ne 0 ]; then
        echo "[ERROR] Frontend build failed."
        exit 1
    fi
fi

echo "=== Imagine — FastAPI Server ==="
echo "  URL:  http://localhost:${PORT}"
echo "  Docs: http://localhost:${PORT}/docs"
echo ""

cd "$PROJECT_ROOT"
exec python -m backend.server.app
