#!/bin/bash
# run-full.sh — Build frontend + Start FastAPI server
# Usage: bash scripts/run-full.sh [port]
#
# 1. Builds frontend (vite build)
# 2. Starts FastAPI server serving the built SPA

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PORT="${1:-8000}"

# 1. Build frontend
echo "=== Imagine — Full Build + Server ==="
echo "[1/2] Building frontend..."
cd "$PROJECT_ROOT/frontend"
npm run build
if [ $? -ne 0 ]; then
    echo "ERROR: Frontend build failed"
    exit 1
fi
echo "  Build complete."

# 2. Start server
echo "[2/2] Starting FastAPI server..."
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "  URL:  http://localhost:${PORT}"
echo "  Docs: http://localhost:${PORT}/docs"
echo ""

exec python -m backend.server.app
