#!/bin/bash
# run-server.sh — Start FastAPI server (backend only)
# Usage: bash scripts/run-server.sh [port]
#
# The server serves the frontend SPA from frontend/dist/.
# Build frontend first: cd frontend && npm run build

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${1:-8000}"

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=== Imagine — FastAPI Server ==="
echo "  URL:  http://localhost:${PORT}"
echo "  SPA:  frontend/dist/ (pre-built)"
echo "  Docs: http://localhost:${PORT}/docs"
echo ""

exec python -m backend.server.app
