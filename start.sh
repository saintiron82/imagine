#!/bin/bash
# start.sh — Electron dev mode (Vite + Electron)
# Usage: bash start.sh

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Activate venv
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "=== Imagine — Electron Dev ==="
echo "  Vite:     http://localhost:9274"
echo "  Electron: auto-launch after Vite ready"
echo ""

cd "$PROJECT_ROOT/frontend"
exec npm run electron:dev
