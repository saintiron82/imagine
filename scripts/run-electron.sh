#!/bin/bash
# run-electron.sh — Start Electron dev mode (Vite + Electron)
# Usage: bash scripts/run-electron.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/frontend"

echo "=== Imagine — Electron Dev Mode ==="
echo "  Vite:     http://localhost:9274"
echo "  Electron: auto-launch after Vite ready"
echo ""

exec npm run electron:dev
