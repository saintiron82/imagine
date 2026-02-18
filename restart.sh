#!/bin/bash
# restart.sh — Kill all Imagine processes, bump version tag, restart electron:dev
# Usage: bash restart.sh  (from project root)

set -e

# Project root is where this script is
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT/frontend"

echo "=== Imagine Restart ==="

# 1. Kill all existing Imagine-related processes
echo "[1/3] Killing existing processes..."

# Kill Vite dev server on port 9274
lsof -ti :9274 2>/dev/null | xargs kill 2>/dev/null && echo "  Killed Vite on :9274" || true

# Kill Electron instances
pgrep -f "electron \." 2>/dev/null | while read pid; do
    cmdline=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if echo "$cmdline" | grep -qi "imagine"; then
        kill "$pid" 2>/dev/null && echo "  Killed Electron PID $pid"
    fi
done

# Kill Python Backend Processes (Search Daemon, Pipeline)
pkill -f "api_search.py" 2>/dev/null && echo "  Killed Search Daemon" || true
pkill -f "ingest_engine.py" 2>/dev/null && echo "  Killed Ingest Engine" || true
pkill -f "Imagine-Search" 2>/dev/null && echo "  Killed Imagine-Search process" || true

# Kill any leftover npm scripts
pgrep -f "npm run electron:dev" 2>/dev/null | xargs kill 2>/dev/null || true
pgrep -f "npm run dev" 2>/dev/null | xargs kill 2>/dev/null || true

sleep 1

# Verify port 9274 is free
if lsof -ti :9274 >/dev/null 2>&1; then
    echo "  Force killing remaining on :9274..."
    lsof -ti :9274 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# 2. Bump version tag in StatusBar
echo "[2/3] Bumping version..."
TODAY=$(date +%Y%m%d)
STATUSBAR="src/components/StatusBar.jsx"

# Extract current build number for today
# Matches format: vX.Y.Z.YYYYMMDD_NN
CURRENT=$(sed -n "s/.*v[0-9]*\.[0-9]*\.[0-9]*\.${TODAY}_\([0-9]*\).*/\1/p" "$STATUSBAR" 2>/dev/null | head -1)

if [ -z "$CURRENT" ]; then
    # Different date or first build of day -> start at 01
    NEXT_PADDED="01"
else
    NEXT=$(( 10#$CURRENT + 1 ))
    NEXT_PADDED=$(printf "%02d" "$NEXT")
fi

# Base version (v3.6.0) - update this if major/minor version changes
BASE_VER="v3.6.0"
NEW_VERSION="${BASE_VER}.${TODAY}_${NEXT_PADDED}"

# Replace version string in file (handles any previous vX.Y.Z.Date_Build format)
sed -i '' "s/v[0-9]*\.[0-9]*\.[0-9]*\.[0-9]*_[0-9]*/${NEW_VERSION}/" "$STATUSBAR"
echo "  Version: $NEW_VERSION"

# 3. Start fresh
echo "[3/3] Starting electron:dev..."
exec npm run electron:dev
