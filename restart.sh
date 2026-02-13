#!/bin/bash
# restart.sh — Kill all Imagine processes, bump version tag, restart electron:dev
# Usage: bash restart.sh  (from project root)

set -e
cd "$(dirname "$0")/frontend"

echo "=== Imagine Restart ==="

# 1. Kill all existing Imagine-related processes
echo "[1/3] Killing existing processes..."

# Kill Vite dev server on port 9274
lsof -ti :9274 2>/dev/null | xargs kill 2>/dev/null && echo "  Killed Vite on :9274" || true

# Kill Electron instances running from this project
pgrep -f "electron \." 2>/dev/null | while read pid; do
    cmdline=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if echo "$cmdline" | grep -qi "imagine"; then
        kill "$pid" 2>/dev/null && echo "  Killed Electron PID $pid"
    fi
done

# Kill any leftover concurrently/wait-on/npm for this project
pgrep -f "wait-on tcp:9274" 2>/dev/null | xargs kill 2>/dev/null || true
pgrep -f "npm run electron:dev" 2>/dev/null | xargs kill 2>/dev/null || true
pgrep -f "npm run dev" 2>/dev/null | xargs kill 2>/dev/null || true

sleep 1

# Verify port is free
if lsof -ti :9274 >/dev/null 2>&1; then
    echo "  Force killing remaining on :9274..."
    lsof -ti :9274 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# 2. Bump version tag in StatusBar
echo "[2/3] Bumping version..."
TODAY=$(date +%Y%m%d)
STATUSBAR="src/components/StatusBar.jsx"

# macOS-compatible: extract build number after date_
CURRENT=$(sed -n "s/.*v3\.5\.1\.${TODAY}_\([0-9]*\).*/\1/p" "$STATUSBAR" 2>/dev/null | head -1)
if [ -z "$CURRENT" ]; then
    # Different date or not found — start at 01
    NEXT_PADDED="01"
else
    NEXT=$(( 10#$CURRENT + 1 ))
    NEXT_PADDED=$(printf "%02d" "$NEXT")
fi
NEW_VERSION="v3.5.1.${TODAY}_${NEXT_PADDED}"

sed -i '' "s/v3\.5\.1\.[0-9]*_[0-9]*/${NEW_VERSION}/" "$STATUSBAR"
echo "  Version: $NEW_VERSION"

# 3. Start fresh
echo "[3/3] Starting electron:dev..."
exec npm run electron:dev
