#!/bin/bash
# Build backend_cli using PyInstaller (macOS/Linux)
# Run from project root: bash scripts/build-backend.sh

set -e

echo "=== Imagine Backend Build ==="
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.11+ first."
    exit 1
fi

# Install PyInstaller if needed
if ! python3 -m PyInstaller --version &> /dev/null 2>&1; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Install backend dependencies
echo "Installing backend dependencies..."
pip3 install -r requirements.txt
pip3 install uvicorn fastapi python-jose passlib python-multipart

# Build
echo
echo "Building backend_cli..."
python3 -m PyInstaller backend_cli.spec --noconfirm

echo
echo "=== Build complete ==="
echo "Output: dist/backend_cli/backend_cli"
echo
echo "Next: cd frontend && npm run build && npx electron-builder"
