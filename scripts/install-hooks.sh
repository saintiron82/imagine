#!/bin/bash
# Install git hooks via symlinks
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ln -sf "$SCRIPT_DIR/git-hooks/pre-push" "$PROJECT_ROOT/.git/hooks/pre-push"
chmod +x "$SCRIPT_DIR/git-hooks/pre-push"

echo "Git hooks installed."
