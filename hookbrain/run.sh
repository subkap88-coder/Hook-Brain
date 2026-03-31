#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"
source venv/bin/activate

# Install web deps into the venv if not already present
python -c "import flask" 2>/dev/null    || pip install flask --quiet
python -c "import anthropic" 2>/dev/null || pip install anthropic --quiet

echo "Starting HookBrain on http://127.0.0.1:5050"
python hookbrain/app.py
