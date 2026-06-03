#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so ruff finds its config and checks the whole tree.
cd "$(git rev-parse --show-toplevel)"

# Locate ruff inside the project venv (handles Linux/macOS and Windows git-bash).
if [ -x ".venv/bin/ruff" ]; then
  RUFF=".venv/bin/ruff"
elif [ -x ".venv/Scripts/ruff.exe" ]; then
  RUFF=".venv/Scripts/ruff.exe"
else
  echo "error: ruff not found in .venv" >&2
  echo "  Set up the environment first:" >&2
  echo "    ./scripts/setup.sh" >&2
  exit 1
fi

echo "Running ruff on all Python files in the repo..."

# Either command failing exits non-zero and blocks the commit.
"$RUFF" check .          # lint
"$RUFF" format --check . # verify formatting

echo "ruff: all checks passed."
