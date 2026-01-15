#!/usr/bin/env bash
set -euo pipefail

# Your spec creates: dist/WISER/WISER_Bin
# Phase B copied dist/* -> /out, then Phase C copies /out -> /app
BIN="/app/WISER/WISER_Bin"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: Expected executable not found or not executable: $BIN"
  echo "Contents of /app:"
  ls -lah /app || true
  echo "Contents of /app/WISER (if exists):"
  ls -lah /app/WISER || true
  exit 1
fi

echo "Running smoke test: $BIN --test_mode"

# Run under Xvfb so Qt has a display even on headless runners.
# If your test_mode truly never initializes Qt, this still works fine.
xvfb-run -a "$BIN" --test_mode

echo "Smoke test passed."
