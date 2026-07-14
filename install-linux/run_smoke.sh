#!/usr/bin/env bash
set -euo pipefail

# Your spec creates: dist/WISER/WISER_Bin
# Phase B copied dist/* -> /out, then Phase C copies /out -> /app
BIN="/app/WISER/WISER_Bin"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: Expected executable not found or not executable: $BIN"
  echo "Contents of /app:"
  ls -lah /app || true
  exit 1
fi

# --- FIX START: Robust Libgomp Discovery ---
# Debian arm64 won't run without libgomp on path

# 1. Initialize to empty string to prevent "unbound variable" errors (set -u)
GOMP_PATH=""

# 2. Attempt to find the library (silence errors with || true)
FOUND_LIB=$(find /app -name "libgomp.so.1" | head -n 1 || true)

# 3. Only set GOMP_PATH if we actually found something
if [[ -n "$FOUND_LIB" ]]; then
    echo "Found libgomp at: $FOUND_LIB"
    GOMP_PATH="$FOUND_LIB"
else
    echo "Info: libgomp.so.1 not found in /app. Skipping preload (safe for AMD64)."
fi
# --- FIX END ---

echo "Running smoke test: $BIN --smoke"

# 4. Conditionally run with or without LD_PRELOAD
if [[ -n "$GOMP_PATH" ]]; then
    # ARM Fix: Preload the library into the app process only
    xvfb-run -a env LD_PRELOAD="$GOMP_PATH" "$BIN" --smoke
else
    # Standard Run (AMD64 or if lib not bundled)
    xvfb-run -a "$BIN" --smoke
fi

echo "Smoke test passed."
