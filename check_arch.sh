#!/usr/bin/env bash
set -euo pipefail

# Enable nullglob so the loop skips if no files match
shopt -s nullglob

DIST_DIR="dist"
errors=0

# Iterate all files under dist/
while IFS= read -r -d '' file; do
  # Only consider Mach-O binaries (skip text, images, etc.)
  if file "$file" | grep -q 'Mach-O'; then
    info=$(lipo -info "$file" 2>/dev/null || true)
    # Case 1: non-fat file, architecture: arm64 (and no x86_64)
    if [[ $info == *"Non-fat file"* && $info == *"arm64"* && $info != *"x86_64"* ]]; then
      echo "⚠️  ARM64-only: $file — $info"
      errors=1
    # Case 2: fat file whose only slice is arm64
    elif [[ $info == *"Architectures in the fat file"* ]]; then
      # Extract everything after “are:”
      arches=${info#*are:\ }
      # Normalize spaces
      arches=($arches)
      if [[ ${#arches[@]} -eq 1 && ${arches[0]} == "arm64" ]]; then
        echo "⚠️  ARM64-only (fat): $file — $info"
        errors=1
      fi
    fi
  fi
done < <(find "$DIST_DIR" -type f -print0)

if (( errors )); then
  echo
  echo "❌  Build check failed: ARM64-only binaries detected."
  exit 1
else
  echo "✅  All good: no ARM64-only binaries in $DIST_DIR."
  exit 0
fi
