#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${1:-dist}"

if [[ ! -e "$DIST_DIR" ]]; then
  echo "Usage: $0 [DIST_DIR]"
  echo "Error: '$DIST_DIR' not found."
  exit 2
fi

# Track whether ANY single-arch of a given family exists
found_arm_single=0
found_x86_single=0

# Collect examples for nicer error messages
declare -a arm_single_files=()
declare -a x86_single_files=()
declare -a unknown_macho=()

# Normalize lipo output into an array of arches
get_arches() {
  local f="$1"
  # Prefer lipo -archs; fall back to parsing 'file' if needed
  if arches=$(lipo -archs "$f" 2>/dev/null); then
    echo "$arches"
    return
  fi
  # Fallback: try to detect from 'file' output
  local finfo
  finfo=$(file -b "$f" 2>/dev/null || true)
  # Common patterns: "Mach-O 64-bit executable arm64", "Mach-O 64-bit bundle x86_64"
  if grep -qi 'Mach-O' <<<"$finfo"; then
    if grep -q 'arm64' <<<"$finfo"; then
      echo "arm64"
      return
    elif grep -q 'x86_64' <<<"$finfo"; then
      echo "x86_64"
      return
    fi
  fi
  echo ""  # unknown
}

is_macho() {
  file -b "$1" 2>/dev/null | grep -q 'Mach-O'
}

# Walk all files
while IFS= read -r -d '' f; do
  is_macho "$f" || continue

  arches_str=$(get_arches "$f")
  # skip if we truly couldn't tell
  if [[ -z "$arches_str" ]]; then
    unknown_macho+=("$f")
    continue
  fi

  # Split into array
  read -r -a ARCHES <<<"$arches_str"

  # Classify
  has_arm=0
  has_x86=0
  for a in "${ARCHES[@]}"; do
    case "$a" in
      arm64|arm64e) has_arm=1 ;;
      x86_64|x86_64h) has_x86=1 ;;
      *) ;; # ignore others if ever present
    esac
  done

  if (( has_arm == 1 && has_x86 == 1 )); then
    # universal2: allowed in either mode
    :
  elif (( has_arm == 1 && has_x86 == 0 )); then
    found_arm_single=1
    arm_single_files+=("$f")
  elif (( has_arm == 0 && has_x86 == 1 )); then
    found_x86_single=1
    x86_single_files+=("$f")
  else
    # Neither arm nor x86 detected (rare); mark unknown
    unknown_macho+=("$f")
  fi
done < <(find "$DIST_DIR" -type f -print0)

# Decision logic
if (( found_arm_single == 1 && found_x86_single == 1 )); then
  echo "❌ Mixed single-arch binaries detected (ARM and x86_64)."
  echo
  echo "ARM-only binaries:"
  for f in "${arm_single_files[@]}"; do
    echo "  $f"
  done
  echo
  echo "x86_64-only binaries:"
  for f in "${x86_single_files[@]}"; do
    echo "  $f"
  done
  exit 1
fi

# If we reach here, mode is consistent (or all universal2)
if (( found_arm_single == 1 )); then
  echo "✅ Consistent build: ARM-only + universal2."
elif (( found_x86_single == 1 )); then
  echo "✅ Consistent build: x86_64-only + universal2."
else
  echo "✅ Consistent build: all universal2 (or Mach-O types without explicit arm/x86 slices)."
fi

# Warn about any Mach-O we couldn't classify (non-fatal)
if (( ${#unknown_macho[@]} > 0 )); then
  echo
  echo "⚠️  Warning: some Mach-O files had unrecognized arch metadata:"
  for f in "${unknown_macho[@]}"; do
    echo "  $f"
  done
  echo "    (Check with: lipo -info '<file>' && file '<file>')"
fi

exit 0
