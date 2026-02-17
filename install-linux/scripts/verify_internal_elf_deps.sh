#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  verify_internal_elf_deps.sh <internal_dir> [abs_prefix]

Arguments:
  internal_dir  Path to PyInstaller _internal directory
  abs_prefix    Absolute build prefix that must not appear in dynamic entries
                (default: /opt/micromamba/envs)

Checks:
  - Fails if any DT_NEEDED entry is absolute (/...)
  - Fails if any RPATH/RUNPATH contains absolute build-time paths (abs_prefix)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

internal_dir="$1"
abs_prefix="${2:-/opt/micromamba/envs}"

if [[ ! -d "$internal_dir" ]]; then
  echo "ERROR: _internal directory not found: $internal_dir" >&2
  exit 1
fi
if ! command -v readelf >/dev/null 2>&1; then
  echo "ERROR: readelf is required (install binutils)." >&2
  exit 1
fi

is_elf() {
  local f="$1"
  readelf -h "$f" >/dev/null 2>&1
}

extract_needed() {
  local f="$1"
  readelf -d "$f" 2>/dev/null \
    | sed -n "s/.*(NEEDED).*Shared library: \[\(.*\)\].*/\1/p"
}

extract_rpath_like() {
  local f="$1"
  readelf -d "$f" 2>/dev/null \
    | sed -n "s/.*(RPATH).*Library rpath: \[\(.*\)\].*/RPATH:\1/p; s/.*(RUNPATH).*Library runpath: \[\(.*\)\].*/RUNPATH:\1/p"
}

offenders=0
scanned=0

check_one() {
  local f="$1"
  local needed rp_line rp_value had_problem
  had_problem=0
  scanned=$((scanned + 1))

  while IFS= read -r needed; do
    [[ -n "$needed" ]] || continue
    if [[ "$needed" == /* ]]; then
      if [[ $had_problem -eq 0 ]]; then
        echo "ELF offender: $f"
        had_problem=1
      fi
      echo "  absolute DT_NEEDED: $needed"
    fi
  done < <(extract_needed "$f")

  while IFS= read -r rp_line; do
    [[ -n "$rp_line" ]] || continue
    rp_value="${rp_line#*:}"
    if [[ "$rp_value" == *"$abs_prefix"* ]]; then
      if [[ $had_problem -eq 0 ]]; then
        echo "ELF offender: $f"
        had_problem=1
      fi
      echo "  absolute ${rp_line%%:*}: $rp_value"
    fi
  done < <(extract_rpath_like "$f")

  if [[ $had_problem -eq 1 ]]; then
    offenders=$((offenders + 1))
    echo "  readelf -d excerpt:"
    readelf -d "$f" | sed -n '/NEEDED\|RPATH\|RUNPATH/p' | sed 's/^/    /'
  fi
}

while IFS= read -r -d '' f; do
  if is_elf "$f"; then
    check_one "$f"
  fi
done < <(find "$internal_dir" -type f -print0)

if [[ $offenders -gt 0 ]]; then
  echo "Verification failed: $offenders offending ELF file(s) under $internal_dir" >&2
  exit 1
fi

echo "Verification passed: scanned $scanned ELF file(s); no absolute DT_NEEDED or build-path RUNPATH/RPATH found."
