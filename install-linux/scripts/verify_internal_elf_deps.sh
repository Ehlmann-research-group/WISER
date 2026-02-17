#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  verify_internal_elf_deps.sh <internal_dir>

Arguments:
  internal_dir  Path to PyInstaller _internal directory

Checks:
  - For any RPATH/RUNPATH containing /opt/micromamba or /home/conda,
    fails unless the last path element is exactly $ORIGIN
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
readonly build_prefix_a="/opt/micromamba"
readonly build_prefix_b="/home/conda"

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

extract_rpath_like() {
  local f="$1"
  readelf -d "$f" 2>/dev/null \
    | sed -n "s/.*(RPATH).*Library rpath: \[\(.*\)\].*/RPATH:\1/p; s/.*(RUNPATH).*Library runpath: \[\(.*\)\].*/RUNPATH:\1/p"
}

value_has_hardcoded_build_path() {
  local path_value="$1"
  local entry
  local entries=()
  IFS=':' read -r -a entries <<< "$path_value"
  for entry in "${entries[@]}"; do
    [[ -n "$entry" ]] || continue
    if [[ "$entry" == *"$build_prefix_a"* || "$entry" == *"$build_prefix_b"* ]]; then
      return 0
    fi
  done
  return 1
}

path_ends_with_origin() {
  local path_value="$1"
  [[ "$path_value" == "\$ORIGIN" || "$path_value" == *":\$ORIGIN" ]]
}

offenders=0
scanned=0

check_one() {
  local f="$1"
  local rp_line rp_value had_problem
  had_problem=0
  scanned=$((scanned + 1))

  while IFS= read -r rp_line; do
    [[ -n "$rp_line" ]] || continue
    rp_value="${rp_line#*:}"
    if value_has_hardcoded_build_path "$rp_value" && ! path_ends_with_origin "$rp_value"; then
      if [[ $had_problem -eq 0 ]]; then
        echo "ELF offender: $f"
        had_problem=1
      fi
      echo "  ${rp_line%%:*} contains build paths but does not end with \$ORIGIN"
      echo "  ${rp_line%%:*} value: $rp_value"
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

echo "Verification passed: scanned $scanned ELF file(s); build-path RPATH/RUNPATH entries are origin-safe."
