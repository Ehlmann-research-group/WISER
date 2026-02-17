#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  patch_internal_elf_deps.sh <internal_dir> [env_name] [abs_prefix] [abs_sqlite]

Arguments:
  internal_dir  Path to PyInstaller _internal directory
  env_name      Optional conda env name (default: wiser-prod)
  abs_prefix    Optional absolute build prefix to scrub
                (default: /opt/micromamba/envs)
  abs_sqlite    Optional absolute sqlite path to rewrite
                (default: <abs_prefix>/<env_name>/lib/libsqlite3.so)

Behavior:
  - Scans all ELF files under _internal (shared libs and extension modules)
  - Rewrites absolute DT_NEEDED entries to sonames via patchelf --replace-needed
  - Ensures local lookup with patchelf --force-rpath --set-rpath '$ORIGIN'
  - Creates sqlite aliases:
      libsqlite3.so -> latest libsqlite3.so.*
      libsqlite3.so.0 -> latest libsqlite3.so.*
      libsqlite3.so.3 -> latest libsqlite3.so.* (best-effort compatibility)
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
env_name="${2:-wiser-prod}"
abs_prefix="${3:-/opt/micromamba/envs}"
abs_sqlite="${4:-${abs_prefix}/${env_name}/lib/libsqlite3.so}"

if [[ ! -d "$internal_dir" ]]; then
  echo "ERROR: _internal directory not found: $internal_dir" >&2
  exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
  echo "ERROR: readelf is required (install binutils)." >&2
  exit 1
fi
if ! command -v patchelf >/dev/null 2>&1; then
  echo "ERROR: patchelf is required." >&2
  exit 1
fi

echo "Patching ELF deps under: $internal_dir"
echo "Absolute prefix scrub target: $abs_prefix"
echo "Absolute sqlite target: $abs_sqlite"

ensure_sqlite_aliases() {
  local latest
  latest="$(
    ls -1 "$internal_dir"/libsqlite3.so.* 2>/dev/null \
      | sed 's#^.*/##' \
      | sort -V \
      | tail -n 1 || true
  )"
  if [[ -z "$latest" ]]; then
    echo "No bundled libsqlite3.so.* found in _internal; skipping alias creation."
    return 0
  fi

  (
    cd "$internal_dir"
    ln -sfn "$latest" libsqlite3.so
    ln -sfn "$latest" libsqlite3.so.0
    ln -sfn "$latest" libsqlite3.so.3
    echo "SQLite aliases:"
    ls -lah libsqlite3.so*
  )
}

is_elf() {
  local f="$1"
  readelf -h "$f" >/dev/null 2>&1
}

extract_needed() {
  local f="$1"
  readelf -d "$f" 2>/dev/null \
    | sed -n "s/.*(NEEDED).*Shared library: \[\(.*\)\].*/\1/p"
}

patch_one() {
  local f="$1"
  local needed old base
  while IFS= read -r needed; do
    [[ -n "$needed" ]] || continue
    if [[ "$needed" == /* ]]; then
      old="$needed"
      base="$(basename "$old")"
      if [[ "$old" == "$abs_sqlite" ]]; then
        base="libsqlite3.so.0"
      fi
      if [[ "$old" != "$base" ]]; then
        echo "replace-needed: $f"
        echo "  $old -> $base"
        patchelf --replace-needed "$old" "$base" "$f"
      fi
    fi
  done < <(extract_needed "$f")

  # Ensure bundle-local lookup; this avoids accidental build-host library resolution.
  patchelf --force-rpath --set-rpath '$ORIGIN' "$f"
}

ensure_sqlite_aliases

patched_count=0
while IFS= read -r -d '' f; do
  if ! is_elf "$f"; then
    continue
  fi
  patch_one "$f"
  patched_count=$((patched_count + 1))
done < <(find "$internal_dir" -type f \( -name "*.so" -o -name "*.so.*" -o -name "*.pyd" \) -print0)

# Catch extension modules without a typical suffix (e.g. *_gdal*.so already covered,
# but this loop also includes executables/other ELF files not matched above).
while IFS= read -r -d '' f; do
  if ! is_elf "$f"; then
    continue
  fi
  case "$f" in
    *.so|*.so.*|*.pyd) continue ;;
  esac
  patch_one "$f"
  patched_count=$((patched_count + 1))
done < <(find "$internal_dir" -type f -print0)

echo "Patched ELF file count: $patched_count"
