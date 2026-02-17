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
                (default: /opt/micromamba)
  abs_sqlite    Optional absolute sqlite path to rewrite
                (default: /opt/micromamba/envs/<env_name>/lib/libsqlite3.so)

Behavior:
  - Scans all ELF files under _internal (shared libs and extension modules)
  - Rewrites only /opt/micromamba* DT_NEEDED entries via patchelf --replace-needed
  - If RPATH/RUNPATH contains hard-coded build paths (/opt/micromamba, /home/conda),
    appends $ORIGIN as the final path element (without removing existing entries)
  - Does not force-convert RPATH<->RUNPATH and does not overwrite to only $ORIGIN
  - Creates sqlite aliases:
      libsqlite3.so -> latest libsqlite3.so.*
      libsqlite3.so.0 -> latest libsqlite3.so.*
      libsqlite3.so.3 -> latest libsqlite3.so.* (only when bundle actually needs it)
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
abs_prefix="${3:-/opt/micromamba}"
abs_sqlite="${4:-/opt/micromamba/envs/${env_name}/lib/libsqlite3.so}"
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

extract_runpath() {
  local f="$1"
  readelf -d "$f" 2>/dev/null \
    | sed -n "s/.*(RUNPATH).*Library runpath: \[\(.*\)\].*/\1/p"
}

extract_rpath() {
  local f="$1"
  readelf -d "$f" 2>/dev/null \
    | sed -n "s/.*(RPATH).*Library rpath: \[\(.*\)\].*/\1/p"
}

bundle_needs_sqlite3_so3() {
  local f needed
  while IFS= read -r -d '' f; do
    if ! is_elf "$f"; then
      continue
    fi
    while IFS= read -r needed; do
      [[ -n "$needed" ]] || continue
      case "$needed" in
        libsqlite3.so.3|*/libsqlite3.so.3) return 0 ;;
      esac
    done < <(extract_needed "$f")
  done < <(find "$internal_dir" -type f -print0)
  return 1
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

append_origin_suffix_if_needed() {
  local path_value="$1"
  local updated="$path_value"
  if ! value_has_hardcoded_build_path "$path_value"; then
    printf '%s' "$updated"
    return 0
  fi

  if ! path_ends_with_origin "$path_value"; then
    updated="${path_value}:\$ORIGIN"
  fi
  printf '%s' "$updated"
}

update_runtime_path_if_needed() {
  local f="$1"
  local runpath rpath new_path

  runpath="$(extract_runpath "$f")"
  rpath="$(extract_rpath "$f")"

  if [[ -n "$runpath" ]]; then
    new_path="$(append_origin_suffix_if_needed "$runpath")"
    if [[ "$new_path" != "$runpath" ]]; then
      echo "set-runpath: $f"
      echo "  old RUNPATH: ${runpath}"
      echo "  new RUNPATH: ${new_path}"
      patchelf --set-rpath "$new_path" "$f"
    fi
    return 0
  fi

  if [[ -n "$rpath" ]]; then
    new_path="$(append_origin_suffix_if_needed "$rpath")"
    if [[ "$new_path" != "$rpath" ]]; then
      echo "set-rpath: $f"
      echo "  old RPATH: ${rpath}"
      echo "  new RPATH: ${new_path}"
      patchelf --set-rpath "$new_path" "$f"
    fi
  fi
}

patch_one() {
  local f="$1"
  local needed old base
  while IFS= read -r needed; do
    [[ -n "$needed" ]] || continue
    if [[ "$needed" == "$abs_prefix"* ]]; then
      old="$needed"
      if [[ "$old" == "$abs_sqlite" || "$old" == */libsqlite3.so ]]; then
        base="libsqlite3.so.0"
      else
        base="$(basename "$old")"
      fi
      if [[ "$old" != "$base" ]]; then
        echo "replace-needed: $f"
        echo "  $old -> $base"
        patchelf --replace-needed "$old" "$base" "$f"
      fi
    fi
  done < <(extract_needed "$f")

  update_runtime_path_if_needed "$f"
}

ensure_sqlite_aliases
need_sqlite3_so3=0
if bundle_needs_sqlite3_so3; then
  need_sqlite3_so3=1
fi

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

if [[ $need_sqlite3_so3 -eq 1 ]]; then
  latest_sqlite="$(
    ls -1 "$internal_dir"/libsqlite3.so.* 2>/dev/null \
      | sed 's#^.*/##' \
      | sort -V \
      | tail -n 1 || true
  )"
  if [[ -n "$latest_sqlite" ]]; then
    (
      cd "$internal_dir"
      ln -sfn "$latest_sqlite" libsqlite3.so.3
      echo "Created sqlite compatibility alias: libsqlite3.so.3 -> $latest_sqlite"
    )
  fi
else
  echo "No DT_NEEDED requires libsqlite3.so.3; not creating that alias."
fi

echo "Patched ELF file count: $patched_count"
