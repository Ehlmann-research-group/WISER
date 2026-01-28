#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "INFO: $*" >&2
}

usage() {
  cat >&2 <<'EOF'
Usage:
  bash install_wiser_appimage.sh [path-to-AppImage]

If no argument is provided, the script searches the current directory for exactly one
WISER*.AppImage.
EOF
}

# --- 1) Determine input AppImage ---
appimage_input="${1:-}"

if [[ -z "${appimage_input}" ]]; then
  shopt -s nullglob
  matches=( ./WISER*.AppImage )
  shopt -u nullglob
  if [[ "${#matches[@]}" -eq 0 ]]; then
    usage
    die "No argument provided and no WISER*.AppImage found in current directory."
  elif [[ "${#matches[@]}" -gt 1 ]]; then
    usage
    echo "Found multiple matches:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    die "Please pass the desired AppImage path explicitly."
  fi
  appimage_input="${matches[0]}"
fi

# Resolve to an absolute path for reliability
if [[ ! -f "${appimage_input}" ]]; then
  die "AppImage not found: ${appimage_input}"
fi

# --- 2) Install AppImage to ~/Applications/WISER.AppImage (copy, atomic replace) ---
home_dir="${HOME}"
apps_dir="${home_dir}/Applications"
dest_appimage="${apps_dir}/WISER.AppImage"

mkdir -p "${apps_dir}"

tmp_copy="$(mktemp "${apps_dir}/.WISER.AppImage.tmp.XXXXXX")"
# Preserve file content; permissions will be set explicitly after.
cp -f "${appimage_input}" "${tmp_copy}"
chmod +x "${tmp_copy}"
mv -f "${tmp_copy}" "${dest_appimage}"
chmod +x "${dest_appimage}"

# --- 3) Write per-user desktop entry ---
desktop_dir="${home_dir}/.local/share/applications"
desktop_file="${desktop_dir}/wiser.desktop"
mkdir -p "${desktop_dir}"

# Ensure LF line endings by writing with printf.
# Exec must be absolute path; do not leave $HOME unexpanded in the written file.
exec_path="${dest_appimage}"

printf '%s\n' \
  "[Desktop Entry]" \
  "Type=Application" \
  "Name=WISER" \
  "Comment=Workbench for Imaging Spectroscopy Exploration and Research" \
  "Exec=${exec_path}" \
  "Icon=wiser" \
  "Terminal=false" \
  "Categories=Education;Science;" \
  > "${desktop_file}"

# --- 4) Install icon (extract from AppImage if possible) ---
xdg_dir="${XDG_DATA_DIRS}"
icon_dest_dir="${xdg_dir}/icons/hicolor/256x256/apps"
icon_dest="${icon_dest_dir}/wiser.png"
mkdir -p "${icon_dest_dir}"

icon_installed="no"
extract_tmpdir="$(mktemp -d)"
cleanup() {
  # Only delete our controlled temp dir
  rm -rf "${extract_tmpdir}"
}
trap cleanup EXIT

# Try extraction; AppImage extraction writes "squashfs-root" in the current directory.
(
  cd "${extract_tmpdir}"
  # Some AppImages require execute bit to extract; we set it above.
  "${dest_appimage}" --appimage-extract >/dev/null 2>&1 || exit 0
)

squash_dir="${extract_tmpdir}/squashfs-root"
if [[ -d "${squash_dir}" ]]; then
  # Prefer PNGs that look like icons. Heuristic:
  # - Prefer under usr/share/icons/... and higher resolutions (e.g., 512, 256)
  # - Otherwise any .png; choose the largest by file size as a proxy.
  best_png=""

  # 1) Strong preference: hicolor theme PNGs
  mapfile -t themed_pngs < <(
    find "${squash_dir}/usr/share/icons" -type f -name '*.png' 2>/dev/null || true
  )

  if [[ "${#themed_pngs[@]}" -gt 0 ]]; then
    # Prefer largest file size among themed icons
    best_png="$(printf '%s\0' "${themed_pngs[@]}" | xargs -0 ls -S 2>/dev/null | head -n 1 || true)"
  fi

  # 2) Fallback: top-level or anywhere PNGs (including e.g. WISER.png)
  if [[ -z "${best_png}" ]]; then
    mapfile -t all_pngs < <(
      find "${squash_dir}" -type f -name '*.png' 2>/dev/null || true
    )
    if [[ "${#all_pngs[@]}" -gt 0 ]]; then
      best_png="$(printf '%s\0' "${all_pngs[@]}" | xargs -0 ls -S 2>/dev/null | head -n 1 || true)"
    fi
  fi

  if [[ -n "${best_png}" && -f "${best_png}" ]]; then
    cp -f "${best_png}" "${icon_dest}"
    icon_installed="yes"
  else
    info "No PNG icon found inside extracted AppImage; continuing without icon."
  fi
else
  info "AppImage extraction did not produce squashfs-root; continuing without icon."
fi

# --- 5) Refresh desktop/icon caches if tools exist ---
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${desktop_dir}" || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f "${home_dir}/.local/share/icons/hicolor" || true
fi

# --- 6) Final summary ---
echo
echo "WISER AppImage installed to:"
echo "  ${dest_appimage}"
echo "Desktop launcher written to:"
echo "  ${desktop_file}"
if [[ "${icon_installed}" == "yes" ]]; then
  echo "Icon installed to:"
  echo "  ${icon_dest}"
else
  echo "Icon: (not installed) - no suitable PNG found inside the AppImage."
fi
echo
echo "WISER should appear in your desktop app launcher/search soon."
echo "If it does not, try logging out and back in."
