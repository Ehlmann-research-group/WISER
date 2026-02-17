# syntax=docker/dockerfile:1

# ============================================================
# Multi-stage Fedora build:
#   phasea = base OS + deps
#   phaseb = build WISER + produce /out/WISER.tar.gz
#   phasec = smoke test using Xvfb + run_smoke.sh + package
#           app image, install script, and directions 
#
# Build examples:
#   # Build artifact stage (for docker cp):
#   docker buildx build --platform linux/amd64 -f install-linux/multistage_fedora/Dockerfile \
#     --target phaseb -t wiser-phaseb:fedora39 --load .
#
#   # Build smoke stage (runnable image):
#   docker buildx build --platform linux/amd64 -f install-linux/multistage_fedora/Dockerfile \
#     --target phasec -t wiser-phasec:fedora39 --load .
#
# ============================================================

ARG BASE_IMAGE=fedora:39

# -----------------------------
# Phase A (Fedora): deps
# -----------------------------
FROM ${BASE_IMAGE} AS phasea
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN dnf -y update \
 && dnf -y install \
    ca-certificates curl wget git \
    gcc gcc-c++ make \
    pkgconf-pkg-config \
    patchelf file binutils \
    mesa-libEGL mesa-libGL mesa-dri-drivers \
    libglvnd-glx \
    glib2 dbus-libs \
    libxkbcommon-x11 \
    libxcb libxcb-devel \
    xcb-util xcb-util-image xcb-util-keysyms xcb-util-wm xcb-util-renderutil \
    libXrender libXext libXfixes libXinerama libXrandr \
    fontconfig \
    strace pax-utils \
 && dnf -y clean all \
 && rm -rf /var/cache/dnf

WORKDIR /work
CMD ["bash"]

# -----------------------------
# Phase B (Fedora): build + bundle fixups + /out/WISER.tar.gz
# -----------------------------
FROM phasea AS phaseb
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# 1) Copy lock + spec + repo
WORKDIR /work
COPY etc/prod-conda-lock.yml /work/prod-conda-lock.yml
COPY WISER-ubuntu.spec /work/WISER-ubuntu.spec
COPY . /work

RUN [ -d /work/dist ]  && rm -rf /work/dist  || true \
 && [ -d /work/build ] && rm -rf /work/build || true

# 2) Install micromamba (arch-aware)
ARG TARGETARCH
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN set -eux; \
    case "$TARGETARCH" in \
      amd64)  mm_plat="linux-64" ;; \
      arm64)  mm_plat="linux-aarch64" ;; \
      *) echo "Unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    dnf -y update; \
    dnf -y install ca-certificates curl tar bzip2 findutils file make; \
    dnf -y clean all; rm -rf /var/cache/dnf; \
    mkdir -p "$MAMBA_ROOT_PREFIX"; \
    curl -Ls "https://micro.mamba.pm/api/micromamba/${mm_plat}/latest" \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba; \
    micromamba --version

# 3) Create env
ARG ENV_NAME=wiser-prod
ENV WISER_ENV=prod
RUN micromamba create -y -n "${ENV_NAME}" -f /work/prod-conda-lock.yml \
 && micromamba clean -a -y

# 4) Build (your Fedora version with LD_LIBRARY_PATH)
RUN micromamba run -n "${ENV_NAME}" bash -lc '\
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; \
  cd /work; \
  make build-linux \
'

# Copy specific libs from conda env into _internal (Fedora-specific)
RUN set -euxo pipefail; \
  internal_dir="$(find /work/dist -type d -name _internal -print -quit)"; \
  test -n "$internal_dir"; \
  env_lib="/opt/micromamba/envs/${ENV_NAME}/lib"; \
  for pattern in "libcurl.so.*" "libssl.so.*" "libcrypto.so.*"; do \
    for src in $env_lib/$pattern; do \
      [ -e "$src" ] || continue; \
      cp -av "$src" "$internal_dir/"; \
    done; \
  done

RUN set -eux; \
  internal_dir="$(find /work/dist -type d -name _internal -print -quit)"; \
  objdump -T "$internal_dir"/libcurl.so.* 2>/dev/null | grep -F OPENSSL_ | sort -u | head -n 10 || true; \
  readelf --version-info "$internal_dir"/libssl.so.* 2>/dev/null | grep -E "Name: OPENSSL_" | sort -u || true

# Bundle ELF dependency scrub/verify:
# Conda absolute paths can leak into DT_NEEDED/RUNPATH (notably GDAL -> sqlite),
# and some bundled libs carry hard-coded build paths in RPATH/RUNPATH.
# We do not force-convert tags; we append $ORIGIN when those build paths appear.
RUN set -euxo pipefail; \
    internal_dir="$(find /work/dist -type d -name _internal -print -quit)"; \
    test -n "$internal_dir"; \
    sed -i 's/\r$//' /work/install-linux/scripts/patch_internal_elf_deps.sh /work/install-linux/scripts/verify_internal_elf_deps.sh; \
    bash /work/install-linux/scripts/patch_internal_elf_deps.sh "$internal_dir" "${ENV_NAME}"; \
    bash /work/install-linux/scripts/verify_internal_elf_deps.sh "$internal_dir"

# Exportable output + tarball
RUN set -euxo pipefail; \
    mkdir -p /out; \
    if [ -d /work/dist ]; then cp -a /work/dist/. /out/; fi; \
    if [ -d /work/build ]; then cp -a /work/build/. /out/build/; fi; \
    \
    # --- NEW: include icon in the artifact so smoke stage can build AppDir ---
    if [ -f /work/icons/wiser.iconset/icon_256x256.png ]; then \
      cp -a /work/icons/wiser.iconset/icon_256x256.png /out/wiser.png; \
    else \
      echo "WARNING: icon not found at /work/icons/wiser.iconset/icon_256x256.png; AppImage will be built without icon"; \
    fi; \
    \
    ls -lah /out; \
    cd /out; \
    tar -C /out -czf /tmp/WISER.tar.gz .; \
    mv /tmp/WISER.tar.gz /out/WISER.tar.gz; \
    ls -lah /out/WISER.tar.gz

CMD ["bash"]

# -----------------------------
# Phase C (Fedora): smoke test image
#   - consumes /out/WISER.tar.gz from phaseb
#   - builds WISER AppImage (arch-aware) + adds installer + SHA256SUMS + RUN.txt
#   - repackages to /app/WISER.tar.gz
#   - then runs run_smoke.sh
# -----------------------------
FROM phasea AS phasec
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ARG TARGETARCH

# Install Xvfb (Fedora)
RUN dnf -y update \
 && dnf -y install xorg-x11-server-Xvfb \
 && dnf -y clean all \
 && rm -rf /var/cache/dnf

# Copy the artifact from Phase B into this image
WORKDIR /app
COPY --from=phaseb /out/ /app/

# Writable HOME/config area
ENV HOME=/tmp/home
ENV XDG_CONFIG_HOME=/tmp/xdg-config
RUN mkdir -p "$HOME" "$XDG_CONFIG_HOME" && chmod -R 777 /tmp/home /tmp/xdg-config

# AppImage templates (avoid heredocs inside RUN)
COPY install-linux/appimage/AppRun /tmp/appimage/AppRun
COPY install-linux/appimage/WISER.desktop /tmp/appimage/WISER.desktop
COPY install-linux/appimage/install_wiser_appimage.sh /tmp/appimage/install_wiser_appimage.sh

# ------------------------------------------------------------
# Build AppImage from the PyInstaller artifact tarball, then
# re-pack to /app/WISER.tar.gz (contains AppImage + SHA256SUMS + RUN.txt + installer)
# ------------------------------------------------------------
RUN set -euxo pipefail; \
    \
    # 1) Extract the build artifact tarball to /work
    test -f /app/WISER.tar.gz; \
    rm -rf /work && mkdir -p /work; \
    tar -xzf /app/WISER.tar.gz -C /work; \
    \
    # Expect one-folder pyinstaller layout inside extracted artifact:
    test -d /work/WISER; \
    test -f /work/WISER/WISER_Bin; \
    test -d /work/WISER/_internal; \
    \
    # 2) Pick appimagetool URL + AppImage arch label from TARGETARCH
    case "${TARGETARCH}" in \
      amd64)  appimage_arch="x86_64";  tool_url="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" ;; \
      arm64)  appimage_arch="aarch64"; tool_url="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-aarch64.AppImage" ;; \
      *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    \
    # 3) Download appimagetool + chmod
    curl -L --fail -o /tmp/appimagetool.AppImage "${tool_url}"; \
    chmod +x /tmp/appimagetool.AppImage; \
    \
    # 4) Build WISER.AppDir structure
    rm -rf /work/WISER.AppDir; \
    mkdir -p /work/WISER.AppDir/usr/bin; \
    \
    # Copy PyInstaller bundle into AppDir payload dir (WISER_Bin + _internal)
    cp -a /work/WISER/. /work/WISER.AppDir/usr/bin/; \
    \
    # Install AppRun + desktop file (and make AppRun executable)
    install -m 0755 /tmp/appimage/AppRun /work/WISER.AppDir/AppRun; \
    install -m 0644 /tmp/appimage/WISER.desktop /work/WISER.AppDir/WISER.desktop; \
    sed -i 's/\r$//' /work/WISER.AppDir/AppRun; \
    sed -i 's/\r$//' /work/WISER.AppDir/WISER.desktop; \
    \
    # Icon (optional): phaseb should have copied this into the artifact as /out/wiser.png
    if [ -f /work/wiser.png ]; then \
      install -m 0644 /work/wiser.png /work/WISER.AppDir/wiser.png; \
    else \
      echo "WARNING: /work/wiser.png not found; continuing without icon"; \
    fi; \
    \
    # 5) Run appimagetool -> AppImage (with extraction fallback)
    out_appimage="/work/WISER-${appimage_arch}.AppImage"; \
    if ARCH="${appimage_arch}" /tmp/appimagetool.AppImage /work/WISER.AppDir "${out_appimage}"; then \
      echo "appimagetool ran directly."; \
    else \
      echo "Direct appimagetool execution failed; falling back to --appimage-extract."; \
      rm -rf /tmp/squashfs-root; \
      (cd /tmp && /tmp/appimagetool.AppImage --appimage-extract); \
      ARCH="${appimage_arch}" /tmp/squashfs-root/AppRun /work/WISER.AppDir "${out_appimage}"; \
    fi; \
    test -f "${out_appimage}"; \
    \
    # 6) Release bundle: AppImage + installer + SHA256SUMS + RUN.txt
    rm -rf /work/release && mkdir -p /work/release; \
    mv "${out_appimage}" "/work/release/WISER-${appimage_arch}.AppImage"; \
    install -m 0755 /tmp/appimage/install_wiser_appimage.sh /work/release/install_wiser_appimage.sh; \
    sed -i 's/\r$//' /work/release/install_wiser_appimage.sh; \
    (cd /work/release && sha256sum "WISER-${appimage_arch}.AppImage" > SHA256SUMS); \
    printf '%s\n' \
      "WISER AppImage (${appimage_arch})" \
      "" \
      "Option A: Run portable (no installation)" \
      "  chmod +x WISER-${appimage_arch}.AppImage" \
      "  ./WISER-${appimage_arch}.AppImage" \
      "" \
      "Option B: Install desktop launcher (per-user, no sudo, recommended)" \
      "  bash install_wiser_appimage.sh WISER-${appimage_arch}.AppImage" \
      "" \
      "Uninstall:" \
      "  - Portable: delete the AppImage file" \
      "  - Desktop install: delete ~/Applications/WISER.AppImage and ~/.local/share/applications/wiser.desktop" \
      > /work/release/RUN.txt; \
    \
    # 7) Re-tar to the required output path
    tar -C /work/release -czf /tmp/WISER.tar.gz .; \
    mv /tmp/WISER.tar.gz /app/WISER.tar.gz; \
    ls -lah /app/WISER.tar.gz

# Smoke test script
COPY install-linux/run_smoke.sh /usr/local/bin/run_smoke.sh
RUN chmod +x /usr/local/bin/run_smoke.sh

ENTRYPOINT ["/usr/local/bin/run_smoke.sh"]
