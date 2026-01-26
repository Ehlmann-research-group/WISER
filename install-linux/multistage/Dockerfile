# syntax=docker/dockerfile:1

# ============================================================
# Combined Phase A + B + C (multi-stage)
#
# Build args:
#   BASE_IMAGE   e.g. ubuntu:20.04  or  debian:11
#   ENV_NAME     e.g. wiser-prod
# ============================================================

ARG BASE_IMAGE=ubuntu:20.04
ARG ENV_NAME=wiser-prod

# ----------------------------
# Phase A (base)
# ----------------------------
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    build-essential pkg-config \
    patchelf file \
    # Qt/OpenGL/EGL runtime deps (PySide2 common deps)
    libegl1 libgl1 \
    libgl1-mesa-dri \
    libglx0 \
    libglib2.0-0 libdbus-1-3 \
    libxkbcommon-x11-0 \
    libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
    libfontconfig1 \
    strace pax-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# ----------------------------
# Phase B (build + bundle fixups + tarball)
# ----------------------------
FROM base AS build

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Copy repo + lock + spec
WORKDIR /work
COPY etc/prod-conda-lock.yml /work/prod-conda-lock.yml
COPY WISER-ubuntu.spec /work/WISER-ubuntu.spec
COPY . /work

RUN [ -d /work/dist ]  && rm -rf /work/dist  || true \
 && [ -d /work/build ] && rm -rf /work/build || true

# Micromamba install (arch-aware)
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ARG TARGETARCH
RUN set -eux; \
    case "${TARGETARCH}" in \
      amd64)  mm_arch="linux-64" ;; \
      arm64)  mm_arch="linux-aarch64" ;; \
      *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl tar bzip2 make; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p "${MAMBA_ROOT_PREFIX}"; \
    curl -Ls "https://micro.mamba.pm/api/micromamba/${mm_arch}/latest" \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba; \
    micromamba --version

# Create env
ARG ENV_NAME
ENV WISER_ENV=prod
RUN micromamba create -y -n "${ENV_NAME}" -f /work/prod-conda-lock.yml \
 && micromamba clean -a -y

# Run build (avoid bash -l; keep env PATH intact)
RUN micromamba run -n "${ENV_NAME}" make -C /work build-linux

# ---- Bundle fixups ----
RUN set -euxo pipefail; \
    internal_dir="$(find /work/dist -type d -name _internal -print -quit 2>/dev/null || true)"; \
    if [ -z "$internal_dir" ]; then \
      echo "No _internal directory found under /work/dist; skipping fixups."; \
      exit 0; \
    fi; \
    absolute_internal_dir="$(cd "$internal_dir" && pwd)"; \
    cd "$absolute_internal_dir"; \
    pick_sqlite_target() { ls -1 libsqlite3.so.* 2>/dev/null | sort -V | tail -n 1; }; \
    \
    echo "== Step 1: sqlite SONAME/aliases =="; \
    if ls -1 libsqlite3.so.* >/dev/null 2>&1 && [ ! -e libsqlite3.so ]; then \
      target="$(pick_sqlite_target)"; \
      ln -sf "$target" libsqlite3.so; \
    fi; \
    if [ -e libsqlite3.so ] && [ ! -e libsqlite3.so.3 ]; then \
      ln -sf libsqlite3.so libsqlite3.so.3; \
    fi; \
    if ls -1 libsqlite3.so.* >/dev/null 2>&1 && [ ! -e libsqlite3.so.0 ]; then \
      target="$(pick_sqlite_target)"; \
      ln -sf "$target" libsqlite3.so.0; \
    fi; \
    \
    echo "== Step 2: GDAL RPATH to \$ORIGIN =="; \
    found_any=0; \
    while IFS= read -r -d '' f; do \
      found_any=1; \
      patchelf --force-rpath --set-rpath '$ORIGIN' "$f"; \
    done < <(find "$absolute_internal_dir" -type f \( -iname "*gdal*.so*" -o -iname "libgdal.so*" \) -print0); \
    if [ "$found_any" -eq 0 ]; then echo "No GDAL libs found; skipping."; fi; \
    \
    echo "== Step 3: replace-needed (only if absolute DT_NEEDED present) =="; \
    gdal_so="$(find "$absolute_internal_dir" -maxdepth 1 -type f -name 'libgdal.so*' -print -quit 2>/dev/null || true)"; \
    if [ -z "$gdal_so" ]; then echo "No libgdal.so* found; skipping."; exit 0; fi; \
    ABS_SQLITE="/opt/micromamba/envs/${ENV_NAME}/lib/libsqlite3.so"; \
    if readelf -d "$gdal_so" | grep -Fq "$ABS_SQLITE"; then \
      patchelf --replace-needed "$ABS_SQLITE" "libsqlite3.so.0" "$gdal_so"; \
    fi; \
    readelf -d "$gdal_so" | grep -i sqlite || true

# Exportable output + tarball
RUN set -euxo pipefail; \
    mkdir -p /out; \
    if [ -d /work/dist ]; then cp -a /work/dist/. /out/; fi; \
    if [ -d /work/build ]; then cp -a /work/build/. /out/build/; fi; \
    ls -lah /out; \
    cd /out; \
    tar -C /out -czf /tmp/WISER.tar.gz .; \
    mv /tmp/WISER.tar.gz /out/WISER.tar.gz; \
    ls -lah /out/WISER.tar.gz

# ----------------------------
# Export-only stage (for copying to host via --output type=local)
# ----------------------------
FROM scratch AS artifact
COPY --from=build /out/WISER.tar.gz /WISER.tar.gz

# ----------------------------
# Phase C (smoke test image that CONSUMES the artifact)
# ----------------------------
FROM base AS smoke

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends xvfb \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/ /app/

ENV HOME=/tmp/home
ENV XDG_CONFIG_HOME=/tmp/xdg-config
RUN mkdir -p "$HOME" "$XDG_CONFIG_HOME" && chmod -R 777 /tmp/home /tmp/xdg-config

COPY install-linux/run_smoke.sh /usr/local/bin/run_smoke.sh
RUN chmod +x /usr/local/bin/run_smoke.sh

ENTRYPOINT ["/usr/local/bin/run_smoke.sh"]
