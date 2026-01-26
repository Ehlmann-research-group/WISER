# syntax=docker/dockerfile:1

# Phase B (Fedora): build WISER + run PyInstaller, output to /out

ARG PHASEA_IMAGE=wiser-phasea:fedora39
FROM ${PHASEA_IMAGE}

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# 1) Copy lock + spec into image
WORKDIR /work
COPY etc/prod-conda-lock.yml /work/prod-conda-lock.yml
COPY WISER-ubuntu.spec /work/WISER-ubuntu.spec
COPY . /work

RUN [ -d /work/dist ]  && rm -rf /work/dist  || true \
 && [ -d /work/build ] && rm -rf /work/build || true

# 2) Install micromamba (Docker-safe install: put binary on PATH)
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

# 3) Create env from conda-lock file
ARG ENV_NAME=wiser-prod
ENV WISER_ENV=prod
RUN micromamba create -y -n "${ENV_NAME}" -f /work/prod-conda-lock.yml \
 && micromamba clean -a -y

# 4) Run PyInstaller using the spec file
RUN micromamba run -n "${ENV_NAME}" bash -lc '\
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; \
  cd /work; \
  make build-linux \
'

RUN set -euxo pipefail; \
  internal_dir="$(find /work/dist -type d -name _internal -print -quit)"; \
  test -n "$internal_dir"; \
  env_lib="/opt/micromamba/envs/${ENV_NAME}/lib"; \
  for pattern in 'libcurl.so.*' 'libssl.so.*' 'libcrypto.so.*'; do \
    for src in $env_lib/$pattern; do \
      [ -e "$src" ] || continue; \
      cp -av "$src" "$internal_dir/"; \
    done; \
  done

RUN set -eux; \
  internal_dir="$(find /work/dist -type d -name _internal -print -quit)"; \
  objdump -T "$internal_dir"/libcurl.so.* 2>/dev/null | grep -F OPENSSL_ | sort -u | head -n 10 || true; \
  readelf --version-info "$internal_dir"/libssl.so.* 2>/dev/null | grep -E 'Name: OPENSSL_' | sort -u || true

# -------------------------------------------------------------------
# Bundle fixups: sqlite aliases + GDAL RPATH + (optional) replace-needed
# These are written to be robust across distros:
#  - only patch if files exist
#  - only replace-needed if the absolute DT_NEEDED actually appears
# -------------------------------------------------------------------

# Ensure sqlite SONAME/aliases exist in the bundle (_internal)
RUN set -euxo pipefail; \
    internal_dir="$(find /work/dist -type d -name _internal -print -quit 2>/dev/null || true)"; \
    if [ -z "$internal_dir" ]; then \
      echo "No _internal directory found; skipping sqlite alias step."; \
      exit 0; \
    fi; \
    absolute_internal_dir="$(cd "$internal_dir" && pwd)"; \
    cd "$absolute_internal_dir"; \
    pick_sqlite_target() { \
      ls -1 libsqlite3.so.* 2>/dev/null | sort -V | tail -n 1; \
    }; \
    if ls -1 libsqlite3.so.* >/dev/null 2>&1 && [ ! -e libsqlite3.so ]; then \
      target="$(pick_sqlite_target)"; \
      echo "Creating symlink: libsqlite3.so -> $target"; \
      ln -sf "$target" libsqlite3.so; \
    fi; \
    if [ -e libsqlite3.so ] && [ ! -e libsqlite3.so.3 ]; then \
      echo "Creating symlink: libsqlite3.so.3 -> libsqlite3.so"; \
      ln -sf libsqlite3.so libsqlite3.so.3; \
    fi; \
    if ls -1 libsqlite3.so.* >/dev/null 2>&1 && [ ! -e libsqlite3.so.0 ]; then \
      target="$(pick_sqlite_target)"; \
      echo "Creating symlink: libsqlite3.so.0 -> $target"; \
      ln -sf "$target" libsqlite3.so.0; \
    fi; \
    echo "SQLite files now:"; \
    ls -lah libsqlite3.so* 2>/dev/null || true

# Patch GDAL extension + libgdal to look in PyInstaller _internal first
RUN set -euxo pipefail; \
    internal_dir="$(find /work/dist -type d -name _internal -print -quit 2>/dev/null || true)"; \
    if [ -z "$internal_dir" ]; then \
      echo "No _internal directory found; skipping GDAL rpath step."; \
      exit 0; \
    fi; \
    absolute_internal_dir="$(cd "$internal_dir" && pwd)"; \
    found_any=0; \
    while IFS= read -r -d '' f; do \
      found_any=1; \
      echo "patchelf --set-rpath '\$ORIGIN' -> $f"; \
      patchelf --force-rpath --set-rpath '$ORIGIN' "$f"; \
    done < <(find "$absolute_internal_dir" -type f \( -iname "*gdal*.so*" -o -iname "libgdal.so*" \) -print0); \
    if [ "$found_any" -eq 0 ]; then \
      echo "No GDAL libs found in _internal; skipping rpath step."; \
    fi

# Patch libgdal.so.* to stop requiring sqlite via an absolute conda path (only if present)
RUN set -euxo pipefail; \
    internal_dir="$(find /work/dist -type d -name _internal -print -quit 2>/dev/null || true)"; \
    if [ -z "$internal_dir" ]; then \
      echo "No _internal directory found; skipping replace-needed step."; \
      exit 0; \
    fi; \
    absolute_internal_dir="$(cd "$internal_dir" && pwd)"; \
    gdal_so="$(find "$absolute_internal_dir" -maxdepth 1 -type f -name 'libgdal.so*' -print -quit 2>/dev/null || true)"; \
    if [ -z "$gdal_so" ]; then \
      echo "No libgdal.so* found; skipping replace-needed step."; \
      exit 0; \
    fi; \
    ABS_SQLITE="/opt/micromamba/envs/${ENV_NAME}/lib/libsqlite3.so"; \
    echo "Checking DT_NEEDED entries for sqlite in: $gdal_so"; \
    echo "Looking for absolute path: $ABS_SQLITE"; \
    if command -v readelf >/dev/null 2>&1 && ( readelf -d "$gdal_so" | grep -Fq "$ABS_SQLITE" ); then \
      echo "Found absolute sqlite DT_NEEDED; rewriting to libsqlite3.so.0"; \
      cd "$absolute_internal_dir"; \
      pick_sqlite_target() { \
        ls -1 libsqlite3.so.* 2>/dev/null | sort -V | tail -n 1; \
      }; \
      if [ ! -e libsqlite3.so.0 ]; then \
        if ls -1 libsqlite3.so.* >/dev/null 2>&1; then \
          target="$(pick_sqlite_target)"; \
          echo "Creating symlink: libsqlite3.so.0 -> $target"; \
          ln -sf "$target" libsqlite3.so.0; \
        else \
          echo "ERROR: No libsqlite3.so.* files found to link from."; \
          exit 1; \
        fi; \
      fi; \
      patchelf --replace-needed "$ABS_SQLITE" "libsqlite3.so.0" "$gdal_so"; \
      echo "After patch, sqlite-related entries:"; \
      readelf -d "$gdal_so" | grep -i sqlite || true; \
    else \
      echo "No absolute sqlite DT_NEEDED found; leaving libgdal as-is."; \
      if command -v readelf >/dev/null 2>&1; then \
        readelf -d "$gdal_so" | grep -i sqlite || true; \
      else \
        echo "readelf not found."; \
      fi; \
    fi


# 5) Put artifacts somewhere easy to export + tarball it (Linux-safe transfer)
RUN set -euxo pipefail; \
    mkdir -p /out; \
    if [ -d /work/dist ]; then cp -a /work/dist/. /out/; fi; \
    if [ -d /work/build ]; then cp -a /work/build/. /out/build/; fi; \
    ls -lah /out; \
    cd /out; \
    tar -C /out -czf /tmp/WISER.tar.gz .; \
    mv /tmp/WISER.tar.gz /out/WISER.tar.gz; \
    ls -lah /out/WISER.tar.gz

CMD ["bash"]
