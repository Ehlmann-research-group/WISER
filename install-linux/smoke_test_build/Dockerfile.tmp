# syntax=docker/dockerfile:1

# Phase C: smoke test the PyInstaller output from Phase B

ARG PHASEB_IMAGE=wiser-phaseb:ubuntu20.04-amd64
ARG PHASEA_IMAGE=wiser-phasea:ubuntu20.04-amd64

FROM ${PHASEB_IMAGE} AS artifact
FROM ${PHASEA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Install xvfb (NO sudo; container runs as root unless USER is set)
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy the artifact from Phase B into this image
WORKDIR /app
COPY --from=artifact /out/ /app/

# Writable HOME/config area (your app writes config)
ENV HOME=/tmp/home
ENV XDG_CONFIG_HOME=/tmp/xdg-config
RUN mkdir -p "$HOME" "$XDG_CONFIG_HOME" && chmod -R 777 /tmp/home /tmp/xdg-config

# Smoke test script (must exist in build context)
COPY install-linux/run_smoke.sh /usr/local/bin/run_smoke.sh
RUN chmod +x /usr/local/bin/run_smoke.sh

ENTRYPOINT ["/usr/local/bin/run_smoke.sh"]
