# syntax=docker/dockerfile:1

# Phase A (Fedora): base OS + common build/runtime deps for PySide2/Qt
# Build with: docker build --platform=linux/amd64 -t wiser-phasea:fedora39-amd64 -f docker/phaseA/Dockerfile .

ARG BASE_IMAGE=fedora:39
FROM ${BASE_IMAGE}

# Make shell strict for RUN lines
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

# Core tooling + common native build prereqs + common Qt runtime libs that often bite PySide2 apps
RUN dnf -y update \
 && dnf -y install \
    ca-certificates curl wget git \
    # build tools (rough equivalent of build-essential)
    gcc gcc-c++ make \
    pkgconf-pkg-config \
    patchelf file \
    # Qt/OpenGL/EGL runtime deps (Fedora names)
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

# Default command (can be overridden)
CMD ["bash"]
