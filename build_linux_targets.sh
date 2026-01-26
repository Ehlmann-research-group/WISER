#!/usr/bin/env bash
set -euxo pipefail

# ============================================================
# Valid arguments (exactly one required):
#   ubuntu2004
#   debian11
#   fedora39
# Architecture argument (exactly one required):
#   amd  -> amd64
#   arm  -> arm64
#
# Example:
#   ./script.sh fedora39 amd
# ============================================================

BAKE_FILE="install-linux/docker_bake.hcl"

# ----- Argument parsing -----
if [[ $# -ne 2 ]]; then
  echo "ERROR: Expected 2 arguments: <distro> <arch>"
  echo "Arch must be one of: amd | arm"
  exit 1
fi

DISTRO="$1"

case "$DISTRO" in
  ubuntu2004|debian11|fedora39)
    ;;
  *)
    echo "ERROR: Invalid distro: $DISTRO"
    echo "Valid values:"
    echo "  ubuntu2004"
    echo "  debian11"
    echo "  fedora39"
    exit 1
    ;;
esac

# ----- Populate targets -----
PHASEA_TARGETS=("phasea_${DISTRO}")
PHASEB_TARGETS=("phaseb_${DISTRO}")
PHASEC_TARGETS=("phasec_${DISTRO}")

# Debug / sanity check
echo "PHASEA_TARGETS: ${PHASEA_TARGETS[*]}"
echo "PHASEB_TARGETS: ${PHASEB_TARGETS[*]}"
echo "PHASEC_TARGETS: ${PHASEC_TARGETS[*]}"

ARCH_INPUT="$2"

case "$ARCH_INPUT" in
  amd)
    target_arch="amd64"
    ;;
  arm)
    target_arch="arm64"
    ;;
  *)
    echo "ERROR: Invalid arch: $ARCH_INPUT"
    echo "Valid values: amd | arm"
    exit 1
    ;;
esac

echo "Using target architecture: ${target_arch}"

# -------------------------------------------------------------------
# Build Phase A
# -------------------------------------------------------------------
echo "=== BUILDING PHASE A ==="
for tgt in "${PHASEA_TARGETS[@]}"; do
  echo "→ Building ${tgt}"
  docker buildx bake -f "${BAKE_FILE}" \
    --set "${tgt}.platform=linux/${target_arch}" \
    --set '*.output=type=docker' \
    --set '*.cache-from=type=gha' \
    --set '*.cache-to=type=gha,mode=max' \
    --progress=plain \
    "${tgt}"
done

# -------------------------------------------------------------------
# Build Phase B
# -------------------------------------------------------------------
echo "=== BUILDING PHASE B ==="
for tgt in "${PHASEB_TARGETS[@]}"; do
  echo "→ Building ${tgt}"
  docker buildx bake -f "${BAKE_FILE}" \
    --set "${tgt}.platform=linux/${target_arch}" \
    --set '*.output=type=docker' \
    --set '*.cache-from=type=gha' \
    --set '*.cache-to=type=gha,mode=max' \
    --progress=plain \
    "${tgt}"
done

# -------------------------------------------------------------------
# Build Phase C
# -------------------------------------------------------------------
echo "=== BUILDING PHASE C ==="
for tgt in "${PHASEC_TARGETS[@]}"; do
  echo "→ Building ${tgt}"
  docker buildx bake -f "${BAKE_FILE}" \
    --set "${tgt}.platform=linux/${target_arch}" \
    --set '*.output=type=docker' \
    --set '*.cache-from=type=gha' \
    --set '*.cache-to=type=gha,mode=max' \
    --progress=plain \
    "${tgt}"
done

# -------------------------------------------------------------------
# Extract Phase B tarballs
# -------------------------------------------------------------------
echo "=== EXTRACTING LINUX BUILD OUTPUTS ==="

OUTPUT_ROOT="linux_build_output"
mkdir -p "${OUTPUT_ROOT}"

# Extract Phase B tarballs for BOTH amd64 + arm64
for tgt in "${PHASEB_TARGETS[@]}"; do
  distro_key="${tgt#phaseb_}"   # ubuntu2004 / debian11 / fedora39

  # This is the Phase B image tag you built in bake
  img="wiser-phaseb:${distro_key/ubuntu2004/ubuntu20.04}"

  # Pretty name prefix for output folders
  base_name="$(
    echo "${tgt}" \
      | sed 's/^phaseb_//' \
      | sed 's/ubuntu2004/ubuntu_2004/' \
      | sed 's/debian11/debian_11/' \
      | sed 's/fedora39/fedora_39/'
  )"

  echo "Extracting from image: ${img}"

  build_name="${base_name}_${target_arch}"
  out_dir="${OUTPUT_ROOT}/${build_name}"
  mkdir -p "${out_dir}"

  cid="$(docker create --platform "linux/${target_arch}" "${img}")" \
    || { echo "ERROR: could not create container for ${img} (linux/${target_arch})"; exit 1; }

  docker cp "${cid}:/out/WISER.tar.gz" "/tmp/${build_name}.tar.gz" \
    || { echo "ERROR: /out/WISER.tar.gz not found in ${img} (linux/${target_arch})"; docker rm "${cid}" >/dev/null; exit 1; }

  docker rm "${cid}" >/dev/null

  cp "/tmp/${build_name}.tar.gz" "${out_dir}/WISER.tar.gz"
  rm "/tmp/${build_name}.tar.gz"

  echo "✓ ${build_name}"
done

echo "All done building and extracting linux targets"
