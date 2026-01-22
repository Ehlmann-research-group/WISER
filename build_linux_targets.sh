#!/usr/bin/env bash
set -euxo pipefail

BAKE_FILE="install-linux/docker_bake.hcl"

# Phase A targets
PHASEA_TARGETS=(
  # "phasea_ubuntu2004"
  # "phasea_debian11"
  "phasea_fedora39"
)

# Phase B targets
PHASEB_TARGETS=(
  # "phaseb_ubuntu2004"
  # "phaseb_debian11"
  "phaseb_fedora39"
)

# Phase C targets
PHASEC_TARGETS=(
  # "phasec_ubuntu2004"
  # "phasec_debian11"
  "phasec_fedora39"
)

# -------------------------------------------------------------------
# Build Phase A
# -------------------------------------------------------------------
echo "=== BUILDING PHASE A ==="
for tgt in "${PHASEA_TARGETS[@]}"; do
  echo "→ Building ${tgt}"
  docker buildx bake -f "${BAKE_FILE}" \
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

  for arch in amd64 arm64; do
    build_name="${base_name}_${arch}"
    out_dir="${OUTPUT_ROOT}/${build_name}"
    mkdir -p "${out_dir}"

    # Force which platform variant to use
    cid="$(docker create --platform "linux/${arch}" "${img}")" \
      || { echo "ERROR: could not create container for ${img} (linux/${arch})"; exit 1; }

    docker cp "${cid}:/out/WISER.tar.gz" "/tmp/${build_name}.tar.gz" \
      || { echo "ERROR: /out/WISER.tar.gz not found in ${img} (linux/${arch})"; docker rm "${cid}" >/dev/null; exit 1; }

    docker rm "${cid}" >/dev/null

    cp "/tmp/${build_name}.tar.gz" "${out_dir}/WISER.tar.gz"
    rm "/tmp/${build_name}.tar.gz"

    echo "✓ ${build_name}"
  done
done

echo "All done building and extracting linux targets"
