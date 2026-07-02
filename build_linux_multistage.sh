#!/usr/bin/env bash
set -euxo pipefail

# ============================================================
# Valid args:
#   distro: ubuntu2004 | debian11 | fedora39
#   arch:   amd -> amd64 | arm -> arm64
#
# Example:
#   ./build_multistage_linux.sh ubuntu2004 amd
# ============================================================

BAKE_FILE="install-linux/docker_bake.hcl"

if [[ $# -ne 2 ]]; then
  echo "ERROR: Expected 2 arguments: <distro> <arch>"
  echo "  distro: ubuntu2004 | debian11 | fedora39"
  echo "  arch:   amd | arm"
  exit 1
fi

DISTRO="$1"
ARCH_INPUT="$2"

case "$ARCH_INPUT" in
  amd) target_arch="amd64" ;;
  arm) target_arch="arm64" ;;
  *) echo "ERROR: Invalid arch: $ARCH_INPUT (use amd|arm)"; exit 1 ;;
esac

# Map user distro -> bake target + image tag + pretty output prefix
case "$DISTRO" in
  ubuntu2004)
    tgt="multistage_ubuntu"
    img="wiser-multistage:ubuntu20.04"
    base_name="ubuntu_2004"
    ;;
  debian11)
    tgt="multistage_debian"
    img="wiser-multistage:debian11"
    base_name="debian_11"
    ;;
  fedora39)
    tgt="multistage_fedora"
    img="wiser-multistage:fedora39"
    base_name="fedora_39"
    ;;
  *)
    echo "ERROR: Invalid distro: $DISTRO"
    echo "Valid: ubuntu2004 | debian11 | fedora39"
    exit 1
    ;;
esac

echo "=== BUILDING MULTISTAGE ==="
echo "→ Target: ${tgt}"
echo "→ Platform: linux/${target_arch}"
echo "→ Image tag: ${img}"

# Give each distro/arch its own cache scope so the six matrix legs don't evict
# each other's layers out of the shared, space-limited GHA cache.
scope="${base_name}_${target_arch}"

# Base-image pulls and the GHA cache are the flakiest part of the build, so retry
# the bake a few times instead of forcing a manual re-run of the whole job.
bake() {
  docker buildx bake -f "${BAKE_FILE}" \
    --set "${tgt}.platform=linux/${target_arch}" \
    --set "${tgt}.output=type=docker" \
    --set "*.cache-from=type=gha,scope=${scope}" \
    --set "*.cache-to=type=gha,mode=max,scope=${scope}" \
    --progress=plain \
    "${tgt}"
}

attempt=1
max_attempts=3
until bake; do
  if (( attempt >= max_attempts )); then
    echo "ERROR: 'docker buildx bake' failed after ${max_attempts} attempts"
    exit 1
  fi
  echo "buildx bake failed (attempt ${attempt}/${max_attempts}); retrying in 30s..."
  sleep 30
  attempt=$(( attempt + 1 ))
done

echo "=== SMOKE TEST (PHASE C via docker run) ==="
docker run --rm --platform "linux/${target_arch}" "${img}"

echo "=== EXTRACTING ARTIFACT ==="
OUTPUT_ROOT="linux_build_output"
mkdir -p "${OUTPUT_ROOT}"

build_name="${base_name}_${target_arch}"
out_dir="${OUTPUT_ROOT}/${build_name}"
mkdir -p "${out_dir}"

tmp_dir="${TMPDIR:-/tmp}"
tmp_tar="${tmp_dir}/${build_name}.tar.gz"

cid="$(docker create --platform "linux/${target_arch}" "${img}")" \
  || { echo "ERROR: could not create container for ${img} (linux/${target_arch})"; exit 1; }

docker cp "${cid}:/app/WISER.tar.gz" "${tmp_tar}" \
  || { echo "ERROR: /app/WISER.tar.gz not found in ${img} (linux/${target_arch})"; docker rm "${cid}" >/dev/null; exit 1; }

docker rm "${cid}" >/dev/null

cp "${tmp_tar}" "${out_dir}/WISER.tar.gz"
rm "${tmp_tar}"

echo "✓ ${build_name}"
echo "All done"
