variable "PLATFORMS" {
  default = ["linux/amd64", "linux/arm64"]
}

# -------------------------------------------------------------------
# Multi-stage (A+B+C in one Dockerfile)
#   - Image includes:
#       /app/WISER.tar.gz   (artifact for docker cp)
#       ENTRYPOINT runs smoke test (so "docker run" is Phase C)
# -------------------------------------------------------------------

target "multistage_ubuntu" {
  context    = "."
  dockerfile = "install-linux/multistage/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-multistage:ubuntu20.04"]
  args = {
    BASE_IMAGE = "ubuntu:20.04"
  }
  output = ["type=docker"]
}

target "multistage_debian" {
  context    = "."
  dockerfile = "install-linux/multistage/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-multistage:debian11"]
  args = {
    BASE_IMAGE = "debian:11"
  }
  output = ["type=docker"]
}

target "multistage_fedora" {
  context    = "."
  dockerfile = "install-linux/multistage_fedora/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-multistage:fedora39"]
  args = {
    BASE_IMAGE = "fedora:39"
  }
  output = ["type=docker"]
}
