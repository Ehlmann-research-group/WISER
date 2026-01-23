variable "PLATFORMS" {
  default = ["linux/arm64"]
}

# Phase A: two distros, same Dockerfile
target "phasea_ubuntu2004" {
  dockerfile = "install-linux/base_image/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasea:ubuntu20.04"]
  args = { BASE_IMAGE = "ubuntu:20.04" }
  output = ["type=docker"]
}

target "phasea_debian11" {
  dockerfile = "install-linux/base_image/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasea:debian11"]
  args = { BASE_IMAGE = "debian:11" }
  output = ["type=docker"]
}

target "phasea_fedora39" {
  dockerfile = "install-linux/base_image_fedora/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasea:fedora39"]
  args = {
    BASE_IMAGE = "fedora:39"
  }
  output = ["type=docker"]
}

# Phase B: one Dockerfile, parameterized by which Phase A tag to use
target "phaseb_ubuntu2004" {
  depends_on = ["phasea_ubuntu2004"]
  dockerfile = "install-linux/build_wiser/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phaseb:ubuntu20.04"]
  args = { PHASEA_IMAGE = "wiser-phasea:ubuntu20.04" }

  contexts = {
    "wiser-phasea:ubuntu20.04" = "target:phasea_ubuntu2004"
  }
  output = ["type=docker"]
}

target "phaseb_debian11" {
  depends_on = ["phasea_debian11"]
  dockerfile = "install-linux/build_wiser/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phaseb:debian11"]
  args = { PHASEA_IMAGE = "wiser-phasea:debian11" }

  contexts = {
    "wiser-phasea:debian11" = "target:phasea_debian11"
  }
  output = ["type=docker"]
}

target "phaseb_fedora39" {
  depends_on = ["phasea_fedora39"]
  dockerfile = "install-linux/build_wiser_fedora/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phaseb:fedora39"]
  args = {
    PHASEA_IMAGE = "wiser-phasea:fedora39"
  }

  contexts = {
    "wiser-phasea:fedora39" = "target:phasea_fedora39"
  }
  output = ["type=docker"]
}

# Phase C: parameterized by PhaseA + PhaseB tags
target "phasec_ubuntu2004" {
  depends_on = ["phaseb_ubuntu2004"]
  dockerfile = "install-linux/smoke_test_build/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasec:ubuntu20.04"]
  args = {
    PHASEA_IMAGE = "wiser-phasea:ubuntu20.04"
    PHASEB_IMAGE = "wiser-phaseb:ubuntu20.04"
  }

  contexts = {
    "wiser-phasea:ubuntu20.04" = "target:phasea_ubuntu2004"
    "wiser-phaseb:ubuntu20.04" = "target:phaseb_ubuntu2004"
  }
  output = ["type=docker"]
}

target "phasec_debian11" {
  depends_on = ["phaseb_debian11"]
  dockerfile = "install-linux/smoke_test_build/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasec:debian11"]
  args = {
    PHASEA_IMAGE = "wiser-phasea:debian11"
    PHASEB_IMAGE = "wiser-phaseb:debian11"
  }

  contexts = {
    "wiser-phasea:debian11" = "target:phasea_debian11"
    "wiser-phaseb:debian11" = "target:phaseb_debian11"
  }
  output = ["type=docker"]
}

target "phasec_fedora39" {
  depends_on = ["phaseb_fedora39"]
  dockerfile = "install-linux/smoke_test_build_fedora/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasec:fedora39"]
  args = {
    PHASEA_IMAGE = "wiser-phasea:fedora39"
    PHASEB_IMAGE = "wiser-phaseb:fedora39"
  }

  contexts = {
    "wiser-phasea:fedora39" = "target:phasea_fedora39"
    "wiser-phaseb:fedora39" = "target:phaseb_fedora39"
  }
  output = ["type=docker"]
}
