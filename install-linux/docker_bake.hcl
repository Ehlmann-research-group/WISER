variable "PLATFORMS" {
  default = ["linux/amd64"]
}

target "phasea" {
  dockerfile = "install-linux/ubuntu_amd64/Dockerfile"   # or per-distro file
  platforms  = PLATFORMS
  tags       = ["wiser-phasea:ubuntu20.04-amd64"]
}

target "phaseb" {
  depends_on = ["phasea"]
  dockerfile = "install-linux/build_wiser/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phaseb:ubuntu20.04-amd64"]
  args = {
    PHASEA_IMAGE = "wiser-phasea:ubuntu20.04-amd64"
  }
}

target "phasec" {
  depends_on = ["phaseb"]
  dockerfile = "install-linux/smoke_test_build/Dockerfile"
  platforms  = PLATFORMS
  tags       = ["wiser-phasec:ubuntu20.04-amd64"]
  args = {
    PHASEA_IMAGE = "wiser-phasea:ubuntu20.04-amd64"
    PHASEB_IMAGE = "wiser-phaseb:ubuntu20.04-amd64"
  }
}
