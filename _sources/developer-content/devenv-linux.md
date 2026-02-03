# Developing and Building WISER on Linux

You will need conda and python installed. You will need to do `pip install conda-lock` and `pip install pyyaml`. I suggest you do this inside of a conda-environment. You will also need to have `make` installed. Once you have all of this installed you are ready to go into the /etc folder and run the command `make install-dev-env`.

Note that you will also need `git` installed. You will likely want `build-essential` installed as well.

## Developing and Building WISER on Linux

There are many linux distributions available that target many
different instruction sets. Currently, we have made WISER builds
target Ubuntu 20.04, Debian 11, and Fedora 39. WISER builds
should generally work on future versions of these operating
systems due to these operating systems striving for backwards
compatibility. The Instruction Set Architectures (ISAs) that
WISER builds for are amd64 (aka x86_64) and aarch64 (aka arm64).

Currently, we have only tested developing WISER on a Ubuntu 22.04.5
amd64 (x86_64) ISA. The below instructions should work for other linux
distributions.


## How to Install Conda

To build WISER for linux, you must first have the development environments for
WISER set up on linux. WISER uses conda for package management, so you will need
conda to do this. Follow the instructions here to install miniconda:
https://www.anaconda.com/docs/getting-started/miniconda/main.

Once you have installed conda, follow the directions in `Environment setup` to
set up your conda environments.

## How to Build WISER

Once you have your conda environments set up, activate the `wiser-prod`
environment. Then simply run `make build-linux` in the root directory
of the repository. This should make a linux build that targets your current 
linux distribution and ISA. It is recommended to instead use the GitHub Action
to create your linux build as it has code that solves some dynamic library
dependency issues that you will likely encounter if you just build with
`make build-linux`.

The build output will be placed in the root directory under `/dist`. Try
running the output by going to `/dist/WISER` and running the binary by doing
`./WISER_Bin`.

### Continuous Deployment

Currently, the official WISER releases for linux are made in the GitHub action
for `Build and Smoke WISER`. The way that this works for linux is a bit involved,
so I will explain here:

1. First we create a matrix strategy that lets us run jobs in parallel. These jobs will
target Ubuntu 20.04 + amd64, Debian 11 + amd64, Fedora 39 + amd64, Ubuntu 20.04 + arm64,
Debian 11 + arm64, Fedora 39 + arm64.
2. For each of these targets we run the file `./build_linux_multistage.sh` which just takes
the OS and ISA and runs the corresponding docker bake target that's in `docker_bake.hcl`.
The Dockerfile that runs will create an output artifact in `/app/WISER.tar.gz` in the container.
3. The `./build_linux_multistage.sh` file will then copy the artifact in `/app/WISER.tar.gz`
to the host machine. It contains the following files:
    a. `install_wiser_appimage.sh`: Used to install the .AppImage on the users machine
    b. `RUN.txt`: Directions for the user on how to run WISER
    c. `SHA256SUMS`: The SHA256 for verifying WISER-x86_64.AppImaage
    d. `WISER-x86_64.AppImaage`: Contains the actual app
4. The GitHub action will then upload `/app/WISER.tar.gz` to GitHub. Unfortunately, the upload
always zips `WISER.tar.gz` which is a bit redundant as it is already compressed.
