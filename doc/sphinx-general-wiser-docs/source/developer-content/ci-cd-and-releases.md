# CI/CD and Releases

CI/CD stands for continuous integration/continuous
development. It is a practice that is meant to allow
code to be release as frequently as possible. The
continuous integration encourages code to be frequently
merged into main and continuous deployment encourages
stable builds to be made frequently. Currently, the deployment
process for WISER is run manually, but we are looking into
running it nightly or more frequently.

## Continuous Integration

In WISER we will be practicing continuous integration.
Read [this](https://martinfowler.com/articles/branching-patterns.html#ContinuousIntegration)
to learn more. That article has a lot of useful information on branching strategies
if you want to read more.

WISER currently has two main github actions to increase code quality:
linting/formatting and development tests. We use ruff to do our linting
and formatting because it is fast. For testing, we use pytest. Read on
for more information on how the these are set up.

### Linting / Formatting

As previously said, we use ruff to do linting and formatting. We control
the rules for this in _pyproject.toml_. You will see that there are a
few rules that we ignore for linting. Read the comments
in [this](https://github.com/Ehlmann-research-group/WISER/pull/238) PR for more information.

Linting can be done with the command `ruff check .`. If there are any
safe fixes that can be done do `ruff check --fix .`. You should not do
`ruff check --fix --unsafe-fixes .` unless you are very confident.

Formatting can be done with `ruff format`. Note that the command `ruff format`
will change files in order to make the formatting changes. If you do not
want to do this and you just want to check to see if things are formatted
correctly, do `ruff format --check`.

There is a pre-commit hook to perform the linting and formatting. However,
you have to install it the pre-commit first. Pre-commit hooks that are
defined in .pre-commit-config are automatically installed if you run the
command `make install-dev-env` in the _etc/_ folder. You can also run the
commands  `pre-commit clean` then `pre-commit install` while you're in the
dev environment.

The linting / formatting is also done on push and pull-request to `main`
and release branches `rel/**`.

### Dev Testing

Testing is only done on push and pull-request to `main` and release branches
`rel/**`. We use micromamba to make the conda environment to run our tests.
Micromamba is great because it is fast to make the environment and can be
quickly pulled from cache. The cache is made from the file _etc/dev-conda-lock.yml_
and will update when this file is changed. The tests use the cache and
also update the cache, so there are two rules that update the micromamba
environment cache. When running tests, we simply `cd` into _src/tests_ and
run the command `pytest -s .`.

Testing happens after linting.

## Continuous Deployment

We are still figuring out what type of continuous
deployment framework that we want for WISER. Currently,
the build process takes a pretty long time (around 24
minutes for the longest build). This seems a bit
infeasible for developers to run on each push to a pull
request. We need to look into a way to speed up our builds
(<10 minutes would be good).

The deployment code is in the github action `prod-deploy.yml`. It
builds WISER on all three platforms (windows, macos arm, macos intel)
then runs tests inside of each WISER distributable. Running these tests
ensures that WISER actually starts up and that core WISER functionality
has been properly packaged up.

It is important to note that these build artifacts still
need some more work done to them in order to become our
distributables. The Windows artifact needs to be code-signed
then packaged into an installer. The MacOS artifacts
need to be packaged into a .dmg and code-signed.

### "Production" Testing

Currently, we test on our distributables in the github
runners by running WISER with the `--test_mode` flag. We
do this because we want to make sure everything in our
distributables was packaged up correctly. For example, to
test that JP2OpenJPEG.dll was properly packaged into the
build, we would have a test that would try opening up a very
small JP2 file that we can run with a command like `./WISER_Bin --mode test`. So you can see how it is important
to run tests on the build artifact.

We want to build these
artifacts and run these tests on `main` and `rel/**` banches
because these branches are meant to contain very stable
code that can be deployed at a moments notice. However,
currently we are still deciding whether we should run
deployment tests on every PR to `main` and `rel/**` or
nightly.

#### Deployment Tests

We currently have a github action that builds WISER on the github runners then runs
our smoke tests then uploads distribution files to github. Currently, this only happens
by workflow_dispatch, which means a developer must manually make it run.

There are also simple make recipes to build and test wiser locally.
The make commands `smoke-test-win-build` and `smoke-test-mac-build` when run from the root directory
build WISER and run the `--test_mode` locally. This is useful to find build bugs as well.

This github action is under .github\workflows\prod-deploy.yml .

#### Deployment Test Signing

The good thing about the pipeline in the Deployment Tests section
is that we can deploy directly from it. Since the artifacts from these tests were
built on a fresh machine (the github runner), we know its more reproducible than building
it locally! All we have to do is pull the artifact down to our local machine and
sign it. I have [made code to do this](https://github.com/Ehlmann-research-group/WISER/pull/257).

You can run this code by going into the root directory and doing

`make sign-mac LINK=<link-address-of-artifact> MAC_DIST_GITHUB_NAME=<artifact-github-name>`

or simply

`make sign-windows LINK=<link-address-of-artifact>`

for windows.
This step requires you to have Github's CLI tool installed which lets you use the
command `gh`. The logic for this step is in the files /src/devtools/sign_mac.py and
/src/devtools/sign_windows.py. It was originally introduced on
[this](https://github.com/Ehlmann-research-group/WISER/pull/257) branch.

## Releases

WISER releases should always be made from a release branch. Release
notes should accompany releases. Build artifacts
should accompany releases. These build artifacts won't be
signed due to how artifact creation works. Additionally, official
releases should be made on the GitHub through
the [Releases feature](https://github.com/Ehlmann-research-group/WISER/releases)
and the release should be tagged.

## Release Process

The process of creating a release is documented below.

1. Build WISER through the GitHub workflow `prod-deploy`.
2. Do `make sign-mac` or `make sign-windows` to run the signing
   logic. For signing on mac, you must have a valid Apple Developer
   signing certificate tied to a paid Apple Developer Program account.
   For signing on windows, you must have a windows code signing
   certificate. This is tied to an individual or a legal entity.

   a. If you do code sign your own distribution, please do not
   present it to others as a official WISER release unless you have
   been explicitly allowed to do so for a specific release.

   b. Note that WISER currently does not have a code
   signing mechanism for Linux.

3. If you are making an official release, you will need access to
   the [WISER website](https://ehlmann.caltech.edu/wiser/index.html)
   and a stable place to host the WISER distributable. When you have
   access to this, you will add a download link to the website for the
   new release.

4. You will need to make a release on GitHub with the release notes
   and tag the release's commit.

5. You will then need to put the release on
   the [website's release notes page](https://ehlmann.caltech.edu/wiser/release-notes.html).

6. Lastly, you will need to update the plugin API documentation on
   the page for [Plugin Dependencies](https://ehlmann-research-group.
   github.io/WISER-Plugin-API/plugin_dependencies.html) to include all
   of the dependencies in the latest version.

7. Finally, if have the permissions to, then you should send an
   email to wiser-announce@caltech.edu to announce a new release of
   WISER with a summary of what is in the release. If you don't have
   permission to, reach out to someone who does with the email you want
   them to send.

> **Note**: This process can only be done by a maintainer with
> access to all of these resources. This intentionally limits who can
> do official WISER releases. If the community thinks that an official
> WISER release should be done, please reach out to the maintainers.

## GitHub Actions Environment Setup

Creating the conda environment on each CI run is the main performance
bottleneck. Approaches ranked slowest to fastest:

### 1. Create from scratch every run (~7.5 min)

```yaml
- uses: conda-incubator/setup-miniconda@v3
  with:
    auto-update-conda: true
    activate-environment: wiser-source
    environment-file: etc/testing.yml
    channels: conda-forge
    auto-activate-base: false
```

### 2. Cache conda packages, then create env (~4.5 min)

```yaml
- name: Setup Cached Conda Environment
  uses: actions/cache@v4
  env:
    CACHE_NUMBER: 0
  id: cache
  with:
    path: ~/conda_pkgs_dir
    key: ${{ runner.os }}-conda-${{ env.CACHE_NUMBER }}-${{ hashFiles('etc/testing.yml') }}
    restore-keys: |
      ${{ runner.os }}-conda-${{ env.CACHE_NUMBER }}-

- uses: conda-incubator/setup-miniconda@v3
  with:
    auto-update-conda: true
    activate-environment: wiser-source
    environment-file: etc/testing.yml
    channels: conda-forge
    auto-activate-base: false
    use-only-tar-bz2: true  # required for caching to work

- name: Update environment
  run: conda env update -n wiser-source -f etc/testing.yml
  if: steps.cache.outputs.cache-hit != 'true'
```

### 3. Preloaded Docker image (fastest, harder to maintain)

Run tests inside a Docker image that already has the conda environment
installed. The tradeoff is that the Docker image must be rebuilt and pushed
whenever dependencies change.

Useful references:

- [Caching dependencies in GitHub Actions](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/caching-dependencies-to-speed-up-workflows)
- [setup-miniconda caching](https://github.com/conda-incubator/setup-miniconda?tab=readme-ov-file) — search "caching
  environments"
- [Beginner Docker guide](https://learncloudnative.com/blog/2020-04-29-beginners-guide-to-docker)
- [Conda in Dockerfiles](https://pythonspeed.com/articles/activate-conda-dockerfile/)

## Docker Container Reference

### Running Qt (headless) inside Docker

PySide6/PySide2 requires a display. Use `xvfb` inside a container:

```bash
Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1
pytest test_rasterpane
```

See
also: [Stack Overflow — headless Qt](https://stackoverflow.com/questions/65200690/how-to-run-a-qt-application-in-headless-mode-without-showing-my-gui)

**Note:** `pytest-qt` currently only works with PySide6, not PySide2.
See [pytest-qt docs](https://pytest-qt.readthedocs.io/en/latest/intro.html).

### Common Docker Commands

```bash
# List all containers
docker ps -a

# Run a container with WISER source mounted
docker run -it --mount type=bind,src=<path-to-WISER>,dst=/WISER wiser-env:0.1.17 /bin/bash

# Build the Docker image (run from WISER/ root)
docker build -f ./etc/Dockerfile -t wiser-env:0.1.17 .

# Tag for pushing to Docker Hub
docker tag wiser-env:0.1.17 jgarciak/wiser-env:0.1.17

# Push to Docker Hub (must be signed in)
docker image push jgarciak/wiser-env:0.1.17
```

### Known Docker / CI Issues

**NetCDF segfault on Linux:** Opening a `.nc` file inside a Linux Docker
container (tested with GDAL 3.10.1 and 3.9.3) causes a fatal segmentation
fault during `get_band_data_normalized`. The stack trace originates in
`numpy.ma.core.masked_values` called from `dataset.py`. This is a known
issue with no current fix.

**Common conda error:** `does not exist (perhaps a typo or a missing channel)`
— usually caused by a missing conda-forge channel or a typo in the environment
file.

## Maintaining the Plugin API Documentation

The Plugin API documentation lives in a separate repository:
[WISER-Plugin-API](https://github.com/Ehlmann-research-group/WISER-Plugin-API).
It is served as GitHub Pages from the `main` branch.

To rebuild and publish the Plugin API docs:

1. Make sure you have Sphinx and the required extensions installed (activate
   the `wiser-dev` environment or a dedicated docs environment).
2. Go to `WISER/doc/sphinx-extending-wiser/`.
3. Confirm `conf.py` has:
   ```python
   html_baseurl = 'https://ehlmann-research-group.github.io/WISER-Plugin-API/'
   ```
4. Run `make` (or `make html`) to build.
5. Copy the **contents** of the `build/` folder (not the folder itself) into
   the root of the `WISER-Plugin-API` repository.
6. Ensure a `.nojekyll` file exists at the root of `WISER-Plugin-API`.
7. Push to `main` — GitHub Pages will publish automatically.

This step is part of the release process (see the Release Process section above).

---

## Release Checklist

This page documents the process for creating a new WISER release.

## Overview

```{note}
This section is not yet written. It will cover the end-to-end release workflow:
version bumping, building distributables, code signing, CI/CD pipeline, and
publishing releases.
```

## Related

- See the CI/CD and Releases section above for pipeline configuration details.

