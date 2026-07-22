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

The deployment code is in the GitHub Actions workflow `prod-deploy.yml`. It builds WISER on every
platform (Windows, macOS arm/intel, and the six Linux distro/arch targets). The Linux legs
run the full `--test_mode` suite inside each distributable (a hard gate — a failing suite
produces no tarball); Windows and macOS run a fast `--smoke` launch check in CI and the full
`--test_mode` locally. Running these checks ensures WISER actually starts up and that core
functionality was packaged correctly.

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

Note that "smoke test" means two different things in our pipeline. The **Linux** builds run
the full `--test_mode` suite in-container via `install-linux/run_smoke.sh` (despite the
name) — a failing suite fails the build and no tarball is produced. **Windows/macOS** run
only a lightweight `--smoke` launch check in CI (start Qt, show the splash, exit) and rely
on the full `--test_mode` being run locally before signing. Don't "standardize" the Linux
path onto `--smoke` — that would silently drop the Linux test gate.

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

WISER releases should always be made from a release branch and tagged through the
GitHub [Releases feature](https://github.com/Ehlmann-research-group/WISER/releases).
Release notes should accompany every release.

The installers are attached to the GitHub release as **release assets** (not linked
from an Actions run — Actions artifacts expire, require a login, and are not counted
in a release's download totals). Release notes should point users at the assets on the
release page, never at a `…/actions/runs/<id>` URL.

### Asset naming convention

Asset filenames are normalized at build time so a platform + architecture is always
identifiable from the name (the public downloads page groups assets by these tokens):

```
WISER-<version>-windows-x64-setup.exe
WISER-<version>-macos-arm64.dmg
WISER-<version>-macos-x64.dmg
WISER-<version>-linux-<distro>-x64.tar.gz     # distro: ubuntu2004 | debian11 | fedora39
WISER-<version>-linux-<distro>-arm64.tar.gz
```

`x64`/`arm64` are the canonical arch tokens (Intel/amd64 → `x64`). The names are set by
`win-install.nsi` (Windows), `src/devtools/sign_mac.py` (macOS), and the Linux build
step in `prod-deploy.yml` — not by hand on upload.

### How assets reach the release

Assets are **built once and promoted** — the release event never triggers a build, so the
bits you ship are the exact bits you tested.

- **Linux** builds are unsigned, so a CI tarball *is* the final asset. `prod-deploy.yml`
  (run via `workflow_dispatch`) builds and tests each distro and uploads the tarballs as
  run artifacts. When you are ready to release, `publish-release-assets.yml` attaches the
  tarballs from that specific run to the release — it verifies the run succeeded and that
  the tag points at the built commit, so nothing is rebuilt at release time.
- **Windows / macOS** are signed locally with the maintainer's certificates, then uploaded
  with the canonical name via the sign scripts' `--release-tag <tag>` option (see the
  Release Process below).

### Pre-release flag

The web `…/releases/latest` and the API `latest` endpoint both **exclude** releases
flagged `prerelease`. Our beta tags (e.g. `v2.2b1`) are *not* automatically treated as
pre-releases — it is purely the checkbox at publish time. When publishing the release
the public should download, leave **"Set as a pre-release" unchecked** (and keep "Set as
the latest release" on) so `latest` resolves to it.

## Release Process

Assets are built once, tested, then **promoted** to the release — nothing is rebuilt at
release time, so what ships is exactly what you tested.

1. **Bump the version** in `src/wiser/version.py` (and `RELEASE_DATE`) and commit it. The
   Linux asset name is read from `version.py` at build time, so it must be correct
   *before* you build.

2. **Verify app metadata is current**:

   - **Check the year.** Ctrl+F for whatever the current year is throughout the app and
     go through each hit to make sure the year is correct — some may not need to be
     changed. (We should turn this into a single variable in the future.)
   - **Check the feature flags.** Search for `FLAGS.` and check `feature_flags.py`,
     confirming the `FEATURE_GATES` dictionary is correct. Then check
     `app_config.py` and confirm everything referencing `feature_flags.` is correct.

3. **Build and test** — run **Build and Smoke Test WISER** (`prod-deploy.yml`) via
   `workflow_dispatch` on the commit you intend to release. Every platform builds; each
   Linux distro runs the full `--test_mode` suite in-container (a leg only produces a
   tarball if its tests pass), and Windows/macOS run a `--smoke` launch check. Note the
   **run ID** — you will promote this exact run.

4. **Verify locally** — download the Linux tarballs from the run and test if needed. Build,
   `--test_mode`, code-sign, and clean-install-test the Windows/macOS installers locally
   (CI does not produce the signed installers). See the signing certificate requirements
   below.

5. **Create the release** on the *same commit* you built. Publish it as a **pre-release**
   (see the Pre-release flag section) so it stays out of `…/releases/latest` while you
   attach and verify assets. A published pre-release also lets the publish workflow confirm
   the tag matches the built commit.

6. **Attach the Linux assets** — run **Publish release assets**
   (`publish-release-assets.yml`) with the build **run ID** and the release **tag**. It
   checks the run succeeded and that the tag points at the built commit, then attaches the
   six Linux tarballs.

7. **Upload the signed Windows/macOS installers** with `RELEASE_TAG=<tag>` so each is
   uploaded to the release with its canonical name in one step:

   `make sign-mac LINK=<artifact-url> MAC_DIST_GITHUB_NAME=<artifact-name> RELEASE_TAG=<tag>`
   or `make sign-windows LINK=<artifact-url> RELEASE_TAG=<tag>`.

   For signing on macOS, you must have a valid Apple Developer signing certificate tied to
   a paid Apple Developer Program account. For signing on Windows, you must have a Windows
   code signing certificate, tied to an individual or a legal entity.

   a. If you code-sign your own distribution, do not present it to others as an official
   WISER release unless you have been explicitly allowed to for a specific release.

   b. WISER has no code-signing mechanism for Linux; the Linux assets are attached unsigned.

8. **Finalize the release notes** on the GitHub release. Point users at the attached
   release assets — never a `…/actions/runs/<id>` URL.

9. **Go live** — once every asset is present with its canonical name, uncheck **"Set as a
   pre-release"** (and keep "Set as the latest release" on) so `latest` resolves to it.

10. **Update the plugin API documentation** in
    `doc/sphinx-general-wiser-docs/source/extending-wiser/` to reflect any changes to plugin
    interfaces or dependencies in the latest version.

11. **Announce** — if you have permission, email wiser-announce@caltech.edu with a summary of
    the release. If not, reach out to someone who does with the email you want sent.

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

### 3. Preloaded Docker image (fast, harder to maintain)

Run tests inside a Docker image that already has the conda environment
installed. The tradeoff is that the Docker image must be rebuilt and pushed
whenever dependencies change.

Useful references:

- [Caching dependencies in GitHub Actions](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/caching-dependencies-to-speed-up-workflows)
- [setup-miniconda caching](https://github.com/conda-incubator/setup-miniconda?tab=readme-ov-file) — search "caching
  environments"
- [Beginner Docker guide](https://learncloudnative.com/blog/2020-04-29-beginners-guide-to-docker)
- [Conda in Dockerfiles](https://pythonspeed.com/articles/activate-conda-dockerfile/)

#### 4. Micromamba with conda-lock and caching (fastest, and relatively easy to maintain)

Using conda-lock, micromamba doesn't need to solve and micromamba itself is very fast and barebones. I believe when I
tested this in the past it took about 1.5 minutes to make an environment from a cold start.

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

The Plugin API documentation is part of this Sphinx docs site
(`doc/sphinx-general-wiser-docs/`), under the **Extending WISER** section.
It is built and deployed alongside the rest of the WISER documentation via
the existing GitHub Actions workflow (see `docs-preview.yml` / `deploy-docs.yml`).

The community plugin examples and third-party plugins themselves live in a
separate repository:
[WISER-Plugin-Repository](https://github.com/Ehlmann-research-group/WISER-Plugin-Repository).

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

