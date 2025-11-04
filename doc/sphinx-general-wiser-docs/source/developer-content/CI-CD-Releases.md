# CI/CD and Releases

CI/CD stands for continuous integration/continuous
development. It is a practice that is meant to allow
code to be release as frequently as possible. The
continuous integration encourages code to be frequently
merged into main and continuous deployment encourages
stable builds to be made frequently. Currently, the deployment process for WISER is run nightly. Ideally,
we would run it on ever PR to main but this process takes
a long time.

TODO (Joshua-GK): FIgure out a way to run constant builds
(currently I am thinking that nightly builds will work, maybe figure out a way to speed up pyinstaller builds
so we can feasibly run the build process on every PR)

## Continuous Integration

In WISER we will be practicing continuous integration. Read [this](https://martinfowler.com/articles/branching-patterns.html#ContinuousIntegration)
to learn more. That article has a lot of useful information on branching strategies
if you want to read more. 

WISER currently has two main github actions to increase code quality:
linting/formatting and development tests. We use ruff to do our linting
and formatting because it is fast. For testing, we use pytest. Read on
for more information on how the these are set up.

### Linting / Formatting

As previously said, we use ruff to do linting and formatting. We control
the rules for this in _pyproject.toml_. You will see that there are a
few rules that we ignore for linting. Read the comments in [this](https://github.com/Ehlmann-research-group/WISER/pull/238) PR for more information.

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

TODO (Joshua G-K) Finish this section, figure out continuous
deployment framework we want.

We are still figuring out what type of continuous 
deployment framework that we want for WISER. Currently,
the build process takes a pretty long time (around 24
minutes for the longest build). This seems a bit
infeasible for developers to run on each push to a pull
request. We need to look into a way to speed up our builds 
(<10 minutes would be good).

The deployment code is in the github action `prod-deploy.yml`. It builds WISER on all three platforms (windows, macos arm, macos intel) then runs tests inside of each WISER distributable. Running these tests ensures that
WISER actually starts up and that core WISER functionality
has been properly packaged up. 

It is important to note that these build artifacts still
need somne more work done to them to become our
distributables. The Windows artifact needs to be code-signed
then packaged into an installer. The MacOS artifacts
need to be packaged into a .dmg and code-signed. 

### Production Testing

TODO Finish this section.



OLD: This has not been built yet. But we would like a way to test WISER when it is
in production. We would mainly want to test aspects of WISER that need to be
packaged into the build. For example, to test that JP2OpenJPEG.dll were properly
packaged into the build, we would have a test that would try opening up a very
small JP2 file that we can run with a command like `./WISER_Bin --mode test`.
We do want to have a github action for this, but we don't want it to run for a
long time as builds take a while to run (and we don't want to
[waste our github time](https://docs.github.com/en/billing/concepts/product-billing/github-actions#free-use-of-github-actions)). 
I think it would be best to run this test on pushes to `main` or `rel/**`.
This has the downside of the bug not being seen until after the user has pushed
their pull_request (since we will only be pull requesting into `main` and `rel/**`).
However, to make it easier to test, we can make a recipe in a Makefile.

Currently, we test on our distributables in the github
runners by running WISER with the `--test_mode` flag. We
do this because we want to make sure everything in our
distributables was packaged up correctly. For example, to 
test that JP2OpenJPEG.dll were properly packaged into the
build, we would have a test that would try opening up a very
small JP2 file that we can run with a command like `./WISER_Bin --mode test`. So you can see how it is important
to run tests on the build artifact. We want to build these
artifacts and run these tests on `main` and `rel/**` banches
because these branches are meant to contain very stable 
code that can be deployed at a moments notice. However, 
currently we are still deciding whether we should run 
deployment tests on every PR to `main` and `rel/**` or
nightly. 

TODO (Joshua G-K): Figure out if we should do nightly builds
or run builds on each PR to main/rel** branches.

#### Deployment Tests

We currently have a github action that builds WISER on the github runners then runs
our smoke tests then uploads distribution files to github. Currently this doesn't
work well because on Windows, the micromamba environment is hanging at a specific
package. The closest issue I could find online is [this]
(https://github.com/mamba-org/mamba/issues/3575?utm_source=chatgpt.com). The macOS
arm runner works (which is the macOS-15 one), but the macOS intel runner (macOS-13 one)
is very slow so I want to try running the macOS-15 runner but with rosetta2 (so
we can get intel dylibs from it).

This github action is under .github\workflows\prod-deploy.yml . 

#### Deployment Test Signing

The good thing about the pipeline in the [Deployment Tests](#deployment-tests) section
is that we can deploy directly from it. Since the artifacts from these tests were
built on a fresh machine (the github runner), we know its more sturdy than building
it locally! All we have to do is pull the artifact down to our local machine and
sign it. I have [made code to do this](https://github.com/Ehlmann-research-group/WISER/pull/257).

This step requires you to have Github's CLI tool installed which lets you use the
command `gh`. The logic for this step is in the files /src/devtools/sign_mac.py and
/src/devtools/sign_windows.py. It was originally introduced on 
[this](https://github.com/Ehlmann-research-group/WISER/pull/257) branch.

## Releases

WISER releases should always be made from a release branch. Release notes should accompany releases. Build artifacts 
should accompany releases. These build artifacts won't be 
signed. Additionally, official release should be 
made on the github and the release should be tagged.
