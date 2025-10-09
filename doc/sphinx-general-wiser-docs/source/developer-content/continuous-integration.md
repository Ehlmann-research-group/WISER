# Continuous Integration

WISER currently has two main pieces of continuous integration: linting/formatting and development tests. We use ruff to do our linting and formatting because it is fast. For testing, we use pytest. Read on for more information on how the these are set up.

## Linting / Formatting

As previously said, we use ruff to do linting and formatting. We control the rules for this in _pyproject.toml_. You will see that there are a few rules that we ignore for linting. Read the comments in [this](https://github.com/Ehlmann-research-group/WISER/pull/238) PR for more information.

Linting can be done with the command `ruff check .`. If there are any safe fixes that can be done do `ruff check --fix .`. You should not do `ruff check --fix --unsafe-fixes .` unless you are very confident. 

Formatting can be done with `ruff format`. Note that the command `ruff format` will change files in order to make the formatting changes. If you do not want to do this and you just want to check to see if things are formatted correctly, do `ruff format --check`. 

There is a pre-commit hook to perform the linting and formatting. However, you have to install it the pre-commit first. Pre-commit hooks that are defined in .pre-commit-config are automatically installed if you run the command `make install-dev-env` in the _etc/_ folder. You can also run the commands  `pre-commit clean` then `pre-commit install` while you're in the dev environment.

The linting / formatting is also done on push and pull-request to `main` and release branches `rel/**`.

## Dev Testing

Testing is only done on push and pull-request to `main` and release branches `rel/**`. We use micromamba to make the conda environment to run our tests. Micromamba is great because it is fast to make the environment and can be quickly pulled from cache. The cache is made from the file _etc/dev-conda-lock.yml_ and will update when this file is changed. The tests use the cache and also update the cache, so there are two rules that update the micromamba environment cache. When running tests, we simply `cd` into _src/tests_ and run the command `pytest -s .`.

Testing happens after linting.

## Production Testing

TODO Finish this section.

This has not been built yet. But we would like a way to test WISER when it is in production. We would mainly want to test aspects of WISER that need to be packaged into the build. For example, to test that JP2OpenJPEG.dll were properly packaged into the build, we would have a test that would try opening up a very small JP2 file that we can run with a command like `./WISER_Bin --mode test`. We do want to have a github action for this, but we don't want it to run for a long time as builds take a while to run (and we don't want to [waste our github time](https://docs.github.com/en/billing/concepts/product-billing/github-actions#free-use-of-github-actions)). I think it would be best to run this test on pushes to `main` or `rel/**`. This has the downside of the bug not being seen until after the user has pushed their pull_request (since we will only be pull requesting into `main` and `rel/**`). However, to make it easier to test, we can make a recipe in a Makefile.
