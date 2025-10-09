# Continuous Integration

WISER currently has two main pieces of continuous integration: linting/formatting and development tests. We use ruff to do our linting and formatting because it is fast. For testing, we use pytest. Read on for more information on how the these are set up.

## Linting / Formatting

As previously said, we use ruff to do linting and formatting. We control the rules for this in _pyproject.toml_. You will see that there are a few rules that we ignore for linting. Read the comments in [this](https://github.com/Ehlmann-research-group/WISER/pull/238) PR for more information.

Linting can be done with the command `ruff check .`. If there are any safe fixes that can be done do `ruff check --fix .`. You should not do `ruff check --fix --unsafe-fixes .` unless you are very confident. 

Formatting can be done with `ruff format`. Note that the command `ruff format` will change files in order to make the formatting changes. If you do not want to do this and you just want to check to see if things are formatted correctly, do `ruff format --check`. 

There is a pre-commit hook to perform the linting and formatting. However, you have to install it the pre-commit first. Pre-commit hooks that are defined in .pre-commit-config are automatically installed if you run the command `make install-dev-env` in the _etc/_ folder. You can also run the commands  `pre-commit clean` then `pre-commit install` while you're in the dev environment.

The linting / formatting is also done on push and pull-request to `main` and release branches `rel/**`.

## Testing

Testing is only done on push and pull-request to `main` and release branches `rel/**`. 
