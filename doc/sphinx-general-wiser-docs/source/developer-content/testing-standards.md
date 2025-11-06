# Testing Standards

## Purpose & Scope

Reliable testing is essential to maintaining a stable and trustworthy codebase. Our tests protect against regressions when code changes, enable confident refactoring, and increase trust in the correctness of new features. Tests should be treated as production-grade code: readable, maintainable, and updated as the software evolves.

This guide covers our approach to all tests that are in WISER which mainly include **unit tests**, **functional tests**, **integration tests**, **end-to-end tests**, and **performance tests**. Each plays a different role in verifying system correctness, and contributors are expected to write tests appropriate to the type of change they are making.

This document applies to **all contributors**, including new contributors submitting their first PR, committers implementing new features, and maintainers reviewing contributions.

## Testing Philosophy / Principles

- **Fast and deterministic:** Tests should run quickly and produce the same result every time.
- **Isolated:** Tests should not depend on global state, shared data, or hidden side effects.
- **Parallel-safe:** Tests should not interfere with one another so that they can be executed in parallel.
- **Readable and maintainable:** Tests should be easy to understand and evolve as the code evolves.
- **Typical + edge cases:** Tests should cover both the common use-cases and the boundary conditions where failures are likely to occur.
- **Hardware or external dependency tests must be explicitly marked.**

We follow a **test pyramid approach**:
1. **Unit Tests** — small, isolated, fast.
2. **Functional Tests** — test core workflows.
3. **Integration Tests** — exercise interactions between multiple components.
4. **End-to-End Tests** — validate the full user-facing experience.

We do **not** chase coverage numbers for their own sake. Instead, your tests should meaningfully cover the behavior of the feature you are implementing, which will generally result in good functional coverage naturally.

## Test Organization

Tests currently live at:

_src/tests/_

As of **11/05/2025**, we do not yet organize tests by type (unit, integration, etc.), but this may be introduced later as the test suite grows.

### Naming & Structure (pytest)
- Test filenames must start with:  
  `test_*.py`
- If the test works y by clicking through gui elements and only 
tests one piece of functionality, put `_gui` at the end. If it tests 
the interface between two features put `_integ` at  the end. We may change or get rid of this in the future in place of pytest markers.
- Currently, there are no hard set rules dictating what to name the rest of the file, but generally it should be make clear what is being tested and any specifics about what is being tested.
- Test classes should follow Python testing patterns. If using unittest style, inherit from `unittest.TestCase`.
- Test functions should start with `test_`.

Example:
```python
def test_my_feature():
    ...
```

## Test Markers
Test markers are declared in the root `pyproject.toml`:
```
[tool.pytest.ini_options]
markers = [
  # ...
]
```
Markers are used to classify and selectively run tests (`smoke`, `integration`, `device`, `slow`, etc.).

If a contributor believes a new marker is needed:
1. Write a test using the proposed marker.
2. Open a PR explaining why existing markers are insufficient.
3. Add a descriptive entry for the marker in pyproject.toml.

## Running Tests & CI/CD Integration

### Local
```bash
cd src/tests
pytest .
```

### Selecting Marker Groups
```bash
pytest -m smoke
pytest -m "smoke and not slow"
pytest -m "(device or integration) and not slow"
```

### Continuous Integration (CI)
CI on github runners runs:
```bash
cd src/tests
pytest .
```

### Continuous Deployment (CD)

CD runs a WISER build  in test mode. Example on MacOSX
```bash
./WISER_Bin --test_mode
```
The `--test_mode` executes `pytest .` against _src/tests_ inside the bundled distribution.

## Test Guidelines
| Principle                  | Description                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------- |
| Arrange → Act → Assert     | Organize test logic clearly.                                                          |
| Minimize side-effects      | Avoid unnecessary I/O, network access, or global state changes in unit tests.         |
| Mock/stub appropriately    | Replace external dependencies where full execution is unnecessary.                    |
| Independent and repeatable | Tests must not depend on execution order or shared state.                             |
| Avoid magic numbers        | Use named constants or helper functions to clarify intent.                            |
| Avoid flakiness            | Tests should not depend on timing, network reliability, or nondeterministic behavior. |

## Test Maintenance & Flakiness
### Handling Flaky Tests

If a test behaves inconsistently:

1. **Identify** the flakiness.

2. **Mark** it as `xfail` so CI remains stable.

3. **Submit a PR** marking it xfail.

4. After merge, submit a **follow-up PR** that fixes the underlying issue.

### Isolation Requirements

A properly isolated test:

- Passes regardless of execution order.
- Has no hidden dependencies (network, locale, external services) unless explicitly mocked.
- Does not share mutable state between tests.
- Runs correctly under parallel execution (pytest -n auto).

### Updating Tests

If code changes introduce new behavior or modify APIs:

- Update relevant tests.
- Add tests for newly introduced code paths.
- Fix broken tests that fail for valid reasons.

### Deprecating Tests

You may remove a test if:

- The feature it covers is intentionally removed.
- The test overlaps with other tests and provides no meaningful value.

### Ownership
Everyone is responsible for tests:

- If you introduce new functionality → you write the tests.
- If you refactor functionality → you update the tests.
- If a test breaks unrelated to your change → fix it or report it.

### Code Review

All tests must go through code review and pass CI.

### Test Environment

- CI runs tests in a development-like environment on Linux GitHub runners.
- CD runs tests in environments similar to production (macOS ARM, macOS Intel, Windows).
Note: CD currently tests on the same runner that performs the build. In the future we plan to test distributed artifacts on separate machines (macOS code signing poses additional constraints).

## Tests Are Expected in the Same PR

When introducing new features or changes, tests should be included in the same PR.

If the feature is too large to test immediately:

- Submit a follow-up PR containing the tests.
- The feature must be gated behind a feature flag and clearly annotated:
    ``` python
    # NOT-TESTED: feature not yet covered by test suite. 
    # Tracking: <link to PR with tests>
    ```
    No feature is considered complete until the corresponding tests have been merged.
