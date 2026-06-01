# Testing & QA

This page documents testing standards, quality assurance practices, and
release checklists for WISER development.

## Purpose & Scope

Reliable testing is essential to maintaining a stable and trustworthy codebase.
Tests protect against regressions when code changes, enable confident
refactoring, and increase trust in the correctness of new features. Tests
should be treated as production-grade code: readable, maintainable, and updated
as the software evolves.

This guide covers **unit tests**, **functional tests**, **integration tests**,
**end-to-end tests**, and **performance tests**. Each plays a different role in
verifying system correctness, and contributors are expected to write tests
appropriate to the type of change they are making.

This document applies to **all contributors**, including new contributors
submitting their first PR, committers implementing new features, and
maintainers reviewing contributions.

## Testing Philosophy

- **Fast and deterministic:** Tests should run quickly and produce the same result every time.
- **Isolated:** Tests should not depend on global state, shared data, or hidden side effects.
- **Parallel-safe:** Tests should not interfere with one another so that they can be executed in parallel.
- **Readable and maintainable:** Tests should be easy to understand and evolve as the code evolves.
- **Typical + edge cases:** Tests should cover both the common use-cases and the boundary conditions where failures are
  likely to occur.
- **Hardware or external dependency tests must be explicitly marked.**

We follow a **test pyramid approach**:

1. **Unit Tests** — small, isolated, fast.
2. **Functional Tests** — test core workflows.
3. **Integration Tests** — exercise interactions between multiple components.
4. **End-to-End Tests** — validate the full user-facing experience.

We do **not** chase coverage numbers for their own sake. Tests should
meaningfully cover the behavior of the feature being implemented, which will
generally result in good functional coverage naturally.

## Test Organization

Tests live at `src/tests/`. Tests are not yet organized by type (unit,
integration, etc.), but this may be introduced as the test suite grows.

### Naming & Structure (pytest)

- Test filenames must start with `test_*.py`
- Suffix conventions:
    - `_gui` — tests one piece of functionality by clicking through GUI elements
    - `_integ` — tests the interface between two features
- Test classes should follow Python testing patterns. If using unittest style,
  inherit from `unittest.TestCase`.
- Test functions must start with `test_`.

```python
def test_my_feature():
    ...
```

## Test Markers

Markers are declared in the root `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    # ...
]
```

Markers classify and selectively run tests (`smoke`, `integration`, `device`,
`slow`, etc.).

To propose a new marker:

1. Write a test using the proposed marker.
2. Open a PR explaining why existing markers are insufficient.
3. Add a descriptive entry for the marker in `pyproject.toml`.

## Test Timeouts

To prevent hung or deadlocked tests from burning CI compute, two layers of
timeout protection are in place.

### Per-test timeout (`pytest-timeout`)

Every test has a default budget of **240 seconds**, configured in the root
`pyproject.toml`:

```toml
[tool.pytest.ini_options]
timeout = 240
timeout_method = "thread"
```

When a test exceeds the budget, the watchdog thread dumps the stacks of every
thread (useful for diagnosing deadlocks) and aborts the test with
`Failed: Timeout >240s`. The suite continues with the next test.

If a test legitimately needs longer (e.g. an expensive multi-stage pipeline),
override the budget on that test:

```python
import pytest

@pytest.mark.timeout(600)
def test_long_running_pipeline():
    ...
```

You can also apply the override at module level via `pytestmark`:

```python
pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(600),
]
```

### Job-level timeout (GitHub Actions)

As a last-resort kill switch, the `tests:` job in
`.github/workflows/dev-CI.yml` sets `timeout-minutes: 60`. This catches
anything `pytest-timeout` cannot interrupt — hangs during pytest collection,
hangs in `make generated`, or native-code freezes the watchdog thread cannot
preempt.

## Running Tests

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

### CI/CD

CI runs `pytest .` from `src/tests` on Linux GitHub runners on every PR and
push to `main`.

CD runs WISER in `--test_mode`, which re-runs the test suite against the
bundled distribution:

```bash
./WISER_Bin --test_mode
```

## Test Guidelines

| Principle                  | Description                                                                           |
|----------------------------|---------------------------------------------------------------------------------------|
| Arrange → Act → Assert     | Organize test logic clearly.                                                          |
| Minimize side-effects      | Avoid unnecessary I/O, network access, or global state changes in unit tests.         |
| Mock/stub appropriately    | Replace external dependencies where full execution is unnecessary.                    |
| Independent and repeatable | Tests must not depend on execution order or shared state.                             |
| Avoid magic numbers        | Use named constants or helper functions to clarify intent.                            |
| Avoid flakiness            | Tests should not depend on timing, network reliability, or nondeterministic behavior. |

## Test Maintenance & Flakiness

### Handling Flaky Tests

1. **Identify** the flakiness.
2. **Mark** it as `xfail` so CI remains stable.
3. **Submit a PR** marking it xfail.
4. After merge, submit a **follow-up PR** fixing the underlying issue.

### Isolation Requirements

A properly isolated test:

- Passes regardless of execution order.
- Has no hidden dependencies (network, locale, external services) unless explicitly mocked.
- Does not share mutable state between tests.
- Runs correctly under parallel execution (`pytest -n auto`).

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

## Tests Are Expected in the Same PR

When introducing new features or changes, tests should be included in the same PR.

If the feature is too large to test immediately:

- Submit a follow-up PR containing the tests.
- Gate the feature behind a feature flag and annotate it:

```python
# NOT-TESTED: feature not yet covered by test suite.
# Tracking: <link to PR with tests>
```

No feature is considered complete until corresponding tests are merged.

## File Type Testing Checklist

When adding support for a new file type, verify the following after the class
is integrated. Go through `RasterDataImpl` and confirm each function still makes
sense with the new class and that the class can remain multi-threaded.

### General Checklist (apply to every new file type)

- [ ] Band math can be done
    - [ ] Multi-threaded band math can be done
- [ ] ROI average
- [ ] All panes, zoom in, zoom out, aspect ratio fit
- [ ] Has wavelengths
- [ ] Has all metadata
- [ ] Has geo transform
- [ ] Stretch builder works
- [ ] Band chooser works (greyscale and colour map)
- [ ] Georeferencing
    - Note: linking views for bad warps may be broken
- [ ] Similarity transform tooling works
- [ ] Continuum removal
- [ ] Saving works

### NetCDF

All items above verified as of last update. Remaining notes:

- Linking views for bad warps may still be broken.
- Bug: right after saving a georeferenced dataset and re-opening it, it
  may not report that views can be linked.

### PDS4

Most items verified. Outstanding items:

- Multi-threaded band math — not yet verified.
- All panes / zoom — not yet verified.
- Has wavelengths — not yet verified (some PDS4 files lack wavelength metadata;
  a prompt to upload wavelengths from a text file is planned).
- Has all metadata — partial: data-ignore value detection added; upload
  wavelengths from file is planned (see below).
- Georeferencing: loading and saving GCPs not yet verified.
- Continuum removal — not yet verified.

**Planned: wavelength upload for PDS4**

When a PDS4 file has no wavelengths, prompt the user to upload them from a
text file. The upload dialog must let the user specify: delimiter, whether a
header row exists, whether wavelengths are in columns or rows, which column or
row contains the wavelengths, and wavelength units. Show validation results
and prevent confirmation if the wavelengths do not match the band count. Use
the existing import-spectra-text dialog as the starting point.

### PDS3

- Cannot open `.dat` files.
- Planned: try-your-luck PDR file opener.

### JP2

- Ask whether JP2 files always have an associated wavelength or are assumed RGB.
- Has wavelengths — not yet verified.
- Has all metadata — not yet verified.
- Continuum removal — not yet verified.
- Saving — not yet verified.

---

## QA Checklist Before Release

Run through all items below before cutting a new WISER release. Mark items
that have known bugs with a note.

- [ ] All file types and sizes: ROI average works
- [ ] All file types and sizes: colour map can be changed
- [ ] All file types and sizes: colour map works with 1-band images
- [ ] Files can be saved correctly (including subset save)
    - Known: when images are saved from a compute dataset, they do not inherit
      the spatial reference system or data-ignore value from the parent dataset
      (fixed in recent release — verify).
- [ ] Band math works with all functions, file types, and sizes
- [ ] Geographic linking works
- [ ] Clicking context pane and zoom pane properly changes main view, especially
  when rasters are linked
- [ ] Spectrum can be saved, collected, and imported in the spectrum plot
- [ ] Opening all file types at once
- [ ] Stretch builder preserves WISER state
    - Known: stretch builder was broken for large images (float32/GDAL FLOAT16
      list issue — verify fix).
- [ ] "Go to pixel" and "Go to coordinate" work
- [ ] All example plugin types still work
    - How: open `WISER/src/example_plugins/`, go to Settings → Plugins, add the
      path to each example plugin folder.
- [ ] WISER settings dialog works
- [ ] Multiple band math operations in sequence work

---

## Notes on Stretch Builder and Band Math Tests

Stretch builder cache behaviour to verify:

- Caches link states for min/max and slider link.
- After linking, remembers the values provided by the link and does not revert
  to individual values.
- Caches individual min, max, and stretch states.
- Changing the slider alters the correct display band in the raster view.

Band math tests to verify:

- Performing band math and retrieving output datasets.
- Performing operations on band-math output datasets.

