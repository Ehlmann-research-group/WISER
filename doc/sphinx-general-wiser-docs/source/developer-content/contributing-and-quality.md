# Contributing and Code Quality

This page covers branching conventions, code review practices, and testing standards for WISER development.

## Branching Strategy

WISER will be using a [trunk based branching strategy](https://trunkbaseddevelopment.com/) with continuous integration.
Release branches will be created to strictly enforce feature freezes. More explanation below.

A trunk based branching strategy means that we have our main branch and when
you want to create a feature, you will open up a feature branch from main and
make your changes in that feature branch. You should constantly pull from `main` to ensure your feature branch is up to
date. This next part is very **IMPORTANT**.
[Continuous integration](https://martinfowler.com/articles/branching-patterns.html#continuous-integration)
means that you should be integrating your changes into `main` very frequently.
We want very small commits to be made into main as this makes it easier to review
changes. Only on very rare occasions should a commit exceed 250 lines of code. As
the article linked above says, you will 'need to get used to the idea of reaching
frequent integration points with a partially built feature'. You will also have
to find a way to make sure your feature is gated from production until it is ready.
We use feature flags to do this.

We should have good confidence that the code we merge into main is not buggy.
To have this confidence we need to ensure our testing
suite has good coverage and handles edge cases well. Currently, I would not say
WISER's testing suite has good coverage. This is due to the fact that the core body
of WISER was not written with GUI tests and this issue was compounded by how hard it is to write GUI tests for PySide2
applications. There is currently no good
GUI testing suite for PySide2 applications.

Next we have release branches. Release branches will be made from main. When
release branches are made, there should no longer be any new features added
to them. All future commits on the release branch should be for fixing bugs.
After each commit is made on the release branch, you will have to merge the
release branch into `main` so main has that fix.

---

## Code Review and Quality Guide

This document explains how **code review** and **code submittal** work in WISER.  
Its purpose is to help contributors and maintainers write consistent, maintainable, and
readable code - and to make the review process efficient and collaborative.

## Overview

Code review and submittal in WISER are centered on producing **high-quality, maintainable code**
that aligns with the project’s design philosophy and long-term vision.  
This guide covers:

- [Code Review Guidelines](#code-review-guidelines)
- [Commit Message Standards](#commit-message-standards)
- [Commenting Style](#commenting-style)
- [Naming and Code Style](#naming-and-code-style)

All pull requests (PRs) should be tied to an **existing issue**.  
If no issue exists, **create one first** - we provide templates for common issue types to make this easy.

(code-review-guidelines)=

## Code Review Guidelines

Code review in WISER focuses on ensuring that contributions meet both **functional** and **design** expectations.

### What We Look For

| Category                          | Description                                                                                                                         |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Maintainability & Readability** | Is the code easy to understand and modify? Are variables and functions well named?                                                  |
| **Design Quality**                | Is the code modular, well-structured, and aligned with WISER’s architecture?                                                        |
| **Correctness**                   | Does the code do what it’s supposed to do? Are there tests verifying its behavior?                                                  |
| **Simplicity**                    | Can the code be simplified without losing clarity or function?                                                                      |
| **Style Conformance**             | Are naming conventions, comment formats, and indentation consistent with the project’s style?                                       |
| **Documentation Needs**           | Does this change need documentation? Should it be added before or after merging? (It’s okay to open a second PR for docs or tests.) |
| **Performance**                   | Are there any obvious inefficiencies or bottlenecks?                                                                                |
| **Duplication**                   | Does similar code already exist elsewhere? If so, can it be reused?                                                                 |
| **Scope & Focus**                 | Is the code single-purpose? Avoid adding multiple unrelated changes in one PR.                                                      |

### PR Expectations

- Every PR should have a **clear purpose** and be linked to an **issue**.
- Keep PRs **small and focused**. Avoid exceeding a few hundred lines of changes where possible.
- If documentation or tests are pending, you may open a **follow-up PR**.
- Reviewers are encouraged to comment on **design clarity, naming, and simplicity** just as much as functionality.

(commit-message-standards)=
## Commit Message Standards


### Format

A proper commit message should look like this:

> Short subject line (max 50 chars)
>
>Detailed body explaining what changed and why.

Wrap lines at 50 characters.

We use semantic commit messages for the title
and summary. It's described [here](https://sparkbox.com/foundry/semantic_commit_messages).
Use the below prefixes at the start of each PR merge
commit title (it would also help the reviewer to
do this for all other commit messages):

- feat:     (feature)
- fix:      (fix)
- docs:     (documentation)
- style:    (style change)
- refactor: (code refactor)
- perf:     (performance improvement)
- test:     (adding tests)
- hotfix:   (quick urgen fix)
- build:    (changes to build system)
- ci:       (changes to continuous integration)
- chore:    (miscellaneous task)
- rev:      (revision)

### Guidelines

- Keep commit messages **clear and concise**.
- Describe **what** was done and **why**.
- Long, detailed messages are helpful during PR review but are less important after a squash merge.
- Avoid long unbroken lines; insert newlines around 50 characters.

### Merge Commits

When squash merging a PR:

- The **PR title** becomes the **merge commit title**, followed by the PR number (e.g.,
  `feat: add spectral viewer #298`).
- The **PR description** serves as the commit body.
- After merging:
    - **Delete your branch** both locally and on GitHub.
    - **Pull from `main`** to sync your local repository.

(commenting-style)=

## Commenting Style

Comments clarify *why* code exists — not just *what* it does.

> Reference: [Do’s and Don’ts of Commenting Code](https://blog.openreplay.com/dos-and-donts-of-commenting-code/)

### General Principles

- **Explain the “why.”** Use comments to clarify design decisions, algorithms, or tricky logic.
- **Avoid redundancy.** Don’t restate what the code already clearly expresses.
- **Comment for others.** Write for future maintainers and collaborators.
- **Use inline comments sparingly.** Prefer comments at function or block level for clarity.
- **At API boundaries**, explain how to use or extend the code.

### Formatting Conventions

| Type                          | Example                                                                    | Description                                |
|-------------------------------|----------------------------------------------------------------------------|--------------------------------------------|
| **File/Module Header**        | `"""Brief description of what this file/module does and why it exists."""` | At the top of every file.                  |
| **Class/Function Docstrings** | `"""Explain purpose, behavior, and key parameters."""`                     | Triple quotes.                             |
| **Block Comments**            | `"""Describe logic behind a block of code."""`                             | Triple quotes.                             |
| **Inline Comments**           | `# Explain non-obvious line or condition`                                  | Use `#` sparingly for complex expressions. |

---

(naming-and-code-style)=

## Naming and Code Style

Code should be **self-describing**: good names make code easier to read and maintain.

### Python Naming Conventions

| Element       | Convention   | Example          |
|---------------|--------------|------------------|
| **Classes**   | `PascalCase` | `ImageProcessor` |
| **Functions** | `snake_case` | `load_dataset()` |
| **Variables** | `snake_case` | `spectral_band`  |
| **Constants** | `ALL_CAPS`   | `MAX_CACHE_SIZE` |

### GUI Widget Naming

Since WISER uses **PySide2**, a wrapper around **Qt** (a C++ library), some functions may follow Qt’s C++ naming
conventions - this is acceptable when overriding or subclassing Qt components.

To maintain clarity in GUI code, widgets should use consistent prefixes:

| Widget Type | Prefix   | Example                |
|-------------|----------|------------------------|
| Push Button | `btn_`   | `btn_apply_stretch`    |
| Label       | `lbl_`   | `lbl_filename`         |
| Widget      | `wdgt_`  | `wdgt_spectral_viewer` |
| Combo Box   | `cbox_`  | `cbox_bandselector`    |
| Tool Button | `tbtn_`  | `tbtn_zoom_in`         |
| Line Edit   | `ledit_` | `ledit_path_input`     |
| Spin Box    | `sbox_`  | `sbox_frame_index`     |

### Private and public attributes/methods

In a class, a private attribute (variable) should have an underscore infront of it. If it does not
have an underscore infront of it, then it's assumped to be public. Example:

```python
self._update_table  # Private
# vs
self.update_table  # Public
```

Python does not enforce variables that have an underscore prefix to be private,
but it is common practice to not access them directly. If you need to, consider
first why the variable is private and then make a getter for it.

The same naming convention rules apply for private and public methods (functions).

### Enforced Code Style

Python formatting is **enforced automatically**:

- Style rules are defined in **`pyproject.toml`**.
- They are validated by **pre-commit hooks** and **GitHub Actions**.
- Always run pre-commit checks locally before pushing to avoid CI failures.

## Code Design

Code design is very important to the maintainability of code and how hard
future refactors will be. Making a set of rules that gives us perfect maintainability
and minimal refactors seems impossible, so these rules may be updated as we see fit.

### General Design Rules

1. Try to keep functions backwards compatible. We do not want to break something that
   a plugin (or internal WISER functionality) relies on.
2. Document contracts, not just parameters
    - What does this function guarantee? What does it expect?
    - E.g. “Raises ValueError if array has NaNs.”
3. Use exceptions over error codes
    - Return useful values, raise on error.
    - Don’t overload **None** or **False** to mean “something exploded”. You can still return **None** or **False** just
      not in-place of an error.
4. Use the logging module with module-level loggers.
    - Don’t log at ERROR for things the user can reasonably fix (e.g. invalid user input).

## Summary

In WISER, code review and submission are designed to:

- Maintain **clarity**, **simplicity**, and **design consistency**.
- Encourage **collaboration** and **shared responsibility** for quality.
- Keep the codebase healthy, documented, and performant.

By following these guidelines, we ensure that WISER remains sustainable and welcoming to contributors - now and in the
future.

---

## Testing Standards

## Purpose & Scope

Reliable testing is essential to maintaining a stable and trustworthy codebase. Our tests protect against regressions
when code changes, enable confident refactoring, and increase trust in the correctness of new features. Tests should be
treated as production-grade code: readable, maintainable, and updated as the software evolves.

This guide covers our approach to all tests that are in WISER which mainly include **unit tests**, **functional tests**,
**integration tests**, **end-to-end tests**, and **performance tests**. Each plays a different role in verifying system
correctness, and contributors are expected to write tests appropriate to the type of change they are making.

This document applies to **all contributors**, including new contributors submitting their first PR, committers
implementing new features, and maintainers reviewing contributions.

## Testing Philosophy / Principles

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

We do **not** chase coverage numbers for their own sake. Instead, your tests should meaningfully cover the behavior of
the feature you are implementing, which will generally result in good functional coverage naturally.

## Test Organization

Tests currently live at:

_src/tests/_

As of **11/05/2025**, we do not yet organize tests by type (unit, integration, etc.), but this may be introduced later
as the test suite grows.

### Naming & Structure (pytest)

- Test filenames must start with:  
  `test_*.py`
- If the test works y by clicking through gui elements and only
  tests one piece of functionality, put `_gui` at the end. If it tests
  the interface between two features put `_integ` at the end. We may change or get rid of this in the future in place of
  pytest markers.
- Currently, there are no hard set rules dictating what to name the rest of the file, but generally it should be make
  clear what is being tested and any specifics about what is being tested.
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

CD runs a WISER build in test mode. Example on MacOSX

```bash
./WISER_Bin --test_mode
```

The `--test_mode` executes `pytest .` against _src/tests_ inside the bundled distribution.

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
  Note: CD currently tests on the same runner that performs the build. In the future we plan to test distributed
  artifacts on separate machines (macOS code signing poses additional constraints).

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

## File Type Testing Checklist

When adding support for a new file type, verify the following behaviours after
the class is integrated. Go through `RasterDataImpl` and confirm each function
still makes sense with the new class and that the class can remain
multi-threaded.

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
