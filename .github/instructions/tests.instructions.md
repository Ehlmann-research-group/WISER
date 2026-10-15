---
applyTo: "src/tests/**,src/test_utils/**"
---

# Tests

Review test code as production code. Two questions matter more than coverage: does the
test **fail when the behavior breaks**, and does it exercise the path a real user takes?

## Weakened tests are a finding, not a cleanup

When a diff modifies an existing test rather than adding one, check whether the change
fixes the test or hides the bug. Flag, and ask for the justification:

- A tolerance widened (`assertAlmostEqual` places reduced, `rtol`/`atol` raised) without
  a stated numerical reason.
- An assertion made weaker: an exact value replaced by a range, an equality replaced by
  `assertIsNotNone`, a shape check replaced by a truthiness check.
- A `pytest.raises` removed, or narrowed to a broader exception type.
- A new `skip`, `xfail`, or a case dropped from a `parametrize` list.
- An expected value retuned to match the new output rather than derived from what the
  change should produce. If the science result changed, the PR must say why it is now
  correct — a ground-truth fixture comparison is the way to settle it.
- A test that asserts only that nothing raised. That passes against almost any bug.

## Drive the real application

- `WiserTestModel` (`src/test_utils/test_model.py`) boots the app and exposes
  `load_dataset`, `close_dataset`, `import_spectral_library`, `import_ascii_spectra`,
  the collected-spectra accessors, and message-box handling. Use it rather than
  bootstrapping widgets ad hoc — a test that constructs one widget in isolation misses
  the signal wiring that is where GUI bugs actually live.
- Prefer the **real ENVI fixtures** in `src/test_utils/test_datasets/` over synthetic
  arrays. They carry real headers, and several exist specifically to exercise the hard
  cases: data-ignore values and bad bands (`caltech_15_20_20_data_ignore_bb`,
  `caltech_425_6_6_data_ignore`), µm wavelengths (`circuit_4_100_150_um`), rotated
  geotransforms (`caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif`), and GeoTIFF,
  NetCDF, and JP2 formats. Ground-truth outputs exist for MNF, MTMF, decorrelation
  stretch, and linear unmixing — a numerical change should be checked against them.
- **The large scenes are gitignored.** `ang20171108t*`, `f230918t*`, `HRL000040FF*`,
  EMIT, and PACE files are not in the checkout and are not available in CI. A test that
  references one will fail for everyone else. If a change needs large-cube behavior
  covered, the test must synthesize the cube (an ENVI file is a flat binary plus a text
  header, so a sparse multi-GB file is cheap) and assert on **scaling** — peak memory or
  read count at two sizes — rather than on an absolute timing.

## Suite mechanics

- Tests run headless under `xvfb-run`; the local equivalent of CI is
  `make generated && cd src/tests && pytest -s .`. A test needing a real display, a
  window manager, user interaction, or network access will hang or fail in CI.
- Markers are declared in `pyproject.toml` under `--strict-markers`, so an undeclared
  marker is a collection error, not a warning. A new marker needs its `pyproject.toml`
  entry and a stated reason the existing ones (`smoke`, `unit`, `functional`,
  `integration`, `e2e`, `performance`, `slow`, `multiprocessing`, `scheduler`,
  `storage`, `task_manager`, `build`, `regression`) do not fit.
- The per-test timeout is 240 s (`timeout_method = "thread"`). A test that approaches it
  will be flaky on a loaded runner. Flag a new long-running test and ask whether it can
  work on a smaller fixture.
- Naming: files are `test_*.py`, functions `test_*`; `_gui` marks a test that clicks
  through the interface, `_integ` a test of the boundary between two features. The suite
  mixes `unittest` and pytest styles — both are acceptable, consistency within a file is
  not worth a comment.
- `src/tests/conftest.py` caps `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` **before**
  NumPy is imported, mirroring the block in `src/wiser/__main__.py`. Without it, process-
  pool tests oversubscribe cores and fail as `BrokenProcessPool` or an OpenBLAS
  allocation error — order-dependent and hard to diagnose. Flag any import added above
  that block, and any change to one copy that is not made to the other.
- Tests must be isolated and parallel-safe: no reliance on execution order, no shared
  mutable global, no writing into `src/test_utils/test_datasets/`. Temporary outputs go
  to a `tmp_path`.
- Storage and scheduler tests must release their refs and shut down their pools, or they
  leak shared memory and hang the session at teardown.

## Coverage worth asking for

Ask for a test when the diff adds a **failure path** that is currently unasserted — a
malformed header, a missing CRS, an empty ROI, a cancelled task, a file that disappears
mid-read. The repository's own standard is that tests cover typical *and* edge cases,
and the edge cases are where hyperspectral data actually differs from the fixtures.

Do not ask for tests of generated UI code, of trivial getters, or to raise a coverage
percentage. The project explicitly does not chase coverage numbers.
