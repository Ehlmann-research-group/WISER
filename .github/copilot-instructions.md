# Copilot instructions — WISER

WISER is a desktop GUI for analyzing hyperspectral imagery, built on PySide6 + GDAL +
NumPy. Read `AGENTS.md` first for architecture and build commands; this file defines how
to **review** a change.

Three facts shape every review here:

1. **The data is enormous.** Real scenes are hyperspectral cubes of 10-100 GB. The
   repository's test fixtures are tiny by comparison (the largest tracked ones are a few
   MB), so code that passes CI can still make the application unusable in the field.
2. **It ships to scientists on three platforms.** macOS (Intel + Apple Silicon),
   Windows, and Linux, as frozen PyInstaller bundles — not as a library developers
   `pip install`. A change that works from source can still break the shipped product.
3. **The team is 1-2 developers.** Prefer the simplest change that is correct. Do not
   ask for speculative abstraction, and do not request a rewrite where a fix will do.

## Review priorities

Comment in this order. If you can only make a few comments, make them from P0.

### P0 — will produce a wrong number, a crash, or an unusable app

- **Silent scientific error.** A value that is wrong rather than absent: dropped
  no-data handling, an ignored bad-bands mask, a unit assumption, a flipped axis. These
  are the worst defects in this codebase because nothing fails — a scientist publishes
  the number. See `.github/instructions/raster-and-science.instructions.md`.
- **Unbounded memory against a real cube.** Any new full-cube materialization, dtype
  promotion, or copy on the data path. See "Scale analysis" below.
- **Blocking the GUI thread.** Compute or I/O on the Qt main thread freezes the whole
  application for the minutes-to-hours a real cube takes.
- **Platform-specific breakage.** Code that only works on the author's OS, or only from
  source and not in a frozen bundle. See
  `.github/instructions/packaging-and-platform.instructions.md`.

### P1 — will bite a user or a maintainer soon

- Resource lifecycle: leaked GDAL handles, shared memory, temp files, storage leases.
- Cancellation and progress: a long task the user cannot stop or watch.
- Backward compatibility of anything persisted — `.wiserproj`, `wiser-conf.json`,
  spectral library and ENVI header output, the plugin base classes.
- Swallowed failure: a new `except Exception` that leaves the user with no error and
  the program in an undefined state.
- Missing or weak tests on a code path that can be tested headlessly.

### P2 — worth saying once, briefly

Naming, structure, duplication, docstrings. One comment, not a campaign.

## Scale analysis (required when the diff touches the data path)

"The data path" means anything under `src/wiser/raster/`, `src/wiser/bandmath/`,
`src/wiser/utils/` (task/scheduler/storage), or any GUI code that reads pixels.

You cannot test at 100 GB — the large scenes are gitignored and not in your checkout.
So reason about it explicitly and say so in the comment. For each new or modified data
access, state:

- **Peak memory as a formula**, not a number: in terms of `bands x rows x cols x
  itemsize`. A 60 GB `float32` cube becomes 120 GB the moment something calls
  `.astype(np.float64)`. Flag every dtype promotion on a full array.
- **How many full passes over the cube** the change adds. Each pass on a 60 GB file is
  minutes of disk I/O. Two passes that could be one is a real finding.
- **Whether access matches the file's interleave.** Pulling one spectrum (all bands at
  one pixel) from a BSQ file touches every band plane; pulling one band from a BIP file
  touches every pixel. An access pattern that fights the interleave is orders of
  magnitude slower, not a few percent.
- **Whether the result is cached, and whether the cache is bounded.**
  `RasterDataSet.get_image_data()` caches the *entire cube*. A new call to it, or a new
  `add_cache_item`, is a potential out-of-memory on a real scene.
- **Whether the work is chunked and scheduled.** Heavy work belongs in a `SemanticTask`
  / `TaskStage` pipeline, not in a dialog's button handler. See
  `.github/instructions/concurrency-and-memory.instructions.md`.

If the change scales linearly in cube size with no chunking, say so plainly and estimate
what it does at 50 GB. That is more useful than a style comment.

## Edge cases to probe on any data-handling change

Check the diff against these, and name the specific one you think breaks:

- Single-band datasets, and cubes with one row or one column.
- Datasets with **no** CRS or geotransform, and datasets with a rotated (non-north-up)
  geotransform.
- All-no-data regions, and a cube where an entire band is no-data.
- A no-data value that is `0`, `NaN`, or a valid measurement elsewhere in the cube.
- Bad bands: a contiguous run of them, and bad bands at the first or last index.
- Non-monotonic or missing wavelengths, and wavelengths in µm where nm is assumed.
- Integer overflow on index arithmetic: `bands * rows * cols` exceeds `int32` on a real
  cube.
- Empty selections: an ROI of zero pixels, a spectral library with no spectra.
- A file the user deletes or a drive that unmounts mid-task.

## Verifying your review

If you can run code (coding agent, not code review), verify before you assert:

```
make generated
xvfb-run -a bash -lc 'cd src/tests && pytest -s .'
```

Drive the real UI rather than testing functions in isolation. `WiserTestModel`
(`src/test_utils/test_model.py`) boots the application and gives you `load_dataset`,
`import_spectral_library`, and the spectrum-plot accessors. The tracked ENVI fixtures in
`src/test_utils/test_datasets/` are real imagery with real headers — including cubes
with data-ignore values and bad bands (`caltech_15_20_20_data_ignore_bb`,
`caltech_425_6_6_data_ignore`), µm wavelengths (`circuit_4_100_150_um`), and GeoTIFF,
NetCDF, and JP2 samples. Prefer them over synthetic arrays.

To probe large-cube behavior, generate a synthetic cube rather than claiming you tested
at scale: an ENVI file is a flat binary plus a text `.hdr`, so a few-GB sparse file
costs almost nothing to create. Then check that opening and rendering it does not
materialize the whole array — measure peak RSS at two cube sizes and confirm the growth
is sublinear. Never report a scale result you did not measure.

## What not to comment on

- Formatting and import order. `ruff format` and `ruff check` gate CI; do not duplicate
  them. Note that the config ignores `E722`, `F401`, `F403`, and `F405`, so a bare
  `except:` will pass lint — that one is still worth a comment on its merits.
- Files under `src/wiser/gui/generated/` — generated by `make generated` from
  `.ui`/`.qrc` sources. Review the source, never the output.
- The Caltech-era identifiers (`edu.caltech.gps.WISER`, `QSettings("Caltech", ...)`, the
  Windows installer Publisher). They are deliberately frozen; renaming them breaks app
  identity or orphans user settings. Flag any diff that *changes* them.
- Requests for type annotations on untouched surrounding code. The codebase predates the
  current standards; ask only for the lines the diff adds.
- Test coverage numbers as a goal in themselves.

## Comment style

Be specific and short. Name the file, the line, the input that breaks it, and what
happens. "This calls `get_image_data()` inside the band-change handler, so switching
bands on a 40 GB cube loads the full cube on the GUI thread" is useful. "Consider
performance implications" is not.

State your confidence. If you are inferring behavior you could not execute, say so —
a confident wrong comment costs a maintainer more than no comment.
