# Developer Guide

Everything needed to build, test, extend and understand WISER.

If you want to add functionality **without** modifying WISER, you probably want
{doc}`Extending WISER <../extending-wiser/index>` instead — the plugin API adds
Tools-menu actions, context-menu operations and band-math functions with no
rebuild.

---

## Start here

```{list-table}
:header-rows: 1
:widths: 34 66

* - Page
  - Covers
* - {doc}`Environment Setup <environment-setup>`
  - Conda lockfiles, per-platform environments, dependency management, building installers
* - {doc}`Contributing and Code Quality <contributing-and-quality>`
  - Branching, commits, the DCO, review expectations, style and linting
* - {doc}`Testing and QA <testing-and-qa>`
  - Test layout, pytest markers, the GUI test harness, what to write for a change
* - {doc}`CI/CD and Releases <ci-cd-and-releases>`
  - The pipeline, artefacts, versioning, release procedure
* - {doc}`Design Documents <design-documents>`
  - How and when to write one before a large change
```

Code signing is platform-specific: {doc}`macOS <codesign-mac>` ·
{doc}`Windows <codesign-win>`.

---

## Architecture

Read these before changing anything that crosses subsystem boundaries.

```{list-table}
:header-rows: 1
:widths: 34 66

* - Page
  - Covers
* - {doc}`System Design <system-design>`
  - The overall shape of the application: processes, threads, the task system, storage
* - {doc}`Application State <app-state>`
  - `ApplicationState` — datasets, ROIs, spectra, stretches, and the signals that tie them together
* - {doc}`Data Caching <data-caching>`
  - How raster data is held in memory and on disk
* - {doc}`Plugin System <plugin-system>`
  - How plugins are discovered, loaded and dispatched
* - {doc}`Plugin Dependencies <plugin-dependencies>`
  - How a plugin's third-party dependencies are resolved at runtime
```

---

## Subsystem internals

One page per subsystem: the classes involved, the signals they exchange, and
which component owns what.

```{list-table}
:header-rows: 1
:widths: 34 66

* - Page
  - Covers
* - {doc}`Viewport System <viewport-system>`
  - Panes, scroll and zoom state, how the views stay in step
* - {doc}`Rendering Pipeline <rendering-pipeline>`
  - Band data to pixels: normalisation, conditioners, stretches, colormaps
* - {doc}`Stretch Builder <stretch-builder>`
  - The contrast-stretch dialog and the stretch classes behind it
* - {doc}`Band Chooser <band-chooser>`
  - Band selection and colormap handling
* - {doc}`Spectrum Plot <spectrum-plot>`
  - The plot widget, spectrum sources, collection and display
* - {doc}`Band Math Internals <bandmath-internals>`
  - Grammar, parser, analyzer, evaluator, chunking and batch jobs
* - {doc}`Raster Format Dispatch <raster-format-dispatch>`
  - How a file is matched to a loader, and how new formats are added
* - {doc}`Georeferencer Internals <georeferencer-internals>`
  - GCPs, transform fitting, residuals, warping
* - {doc}`CRS Creator Internals <crs-creator-internals>`
  - Building and validating custom coordinate reference systems
* - {doc}`Mosaic Internals <mosaic-internals>`
  - Ingestion, tiling, compositing, materialisation, export
* - {doc}`Save to Project <save-to-project>`
  - The `.wiserproj` format, the save plan, and self-contained bundles
```

{doc}`Code Documentation <code-documentation>` describes the conventions these
pages follow.

---

## Regenerating the tutorial screenshots

Every figure in the {doc}`tutorials <../tutorials/index>` is produced by a
script that drives the real application, so the images cannot drift away from
the UI:

```bash
# make sure the generated Qt modules exist
make -C src/wiser/gui && make -C src generated

# list the scenes
python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py --list

# re-shoot one, or all of them
python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py --only first_look
python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py

# headless Linux
xvfb-run -a python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py
```

Each figure comes from a **scene** — a function that opens the same data the
tutorial uses, drives the same dialogs, and grabs the widget it is describing.
When you change a dialog, re-run its scene and commit the new PNG with the code
change. Adding a figure means adding or extending a scene, not hand-capturing a
screenshot.

Some scenes need a lab dataset that is too large to commit. Each one calls
`require()` and skips itself with a message when the file is absent, so the
scenes that run on bundled fixtures always work:

| Scene prefix | Dataset | Where to get it |
|---|---|---|
| `avng_` | AVIRIS-NG Caltech subset, 551 MB | {doc}`Lab A <../tutorials/labs/lab-aviris-ng-urban>` |
| `cuprite_` | AVIRIS-Classic Cuprite window, 2.05 GB | {doc}`Lab B <../tutorials/labs/lab-cuprite-minerals>` |
| `crism_` | CRISM Jezero MTRDR cube, 640 MB | {doc}`Lab C <../tutorials/labs/lab-mars-crism>` |

Each lab's "Get the data" section is the download recipe, and the files belong
in `src/test_utils/test_datasets/`, where `.gitignore` already excludes them.

```{admonition} Two traps the harness works around
:class: note
`WiserTestModel.run()` ends in `QApplication.quit()`, which in Qt 6 closes
every top-level window — including result windows a task has just opened, such
as the PCA scree plot, and modeless dialogs created with `WA_DeleteOnClose`.
The harness provides `soft_pump()` for use while such a window is on screen,
and swaps in a plain `QMainWindow.closeEvent` so `AppServices` survives the
loop.

`MainView` opens the Savitzky–Golay and smoothing dialogs with `exec_()`, which
blocks a script; the filters scene builds the same dialogs directly instead.

Offscreen, a scroll area never gets the resize event that would refresh its
scrollbar ranges after the image is rescaled, so `frame_region()` sets the
range from the pixmap itself before scrolling. It also runs from inside
`shot(..., frame=BOX)`, after a throwaway `grab()`: the first grab of a scene
is what makes the dock panes claim their space, and framing before that measures
the wrong viewport.
```

---

## Building the documentation

```bash
cd doc/sphinx-general-wiser-docs
pip install -r requirements.txt
make html          # output in build/html
```

The build should be warning-free apart from `autodoc2.dup_item` notices. Treat
a new warning as a broken link.

```{toctree}
:hidden:
:maxdepth: 1

environment-setup
contributing-and-quality
testing-and-qa
ci-cd-and-releases
codesign-mac
codesign-win
design-documents
system-design
app-state
data-caching
plugin-system
plugin-dependencies
code-documentation
viewport-system
rendering-pipeline
stretch-builder
band-chooser
spectrum-plot
bandmath-internals
raster-format-dispatch
georeferencer-internals
crs-creator-internals
mosaic-internals
save-to-project
```
