# Design Documents

These documents capture requirements, proposed designs, and architecture decisions for features in development or under
consideration.

## Plugin Architecture and Design

> **Note:** This is an internal design document discussing plugin isolation,
> dependency management, and implementation considerations. It is not the
> user-facing plugin development guide. For documentation on how to write
> plugins for WISER, see the [Extending WISER](../extending-wiser/index) section.

This document discusses plugin support for WISER, including the desired
features, potential implementation issues, and possible design approaches.

### Desired Features

WISER is intended to be a research tool. As such, it should be extensible by
researchers as they develop new data processing techniques.

The integration points for WISER extension are as follows:

* Plugins may be exposed in a "Tools" drop-down menu in the global application
  toolbar. This is the most general way to extend WISER. Such tool plugins
  may be written as Python modules.

* Plugins may be exposed as pop-up context menu entries, when the user e.g.
  right-clicks on a dataset, ROI, spectrum, or other such object in the GUI.
  The plugin is intended to operate on the specific kind of object that was
  selected.  (Example:  A plugin that provides a custom processing operation
  over the pixels in the ROI right-clicked by the user.)

* Plugins may also expose custom functions in the band-math functionality, if
  users want to provide custom band-math operations for exploring their data.

None of these options are necessarily aimed at the WISER Python Console, since
the console is expected to be able to import Python modules on its own.

### Implementation Questions and Issues

Several implementation questions and issues present themselves with this
functionality. They fall into various categories.

### Knowledge Required for Implementation

WISER will provide some kind of programmatic API for integrating plugins into
the application. It is reasonable to expect users to know this API. It will
be well-documented, and over time we will refine it to be powerful and easy to
use.

**Are we expecting users to know Qt5 and PySide2?**  These libraries are quite
involved, and we probably don't want to require users to know about them. On
the other hand, if users _are_ familiar with these libraries, we would like the
user to be able to use them to create more sophisticated UIs.

This suggests providing a library of common UI interactions for users to use in
their plugins. For example:

* `ds: RasterDataSet = app.choose_dataset_ui()`
* `spectrum: SpectrumInfo = app.choose_spectrum_ui()`

We want to provide the _minimum barrier to entry_ for people wishing to extend
WISER.

### Plugin Quality

**Do we want to try to isolate WISER from bad plugin behaviors, such as
long-running tasks, infinite loops, and buggy/crashing behavior?**

It would seem desirable to do this. Because of this, we should consider running
plugins (or giving users the option of running plugins) in separate processes.
Perhaps an option can be provided to turn this on or off, so that
lightweight/reliable plugins can be kept within the WISER process. It should be
noted that supporting running plugins in separate processes will require us to
rethink how the user can create plugins that interact with the GUI. As of
04/14/2026, users just have to create their PySide2 widget and show it, but a
separate process can't easily do this.

WISER needs to provide a long-running-task abstraction for plugins to leverage,
or for WISER to leverage when invoking plugins, to keep them from killing UI
interactivity. We already need this to support large data files, so this will
be a high priority to build early on, for the sake of usability.

### Dependencies

**How do we reconcile the library dependencies of plugins, with the library
dependencies of WISER?**  WISER has a set of Python dependencies. Plugins may
have additional dependencies outside of WISER's dependencies. Also, plugins
may have dependencies that are incompatible with WISER's dependencies. We need
to consider how to support plugins in these scenarios. As of 04/14/2026, WISER
does not support plugins that have incompatible dependencies.

This suggests that WISER should support plugins of two main "flavors":  plugins
that work within the WISER dependencies, and plugins that run out-of-process,
possibly against some separate Python environment.  (A special case of this
could be plugins that run within a Docker container, or that interface with
software running in a Docker container.)

This is further affected by whether WISER is being used in an internal
development setting (where WISER's source code is available to the developer,
and the developer can install other dependencies), or whether it is being used
in a "frozen application" setting (where WISER has been frozen, along with its
dependencies). In the frozen-app situation, WISER's dependencies cannot be
extended.  (This may not be possible in a frozen-app context, but WISER can spawn a separate Python process with its own
environment and
dependencies.)

### Known Plugin Dependency Issue and Fixes

#### The Problem (Pre-release 1.3b1)

When PyInstaller builds a frozen WISER application it recursively resolves
all imports and stores them in `_internal/`. Before release 1.3b1, if a
plugin used a submodule that PyInstaller had pruned (e.g. `scipy.io` when
WISER itself only used top-level `scipy`), that submodule would not be found
at runtime and the plugin would fail to load.

**Bandage solution (releases 1.2b1 and earlier):** The PyInstaller spec was
updated to include all submodules of WISER's Python dependencies explicitly.
This is done in `pyinstaller_hooks/` and ensures submodules are not pruned.
All `rel/1.2b1` and `rel/1.2b1-intelmac` branches (Windows, ARM Mac, Intel
Mac) carry this fix.

To verify the fix:

1. Build a frozen WISER (no need to code-sign or notarize).
2. Add the `pca_plugin` (it depends on `scipy.io`).
3. Confirm no `scipy.io` import error occurs.

**Longer-term direction (under investigation):** Allow plugins to declare
their own Python dependencies (separate from WISER's conda environment) using
`uv` in a subprocess. Rough sketch:

1. A Python bootloader script installs `uv` if not present (downloaded from
   the internet, not bundled with PyInstaller).
2. On plugin load, WISER spawns a subprocess via `uv` that provides the
   plugin's declared dependencies.
3. WISER must be packaged as a wheel (`python -m zipapp`) so it can be
   imported by the plugin subprocess.
4. Plugins that ship as wheels are supported; dev-environment (loose source)
   plugins are also supported.

Constraints identified:

- GDAL and similar packages that require compiled C extensions cannot be
  installed via pip alone — they need conda or system packages.
- `PySide6` is required (over `PySide2`) for this approach because PySide6
  has pip wheels for all major platforms.
- Architecture-specific DLLs can only be installed by pip if the corresponding
  package itself supports pip installation.

Tooling evaluated: Bazel, Nuitka (requires PySide6), Poetry, cx_Freeze.
None has been adopted yet.

#### Replicating the Pre-fix Bug

Add the path to a conda environment's `site-packages` (e.g.
`C:\Users\<user>\anaconda3\envs\plugin_lib\Lib\site-packages`) to plugin
directories, and add `pca_plugin.PCAPlugin` to plugins in settings.
`scipy.io` will not be found because PyInstaller did not include it.

#### Background References

- [Is Python interpreted or compiled?](https://stackoverflow.com/questions/6889747/is-python-interpreted-or-compiled-or-both) —
  Python is first compiled to bytecode and then interpreted, which is faster than pure interpretation.

---

## Batch Processing

> **Status:** In development. Working branch: `feat/batch-proc`.

This document captures the requirements and design notes for batch processing
in WISER.

### Overview

We want a way to queue a batch of tasks and run processes on them. Tasks should
have a progress bar and the ability to be cancelled. This should also allow
many operations that previously blocked the Qt thread to run without hanging.

There are two primary use cases: batch processing for plugins and batch
processing for band math.

### Plugin Batch Processing

#### UI

A "Batch Processing" button in the Tools menu opens a dialog that lets the user
select batch-processing plugins and, for each plugin, select the datasets to run
it on. Dataset selection could be done with a series of dropdowns similar to
the band-math variable binding UI.

#### Backend

- Load all plugin functions and their parameters.
- Spawn a separate process for each plugin run up to a configurable limit; once
  the limit is reached, queue remaining runs per process.
- Shared memory will be required to pass dataset data to worker processes.

#### Batch Processing Plugin Class

A batch-processing plugin class must:

- Accept inputs of types: image cube, image band, spectrum, or number. Any
  number of inputs is allowed, but types must be declared upfront so the GUI can
  present appropriate binding controls.
- Handle its own processing (i.e., be serialisable and runnable in a subprocess).
- Declare the type(s) of objects it produces as output.

The plugin must expose:

- A function listing input parameters (name + type) — used to build the GUI.
- A function listing output parameters (name + type) — used to determine where
  to save results.

#### Order of Work

1. Confirm that multiprocessing in WISER can share dataset data across
   processes.
2. Write the `BatchProcessingPlugin` base class; confirm it is picklable and
   can be executed in a subprocess.
3. Write the GUI for the batch processing dialog.

### Band Math Batch Processing

#### UI Changes

- A toggle button in a new row (Row A, index 0) enables/disables batch mode.
  When enabled, additional controls appear.
- **Row A**: "Input Folder Path:" label + disabled read-only line edit (display
  only) + "Select Folder" push button.
- **Row B**: "Output Folder Path:" label + similar folder picker. Two check
  boxes: "Load Into WISER" and "Save on File System". At least one must be
  selected before "Create Batch Job" is enabled.
- The "Result name (optional):" label changes to "Result suffix (required):" in
  batch mode.
- A "Create Batch Job" button (disabled when batch mode is off) appends the
  current expression and variable bindings to a batch job table. The expression
  is not reset after adding. Duplicate job detection with a confirmation prompt.
- When batch mode is enabled, two additional variable types appear: "Image
  (batch)" and "Image band (batch)". The "Image band (batch)" binding lets the
  user choose between band number or wavelength value.

#### Band Math Backend

- Add `BandMathVariableType` entries for batch inputs.
- `BandMathExprInfo` must detect batch inputs and handle them without changing
  the standard (non-batch) code paths.
- When evaluating, extract each file from the input folder and feed it into the
  standard band-math function. Results go to the output folder and/or are loaded
  into WISER as specified.
- New state variables needed: `_input_folder`, `_output_folder`,
  `_load_into_wiser`, `_save_on_filesystem`.

#### Known Technical Challenges

GDAL `Dataset` objects are not serialisable (cannot be pickled) and cannot be
memory-mapped. Windows does not support `fork`, so process spawning with GDAL
objects is non-trivial. Options being explored:

- Pass file paths to worker processes and re-open datasets there.
- Use shared memory for raw array data extracted before spawning.

Relevant background:

- [Picklability of GDAL files](https://chatgpt.com/c/689ce320-4f9c-8330-8bd0-2e97b3621e77)
- [Spawn process copying data](https://chatgpt.com/c/689ce040-289c-8323-89b6-8c6987d448a1)

### Notes on Concurrency

From Nia (internal discussion):

> I typically use `from concurrent.futures import ProcessPoolExecutor,
> ThreadPoolExecutor`. QThreads are useful if you need to emit Qt signals, but
> are otherwise unnecessary. For most compute work, Python's standard thread and
> process pool executors are preferred.

---

## Georeferencer Design

This document captures requirements and proposed design for the WISER
Georeferencer tool.

### Problem Statement

Allow users to apply a spatial reference system and geo transform to an image
by adding ground control points (GCPs) that map pixel coordinates in the
target image to known geographic coordinates. This can be done using a
reference image that already has spatial information, or by manually entering
reference points.

### Scope

The tool is confined to its dialog window and its supporting classes:

- **GeoreferencerPane**: handles UI updates when the user clicks the target
  dataset to place GCPs.
- **GeoReferenceTaskDelegate**: handles the logic of adding GCPs when the user
  clicks between the target and reference image.

### Goals

- The user can add GCPs to the target image.
- GCPs can be added using either a reference image or manual entry.

### Background

WISER previously had no system to attach geographic information to datasets.
The Georeferencer is the first step in doing so. It is particularly useful for
hyperspectral datasets that lack embedded spatial reference information.

### Functional Requirements

- Open any dataset currently loaded in WISER as the target image.
- Open any dataset with spatial information as the reference image.
- Handle out-of-memory datasets during georeferencing.
- Georeferencing computation must not block the main thread.

### Proposed Design

#### High-Level Architecture

- **GeoreferencerDialog**: top-level dialog containing two GeoreferencerPane
  instances and the GeoReferencerTaskDelegate.
- **GeoreferencerPane** (x2): one for the target image, one for the reference
  image. Updates display when user clicks to place a GCP.
- **GeoReferencerTaskDelegate**: coordinates GCP placement between the two
  panes.

#### Data Model

To be documented as implementation proceeds.

#### UI/UX

Mockups to be added.
