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

---

## Batch Processing

Researchers frequently need to apply the same processing step — a plugin algorithm,
a band-math expression, a spectral analysis — across many datasets at once. Today,
WISER handles each operation interactively and sequentially on the main thread,
which does not scale to multi-file workflows and can freeze the UI on large inputs.

The goal of batch processing is to let users define a set of tasks, submit them
as a queue, track their progress, and cancel individual runs — all without
interrupting the rest of the application. This capability is needed in two areas:
running plugins over collections of datasets, and applying band-math expressions
to folders of images. The band-math is case is already implemented.

### Known Technical Challenges

GDAL `Dataset` objects are not serializable (cannot be pickled) and cannot be
memory-mapped. Windows does not support `fork`, so process spawning with GDAL
objects is non-trivial. Options being explored:

- Pass file paths to worker processes and re-open datasets there.
- Use shared memory for raw array data extracted before spawning.

Relevant background:

- [GDAL and multiprocessing](https://gdal.org/tutorials/multiprocessing.html)
- [Pickle error with GDAL Dataset](https://gis.stackexchange.com/questions/361703/pickle-error-with-gdal-dataset)
