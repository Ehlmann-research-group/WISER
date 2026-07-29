# Design Documents

These documents capture requirements, proposed designs, and architecture decisions for features in development or under
consideration.

## Plugin Architecture and Design

> **Note:** This section captures *open* design questions about plugin
> isolation and quality. For how WISER actually loads and runs plugins today,
> see [Plugin System Internals](plugin-system.md); for the dependency model,
> known issues, and the out-of-process future direction, see
> [Plugin Dependencies](plugin-dependencies.md). For how to *write* a plugin,
> see the [Extending WISER](../extending-wiser/index) section.

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

**Are we expecting users to know Qt5/6 and PySide2/6?**  These libraries are quite
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
04/14/2026, users just have to create their PySide2/6 widget and show it, but a
separate process can't easily do this.

WISER needs to provide a long-running-task abstraction for plugins to leverage,
or for WISER to leverage when invoking plugins, to keep them from killing UI
interactivity. We already need this to support large data files, so this will
be a high priority to build early on, for the sake of usability.

### Dependencies

How WISER reconciles plugin dependencies with its own — the `sys.path`
precedence mechanism, why it follows from the in-process execution model, the
known PyInstaller submodule-pruning issue and its fix, and the proposed
out-of-process (`uv`) future direction — is documented in
[Plugin Dependencies](plugin-dependencies.md).

The open design question that remains is whether to support a second class of
plugin that runs **out of process** against a separate Python environment
(possibly a Docker container), so that plugins with dependencies *incompatible*
with WISER's bundled set can be supported. As of 04/14/2026 such plugins are not
supported.

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
