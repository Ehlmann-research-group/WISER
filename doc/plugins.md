# WISER Plugin Support

This document discusses plugin support for WISER, including the desired
features, potential implementation issues, and possible design approaches.

# Desired Features

WISER is intended to be a research tool.  As such, it should be extensible by
researchers as they develop new data processing techniques.

The integration points for WISER extension are as follows:

*   Plugins may be exposed in a "Tools" drop-down menu in the global application
    toolbar.  This is the most general way to extend WISER.  Such tool plugins
    may be written as Python modules, or may be separate programs invoked as
    child processes by WISER.

*   Plugins may be exposed as pop-up context menu entries, when the user e.g.
    right-clicks on a dataset, ROI, spectrum, or other such object in the GUI.
    The plugin is intended to operate on the specific kind of object that was
    selected.  (Example:  A plugin that provides a custom processing operation
    over the pixels in the ROI right-clicked by the user.)

*   Plugins may also expose custom functions in the band-math functionality, if
    users want to provide custom band-math operations for exploring their data.

None of these options are necessarily aimed at the WISER Python Console, since
the console is expected to be able to import Python modules on its own.

# Implementation Questions and Issues

Several implementation questions and issues present themselves with this
functionality.  They fall into various categories.

## Knowledge Required for Implementation

WISER will provide some kind of programmatic API for integrating plugins into
the application.  It is reasonable to expect users to know this API.  It will
be well-documented, and over time we will refine it to be powerful and easy to
use.

**Are we expecting users to know Qt6 and PySide6?**  These libraries are quite
involved, and we probably don't want to require users to know about them.  On
the other hand, if users _are_ familiar with these libraries, we would like the
user to be able to use them to create more sophisticated UIs.

This suggests providing a library of common UI interactions for users to use in
their plugins.  For example:

*   `ds: RasterDataSet = app.choose_dataset_ui()`
*   `spectrum: SpectrumInfo = app.choose_spectrum_ui()`

We want to provide the _minimum barrier to entry_ for people wishing to extend
WISER.

## Plugin Quality

**Do we want to try to isolate WISER from bad plugin behaviors, such as
long-running tasks, infinite loops, and buggy/crashing behavior?**

It would seem desirable to do this.  Because of this, we should consider running
plugins (or giving users the option of running plugins) in separate processes.
Perhaps an option can be provided to turn this on or off, so that
lightweight/reliable plugins can be kept within the WISER process.

WISER needs to provide a long-running-task abstraction for plugins to leverage,
or for WISER to leverage when invoking plugins, to keep them from killing UI
interactivity.  We already need this to support large data files, so this will
be a high priority to build early on, for the sake of usability.

## Dependencies

**How do we reconcile the library dependencies of plugins, with the library
dependencies of WISER?**  WISER has a set of Python dependencies.  Plugins may
have additional dependencies outside of WISER's dependencies.  Also, plugins
may have dependencies that are incompatible with WISER's dependencies.  We need
to consider how to support plugins in these scenarios.

This suggests that WISER should support plugins of two main "flavors":  plugins
that work within the WISER dependencies, and plugins that run out-of-process,
possibly against some separate Python environment.  (A special case of this
could be plugins that run within a Docker container, or that interface with
software running in a Docker container.)

This is further affected by whether WISER is being used in an internal
development setting (where WISER's source code is available to the developer,
and the developer can install other dependencies), or whether it is being used
in a "frozen application" setting (where WISER has been frozen, along with its
dependencies).  In the frozen-app situation, WISER's dependencies cannot be
extended.  (Or, at least, Donnie doesn't know how to do this.)  But, in that
case, WISER can spawn a separate Python process with its own environment and
dependencies.
