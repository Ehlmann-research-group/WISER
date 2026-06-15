# Plugin System Internals

This page documents how WISER *itself* discovers, loads, registers, and runs
plugins. It is written for WISER core developers working on the plugin
machinery. If you are writing a plugin, start with the user-facing
{doc}`Extending WISER <../extending-wiser/index>` guide instead; for the
dependency model and its implications, see
{doc}`Plugin Dependencies <plugin-dependencies>`.

## Overview

WISER exposes three plugin integration points, each backed by a base class in
`wiser.plugins`:

| Integration point | Base class | When WISER calls into it |
| --- | --- | --- |
| Tools menu | `ToolsMenuPlugin` | Once, at startup, while building the Tools menu |
| Context menus | `ContextMenuPlugin` | Every time a context menu is constructed |
| Band Math | `BandMathPlugin` | When the Band Math dialog collects available functions |

All three are plain Python classes that subclass one of these base types. WISER
loads them **in-process** (see [Execution model](#execution-model)) and keeps a
single live instance of each configured plugin for the lifetime of the
application.

## Plugin base types

The base types and the context enum live in
[`src/wiser/plugins/types.py`](../../../../src/wiser/plugins/types.py) and are
re-exported from [`src/wiser/plugins/__init__.py`](../../../../src/wiser/plugins/__init__.py):

- `Plugin` — the `abc.ABC` root type for every plugin.
- `ToolsMenuPlugin.add_tool_menu_items(tool_menu, wiser)` — add `QAction`s or
  submenus to the Tools `QMenu`.
- `ContextMenuPlugin.add_context_menu_items(context_type, context_menu, context)`
  — add entries to a context menu, based on the `context_type`.
- `BandMathPlugin.get_bandmath_functions()` — return a
  `Dict[str, BandMathFunction]` of custom band-math functions.

`BandMathFunction` (the per-function implementation returned by a
`BandMathPlugin`) is defined separately in
[`src/wiser/bandmath/types.py`](../../../../src/wiser/bandmath/types.py); its
`analyze()` and `apply()` methods are abstract.

The `ContextMenuType` enum (`RASTER_VIEW`, `SPECTRUM_PLOT`, `DATASET_PICK`,
`SPECTRUM_PICK`, `ROI_PICK`) is the de-facto contract between WISER and
context-menu plugins: it determines both *when* a plugin is called and *what*
keys appear in the `context` dict (see
[Per-type registration](#per-type-registration)).

## Discovery and configuration

A plugin is identified by two pieces of configuration, defined in
[`src/wiser/gui/app_config.py`](../../../../src/wiser/gui/app_config.py):

- `plugin_paths` *(list of directories)* — added to `sys.path` so plugin
  modules and their dependencies are importable.
- `plugins` *(list of fully-qualified class names, FQCNs)* — e.g.
  `example_plugins.bandmath_plugin.SpectralAnglePlugin` — the classes WISER
  instantiates at startup.

Users populate these through **Settings → Plugins**, implemented in
[`src/wiser/gui/app_config_dialog.py`](../../../../src/wiser/gui/app_config_dialog.py).
When a user adds a plugin by file, `_discover_plugin_classes()` /
`_derive_paths_and_module()`:

1. Walk parent directories looking for the nearest `__init__.py` to find the
   package root, and derive the FQCN of the module.
2. Temporarily prepend the base directory to `sys.path`, import the module, and
   scan it with `inspect`.
3. Collect every class that is an `issubclass` of one of the `PluginBases`
   (`ContextMenuPlugin`, `ToolsMenuPlugin`, `BandMathPlugin`).

The **Verify All** button attempts to load and initialize every registered
plugin and surfaces failures (details go to the WISER log).

There is no manifest file: discovery is reflection-based, keyed entirely off
the base-class hierarchy.

## Load lifecycle

All loading happens in `App._init_plugins()` in
[`src/wiser/gui/app.py`](../../../../src/wiser/gui/app.py) (around line 434),
called during application startup:

```python
plugin_paths = self._app_state.get_config("plugin_paths")
for p in plugin_paths:
    if not os.path.isdir(p):
        logger.warning(f'Plugin-path "{p}" doesn\'t exist; ignoring')
        continue
    if p not in sys.path:
        sys.path.append(p)          # appended → searched AFTER WISER's own paths
```

After the paths are registered, two groups of plugins are instantiated:

- **Permanent plugins** — built-ins (`ContinuumRemovalPlugin`,
  `SavGolayPlugin`) instantiated directly. They are kept as plugins
  deliberately, as living examples of the API.
- **User plugins** — each configured FQCN is instantiated via
  `plugins.utils.instantiate()` in
  [`src/wiser/plugins/utils.py`](../../../../src/wiser/plugins/utils.py), which
  splits the FQCN, `importlib.import_module()`s the module, and calls the class
  with no arguments.

Every instance is validated with `plugins.utils.is_plugin()` and stored in the
`ApplicationState` via `add_plugin()` /
`get_plugins()` ([`src/wiser/gui/app_state.py`](../../../../src/wiser/gui/app_state.py)).
`get_plugins()` is the single source of truth that the three registration paths
below iterate over.

(per-type-registration)=
## Per-type registration

### Tools-menu plugins

Tools-menu registration is **push-based and one-shot**. During
`_init_plugins()`, immediately after a plugin is added, WISER checks its type
and lets it contribute to the Tools menu:

```python
if isinstance(plugin, plugins.ToolsMenuPlugin):
    plugin.add_tool_menu_items(self._tools_menu, self._app_state)
```

The plugin receives the live Tools `QMenu` and the `ApplicationState`, and adds
its actions/submenus then. Because this only runs at startup, Tools-menu
contributions are fixed for the session.

### Context-menu plugins

Context-menu registration is **pull-based and repeated**: WISER calls plugins
*every time* it builds a context menu. The dispatcher is
`add_plugin_context_menu_items()` in
[`src/wiser/gui/plugin_utils.py`](../../../../src/wiser/gui/plugin_utils.py):

```python
for plugin_name, plugin in app_state.get_plugins().items():
    if isinstance(plugin, plugins.ContextMenuPlugin):
        context = kwargs.copy()           # each plugin gets its own copy
        context["wiser"] = app_state
        context["app_services"] = app_services
        try:
            plugin.add_context_menu_items(context_type, menu, context)
        except:
            logger.exception(...)          # one bad plugin can't break the menu
```

A single menu may trigger several calls with different `context_type` values
(e.g. `RASTER_VIEW`, then `DATASET_PICK`, then `ROI_PICK`). The keys present in
`context` depend on the `context_type`:

| `context_type` | Extra `context` keys (besides `wiser`, `app_services`) |
| --- | --- |
| `RASTER_VIEW` | `dataset`, `display_bands` |
| `SPECTRUM_PLOT` | *(none)* |
| `DATASET_PICK` | `dataset`, `display_bands`, `ds_coord` |
| `SPECTRUM_PICK` | `spectrum` |
| `ROI_PICK` | `dataset`, `display_bands`, `roi`, `ds_coord` |

Each plugin is handed a **copy** of the context, so a misbehaving plugin that
mutates its context cannot affect the others.

### Band-math plugins

Band-math functions are collected on demand by `get_plugin_fns(app_state)` in
[`src/wiser/gui/util.py`](../../../../src/wiser/gui/util.py) (line 651), invoked
when the Band Math dialog is set up
([`src/wiser/gui/bandmath_dialog.py`](../../../../src/wiser/gui/bandmath_dialog.py)).
It iterates **all** plugins and calls `get_bandmath_functions()` on each inside
a `try/except` — plugins that don't implement the method are simply skipped, so
no explicit `isinstance` check is needed. Returned function names are
lower-cased (band-math is case-insensitive) and duplicates across plugins
produce a warning, with the last definition winning.

(execution-model)=
## Execution model

Plugins run **in WISER's own process and Python interpreter**. WISER does not
spawn a new process or interpreter to load or run a plugin — `instantiate()`
imports the plugin module directly into the running process, and every plugin
method (menu callbacks, band-math `apply()`, etc.) executes on WISER's threads.

This has two consequences that are documented in detail on the
{doc}`Plugin Dependencies <plugin-dependencies>` page:

- Plugins share the interpreter's single set of imported modules, so they
  cannot use a package version that conflicts with one WISER has already
  imported.
- Because plugin paths are *appended* to `sys.path`, WISER's own dependencies
  resolve first and take precedence.

It also means a long-running or blocking plugin call runs on WISER's threads;
robust offloading of plugin work to background processes is an open design
question (see {doc}`Design Documents <design-documents>`).

## Exception isolation

The plugin machinery is defensive at each boundary so that a single faulty
plugin degrades gracefully rather than crashing WISER:

- **Instantiation** — `_init_plugins()` wraps each `instantiate()` call in
  `try/except`, logging and skipping plugins that fail to load. Instances that
  load but are not recognized plugin types are skipped via `is_plugin()`.
- **Context menus** — `add_plugin_context_menu_items()` catches and logs any
  exception raised while a plugin populates the menu.
- **Band math** — `get_plugin_fns()` swallows exceptions per plugin while
  collecting functions.
- **`@log_exceptions`** — `wiser.plugins.log_exceptions` is a decorator plugin
  authors can apply to their own methods so that any exception is written to the
  WISER log instead of being lost.

## API versioning

There is currently **no explicit plugin API version negotiation** — no version
attribute on the base classes, and no compatibility check at load time. Plugins
are expected to target the WISER version they were written against. In practice
the `ContextMenuType` enum is the closest thing to a compatibility surface: new
context types can be added without breaking existing plugins, because a plugin
only acts on the `context_type` values it recognizes. Introducing a real
versioning scheme is a known future improvement.
