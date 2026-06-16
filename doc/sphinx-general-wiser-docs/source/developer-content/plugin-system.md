# Plugin System Internals

This page documents how WISER's plugin system works internally — how plugins
are discovered, loaded, instantiated, and wired into the application, how each
of the three plugin types is invoked at runtime, and how plugin dependencies
are resolved against WISER's own.

This is a code-internals reference for WISER developers. The user-facing guide
for *writing* plugins lives in the
{doc}`Extending WISER <../extending-wiser/index>` section; this page covers the
machinery on WISER's side that makes those plugins run.

## Overview

WISER exposes three plugin integration points, each backed by an abstract base
class in `wiser.plugins`:

| Integration point | Base class | Invocation style |
|---|---|---|
| Tools menu | `ToolsMenuPlugin` | Push, once at startup |
| Context menus | `ContextMenuPlugin` | Pull, every time a menu is built |
| Band Math | `BandMathPlugin` | Pull, when the Band Math dialog opens |

A plugin is a plain Python class that subclasses one of these. WISER never
spawns a separate process or interpreter for a plugin — every plugin is
imported into and runs inside WISER's own process (see
[Execution model](#execution-model)).

```{mermaid}
flowchart LR
    CFG["Config\nplugin_paths + plugins (FQCNs)"] --> INIT["App._init_plugins()\napp.py"]
    INIT -->|"sys.path.append(path)"| SP["sys.path\n(plugin paths appended last)"]
    INIT -->|"instantiate(FQCN)"| IMP["importlib.import_module\nplugins/utils.py"]
    IMP --> STORE["ApplicationState\nadd_plugin / get_plugins"]
    STORE --> TOOLS["ToolsMenuPlugin\nadd_tool_menu_items()"]
    STORE --> CTX["ContextMenuPlugin\nadd_context_menu_items()"]
    STORE --> BM["BandMathPlugin\nget_bandmath_functions()"]
```

---

## Plugin Types

**File:** `src/wiser/plugins/types.py` (re-exported from
`src/wiser/plugins/__init__.py`)

All plugin types descend from a single abstract root, `Plugin(abc.ABC)`. The
band-math *function* type lives separately in `src/wiser/bandmath/types.py`,
because it belongs to the band-math engine rather than the plugin API.

```{mermaid}
classDiagram
    direction TB

    class Plugin {
        plugins/types.py
        «abstract»
    }
    class ToolsMenuPlugin {
        plugins/types.py
        +add_tool_menu_items(tool_menu, wiser)
    }
    class ContextMenuPlugin {
        plugins/types.py
        +add_context_menu_items(context_type, context_menu, context)
    }
    class BandMathPlugin {
        plugins/types.py
        +get_bandmath_functions() Dict~str,BandMathFunction~
    }
    class BandMathFunction {
        bandmath/types.py
        «abstract»
        +analyze(args) BandMathExprInfo
        +apply(args) BandMathValue
        +get_description() str
    }

    Plugin <|-- ToolsMenuPlugin
    Plugin <|-- ContextMenuPlugin
    Plugin <|-- BandMathPlugin
    BandMathPlugin ..> BandMathFunction : returns
```

`is_plugin(obj)` (`src/wiser/plugins/utils.py`) is the single predicate WISER
uses to decide whether an object is a recognized plugin — it returns `True`
for instances of any of the three concrete base types. The tuple of base types
is also referenced as `PluginBases` in the settings dialog for discovery.

### ContextMenuType

`ContextMenuType` (an `enum.Enum` in `plugins/types.py`) is the contract between
WISER and context-menu plugins. It tells the plugin *what kind* of context menu
is being built and therefore which keys to expect in the `context` dict:

| Member | Value | Meaning |
|---|---|---|
| `RASTER_VIEW` | 1 | General raster-view operation (not cursor-specific) |
| `SPECTRUM_PLOT` | 2 | General spectrum-plot operation |
| `DATASET_PICK` | 10 | Location-specific dataset operation |
| `SPECTRUM_PICK` | 11 | A specific spectrum was picked |
| `ROI_PICK` | 12 | A specific ROI was picked |

---

## Loading Pipeline

**File:** `src/wiser/gui/app.py` — `App._init_plugins()` (~line 434)

`_init_plugins()` runs once during application startup and is the entry point
for everything below. It performs three steps in order.

### Step 1 — Register plugin paths

```python
plugin_paths = self._app_state.get_config("plugin_paths")
for p in plugin_paths:
    if not os.path.isdir(p):
        logger.warning(f'Plugin-path "{p}" doesn\'t exist; ignoring')
        continue
    if p not in sys.path:
        sys.path.append(p)          # appended → searched AFTER WISER's own paths
```

Each configured directory is **appended** to `sys.path`. The append (rather than
insert) is what makes WISER's own dependencies take precedence — see
[Dependency handling](#dependency-handling).

### Step 2 — Instantiate permanent plugins

A small set of built-in plugins is instantiated directly (not via config), kept
as plugins partly to serve as living examples of the API:

```python
permanent_plugins = [
    ("ContinuumRemovalPlugin", ContinuumRemovalPlugin()),
    ("SavGolayPlugin", SavGolayPlugin()),
]
```

### Step 3 — Instantiate user plugins

Every fully-qualified class name (FQCN) in the `plugins` config list is
instantiated through `plugins.utils.instantiate()`:

**File:** `src/wiser/plugins/utils.py`

```python
def instantiate(fully_qualified_class_name: str) -> Plugin:
    parts = fully_qualified_class_name.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]
    module_obj = importlib.import_module(module_name)
    class_obj = getattr(module_obj, class_name)
    return class_obj()          # instantiated with no arguments
```

Each instance is validated with `is_plugin()` and stored in `ApplicationState`
via `add_plugin(name, plugin)`. Immediately after storage, if the plugin is a
`ToolsMenuPlugin` its menu items are registered (Step "Tools menu" below).

**Controls:**
- Adding plugin paths to `sys.path`
- Instantiating permanent and user plugins
- Storing plugins in `ApplicationState`
- One-shot Tools-menu registration

**Does not control:**
- Discovering which classes are plugins (done by the settings dialog)
- Context-menu and band-math invocation (pull-based, happen later on demand)

### Plugin storage

**File:** `src/wiser/gui/app_state.py`

`ApplicationState` is the registry. `add_plugin(class_name, plugin)` stores the
instance (raising if the name is already registered), and `get_plugins()`
returns a dict copy of all registered plugins. Every runtime registration path
below iterates over `get_plugins()`.

---

## Discovery (Settings → Plugins)

**File:** `src/wiser/gui/app_config_dialog.py`

The configuration that `_init_plugins()` consumes is produced by the
**Settings → Plugins** dialog. There is **no manifest file** — discovery is
reflection-based:

1. `_derive_paths_and_module()` walks parent directories from the chosen `.py`
   file looking for the nearest `__init__.py` to find the package root, then
   derives the module's fully-qualified name.
2. `_discover_plugin_classes()` temporarily prepends the base directory to
   `sys.path`, imports the module, and scans it with `inspect`, collecting every
   class that is an `issubclass` of one of the `PluginBases`
   (`ContextMenuPlugin`, `ToolsMenuPlugin`, `BandMathPlugin`).

The dialog stores two config values (defined in `src/wiser/gui/app_config.py`):

| Config key | Type | Purpose |
|---|---|---|
| `plugin_paths` | list of dirs | Added to `sys.path` so plugin modules and their deps import |
| `plugins` | list of FQCNs | The classes `_init_plugins()` instantiates |

The dialog's **Verify All** button instantiates every registered plugin to
surface load errors early; failures are written to the WISER log.

---

## Per-Type Registration

### Tools-menu plugins

Tools-menu registration is **push-based and one-shot**. Inside
`_init_plugins()`, right after a plugin is stored, WISER checks its type and
lets it contribute to the live Tools `QMenu`:

```python
if isinstance(plugin, plugins.ToolsMenuPlugin):
    plugin.add_tool_menu_items(self._tools_menu, self._app_state)
```

The plugin receives the menu and the `ApplicationState`, and adds its actions or
submenus then. Because this runs only at startup, a plugin's Tools-menu
contributions are fixed for the session.

### Context-menu plugins

Context-menu registration is **pull-based and repeated** — WISER calls every
context-menu plugin *each time* it builds a context menu. The dispatcher is:

**File:** `src/wiser/gui/plugin_utils.py` — `add_plugin_context_menu_items()`

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

A single user-facing menu may produce several calls with different
`context_type` values (e.g. `RASTER_VIEW`, then `DATASET_PICK`, then
`ROI_PICK`). The keys present in `context` depend on the `context_type`:

| `context_type` | Extra `context` keys (besides `wiser`, `app_services`) |
|---|---|
| `RASTER_VIEW` | `dataset`, `display_bands` |
| `SPECTRUM_PLOT` | *(none)* |
| `DATASET_PICK` | `dataset`, `display_bands`, `ds_coord` |
| `SPECTRUM_PICK` | `spectrum` |
| `ROI_PICK` | `dataset`, `display_bands`, `roi`, `ds_coord` |

Each plugin is handed a **copy** of the context, so a plugin that mutates its
context cannot affect the others.

### Band-math plugins

Band-math functions are collected on demand by `get_plugin_fns(app_state)` when
the Band Math dialog is set up.

**File:** `src/wiser/gui/util.py` — `get_plugin_fns()` (~line 651), called from
`src/wiser/gui/bandmath_dialog.py`

```python
def get_plugin_fns(app_state):
    functions = {}
    for plugin_name, plugin in app_state.get_plugins().items():
        try:
            plugin_fns = plugin.get_bandmath_functions()
            # lowercase every name (band math is case-insensitive)
            # warn on duplicate names across plugins
            functions.update(plugin_fns)
        except:
            pass
    return functions
```

Note it calls `get_bandmath_functions()` on **every** plugin and relies on the
`try/except` to skip plugins that don't implement it — there is no
`isinstance` check. Returned names are lower-cased (band math is
case-insensitive) and duplicate names across plugins emit a warning, with the
last definition winning. The returned `BandMathFunction` instances are invoked
by the band-math engine exactly like built-in operators — see
{doc}`Band Math Internals <bandmath-internals>`.

```{mermaid}
flowchart TD
    INIT["App._init_plugins()"]
    INIT -->|"at startup"| T["ToolsMenuPlugin.add_tool_menu_items()\npush · once"]
    GP["get_plugins() registry\n(ApplicationState)"]
    INIT --> GP
    MENU["User opens a context menu"] -->|"per build"| C["add_plugin_context_menu_items()\npull · repeated"]
    C --> GP
    DLG["User opens Band Math dialog"] -->|"on open"| B["get_plugin_fns()\npull · on demand"]
    B --> GP
```

---

(execution-model)=
## Execution model

Plugins run **in WISER's own process and Python interpreter**. WISER does not
spawn a new process or interpreter to load or run a plugin: `instantiate()`
imports the plugin module into the running process, and every plugin method
(menu callbacks, band-math `apply()`, etc.) executes on WISER's threads.

Two consequences follow:

- A long-running or blocking plugin call runs on WISER's threads. Isolating
  WISER from misbehaving plugins (separate processes, long-task abstractions) is
  an open design question — see
  {doc}`Design Documents <design-documents>`.
- All plugins share the interpreter's single set of imported modules, which
  drives the dependency-precedence behavior below.

### Exception isolation

The machinery is defensive at each boundary so a single faulty plugin degrades
gracefully rather than crashing WISER:

| Boundary | Location | Behavior on failure |
|---|---|---|
| Instantiation | `_init_plugins()` | `try/except` logs and skips the plugin |
| Type check | `is_plugin()` | Non-plugin objects are skipped |
| Context menu | `add_plugin_context_menu_items()` | Exception caught and logged; menu still builds |
| Band math | `get_plugin_fns()` | Exception swallowed per plugin |

WISER also provides `wiser.plugins.log_exceptions`, a decorator plugin authors
can apply to their own methods so exceptions land in the WISER log.

---

(dependency-handling)=
## Dependency handling

Because plugins run in WISER's single interpreter, dependency resolution comes
down to one detail: **how plugin paths are added to `sys.path`**. In
`_init_plugins()` they are **appended**, so they land at the *end* of the search
path. Python resolves imports in `sys.path` order, so WISER's own dependencies —
in a frozen build, the modules bundled under `sys._MEIPASS` by PyInstaller — are
found **first**.

> If a plugin and WISER both depend on the same package, **WISER's version
> wins.** The plugin's copy on a later `sys.path` entry is never reached.

This is a deliberate consequence of the [execution model](#execution-model): a
single interpreter holds exactly one copy of any imported module, so WISER
cannot hand a plugin a conflicting version of a package it has already imported.
Appending after WISER's own paths is the safest available option — a plugin can
add *new* packages it needs, but can never shadow one of WISER's libraries and
destabilize the core application.

**Implications for plugin authors** (documented user-side in
{doc}`Extending WISER <../extending-wiser/index>`):

- Pin plugin dependency versions to match the WISER release being targeted.
  WISER publishes per-release dependency lists; mismatches usually work but can
  cause subtle behavior.
- Plugins whose dependencies are *incompatible* with WISER's bundled versions
  are not supported today — in a frozen build WISER's dependency set cannot be
  extended or overridden.

The proposed longer-term direction — running plugins out of process against
their own environment (e.g. via `uv`) so incompatible dependencies become
possible — is captured in
{doc}`Design Documents <design-documents>`.
