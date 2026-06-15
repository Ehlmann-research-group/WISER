# Plugin Dependencies Internals

This page explains how WISER handles a plugin's third-party Python
dependencies, **why** it works the way it does, and what that means for plugin
authors. It is the developer-facing companion to the load mechanics described in
{doc}`Plugin System Internals <plugin-system>`. For the author-facing "how to
configure plugin paths" instructions, see
{doc}`Extending WISER <../extending-wiser/index>`.

## The dependency model

WISER does **not** install a plugin's dependencies for it. A plugin author is
responsible for making their dependencies importable, and points WISER at them
through the `plugin_paths` setting (**Settings → Plugins**). A path may be:

- the plugin's own source directory, and/or
- the `site-packages` of a virtualenv or conda environment where the plugin's
  dependencies are installed.

At startup, every configured path is added to `sys.path` so those packages
become importable by the running interpreter.

## Precedence mechanism

The key detail is **how** plugin paths are added to `sys.path`. In
`App._init_plugins()`
([`src/wiser/gui/app.py`](../../../../src/wiser/gui/app.py), ~line 441):

```python
for p in plugin_paths:
    if not os.path.isdir(p):
        logger.warning(f'Plugin-path "{p}" doesn\'t exist; ignoring')
        continue
    if p not in sys.path:
        sys.path.append(p)          # APPENDED to the end of sys.path
```

Paths are **appended**, so they land at the *end* of `sys.path`. Python resolves
imports by searching `sys.path` in order, which means WISER's own dependencies —
in a packaged build, the modules bundled under `sys._MEIPASS` by PyInstaller —
are found **first**. Plugin paths are consulted only for packages WISER hasn't
already provided.

The practical rule:

> If a plugin and WISER both depend on the same package, **WISER's version
> wins**. The plugin's copy on a later `sys.path` entry is never reached,
> because the import is already satisfied by WISER's bundled copy.

## Why this design

This behavior follows directly from the in-process
[execution model](#execution-model). WISER loads plugins into
its own interpreter rather than spawning a separate process per plugin. A single
interpreter has exactly one copy of any imported module, so WISER fundamentally
*cannot* give a plugin a different version of a package it has already imported —
there is nowhere to put the second version.

Given that constraint, appending plugin paths after WISER's is the safest
option available: it guarantees WISER always runs against the dependency
versions it was built and tested with, so a plugin can never destabilize the
core application by shadowing one of WISER's libraries. Plugins still get access
to any *additional* packages they need; they just don't get to override the ones
WISER ships.

Running plugins in a fresh interpreter would remove this constraint, but at a
large cost in complexity (cross-process GUI interaction, serialization,
packaging WISER for import by the subprocess). That is an active area of design
work — see [Future direction](#future-direction).

## Implications for authors

- **Pin to the WISER release you target.** WISER ships pinned environments;
  match your plugin's dependency versions to the version and platform of WISER
  you run against. Per-release dependency lists are published in the
  {doc}`Extending WISER <../extending-wiser/index>` guide (for example
  `extending-wiser/resources/rel-1.3b1/{windows,arm-mac,intel-mac}-dependencies.txt`).
  Mismatched versions usually work, but can produce subtle, hard-to-diagnose
  behavior.
- **Incompatible dependencies are not supported today.** If a plugin needs a
  version of a package that is incompatible with WISER's bundled version, there
  is currently no way to satisfy it — in a frozen build WISER's dependency set
  cannot be extended or overridden.
- **Compiled C-extension packages are special.** Packages like GDAL that require
  compiled C extensions generally cannot be installed with `pip` alone; they
  need conda or system packages. Plan plugin environments accordingly.

## Known issue: PyInstaller submodule pruning (pre-1.3b1)

When PyInstaller builds a frozen WISER application it recursively resolves all
imports and stores them under `_internal/`. Before release 1.3b1, if a plugin
imported a submodule that PyInstaller had pruned because WISER itself never
imported it (e.g. `scipy.io` when WISER only used top-level `scipy`), that
submodule was missing at runtime and the plugin failed to load.

**Bandage fix (releases 1.2b1 and earlier):** the PyInstaller spec was updated
to explicitly include all submodules of WISER's dependencies via hooks in
`pyinstaller_hooks/`, so submodules are no longer pruned. The `rel/1.2b1` and
`rel/1.2b1-intelmac` branches (Windows, ARM Mac, Intel Mac) carry this fix.

**To verify the fix:**

1. Build a frozen WISER (no code-signing or notarization needed).
2. Add the `pca_plugin` (it depends on `scipy.io`).
3. Confirm no `scipy.io` import error occurs.

**To replicate the pre-fix bug:** add a conda environment's `site-packages`
(e.g. `C:\Users\<user>\anaconda3\envs\plugin_lib\Lib\site-packages`) to plugin
paths, add `pca_plugin.PCAPlugin` to the plugins list, and on a pre-fix build
`scipy.io` will not be found.

(future-direction)=
## Future direction

The longer-term goal (under investigation) is to let plugins declare their own
Python dependencies, independent of WISER's bundled environment, by running them
**out of process**. The rough sketch:

1. A bootloader installs `uv` if it isn't already present (downloaded at
   runtime, not bundled with PyInstaller).
2. On plugin load, WISER spawns a subprocess via `uv` that provides the
   plugin's declared dependencies.
3. WISER is packaged as a wheel (e.g. via `python -m zipapp`) so it can be
   imported by the plugin subprocess.
4. Both wheel-packaged plugins and loose dev-environment plugins are supported.

Identified constraints:

- GDAL and similar compiled-C-extension packages can't be installed by `pip`
  alone — they need conda or system packages.
- `PySide6` is required (over `PySide2`) because it has `pip` wheels for all
  major platforms.
- Architecture-specific DLLs can only be `pip`-installed if the package itself
  ships pip wheels for that platform.

Tooling evaluated (none adopted yet): Bazel, Nuitka (requires PySide6), Poetry,
cx_Freeze.

## See also

- {doc}`Plugin System Internals <plugin-system>` — discovery, loading, and the
  in-process execution model.
- {doc}`Extending WISER <../extending-wiser/index>` — author-facing plugin-path
  setup and per-release dependency lists.
- {doc}`Design Documents <design-documents>` — broader plugin isolation and
  quality design questions.
