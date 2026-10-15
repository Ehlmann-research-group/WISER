---
applyTo: "src/wiser/project/**,src/wiser/plugins/**,src/wiser/config/**,src/wiser/gui/app_config.py,src/wiser/gui/plugin_utils.py,src/example_plugins/**"
---

# Compatibility of persisted state and public surfaces

Anything here outlives the release that wrote it. A user's saved project, their config,
and a third-party plugin all have to keep working across upgrades, and none of them are
covered by CI. Review these diffs as interface changes even when they look internal.

## Project files (`.wiserproj`)

- A `.wiserproj` is a zipped bundle directory carrying a `format_version`.
  `src/wiser/project/migrate.py` documents the policy and the bump checklist — read it
  and check the diff against it rather than inventing rules.
- **Additive changes need no bump**: a new optional field, or a new section an older
  reader will ignore. Flag a diff that bumps `CURRENT_FORMAT_VERSION` for a purely
  additive change — it forces a migration nobody needs.
- **A bump requires the migration.** If a diff renames, removes, retypes, or changes the
  meaning of an existing field, it needs `CURRENT_FORMAT_VERSION` raised *and* a
  registered `migrate_vN_to_vN+1`. A bump with no migration silently orphans every
  project file users have already saved. This is the single highest-cost mistake in this
  directory.
- Migrations must be pure functions over the manifest dict that touch only the sections
  they own and leave `format_version` to the framework.
- Check that a new persister round-trips: what `save` writes, `load` must reconstruct.
  Absolute paths, machine-local temp directories, and user home paths written into a
  project file break it on another machine — projects are shared between researchers.
- A `.wiserproj` is a zip from an untrusted source. Check that extraction rejects
  absolute paths and `..` components rather than writing wherever the archive says.
- The atomic-write pattern (build beside the target, move into place) exists so an
  interrupted save does not destroy the previous project. Flag a new write path that
  truncates the destination first.

## User configuration

- `wiser-conf.json` lives in a **per-version** app-data directory from
  `get_wiser_config_dir()`, so a new version starts from defaults rather than inheriting
  a stale schema. Do not suggest a shared location.
- Every key needs an entry in `ApplicationConfig.DEFAULTS` with its type; the loader
  coerces against that type. A key read without a default returns nothing useful, and a
  type change silently reinterprets existing values.
- Removing or renaming a key strands whatever users already have set. Ask what happens
  to an existing config file on upgrade.

## Feature flags

Two systems, not interchangeable — `doc/sphinx-general-wiser-docs/source/developer-content/contributing-and-quality.md`
explains the distinction and the review should hold the diff to it:

- **`wiser.config.feature_flags.FEATURE_GATES` / `FLAGS.x`** is a deployment-tier gate
  read once from `WISER_ENV` at process start. This is the one for merging unfinished
  work into `main` under trunk-based development. Frozen builds force `prod`.
- **`feature_flags.*` keys in `ApplicationConfig`** are per-user toggles flipped from
  inside the running app.

Flag an in-progress feature merged without a gate — the project merges to `main`
continuously and relies on gating for production safety. Flag a gate reached `prod`
whose entry was left behind; the convention is to remove it. And flag the wrong system:
a tier gate used for a per-user preference cannot be changed without an environment
variable and a relaunch.

## Plugin API

- `ToolsMenuPlugin`, `ContextMenuPlugin`, and `BandMathPlugin` in
  `src/wiser/plugins/types.py` are the supported interfaces, and third-party plugins
  subclass them. Changing a method signature, its arguments, or its return type is a
  breaking change for code not in this repository. Prefer additive changes with a
  default; if a break is unavoidable, it needs to be called out in the PR body.
- The context dict handed to a context-menu plugin is part of that contract. Removing a
  key, or changing what `wiser` / `app_services` provide, breaks plugins silently at
  runtime.
- Plugin loading is config-driven via `plugin_paths` plus fully qualified class names.
  Check that a change to discovery still finds plugins configured the old way.
- `src/example_plugins/` is shipped inside the frozen bundle for `--test_mode`. A change
  to the plugin API that does not update the examples ships a broken example.

## Application identity

The Caltech-era identifiers — the `edu.caltech.gps.WISER` bundle id, the
`QSettings("Caltech", ...)` organization, and the Windows installer Publisher — are
deliberately unchanged. Renaming them breaks code signing and app identity or orphans
every existing user's settings. Flag any diff that changes them; do not suggest
modernizing them.
