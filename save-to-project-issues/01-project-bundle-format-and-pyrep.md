## Is your feature request related to a problem? Please describe.

The "Save to Project File" epic ([00](00-epic-save-load-project-file.md)) needs a concrete, durable on-disk format before any individual state item can be persisted. WISER has two existing serialization styles and neither is currently suited to long-lived project files:

- **pyrep** (`to_pyrep`/`from_pyrep`) — plain Python dicts, used today by `Selection` types ([selection.py](../src/wiser/raster/selection.py)). JSON-friendly and portable, but not used consistently across state items, and many items (e.g. stretches) have no pyrep at all.
- **`Serializable` / `get_serialized_form()`** — a pickle-style mechanism used for **cross-process** transfer (e.g. shipping datasets to worker processes, see `Selection.__getstate__`). It is brittle across code versions and unsafe for files received from others.

Without a single agreed format, every sub-issue would invent its own, producing an inconsistent and unmigratable project file.

## Describe the solution you'd like

Define and implement the **project bundle format** and a **shared pyrep serialization convention** that all other sub-issues build on.

**Bundle layout** — a project is a directory (optionally zipped to a single `.wiserproj` file):

```
my_project.wiserproj/            (dir or zip)
  manifest.json                  # the pyrep of the whole session
  datasets/                      # raster sidecars for in-memory datasets
    ds_7.img / ds_7.hdr ...
  arrays/                        # optional .npy sidecars for large arrays
  format_version                 # integer, for migration
```

**Manifest = pyrep (plain dicts) → JSON.** Decision (see epic discussion): use **pyrep for the manifest**, *not* the `Serializable`/pickle path. Rationale: portability, human-readability, version tolerance, and safety for shared files. The `Serializable` path stays where it belongs (cross-process), and **bulk raster bytes never go in the manifest** — datasets write themselves as native raster sidecars (GDAL/ENVI/GeoTIFF).

**Deliverables:**

1. A `ProjectBundle` abstraction that can create/open a bundle (dir or zip), read/write `manifest.json`, and add/fetch sidecar files (rasters, `.npy` arrays) by a stable relative key.
2. A top-level manifest schema with `format_version`, plus a top-level section per state item (filled in by sub-issues 03–10).
3. A documented **pyrep convention**: every persistable type exposes `to_pyrep() -> dict` and a classmethod/`from_pyrep(data)`; each dict carries a `"type"` tag; large numpy arrays are written to the `arrays/` sidecar area and referenced by key rather than inlined.
4. A `format_version` + the migration seam (the migrate-up mechanism itself; the migration *policy* and test suite are owned by [13](13-project-versioning-and-migration.md)).
5. Round-trip unit tests for the empty/skeleton manifest and the sidecar mechanism.

**Backwards-compatibility strategy (migrate-up on load).** The format is built to evolve from day one, so this is a first-class concern rather than an afterthought:

- **Backward compatibility is guaranteed; forward compatibility is not.** A *newer* WISER must always open *older* project files. An *older* WISER is **not** expected to open files from a newer one — instead, when the loader sees a `format_version` greater than it understands, it **refuses cleanly** ("this project was made with a newer WISER — please upgrade") rather than crashing or silently dropping data.
- **One current-schema loader + a migration chain.** The loader only ever understands the **current** schema. On load, if the file is older, a chain of small *pure* functions (`v1→v2→v3→…→current`) transforms the manifest *pyrep dict* up to the current shape, then the normal load proceeds. All version-handling is isolated in this one migration layer; sub-issues 03–12 only ever deal with the current schema. (Mirrors DB migrations / QGIS.)
- **Lenient additive parsing avoids most migrations.** For non-breaking changes (a new *optional* field), don't bump `format_version` or write a migration — `from_pyrep` ignores unknown keys and supplies defaults for missing ones. Bump the version and write a migration only for **breaking** changes (renamed / removed / restructured fields).
- **Single global `format_version`** for the whole bundle (migrations operate on the whole manifest dict), not per-section versions. Per-section versioning can be added later if one item churns heavily.

The migration-chain harness, the refuse-too-new behavior, and the golden-file regression suite that *enforces* "old files still load" are specified in [13](13-project-versioning-and-migration.md).

## Describe how solution fits WISER's mission

A portable, human-readable, version-tolerant project format is what makes sessions **shareable across machines and OSes** (WISER runs on macOS/Windows/Linux) and **reproducible across WISER releases**. Avoiding pickle also keeps shared project files safe to open.

## Describe alternatives you've considered

- **A single binary/pickled file.** Rejected: not version-tolerant, not safe to share, opaque to debugging.
- **One flat JSON with base64-embedded rasters.** Rejected: huge, slow, defeats native raster tooling.
- **Reusing `Serializable` for the whole manifest.** Rejected for storage (see above); kept for cross-process only.
- **YAML/TOML instead of JSON.** JSON chosen for ubiquity and tooling; the pyrep dicts are format-agnostic so this can change later behind the `ProjectBundle` seam.

## Additional context

- Foundation issue: blocks **all** of 02–13.
- The versioning/migration *policy* and its enforcing test suite live in [13](13-project-versioning-and-migration.md); this issue only provides the `format_version` field and the migration seam.
- Existing pyrep reference implementation to mirror: the `Selection` subclasses in [selection.py](../src/wiser/raster/selection.py).
- Bundle-as-directory-or-zip mirrors how QGIS projects and many geospatial tools ship sidecar data alongside a manifest.
