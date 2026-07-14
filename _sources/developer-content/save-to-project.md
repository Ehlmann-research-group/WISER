# Save to Project File

WISER can persist an entire working session — the loaded datasets, regions of
interest, collected spectra, contrast stretches, spectral libraries, analysis run
histories, user-created coordinate systems, and saved band-math expressions — to a
single portable **project file** and later reopen it into a fresh session. This
page documents the on-disk format, the serialization convention, the dependency
model that decides what is saved faithfully versus snapshotted, and the save/load
orchestration.

The feature lives in `src/wiser/project/`. It is deliberately layered so that each
piece of session state is serialized by an isolated *persister*, and a single
orchestrator drives them in dependency order.

---

## Overview

A project file is a **bundle**: a `manifest.json` describing the session, plus two
sidecar directories for the binary data that does not belong inline in JSON.

| Path | Contents |
|------|----------|
| `manifest.json` | The whole session as type-tagged JSON — one section per state kind. |
| `arrays/` | NumPy `.npy` sidecars for arrays too large for the manifest (e.g. in-memory dataset pixels, endmember snapshots). |
| `datasets/` | ENVI-format raster sidecars for RAM-backed datasets — and, under a self-contained save, for file-backed datasets copied in for portability. |

A bundle is written either as a **directory** or, when the destination ends in
`.wiserproj`, as a single **zip** of that directory. It is saved in one of two modes:

- **Referenced** (default) — file-backed datasets are stored *by their path*, so a
  project that loads only on-disk rasters is tiny; only RAM-backed datasets and their
  derived arrays add bulk. The project reopens as long as those source files stay put.
- **Self-contained** — file-backed datasets are also copied into the bundle (re-saved
  as ENVI sidecars under `datasets/`), so the `.wiserproj` is portable and can be moved
  or shared without its original sources. Chosen per-save with a checkbox in the Save
  dialog (`save_project(..., self_contained=True)`); the load path is unchanged, since a
  copied dataset restores from its sidecar exactly like a RAM-backed one.

```{mermaid}
flowchart TB
    subgraph UI["GUI layer"]
        Menu["File menu<br/>Open / Save / Save As"]
        Dialog["SaveProjectDialog<br/>selection tree: what to save + portability"]
    end
    subgraph Orch["Orchestration — orchestrate.py"]
        Save["save_project()"]
        Load["load_project()"]
    end
    subgraph Pers["Persisters — persisters/*.py"]
        P["datasets · rois · spectra · stretches<br/>libraries · runs · crs · bandmath"]
    end
    subgraph Core["Foundation"]
        Bundle["ProjectBundle<br/>manifest + sidecars, dir/zip"]
        Resolver["DependencyResolver<br/>FAITHFUL / SNAPSHOT / DROP"]
        Migrate["migrate.py<br/>format_version, migrate-up"]
    end
    Menu --> Save & Load
    Dialog --> Resolver
    Save --> P
    Load --> P
    P --> Bundle
    Save --> Resolver
    Load --> Migrate
    Bundle --> Migrate
```

---

## The pyrep convention

Every serializable object is converted to a **pyrep** ("Python representation"): a
JSON-safe `dict` carrying a `"type"` tag that names how to rebuild it. A central
registry maps each tag to a reconstruction function.

- `register_pyrep(tag, from_fn)` registers a reconstructor; `from_pyrep(data)`
  dispatches on `data["type"]`.
- Large arrays are not inlined. A persister calls the bundle to stash the array and
  writes an **array reference** in its place — `array_ref(key)` produces a small
  tagged dict, `is_array_ref(data)` / `array_ref_key(data)` recover it, and the
  bundle resolves it back to a NumPy array on load.

This keeps the manifest human-readable and diff-friendly while binary payloads live
in `arrays/`. All manifest paths are confined to the bundle root on read
(`ProjectBundle` validates sidecar keys and rejects `..`/absolute paths and zip-slip
entries), because a `.wiserproj` may be shared and is therefore untrusted input.

---

## The dependency resolver

Most session state does not stand alone — a contrast stretch belongs to a
`(dataset, band)`, a raster-backed spectrum samples a dataset at a point, an
ROI-average spectrum references an ROI. When the user chooses *not* to save a
RAM-backed dataset, everything downstream of it must be handled coherently. That
decision is the resolver's job.

`DependencyResolver` classifies each dependent item against the set of saved
dataset roots into one of three outcomes:

| Outcome | Meaning |
|---------|---------|
| **FAITHFUL** | The dependency is saved, so the item is stored as a live reference and restores exactly. |
| **SNAPSHOT** | The dependency is cut, but the item can be frozen to self-contained data (e.g. a dataset-backed spectrum saved as raw values). |
| **DROP** | The dependency is cut and the item cannot be snapshotted (e.g. a stretch, which is meaningless without its band) — it is not saved. |

A `Dependency(kind, id)` with an unresolved (`None`) id is treated as a cut edge for
every kind, so a dangling reference snapshots or drops rather than being called
faithful and then failing on load.

Beyond the dataset roots, the resolver also tracks which **ROIs** and which
**standalone items** — a run, spectral library, user CRS, or band-math expression,
kinds nothing else depends on — the user excluded. Excluding an ROI both omits it and
cascades to its ROI-average spectra (they snapshot); excluding a standalone item simply
omits it, with nothing to cascade. `resolver_for_all_datasets(app_state)` builds the
default "save everything" resolver; `resolver_for_selection(app_state, excluded_datasets,
excluded_rois, excluded_items)` builds one from the user's selection. Each `save_*`
persister asks the resolver whether its own items are saved and skips the excluded ones.
`cascade_report` turns a resolver into the human-readable list of consequences the dialog
previews.

---

## Persisters

Each kind of state has a `save_<kind>(app_state, manifest, ...)` /
`load_<kind>(manifest, app_state, ...)` pair in `persisters/`. They share one
contract: **a malformed or unrestorable entry is dropped and reported, never
fatal.** Every `load_*` returns the list of entries it could not restore so the
orchestrator can surface a single "some items could not be opened" warning instead
of aborting the whole project open. Parsing is defensive — missing keys, wrong
types, and bad enum values degrade gracefully.

| Persister | Persists | Notes |
|-----------|----------|-------|
| `datasets` | Loaded datasets | File-backed by reference (or copied into the bundle under a self-contained save), RAM-backed to ENVI sidecars under `datasets/`; a NetCDF subdataset also records which subdataset was selected, and a JSON snapshot of the runtime-editable metadata (data-ignore, bad bands, wavelengths + units, band names, display bands, georeferenced CRS) rides in the manifest and is reapplied on load. Original ids preserved so references stay valid. |
| `rois` | Regions of interest | Standalone; multi-selection geometry round-trips through the pyrep convention. |
| `spectra` | Collected + active spectra | Three kinds: self-contained NumPy, raster-backed (dataset + point + area), ROI-average (dataset + roi). Dataset-backed spectra go faithful when the dataset is saved, freeze to NumPy when cut. |
| `stretches` | Per-`(dataset, band)` contrast stretches | Dataset-cascade leaf: dropped when its dataset is unsaved. Polymorphic across the stretch types. |
| `libraries` | Spectral libraries | `ENVISpectralLibrary` by reference (reopened from its path); `ListSpectralLibrary` inline (members follow the per-member spectra rule). |
| `runs` | PCA / MNF / unmixing / K-Means histories | Every record self-contained; datasets referenced softly by id; run ids re-minted on load. |
| `crs` | User-created coordinate systems | WKT + creator-dialog state; rebuilt via GDAL/`osr`, bad WKT drops the entry, a bad enum degrades one field. |
| `bandmath` | Saved band-math expressions | A plain list of expression strings on `ApplicationState`. |

### Dataset metadata and subdatasets

A file-backed dataset is re-opened from its path on load, which re-derives its
metadata from the source file — so metadata the user edited at runtime (a renamed
band, an adjusted data-ignore value, a hand-marked bad-band list) would be lost on
reopen. Each dataset entry therefore also carries a JSON snapshot of the
runtime-editable metadata: the data-ignore value, the bad-band list, per-band
wavelengths and their unit, band descriptions, the default display bands, and any
georeferencer-assigned CRS (WKT) with its geo-transform. On load the snapshot is
reapplied through the safe per-field setters in a fixed order — data-ignore first,
since it keys the band-statistics cache, and band descriptions after wavelengths,
since rebuilding band info overwrites them. Reapply is best-effort and
field-guarded: a value that no longer fits the reopened dataset (a band-count
mismatch, an unparseable unit) is skipped so the dataset still restores.

A NetCDF **subdataset** needs one extra field. Its `get_filepaths()` returns a GDAL
descriptor (`NETCDF:"/path/file.nc":var`) rather than a plain path, so the entry
records the base `.nc` file together with the `subdataset_name` descriptor and
reopens via `load_from_file(base, subdataset_name=…, interactive=False)` — restoring
the *same* subdataset the user had open instead of re-running the auto-pick heuristic
or dropping the dataset because the descriptor is not a file on disk. Subdataset
selection is NetCDF-only.

Both the metadata snapshot and `subdataset_name` are **additive, optional** manifest
fields, so they require no `format_version` bump: an older manifest simply lacks them
and loads exactly as before.

---

## Save

`save_project(app_state, dest, resolver=None, self_contained=False)` writes the
manifest and sidecars in **dependency order** — datasets first, since they are the
roots the rest of the manifest references by id and they own the sidecar I/O. Without
an explicit resolver, every item is treated as saved; the Save dialog supplies a
user-driven one built from what the user unchecked. `self_contained=True` copies
file-backed datasets into the bundle so the project is portable.

```{mermaid}
sequenceDiagram
    participant U as User
    participant D as SaveProjectDialog
    participant S as save_project
    participant B as ProjectBundle
    U->>D: uncheck items to leave out; optional self-contained
    D->>D: save_tree() — tree + live cascade annotations
    U->>D: confirm
    D->>S: save_project(dest, resolver, self_contained)
    S->>B: create bundle (dir, or temp dir → zip)
    S->>B: save_datasets → sidecars + manifest
    S->>B: save_crs, bandmath, rois, stretches, spectra, libraries, runs
    S->>B: write_manifest
    B-->>U: .wiserproj written
```

### Choosing what to save

The Save dialog is a single checkable tree with three groups — **Datasets**, **ROIs**,
and **Analysis outputs** (runs, libraries, user CRSs, band-math expressions). Every item
is checked by default; unchecking one leaves it out of the project, and unchecking a
group toggles all of its items. The tree is the selection UI *and* the preview: a product
under a dataset or ROI (a stretch, a point spectrum, an ROI-average) is a non-checkable
child that follows its parent, annotated live with the consequence it takes when the
parent is cut — `(snapshot)` for a spectrum that freezes to self-contained values,
`(dropped)` for a stretch that has nothing to apply to. The headless model is
`save_plan.save_tree`, unit-tested on its own; the dialog turns the user's unchecked items
into the resolver handed to `save_project`, plus the self-contained flag.

---

## Load

`load_project(src, app_state, extract_dir=None)` opens a bundle (a directory, or a
`.wiserproj` extracted into `extract_dir`), migrates the manifest up to the current
format, **clears the session**, and restores every item in **topological order** so
each reference resolves — datasets (with their original ids) before the stretches,
spectra, and run records that reference them; ROIs before ROI-average spectra.

The UI updates for free: each `load_*` restores through the signal-emitting
`add_*` / `set_*` accessors on `ApplicationState`, so the granular reload signals
fire inline and derived caches (such as the by-id spectrum index) rebuild
themselves — there is no separate "emit signals" or "rebuild index" pass.

```{mermaid}
sequenceDiagram
    participant U as User
    participant L as load_project
    participant B as ProjectBundle
    participant A as ApplicationState
    U->>L: open .wiserproj
    L->>B: read_manifest() — migrate up / refuse too-new
    L->>A: clear_session()
    L->>A: load_datasets (preserve ids)
    L->>A: load_crs, bandmath, rois
    L->>A: load_stretches, spectra, libraries, runs
    L-->>U: load report (dropped-per-section warning)
```

Restoring datasets with their **original ids** is the invariant that keeps every
downstream reference valid; `clear_session` uses the normal `remove_*` methods so
the removed-signals fire and the UI empties, and it only deletes scratch temp
rasters — never a user's source file.

---

## Versioning and migration

The manifest carries a `format_version`. The guarantee is **backward
compatibility**: any project file written by a released WISER opens in every later
WISER. Forward compatibility is not promised — a file newer than the running WISER
is refused cleanly with `ProjectTooNewError` rather than mis-interpreted.

On load, an older manifest is transformed step-by-step up to the current shape by a
chain of pure `migrate_vN_to_vN+1` functions, so the rest of the load code only ever
sees the current schema. The rule contributors follow:

- **Additive change** (a new *optional* field) needs **no** version bump and **no**
  migration — every `from_pyrep` parses leniently and the orchestrator reads sections
  with `.get`, so unknown keys and sections pass through untouched.
- **Breaking change** (a renamed, removed, restructured, or semantically-changed
  field) bumps `CURRENT_FORMAT_VERSION` and registers a migration via
  `register_migration`.

---

## Extending: adding a new persisted item

1. Write a `save_<kind>` / `load_<kind>` pair in `persisters/`, following the
   drop-not-fatal contract (return the list of unrestorable entries).
2. If the item depends on a dataset or ROI, classify it through the resolver so it
   snapshots or drops when that root is cut. If it is a standalone item the user should
   be able to deselect on its own, give it a `(kind, id)` handle, skip it in `save_*`
   when `resolver.is_saved(...)` is false, and list it under `save_tree`'s
   *Analysis outputs* group so it gets a checkbox.
3. Wire it into `orchestrate._write_bundle` and `orchestrate._restore` at the right
   point in dependency order.
4. Prefer an additive, optional manifest section so no version bump is needed; bump
   `format_version` and add a migration only for a breaking change.
