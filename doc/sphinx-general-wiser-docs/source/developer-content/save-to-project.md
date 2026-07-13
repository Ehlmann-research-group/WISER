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
| `datasets/` | ENVI-format raster sidecars for in-memory (RAM-backed) datasets that have no source file on disk. |

A bundle is written either as a **directory** or, when the destination ends in
`.wiserproj`, as a single **zip** of that directory. File-backed datasets are
stored *by reference* (their path), so a project that loads only on-disk rasters is
tiny; only RAM-backed datasets and their derived arrays add bulk.

```{mermaid}
flowchart TB
    subgraph UI["GUI layer"]
        Menu["File menu<br/>Open / Save / Save As"]
        Dialog["SaveProjectDialog<br/>choose RAM datasets, preview cascade"]
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
faithful and then failing on load. ROIs are standalone and always saved.
`resolver_for_all_datasets(app_state)` builds the default "save everything"
resolver; the Save dialog builds one from the user's exclusions. `cascade_report`
turns a resolver into the human-readable list of consequences the dialog previews.

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
| `datasets` | Loaded datasets | File-backed by reference; RAM-backed to ENVI sidecars under `datasets/`. Original ids preserved on load so references stay valid. |
| `rois` | Regions of interest | Standalone; multi-selection geometry round-trips through the pyrep convention. |
| `spectra` | Collected + active spectra | Three kinds: self-contained NumPy, raster-backed (dataset + point + area), ROI-average (dataset + roi). Dataset-backed spectra go faithful when the dataset is saved, freeze to NumPy when cut. |
| `stretches` | Per-`(dataset, band)` contrast stretches | Dataset-cascade leaf: dropped when its dataset is unsaved. Polymorphic across the stretch types. |
| `libraries` | Spectral libraries | `ENVISpectralLibrary` by reference (reopened from its path); `ListSpectralLibrary` inline (members follow the per-member spectra rule). |
| `runs` | PCA / MNF / unmixing / K-Means histories | Every record self-contained; datasets referenced softly by id; run ids re-minted on load. |
| `crs` | User-created coordinate systems | WKT + creator-dialog state; rebuilt via GDAL/`osr`, bad WKT drops the entry, a bad enum degrades one field. |
| `bandmath` | Saved band-math expressions | A plain list of expression strings on `ApplicationState`. |

---

## Save

`save_project(app_state, dest, resolver=None)` writes the manifest and sidecars in
**dependency order** — datasets first, since they are the roots the rest of the
manifest references by id and they own the sidecar I/O. Without an explicit
resolver, every dataset is treated as saved; the Save dialog supplies a user-driven
one built from the datasets the user unchecked.

```{mermaid}
sequenceDiagram
    participant U as User
    participant D as SaveProjectDialog
    participant S as save_project
    participant B as ProjectBundle
    U->>D: choose RAM datasets to include
    D->>D: save_plan() — preview SNAPSHOT/DROP cascade
    U->>D: confirm
    D->>S: save_project(dest, resolver)
    S->>B: create bundle (dir, or temp dir → zip)
    S->>B: save_datasets → sidecars + manifest
    S->>B: save_crs, bandmath, rois, stretches, spectra, libraries, runs
    S->>B: write_manifest
    B-->>U: .wiserproj written
```

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
2. If the item depends on a dataset, classify it through the resolver so it
   snapshots or drops when its dataset is cut.
3. Wire it into `orchestrate._write_bundle` and `orchestrate._restore` at the right
   point in dependency order.
4. Prefer an additive, optional manifest section so no version bump is needed; bump
   `format_version` and add a migration only for a breaking change.
