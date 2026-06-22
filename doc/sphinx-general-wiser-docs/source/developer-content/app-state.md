# Application State

This page is a map of **every place WISER keeps state** — anything a user loads
in, configures, or any in-RAM / temporary artifact WISER generates from that
data. It exists so a developer can see, in one place, what would have to be
captured to fully reconstruct a WISER session.

"State" here is broader than the [`ApplicationState`](#central-application-state)
class. A lot of state lives inside individual dialogs and tool modules and is
*not* routed through `ApplicationState` at all (the georeferencer's control-point
table, the band-math saved-expression list, the per-dialog analysis
configuration). Whether a piece of data is tracked by `ApplicationState` is
irrelevant to whether it is state — if it is state, it is documented here.

## How to read this page

Every state item is tagged with one of:

- **[SOURCE]** — source-of-truth state. It cannot be recomputed from anything
  else, so it must be captured to faithfully reconstruct the session.
- **[DERIVED]** — rebuildable from `[SOURCE]` state (caches, cached statistics,
  rendered images). It can always be recomputed, so capturing it is an
  optimization, never a requirement.
- **[EPHEMERAL]** — transient UI/runtime state (open dialog references, plot
  windows, running background processes, in-progress form fields). It is not
  worth carrying across sessions.

```{note}
This page deliberately covers *current / loaded* state. Some subsystems keep a
history of past runs (PCA, MNF, linear unmixing) — that history **is** `[SOURCE]`
state because it persists across dialog open/close and is shown back to the user.
But "old runs" beyond what those history managers retain (e.g. previous band-math
executions, past activity-monitor entries) are *not* tracked as state.
```

---

## Master inventory

| Item | Owner — class · attribute | File | Storage | Tag |
|------|---------------------------|------|---------|-----|
| Loaded raster datasets | `ApplicationState._datasets` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM (+ backing file / cache) | **[SOURCE]** |
| Contrast stretches (per dataset+band) | `ApplicationState._stretches` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[SOURCE]** |
| Spectral libraries | `ApplicationState._spectral_libraries` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM (+ file) | **[SOURCE]** |
| Regions of interest | `ApplicationState._regions_of_interest` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[SOURCE]** |
| Active spectrum | `ApplicationState._active_spectrum` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[SOURCE]** |
| Collected spectra | `ApplicationState._collected_spectra` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[SOURCE]** |
| Spectrum registry (by id) | `ApplicationState._all_spectra` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[DERIVED]** |
| K-Means centroids/results | `ApplicationState._kmeans_centroids` | [kmeans.py](../../../../src/wiser/gui/kmeans.py) | RAM | **[SOURCE]** |
| Linear-unmix run history | `ApplicationState._linear_unmix_history` | [linear_unmixing.py](../../../../src/wiser/gui/linear_unmixing.py) | RAM | **[SOURCE]** |
| PCA run history | `ApplicationState._pca_history` | [pca_plugin.py](../../../../src/wiser/gui/permanent_plugins/pca_plugin.py) | RAM | **[SOURCE]** |
| MNF run history | `ApplicationState._mnf_history` | [mnf.py](../../../../src/wiser/gui/mnf.py) | RAM | **[SOURCE]** |
| User-created CRSs | `ApplicationState._user_created_crs` | [reference_creator_dialog.py](../../../../src/wiser/gui/reference_creator_dialog.py) | RAM | **[SOURCE]** |
| Application config | `ApplicationState._config` | [app_config.py](../../../../src/wiser/gui/app_config.py) | RAM (+ settings file) | **[SOURCE]** |
| Georeferencer control-point table | `GeoReferenceDialog._table_entry_list` | [geo_reference_dialog.py](../../../../src/wiser/gui/geo_reference_dialog.py) | RAM (+ `.points`/`.pts`) | **[SOURCE]** |
| Band-math saved expressions | `BandMathDialog._ui.cbox_saved_exprs` | [bandmath_dialog.py](../../../../src/wiser/gui/bandmath_dialog.py) | RAM widget (+ `.txt`) | **[SOURCE]** |
| Render / computation / histogram caches | `DataCache` | [data_cache.py](../../../../src/wiser/raster/data_cache.py) | RAM (FIFO) | **[DERIVED]** |
| Cached per-band stats | `RasterDataSet._cached_band_stats` | [dataset.py](../../../../src/wiser/raster/dataset.py) | RAM | **[DERIVED]** |
| Running background processes | `ApplicationState._running_processes` | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[EPHEMERAL]** |
| Open plot / table / chooser windows | `ApplicationState._generic_spectrum_plots`, … | [app_state.py](../../../../src/wiser/gui/app_state.py) | RAM | **[EPHEMERAL]** |
| Per-dialog form fields (k, components, …) | various dialogs | `src/wiser/gui/` | RAM | **[EPHEMERAL]** |

The rest of this page expands each group.

---

## The big picture

```{mermaid}
flowchart TD
    subgraph inputs["User inputs & file loads"]
        files["Raster / library files"]
        draw["Drawn ROIs, picked pixels"]
        gcps["Georef control points"]
        exprs["Band-math expressions"]
    end

    subgraph tools["Tools that generate artifacts"]
        kmeans["K-Means"]
        unmix["Linear Unmixing"]
        pca["PCA / MNF"]
        bandmath["Band Math"]
        crscreate["CRS Creator"]
    end

    subgraph source["SOURCE stores"]
        appstate["ApplicationState<br/>(datasets, stretches, libraries,<br/>ROIs, spectra, kmeans, histories,<br/>user CRSs, config)"]
        georef["GeoReferenceDialog<br/>(control-point table)"]
        bmstate["BandMathDialog<br/>(saved expressions)"]
    end

    subgraph derived["DERIVED (rebuildable)"]
        cache["DataCache:<br/>render / computation / histogram"]
        stats["cached band stats"]
    end

    files --> appstate
    draw --> appstate
    gcps --> georef
    exprs --> bmstate
    crscreate --> appstate

    kmeans --> appstate
    unmix --> appstate
    pca --> appstate
    bandmath --> appstate

    appstate -.feeds.-> cache
    appstate -.feeds.-> stats
```

`[SOURCE]` state is what a session-reconstruction routine must walk. Note that
two important stores — the georeferencer's control-point table and the band-math
saved-expression list — live **outside** `ApplicationState`, on their respective
dialogs.

---

(central-application-state)=
## Central application state

**File:** [app_state.py](../../../../src/wiser/gui/app_state.py)
**Purpose:** Single object (`ApplicationState`) that owns most — but not all — of
WISER's state and emits Qt signals when it changes.

```{mermaid}
classDiagram
    class ApplicationState {
        +_datasets : dict[int, RasterDataSet]  SOURCE
        +_stretches : dict[(int,int), StretchBase]  SOURCE
        +_spectral_libraries : dict[int, SpectralLibrary]  SOURCE
        +_regions_of_interest : dict[int, RegionOfInterest]  SOURCE
        +_kmeans_centroids : dict[KMeansParameters, KMeansCentroids]  SOURCE
        +_all_spectra : dict[int, Spectrum]  DERIVED
        +_active_spectrum : Spectrum  SOURCE
        +_collected_spectra : list[Spectrum]  SOURCE
        +_user_created_crs : dict[str, tuple]  SOURCE
        +_config : ApplicationConfig  SOURCE
        +_linear_unmix_history : LinearUnmixingHistoryManager  SOURCE
        +_pca_history : PCAHistoryManager  SOURCE
        +_mnf_history : MNFHistoryManager  SOURCE
        +_running_processes : dict  EPHEMERAL
        +_plugins : dict  EPHEMERAL
        +_next_id : int  bookkeeping
        +_current_dir : str  bookkeeping
        +_cache : DataCache  DERIVED
    }
```

**`[SOURCE]` attributes**

- `_datasets` — all loaded raster datasets, keyed by numeric id (see
  [Datasets](#datasets-and-derived-rasters)).
- `_stretches` — the *current* contrast stretch per `(dataset_id, band_index)`
  (see [Contrast stretches](#contrast-stretch-configuration)).
- `_spectral_libraries` — loaded spectral libraries, keyed by id.
- `_regions_of_interest` — all ROIs, keyed by id (see [ROIs](#regions-of-interest)).
- `_kmeans_centroids` — K-Means results keyed by parameters (see
  [Analysis results](#analysis-results-and-run-histories)).
- `_active_spectrum` / `_collected_spectra` — see [Spectra](#spectra-and-spectral-libraries).
- `_user_created_crs` — `{name: (osr.SpatialReference, CrsCreatorState)}` (see
  [Georeferencing](#georeferencing-and-user-created-crss)).
- `_config` — the `ApplicationConfig` object. Application preferences; largely
  mirrors a settings file but is mutated at runtime.
- `_linear_unmix_history`, `_pca_history`, `_mnf_history` — the run-history
  managers (see [Analysis results](#analysis-results-and-run-histories)).

**`[DERIVED]` attributes**

- `_all_spectra` — a by-id registry of every spectrum currently referenced
  (active + collected + plotted). Rebuildable from the `[SOURCE]` collections
  that actually own the spectra.
- `_cache` — the shared `DataCache` (see [Caches](#caches-and-temporary-storage)).

**`[EPHEMERAL]` / runtime attributes**

- `_running_processes`, `_process_pool_manager` — in-flight background tasks.
- `_generic_spectrum_plots`, `_table_display_widgets`,
  `_matplotlib_display_widgets` — open auxiliary windows, kept only so they are
  not garbage-collected.
- `_plugin_*_chooser_dialog`, `_dynamic_input_dialog` — transient chooser dialogs.
- `_plugins` — loaded plugin instances (code, not user data).
- `_next_id`, `_current_dir` — minor session bookkeeping (id counter and the
  last-used directory for file dialogs).

---

(datasets-and-derived-rasters)=
## Datasets and derived rasters

**File:** [dataset.py](../../../../src/wiser/raster/dataset.py)
**Purpose:** `RasterDataSet` represents one raster cube and all of its metadata.
It already subclasses `Serializable`.

A dataset's `[SOURCE]` state:

| Attribute | Meaning |
|-----------|---------|
| `_impl` (`RasterDataImpl`) | Backing store. Either a file on disk (`_impl.get_filepaths()`) or an in-memory array. **In-memory datasets have no file to point at and must have their pixels captured.** |
| `_name`, `_description` | User-facing identity. |
| `_band_unit` | Spectral unit. |
| `SpectralMetadata` → `band_info`, `bad_bands`, `default_display_bands`, `data_ignore_value`, `has_wavelengths` | Per-band metadata (wavelengths, names), bad-band flags, default RGB display bands, nodata value. |
| `SpatialMetadata` → `geo_transform`, `wkt_spatial_reference` | Affine geotransform and CRS as WKT. **Updated by the georeferencer.** |
| `_dirty` | Whether the dataset has unsaved edits. |

`_cached_band_stats` (per-band min/max) is **[DERIVED]** — recomputed on demand.

### Tool outputs are datasets too

Many tools produce a *new* `RasterDataSet` and register it via
`app_state.add_dataset()`. These are first-class `[SOURCE]` artifacts:

- K-Means cluster-label rasters
- Linear-unmixing abundance cubes
- PCA / MNF component rasters
- Band-math result rasters

```{mermaid}
flowchart LR
    load["open_file()"] --> add
    kmeans["K-Means labels"] --> add
    unmix["Unmixing abundances"] --> add
    pcamnf["PCA / MNF components"] --> add
    bm["Band-math result"] --> add
    add["add_dataset()"] --> store["ApplicationState._datasets"]
```

The *provenance* of each tool-generated dataset (which input, which parameters)
is captured separately in the [run histories](#analysis-results-and-run-histories).

---

(spectra-and-spectral-libraries)=
## Spectra and spectral libraries

**Files:** [spectrum.py](../../../../src/wiser/raster/spectrum.py),
[spectral_library.py](../../../../src/wiser/raster/spectral_library.py)

There are three kinds of spectrum state, all `[SOURCE]`:

- **Active spectrum** (`_active_spectrum`) — the single spectrum most recently
  selected (a clicked pixel, an ROI average, a tool output).
- **Collected spectra** (`_collected_spectra`) — spectra the user has explicitly
  set aside (for export or to build a library).
- **Spectral libraries** (`_spectral_libraries`) — loaded/created libraries.

A `Spectrum` carries: values, wavelengths (with units), bad-band mask, `_name`,
`_color`, `_source_name`, and visibility/editable flags. Its cached `_icon` is
`[DERIVED]`. Concrete types include `NumPyArraySpectrum` (in-memory),
`RasterDataSetSpectrum`, and `ROIAverageSpectrum` — note the latter two are
*computed from* a dataset/ROI, so their backing data may be reconstructable from
that source.

A `SpectralLibrary` (`ListSpectralLibrary`, `ENVISpectralLibrary`) holds
`_spectra`, `_name`, `_path` (if file-backed), and `_description`.

```{note}
`_all_spectra` is a by-id index over whatever spectra are currently live; it is
`[DERIVED]` from the active/collected/library collections.
```

---

(regions-of-interest)=
## Regions of interest

**File:** [roi.py](../../../../src/wiser/raster/roi.py)
**Purpose:** `RegionOfInterest` describes a named, colored region as a set of
geometric selections.

`[SOURCE]` state:

| Attribute | Meaning |
|-----------|---------|
| `_id`, `_name`, `_color`, `_description` | Identity and styling. |
| `_selections` (`List[Selection]`) | The geometry — pixels, polygons, rectangles. |
| `_metadata` (`Dict`) | Arbitrary attached metadata. |

Reusable serialization already exists: `roi_to_pyrep(roi)` and
`roi_from_pyrep(data)` in [roi.py](../../../../src/wiser/raster/roi.py) convert an
ROI to/from a plain Python representation — useful building blocks for any
save/restore work.

---

(contrast-stretch-configuration)=
## Contrast-stretch configuration

**Files:** [stretch.py](../../../../src/wiser/raster/stretch.py),
[stretch_builder.py](../../../../src/wiser/gui/stretch_builder.py)

The **current stretch** applied to each band is `[SOURCE]` and lives in
`ApplicationState._stretches`, keyed by `(dataset_id, band_index)`. Each value is
a `StretchBase` subclass:

- `StretchLinear`, `StretchHistEqualize`, `StretchSquareRoot`, `StretchLog2`,
  `StretchDecorrelation`

A stretch is defined by its type plus parameters (conditioner, histogram bounds,
and low/high endpoints). When a dataset is removed, its stretches are removed too
(`remove_dataset()` prunes `_stretches`).

The `StretchBuilder` dialog's working state — histogram bins/edges, raw vs.
normalized band data, slider-link toggles — is `[EPHEMERAL]`/`[DERIVED]`: it is
recomputed each time the dialog opens and only the committed stretch is written
back into `_stretches`.

---

(analysis-results-and-run-histories)=
## Analysis results and run histories

K-Means, linear unmixing, PCA, and MNF all generate result artifacts that are
state. K-Means stores its result keyed by parameters; the other three keep a
**run history** that survives dialog open/close.

### K-Means

**File:** [kmeans.py](../../../../src/wiser/gui/kmeans.py)

- `KMeansParameters` (frozen dataclass) — `dataset_id`, `k`, `init_method`
  (`k-means++` / `random` / `manual`), `num_inits`, `max_iter`, `tol`, `seed`,
  `algorithm` (`lloyd` / `elkan`), and optional `_manual_spectra`. Used as the
  dict key.
- `KMeansCentroids` — the `(k, bands)` centroid array.

Both are `[SOURCE]`, stored in `ApplicationState._kmeans_centroids`. The
cluster-label raster K-Means produces is added as a dataset (see
[Datasets](#datasets-and-derived-rasters)). The dialog's selected dataset, k
value, etc. while editing are `[EPHEMERAL]`.

### Run histories (linear unmixing, PCA, MNF)

**Files:** [run_history.py](../../../../src/wiser/gui/run_history.py),
[linear_unmixing.py](../../../../src/wiser/gui/linear_unmixing.py),
[pca_plugin.py](../../../../src/wiser/gui/permanent_plugins/pca_plugin.py),
[mnf.py](../../../../src/wiser/gui/mnf.py)

All three share `RunHistoryManagerBase`, which holds an in-memory
`List[record]` and re-emits `records_changed` when the list or a referenced
dataset changes. The managers live on `ApplicationState`, so the history
**persists across dialog open/close** — that is exactly why it is `[SOURCE]`.

The records (all `[SOURCE]`):

| Record | Key fields |
|--------|-----------|
| `LinearUnmixingRunRecord` | `run_id`, `timestamp`, `input_dataset_id`, `output_dataset_id`, `input_dataset_name_snapshot`, `output_dataset_name_snapshot`, `endmember_snapshots` (deep copies of the endmember spectra), `sum_to_unity`, `sum_to_unity_weight` |
| `PCARunRecord` | `run_id`, `timestamp`, `input_dataset_id`, `input_dataset_name_snapshot`, `num_components_chosen`, `max_components_available`, `eigenvalues` |
| `MNFRunRecord` | same shape as `PCARunRecord` |

Each record snapshots just enough to revisit the run (spectra/eigenvalues) but
**not** the dataset cubes — if a referenced dataset is closed, the run moves to a
"closed runs" section rather than being deleted.

```{mermaid}
sequenceDiagram
    participant Dialog
    participant Task as Compute task
    participant Mgr as *HistoryManager (on ApplicationState)
    Dialog->>Task: run with chosen params
    Task-->>Dialog: result (raster + eigenvalues/abundances)
    Dialog->>Mgr: add_record(RunRecord)
    Note over Mgr: survives dialog close
    Dialog-->>Dialog: closed
    Dialog->>Mgr: get_records() on reopen
```

```{note}
Only the records the managers retain are state. We do **not** separately track
"old runs" beyond this history.
```

The dialogs' transient configuration (selected dataset, chosen component count,
the endmember table while editing) is `[EPHEMERAL]`.

---

(georeferencing-and-user-created-crss)=
## Georeferencing and user-created CRSs

### User-created CRSs

**File:** [reference_creator_dialog.py](../../../../src/wiser/gui/reference_creator_dialog.py)

When the user builds a coordinate reference system, the finished CRS is stored in
`ApplicationState._user_created_crs` as `{name: (osr.SpatialReference,
CrsCreatorState)}` — **[SOURCE]**. `CrsCreatorState` captures the construction
parameters (projection type, prime meridian, ellipsoid axes, latitudes, scale,
etc.) so the CRS can be re-edited. The in-dialog widgets used while building it
(`_lon_meridian`, `_proj_type`, `_crs_name`, …) are `[EPHEMERAL]` — only the
committed `CrsCreatorState` matters.

### Georeferencer control points

**File:** [geo_reference_dialog.py](../../../../src/wiser/gui/geo_reference_dialog.py)

The georeferencer table is `[SOURCE]` state that lives **on the dialog**, not on
`ApplicationState`:

- `_table_entry_list` — `List[GeoRefTableEntry]`. Each entry holds a
  `_gcp_pair` (ground-control-point pair: pixel ↔ reference coordinate), an
  `_enabled` flag, an `_id`, residuals (`_residual_x`, `_residual_y`), and a
  color.
- `_curr_output_srs` (`GeneralCRS`) — the chosen output CRS.
- `_curr_resample_alg` — GDAL resampling algorithm.
- `_curr_transform_type` (`TRANSFORM_TYPES`) — polynomial order / TPS.

Reusable: the dialog can already load/save control points to QGIS `.points` and
ENVI `.pts` formats.

The *result* of running georeferencing is a warped dataset (added via
`add_dataset()`) and/or an updated `SpatialMetadata` on an existing dataset — both
covered under [Datasets](#datasets-and-derived-rasters).

---

## Band-math expressions

**Files:** [bandmath_dialog.py](../../../../src/wiser/gui/bandmath_dialog.py),
[bandmath-internals.md](bandmath-internals.md)

The **saved-expression list** is `[SOURCE]` state — but note where it lives: it
is held directly in the combo-box widget `_ui.cbox_saved_exprs` (the items in the
combo box *are* the saved expressions), with `_saved_exprs_modified` tracking
unsaved changes. The list is loaded from / saved to a plain `.txt` file (one
expression per line). The current expression text sits in `_ui.ledit_expression`.

`[EPHEMERAL]` / out of scope:

- Variable bindings (`tbl_variables`), transient analysis info (`_expr_info`),
  and batch-job definitions are working state, not persisted session state.
- **Old band-math runs are explicitly not tracked.** Band-math *output rasters*
  are datasets (see [Datasets](#datasets-and-derived-rasters)).

---

(caches-and-temporary-storage)=
## Caches and temporary storage

**File:** [data_cache.py](../../../../src/wiser/raster/data_cache.py)

`DataCache` composes three FIFO, capacity-bounded caches — all **[DERIVED]**:

| Cache | Keyed on | Holds |
|-------|----------|-------|
| `RenderCache` | dataset + bands + stretch + colormap | rendered image arrays |
| `ComputationCache` | dataset + band + normalization | computed band arrays |
| `HistogramCache` | dataset + band + stretch/conditioner + bounds | histogram bins/edges |

Each is a `Cache` with an `OrderedDict` and a byte `_capacity`, evicting oldest
entries when full. None of it needs to be captured — it is all recomputed from
the datasets and stretches.

Temporary on-disk artifacts are likewise `[DERIVED]`/`[EPHEMERAL]`:

- Band-math intermediate files under `TEMP_FOLDER_PATH`
  ([bandmath/utils.py](../../../../src/wiser/bandmath/utils.py)), cleaned up after
  execution.
- Worker/storage-service scratch data used while a background task runs.

---

## Summary: what a session reconstruction must walk

```{mermaid}
flowchart LR
    subgraph must["Must capture — SOURCE"]
        d["Datasets (+ in-memory pixels)"]
        s["Stretches"]
        lib["Spectral libraries"]
        roi["ROIs"]
        spec["Active + collected spectra"]
        km["K-Means results"]
        hist["Run histories (unmix/PCA/MNF)"]
        crs["User-created CRSs"]
        cfg["Config"]
        gcp["Georef control-point table"]
        bm["Band-math saved expressions"]
    end

    subgraph skip["Can skip — DERIVED / EPHEMERAL"]
        cache["DataCache (render/comp/histogram)"]
        stats["Cached band stats"]
        proc["Running processes"]
        wins["Open plot/table windows"]
        forms["In-progress dialog fields"]
    end
```

The non-obvious part of the map: not all `[SOURCE]` state lives on
`ApplicationState`. The **georeferencer control-point table** and the
**band-math saved-expression list** are owned by their dialogs, and **in-memory
datasets** carry their pixels with no file to fall back on. Any reconstruction
routine has to reach into those places too.

If you wanted to have a good reconstruction of WISER's internal state, you would need to keep in mind the above. However, some of these things may be unncessary for a reconstruction.
