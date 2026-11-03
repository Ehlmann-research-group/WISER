# Georeferencer Internals

This page documents the internal architecture of WISER's **Georeferencer** — the
spatial tool that assigns or corrects a raster's coordinate reference system (CRS) by
collecting ground control points (GCPs) and warping the image onto real-world
coordinates. It is intended for developers reading, debugging, or extending the tool.

For the user-facing guide (what the tool does and how to operate it), see
[Spatial Tools](../user-content/spatial-tools.md). The georeferencer's two image panes
reuse the `RasterPane` infrastructure described in the
[Viewport System](viewport-system.md) page.

## Overview

The georeferencer is built from two cooperating halves:

- **The UI / control layer** — `GeoReferencerDialog` owns the whole dialog: two image
  panes (a *target* to be georeferenced and a *reference* providing real-world
  coordinates), a GCP table, and the output controls (CRS, transform type, resampling,
  save path). A `GeoReferencerTaskDelegate` interprets clicks and key presses in the
  panes and turns them into GCPs.

- **The data model** — a small hierarchy of plain Python classes that represent GCPs
  (`GroundControlPoint` and subclasses), GCP pairs (`GroundControlPointPair`), table
  rows (`GeoRefTableEntry`), and coordinate reference systems (`GeneralCRS` and
  subclasses).

The actual transformation math is delegated to **GDAL/OSR**: GCPs become `gdal.GCP`
objects, the transform is a `gdal.Transformer` / `gdal.Warp` call, and CRSs are
`osr.SpatialReference` objects.

The dialog is created lazily from `App.show_geo_reference_dialog` in
`src/wiser/gui/app.py`, passed the shared `ApplicationState` and `app_services` (the latter
provides the work scheduler and progress infrastructure used to run warps off the GUI
thread). The old `main_view` constructor argument was dead and has been removed.

As of WISER#684 the dialog is a thin UI/controller: the GDAL warp engine, GCP file I/O,
and the CRS model have each been extracted into Qt-free modules under `src/wiser/raster/`
(mirroring the Seamless Mosaic's `mosaic_export.py` split), so warp correctness, GCP
round-tripping, and CRS resolution are all unit-testable without a running app.

**Core files:**

| File | Responsibility |
|------|----------------|
| `src/wiser/gui/geo_reference_dialog.py` | Dialog/controller and GCP table; delegates warp, GCP I/O, and CRS to the modules below |
| `src/wiser/gui/geo_reference_config.py` | `GeoReferencerConfig` — presets + lock flags to drive the dialog programmatically |
| `src/wiser/raster/georef_warp.py` | Qt-free warp engine: `build_warp_kwargs`, `compute_residuals`, `warp_dataset_to_path` (+ `TRANSFORM_TYPES`, `RESAMPLE_ALGORITHMS`) |
| `src/wiser/raster/gcp_io.py` | Qt-free GCP persistence: `read_gcp_file`, `write_qgis_points`, `write_envi_pts` |
| `src/wiser/raster/crs_model.py` | Shared Qt-free CRS hierarchy (`GeneralCRS` and subclasses, `COMMON_SRS`) |
| `src/wiser/gui/geo_reference_task_delegate.py` | Input event state machine, GCP data model |
| `src/wiser/gui/geo_reference_pane.py` | `GeoReferencerPane` — a stripped-down `RasterPane` |
| `src/wiser/raster/dataset.py` | Pixel ↔ spatial coordinate helpers used to build GCPs |

---

## Class Hierarchy

```{mermaid}
classDiagram
    direction TB

    class QDialog["QDialog (Qt)"]
    class RasterPane["RasterPane (see Viewport System)"]
    class TaskDelegate["TaskDelegate"]

    class GeoReferencerDialog {
        geo_reference_dialog.py
        +gcp_pair_added : Signal
        +gcp_add_attempt : Signal
        +warp_completed : Signal
        -_table_entry_list : list~GeoRefTableEntry~
        +set_target_dataset() / set_reference_dataset()
        +_apply_config(GeoReferencerConfig)
        +_schedule_residual_recompute()
        +_create_warped_output()
    }

    class GeoReferencerPane {
        geo_reference_pane.py
        -_pane_type : PointSelectorType
        +get_point_selector_type()
        +set_task_delegate()
    }

    class GeoReferencerTaskDelegate {
        geo_reference_task_delegate.py
        -_state : GeoReferencerState
        -_current_point_pair : GroundControlPointPair
        +on_mouse_release()
        +handle_point_click_logic()
        +handle_enter_key_release()
        +handle_escape_key_release()
    }

    class GroundControlPoint {
        <<abstract>>
        +get_spatial_point()
        +get_selector_type()
    }
    class GroundControlPointRasterPane {
        +_point : (col,row)
        +get_scaled_point()
    }
    class GroundControlPointCoordinate {
        +_spatial_coord
        +_srs
    }
    class GroundControlPointPair {
        -_target_gcp
        -_ref_gcp
        +add_gcp()
        +has_both_gcps()
    }
    class GeoRefTableEntry {
        +get_gcp_pair()
        +is_enabled()
        +get_residual_x() / _y()
    }

    class GeneralCRS {
        <<abstract>>
        crs_model.py
        +get_osr_crs()
    }
    class AuthorityCodeCRS
    class UserGeneratedCRS
    class WktGeneratedCRS

    QDialog <|-- GeoReferencerDialog : subclass
    RasterPane <|-- GeoReferencerPane : subclass
    TaskDelegate <|-- GeoReferencerTaskDelegate : subclass

    GroundControlPoint <|-- GroundControlPointRasterPane
    GroundControlPoint <|-- GroundControlPointCoordinate
    GeneralCRS <|-- AuthorityCodeCRS
    GeneralCRS <|-- UserGeneratedCRS
    GeneralCRS <|-- WktGeneratedCRS

    GeoReferencerDialog --> GeoReferencerPane : owns 2 (target + reference)
    GeoReferencerDialog --> GeoReferencerTaskDelegate : owns
    GeoReferencerDialog --> GeoRefTableEntry : owns list
    GeoReferencerDialog --> GeneralCRS : output / reference CRS
    GeoReferencerPane --> GeoReferencerTaskDelegate : forwards events to
    GeoReferencerTaskDelegate --> GroundControlPointPair : builds
    GroundControlPointPair --> GroundControlPoint : target + reference
    GeoRefTableEntry --> GroundControlPointPair : wraps
```

---

## Class Responsibilities

### GeoReferencerDialog

**File:** `src/wiser/gui/geo_reference_dialog.py`

**Purpose:** The top-level controller. Builds and wires the UI, owns the canonical list
of GCP table rows (`_table_entry_list`), and runs both the residual computation and the
final warp.

**Controls:**
- Constructing the two `GeoReferencerPane` instances (TARGET and REFERENCE) and the
  shared `GeoReferencerTaskDelegate`
- The GCP `QTableWidget` and its backing `List[GeoRefTableEntry]`, including add/remove,
  enable/disable, per-row color, and inline coordinate edits
- The output controls: output CRS chooser (`cbox_srs`), resampling algorithm
  (`cbox_interpolation`), transform type (`cbox_poly_order`), and the save path
- Scheduling a debounced, off-thread residual recompute (`_schedule_residual_recompute()`)
  on every relevant change, and launching the threaded output warp
  (`_create_warped_output()`) from the **Run Warp** button
- Saving/loading GCPs to/from disk (thin wrappers over `wiser.raster.gcp_io`)

**Does not control:**
- Per-click GCP state transitions (delegated to `GeoReferencerTaskDelegate`)
- Low-level rendering, zoom, and coordinate conversion (inherited `RasterPane` /
  `RasterView` behavior)
- The transformation math itself (delegated to GDAL)

**Signals:**

| Signal | Argument | Emitted when |
|--------|----------|--------------|
| `gcp_pair_added` | `GroundControlPointPair` | A complete target+reference pair is finalized |
| `gcp_add_attempt` | `GroundControlPoint` | A reference point is added via manual lat/lon entry |
| `warp_completed` | `str` (output path) | A **Run Warp** finishes successfully on its worker thread |

---

### GeoReferencerPane

**File:** `src/wiser/gui/geo_reference_pane.py`

**Purpose:** A purpose-built `RasterPane` subclass (also implements `PointSelector`)
used for both the target and reference image. It strips out features the georeferencer
does not need — dataset adding, ROI/selection tools — and routes raw input events to the
task delegate.

**Controls:**
- Its `PointSelectorType` (`TARGET_POINT_SELECTOR` or `REFERENCE_POINT_SELECTOR`),
  returned by `get_point_selector_type()` — this is how the rest of the system tells the
  two panes apart
- Forwarding `mouseRelease`, `keyPress`, and `keyRelease` events to the task delegate
  (`_onRasterMouseRelease`, `_onRasterKeyRelease`, …), then refreshing the view
- Drawing GCP markers via the delegate's `draw_state()` in `_afterRasterPaint`
- A wider zoom range than a normal pane (up to 64×) for precise point placement

**Does not control:**
- GCP state or pairing logic (delegated)
- ROI/selection tools (deliberately disabled — `_init_select_tools` is a no-op)

---

### GeoReferencerTaskDelegate

**File:** `src/wiser/gui/geo_reference_task_delegate.py`

**Purpose:** The input state machine. A single delegate instance is shared by both panes
and converts the sequence of clicks and ENTER/ESC presses into completed
`GroundControlPointPair`s. It also draws the GCP markers.

**Controls:**
- The current `GeoReferencerState` and the in-progress `_current_point_pair`
- `handle_point_click_logic()` — what a click means in the current state
- `handle_enter_key_release()` / `handle_escape_key_release()` — confirm / undo
- `_on_gcp_add_attempt()` — the manual-entry path (a reference point typed as lat/lon)
- Painting completed and in-progress GCP markers (`draw_state`)
- `check_state()` — defensive assertions that the internal fields are consistent with
  the declared state after every transition

**Does not control:**
- The GCP table or residual computation (it only emits `gcp_pair_added`; the dialog
  reacts)

---

### The GCP data model

**File:** `src/wiser/gui/geo_reference_task_delegate.py`

- **`GroundControlPoint`** (ABC) — the minimum a GCP needs: a spatial point, its CRS,
  and which selector (pane) it belongs to. Note it does **not** require a raster/pixel
  coordinate, because a real-world GCP is fundamentally a spatial coordinate.
- **`GroundControlPointRasterPane`** — a GCP created by clicking a pane. Stores the
  pixel coordinate `_point` and the pane's dataset, and derives the spatial coordinate
  on demand via `dataset.to_geographic_coords(point)`. `set_spatial_point()` does the
  reverse using `dataset.geo_to_pixel_coords_exact()`.
- **`GroundControlPointCoordinate`** — a purely spatial GCP (no pixel coordinate), used
  for the manual reference-point entry path. Holds `_spatial_coord` and an explicit
  `_srs`.
- **`GroundControlPointPair`** — holds one target GCP and one reference GCP. `add_gcp()`
  routes an incoming GCP to the right slot based on its `get_selector_type()`, so order
  of insertion does not matter.

### GeoRefTableEntry

**File:** `src/wiser/gui/geo_reference_dialog.py`

**Purpose:** The model object behind one row of the GCP table. Wraps a
`GroundControlPointPair` plus presentation/derived state: `enabled`, `id`, the computed
residuals (`residual_x`, `residual_y`), and a hex `color`. Column indices are defined by
the `COLUMN_ID` enum.

### The CRS model

**File:** `src/wiser/raster/crs_model.py`

All CRSs are represented through the `GeneralCRS` ABC, whose single contract is
`get_osr_crs() -> osr.SpatialReference`. This lets the dialog treat every CRS source
uniformly (and compare them by WKT via `__eq__`). The hierarchy is Qt-free and now shared
with the Seamless Mosaic's CRS chooser (`mosaic_crs_dialog.py`), so both features offer the
same CRSs from a single source of truth (`geo_reference_dialog.py` re-exports the names for
backwards compatibility):

| Class | Built from |
|-------|------------|
| `AuthorityCodeCRS` | An authority name + code, e.g. `EPSG:4326` (`SetFromUserInput`) |
| `UserGeneratedCRS` | A custom `osr.SpatialReference` from the [CRS Creator](crs-creator-internals.md) |
| `WktGeneratedCRS` | A raw WKT string (e.g. recovered from a loaded GCP file) |

`COMMON_SRS` provides a few built-in `AuthorityCodeCRS` entries (WGS84, Web Mercator,
NAD83 / UTM 15N) that always appear in the output-CRS chooser.

---

## The GCP Collection State Machine

Collecting one GCP pair is a four-step dance: click a point in one pane, press ENTER to
"lock" it, click the corresponding point in the other pane, press ENTER again to commit
the pair. `GeoReferencerState` tracks where the user is in that cycle. (There is a
transient `SECOND_POINT_ENTERED` state that exists only for code clarity — it is set and
then immediately replaced by `NOTHING_SELECTED`.)

```{mermaid}
stateDiagram-v2
    [*] --> NOTHING_SELECTED
    NOTHING_SELECTED --> FIRST_POINT_SELECTED : click a pane
    FIRST_POINT_SELECTED --> FIRST_POINT_SELECTED : click again (re-place point)
    FIRST_POINT_SELECTED --> NOTHING_SELECTED : ESC (discard)
    FIRST_POINT_SELECTED --> FIRST_POINT_ENTERED : ENTER (lock first point)
    FIRST_POINT_ENTERED --> SECOND_POINT_SELECTED : click the OTHER pane
    FIRST_POINT_ENTERED --> FIRST_POINT_SELECTED : ESC (unlock)
    SECOND_POINT_SELECTED --> FIRST_POINT_ENTERED : ESC (remove second point)
    SECOND_POINT_SELECTED --> NOTHING_SELECTED : ENTER (commit pair → gcp_pair_added)
```

Manual reference entry is a shortcut: `_on_gcp_add_attempt()` injects a
`GroundControlPointCoordinate` directly, jumping from `NOTHING_SELECTED` /
`FIRST_POINT_SELECTED` to `FIRST_POINT_ENTERED`, or from `FIRST_POINT_ENTERED` /
`SECOND_POINT_SELECTED` straight to a committed pair.

**Transitions and handlers** (all in `geo_reference_task_delegate.py`):

| From state | Input | Handler | Result |
|------------|-------|---------|--------|
| `NOTHING_SELECTED` | click a pane | `handle_point_click_logic` | new pair, `FIRST_POINT_SELECTED` |
| `FIRST_POINT_SELECTED` | ENTER | `handle_enter_key_release` | `FIRST_POINT_ENTERED` |
| `FIRST_POINT_SELECTED` | ESC | `handle_escape_key_release` | discard, `NOTHING_SELECTED` |
| `FIRST_POINT_ENTERED` | click the *other* pane | `handle_point_click_logic` | `add_gcp`, `SECOND_POINT_SELECTED` |
| `FIRST_POINT_ENTERED` | ESC | `handle_escape_key_release` | back to `FIRST_POINT_SELECTED` |
| `SECOND_POINT_SELECTED` | ENTER | `handle_enter_key_release` | emit `gcp_pair_added`, `NOTHING_SELECTED` |
| `SECOND_POINT_SELECTED` | ESC | `handle_escape_key_release` | remove 2nd GCP, `FIRST_POINT_ENTERED` |
| any | manual ref entry | `_on_gcp_add_attempt` | inject `GroundControlPointCoordinate` |

A guardrail throughout: clicking the *same* pane twice in a row (instead of alternating
target/reference) does not advance the machine — the delegate posts a message telling
the user to press ENTER or ESC first.

---

## Click → GCP → Table Data Flow

The pane forwards raw Qt events to the delegate; the delegate emits `gcp_pair_added`
once a pair is complete; the dialog reacts by adding a table row and recomputing
residuals.

```{mermaid}
sequenceDiagram
    participant User
    participant Pane as GeoReferencerPane
    participant Del as GeoReferencerTaskDelegate
    participant Dlg as GeoReferencerDialog
    participant Table as GCP Table

    User->>Pane: click target image
    Pane->>Del: on_mouse_release()
    Del->>Del: handle_point_click_logic()<br/>→ FIRST_POINT_SELECTED
    User->>Pane: press ENTER
    Pane->>Del: on_key_release()
    Del->>Del: handle_enter_key_release()<br/>→ FIRST_POINT_ENTERED
    User->>Pane: click reference image
    Pane->>Del: on_mouse_release()
    Del->>Del: add_gcp() → SECOND_POINT_SELECTED
    User->>Pane: press ENTER
    Pane->>Del: on_key_release()
    Del->>Dlg: gcp_pair_added.emit(pair)
    Dlg->>Table: _on_gcp_pair_added()<br/>add GeoRefTableEntry row
    Dlg->>Dlg: _schedule_residual_recompute()<br/>(debounced, off-thread)
    Dlg->>Table: _apply_residuals() per row (GUI thread)
```

Editing a cell in the table (`_on_cell_changed`), toggling a row's *enabled* checkbox,
or switching the output CRS / reference CRS / transform type all call
`_schedule_residual_recompute()` so the residual columns stay live — see
[Residual Computation](#residual-computation-schedule_residual_recompute) for how that runs
off the GUI thread.

---

## Transformation Models

The transform type is chosen from the `TRANSFORM_TYPES` enum (defined in
`wiser.raster.georef_warp`). Each maps to a GDAL transformer method and has a minimum GCP
count (`min_points_per_transform`):

| Transform | `TRANSFORM_TYPES` | Min GCPs | GDAL mapping | Use when |
|-----------|-------------------|----------|--------------|----------|
| Affine | `POLY_1` | 3 | `METHOD=GCP_POLYNOMIAL`, `MAX_GCP_ORDER=1` | Pure translate/scale/rotate/shear |
| Polynomial 2 | `POLY_2` | 6 | `METHOD=GCP_POLYNOMIAL`, `MAX_GCP_ORDER=2` | Mild, smooth distortion |
| Polynomial 3 | `POLY_3` | 10 | `METHOD=GCP_POLYNOMIAL`, `MAX_GCP_ORDER=3` | Stronger distortion |
| Thin Plate Spline | `TPS` | 10 | `tps=True`, `METHOD=GCP_TPS`, `MAX_GCP_ORDER=-1` | Local, non-uniform warping; passes through all GCPs |

The mapping lives in one place — `georef_warp.build_warp_kwargs(resample_alg,
transform_type, output_srs)` — which returns the `gdal.Warp` kwargs plus the matching
`gdal.Transformer` options. Both the residual calc and the final warp call it, so they can
never drift apart.

---

(residual-computation-schedule_residual_recompute)=

## Residual Computation (`_schedule_residual_recompute`)

Residuals are recomputed after every change to give the user immediate feedback on how
well each GCP fits the chosen transform. The math itself
(`georef_warp.compute_residuals`) does **not** warp the real image — it builds a 1×1
placeholder dataset purely to drive GDAL's transformer — but because it runs a full
`gdal.Warp` per call it must not block the GUI.

The dialog therefore **debounces and single-flights** the recompute (mirroring
`MosaicView`'s off-thread pixel reads):

1. An edit calls `_schedule_residual_recompute()`, which (re)starts a ~150 ms single-shot
   `QTimer`.
2. When the timer fires, `_recompute_residuals_async()` snapshots the GCPs + SRSs on the
   GUI thread (`_snapshot_residual_inputs()`), bumps an in-flight token
   (`_residual_signature`), and submits `compute_residuals` to
   `app_services.scheduler.submit_thread`.
3. The worker's result is delivered back to the GUI thread via the queued `_residuals_ready`
   signal; `_apply_residuals()` drops it if a newer recompute has superseded the token,
   otherwise writes the residual columns.

(When no scheduler is available — e.g. some unit contexts — it falls back to a synchronous
`_georeference()`.)

```{mermaid}
flowchart TD
    A["_get_entry_gcp_list()<br/>enabled rows → gdal.GCP"] --> B["build output_srs + ref_srs<br/>(OAMS_TRADITIONAL_GIS_ORDER)"]
    B --> C["build_warp_kwargs()<br/>→ warp_kwargs + transformerOptions"]
    C --> D["gdal.Transformer(temp_ds, options)<br/>pixel → output SRS"]
    D --> E["per GCP: TransformPoint(pixel)<br/>→ output-SRS coord"]
    E --> F["CoordinateTransformation<br/>output SRS → reference SRS"]
    F --> G["spatial error = gcp.GCPX/Y − transformed X/Y"]
    G --> H["pixel error = spatial error ÷<br/>warped geotransform pixel size"]
    H --> I["return residuals →<br/>entry.set_residual_x/y() on GUI thread"]
```

Key details:

- `_get_entry_gcp_list()` skips disabled rows and builds
  `gdal.GCP(spatial_x, spatial_y, 0, pixel_x, pixel_y)` — the spatial coordinate comes
  from the *reference* GCP, the pixel coordinate from the *target* GCP.
- Both the output SRS (`_import_current_output_srs`) and reference SRS
  (`_get_reference_srs`) are forced to `OAMS_TRADITIONAL_GIS_ORDER` so axis ordering
  (lat/lon vs lon/lat) does not silently flip coordinates.
- The error is first measured in reference-SRS units, then converted to pixels by
  dividing by the warped geotransform's pixel width/height (`transformed_gt[1]`,
  `transformed_gt[5]`), which is why the residuals are reported in pixels.

---

(warp-output-pipeline-warp_dataset_to_path)=

## Warp / Output Pipeline (`warp_dataset_to_path`)

The **Run Warp** button (`_on_warp_button_clicked` → `_create_warped_output()`) produces
the georeferenced GeoTIFF. `_create_warped_output()` only *validates and launches*: it
gathers the GCPs, resolves the SRSs, builds the warp kwargs, and hands
`georef_warp.warp_dataset_to_path` to `run_with_progress`, which runs it on the work
scheduler with a progress dialog and cancellation. On success `_on_warp_done` emits
`warp_completed(path)`; on failure/cancel `_on_warp_error` surfaces the message. The dialog's
`accept()` no longer warps — the previous double-warp (both `accept()` **and** the button
calling the warp) is gone, so the OK button is now a plain commit/close.

Because hyperspectral cubes can be very large, `warp_dataset_to_path` processes the band
data in RAM-bounded chunks, reporting `progress` and calling `raise_if_cancelled()` between
chunks.

```{mermaid}
flowchart TD
    V["_create_warped_output: validate<br/>+ build_warp_kwargs"] --> R["run_with_progress →<br/>warp_dataset_to_path (worker thread)"]
    R --> P["probe output size:<br/>warp band 0 to /vsimem"]
    P --> Branch{target impl & size}
    Branch -->|"GDALRasterDataImpl"| G["Translate → VRT,<br/>SetGCPs, gdal.Warp whole dataset"]
    Branch -->|"numpy & fits in RAM"| N["OpenNumPyArray,<br/>SetGCPs, gdal.Warp whole array"]
    Branch -->|"too big for RAM"| C["chunk bands by MAX_RAM_BYTES:<br/>warp each chunk, write incrementally<br/>(progress + raise_if_cancelled)"]
    C --> M["driver.Create GTiff,<br/>SetGeoTransform + SetSpatialRef"]
    G --> F["copy_metadata_to_gdal_dataset, FlushCache"]
    N --> F
    M --> F
    F --> S["warp_completed.emit(path)"]
```

`warp_kwargs` carries everything GDAL needs:

| Key | Value | Notes |
|-----|-------|-------|
| `copyMetadata` | `True` | Preserve source metadata |
| `resampleAlg` | a `GRA_*` constant | From the interpolation chooser |
| `dstSRS` | output `osr.SpatialReference` | The chosen output CRS |
| `polynomialOrder` / `tps` | `1`/`2`/`3` or `True` | Set per transform type |
| `transformerOptions` | `["METHOD=…", "MAX_GCP_ORDER=…"]` | Mirrors the transform type |

The available resampling algorithms are discovered dynamically (in
`wiser.raster.georef_warp`):
`RESAMPLE_ALGORITHMS = {name: getattr(gdal, name) for name in dir(gdal) if name.startswith("GRA_")}`
— i.e. all of GDAL's `GRA_NearestNeighbour`, `GRA_Bilinear`, `GRA_Cubic`,
`GRA_CubicSpline`, `GRA_Lanczos`, etc.

In every branch the GCPs are attached to a temporary GDAL dataset via
`SetGCPs(gcps, ref_projection)` *before* warping, so GDAL derives the geometric
transform from the GCPs rather than from any pre-existing geotransform on the source.

---

## GCP Persistence

GCPs can be saved and reloaded in two formats, chosen by file extension. The parsing /
writing is Qt-free and lives in `wiser.raster.gcp_io`; the dialog's
`_on_save_gcps_clicked` / `_on_load_gcps_clicked` are thin wrappers (the writers take the
dialog's flattened `(map_x, map_y, pixel_x, pixel_y, enabled)` rows from `_get_gcp_rows()`):

| Format | Extension | CRS header | Writer / reader (`gcp_io`) |
|--------|-----------|------------|-----------------|
| QGIS points | `.points` | `# CRS, EPSG:code` row (CSV) | `write_qgis_points` / `read_qgis_points_file` |
| ENVI points | `.pts` | `; projection info = {auth, code, units=Degrees}` comment | `write_envi_pts` / `read_envi_pts_file` |

On load, `gcp_io.read_gcp_file` dispatches by extension; the dialog's `load_gcps_and_srs`
rebuilds the `GeoRefTableEntry` rows and the associated `GeneralCRS`. Both readers fall back
to an embedded WKT line if the authority header is missing. `compare_srs_lenient` is used to
reconcile a loaded file's CRS against the current reference CRS.

---

(programmatic-configuration-locking)=

## Programmatic Configuration & Locking

The dialog can be driven entirely through code — not just its own combo boxes — via a
`GeoReferencerConfig` (`src/wiser/gui/geo_reference_config.py`). This is what lets another
feature reuse the dialog with a fixed target, a caller-owned save path, and a custom accept
button, without touching the Tools-menu flow. The Seamless Mosaic's
[**Re-georeferencing a Scene In Place**](mosaic-internals.md#re-georeferencing-a-scene-in-place)
is the production consumer: it opens a task-scoped dialog locked onto a mosaic scene, then
reingests the `warp_completed` output and swaps the corrected scene back into the mosaic.

`show(config=None)` / `exec_(config=None)` accept an optional config and delegate to
`_apply_config`, which first resets the dialog to its classic baseline (repopulate choosers,
clear any prior locks / accept-button label) and then, if a config is given, applies presets
**before** locks:

| Config field | Effect |
|--------------|--------|
| `target_dataset` | `set_target_dataset(ds)` — show it in the target pane |
| `reference_dataset` | `set_reference_dataset(ds)` — show it in the reference pane |
| `reference_crs` | `set_reference_crs(crs)` — manual reference CRS (used when there is no reference dataset) |
| `save_path` | `set_save_path(path)` — preset the output path |
| `allow_change_target` = False | `_set_target_locked(True)` — disable the target chooser |
| `allow_change_reference` = False | `_set_reference_locked(True)` — disable the reference chooser + suppress the manual-ref toggle |
| `allow_change_save_path` = False | `_set_save_path_locked(True)` — make the path read-only, disable the choose button, skip its validation |
| `accept_button_text` | relabel the OK button (e.g. "Save to Mosaic") |

The `set_*` setters are the single shared apply-path: the interactive chooser slots
(`_on_switch_target_dataset` / `_on_switch_reference_dataset`) keep the user-facing
confirm-discard-GCPs prompts, then funnel through the same setters, while a programmatic
caller calls the setters directly (no prompts). `config=None` reproduces the Tools-menu
behavior exactly, so the reused singleton is safe to reopen either way.

---

## Integration with WISER

- **`ApplicationState`** — the dialog reads open datasets to populate the target /
  reference choosers, and reads `get_user_created_crs()` to add each
  [CRS Creator](crs-creator-internals.md) CRS to the output-CRS chooser as a
  `UserGeneratedCRS` (alongside the reference dataset's own CRS and the `COMMON_SRS`
  presets).
- **Viewport reuse** — `GeoReferencerPane` extends `RasterPane`, so it inherits the
  rendering, zoom, and coordinate-conversion machinery documented in the
  [Viewport System](viewport-system.md) page; the georeferencer only adds point
  selection on top.
- **`RasterDataSet`** — GCP spatial coordinates come from
  `RasterDataSet.to_geographic_coords()` and `geo_to_pixel_coords_exact()`, and CRSs
  from `get_spatial_ref()` (`src/wiser/raster/dataset.py`).
- **Work scheduler / progress** — the final warp and the interactive residual recompute
  both run off the GUI thread through `app_services` (`run_with_progress` +
  `scheduler.submit_thread`), the same stack the Seamless Mosaic uses; see
  [Residual Computation](#residual-computation-schedule_residual_recompute) and
  [Warp / Output Pipeline](#warp-output-pipeline-warp_dataset_to_path).
- **Entry point** — `App.show_geo_reference_dialog` (`src/wiser/gui/app.py`) constructs
  the dialog lazily with the shared `ApplicationState` and `app_services`, then
  `exec_()`s it (optionally with a `GeoReferencerConfig`).
