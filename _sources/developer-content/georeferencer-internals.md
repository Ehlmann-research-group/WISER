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
`src/wiser/gui/app.py`, passed the shared `ApplicationState` and the main view.

**Core files:**

| File | Responsibility |
|------|----------------|
| `src/wiser/gui/geo_reference_dialog.py` | Dialog/controller, GCP table, CRS classes, residual + warp pipeline |
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
        -_table_entry_list : list~GeoRefTableEntry~
        -_warp_kwargs : dict
        +_georeference()
        +_create_warped_output()
        +accept()
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
- Triggering `_georeference()` (recompute residuals) on every relevant change, and
  `_create_warped_output()` when the dialog is accepted
- Saving/loading GCPs to/from disk

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

**File:** `src/wiser/gui/geo_reference_dialog.py`

All CRSs are represented through the `GeneralCRS` ABC, whose single contract is
`get_osr_crs() -> osr.SpatialReference`. This lets the dialog treat every CRS source
uniformly (and compare them by WKT via `__eq__`):

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
    Dlg->>Dlg: _georeference() (recompute residuals)
    Dlg->>Table: _update_residuals() per row
```

Editing a cell in the table (`_on_cell_changed`), toggling a row's *enabled* checkbox,
or switching the output CRS / reference CRS / transform type all re-enter
`_georeference()` so the residual columns stay live.

---

## Transformation Models

The transform type is chosen from the `TRANSFORM_TYPES` enum. Each maps to a GDAL
transformer method and has a minimum GCP count (`min_points_per_transform`):

| Transform | `TRANSFORM_TYPES` | Min GCPs | GDAL mapping | Use when |
|-----------|-------------------|----------|--------------|----------|
| Affine | `POLY_1` | 3 | `METHOD=GCP_POLYNOMIAL`, `MAX_GCP_ORDER=1` | Pure translate/scale/rotate/shear |
| Polynomial 2 | `POLY_2` | 6 | `METHOD=GCP_POLYNOMIAL`, `MAX_GCP_ORDER=2` | Mild, smooth distortion |
| Polynomial 3 | `POLY_3` | 10 | `METHOD=GCP_POLYNOMIAL`, `MAX_GCP_ORDER=3` | Stronger distortion |
| Thin Plate Spline | `TPS` | 10 | `tps=True`, `METHOD=GCP_TPS`, `MAX_GCP_ORDER=-1` | Local, non-uniform warping; passes through all GCPs |

These selections are written into `_warp_kwargs` and `_transform_options` in
`_georeference()`, and reused unchanged by `_create_warped_output()`.

---

## Residual Computation (`_georeference`)

`_georeference()` runs after every change to give the user immediate feedback on how
well each GCP fits the chosen transform. It does **not** warp the real image — it builds
a 1×1 placeholder dataset purely to drive GDAL's transformer.

```{mermaid}
flowchart TD
    A["_get_entry_gcp_list()<br/>enabled rows → gdal.GCP"] --> B["build output_srs + ref_srs<br/>(OAMS_TRADITIONAL_GIS_ORDER)"]
    B --> C["assemble _warp_kwargs +<br/>transformerOptions (per transform type)"]
    C --> D["gdal.Transformer(temp_ds, options)<br/>pixel → output SRS"]
    D --> E["per GCP: TransformPoint(pixel)<br/>→ output-SRS coord"]
    E --> F["CoordinateTransformation<br/>output SRS → reference SRS"]
    F --> G["spatial error = gcp.GCPX/Y − transformed X/Y"]
    G --> H["pixel error = spatial error ÷<br/>warped geotransform pixel size"]
    H --> I["entry.set_residual_x/y()<br/>→ dX/dY columns"]
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

## Warp / Output Pipeline (`_create_warped_output`)

Accepting the dialog calls `accept()` → `_create_warped_output()`, which produces the
georeferenced GeoTIFF. Because hyperspectral cubes can be very large, the band data is
processed in RAM-bounded chunks.

```{mermaid}
flowchart TD
    V["validate: save path, enough GCPs, target selected"] --> P["probe output size:<br/>warp band 0 to /vsimem"]
    P --> Branch{target impl & size}
    Branch -->|"GDALRasterDataImpl"| G["Translate → VRT,<br/>SetGCPs, gdal.Warp whole dataset"]
    Branch -->|"numpy & fits in RAM"| N["OpenNumPyArray,<br/>SetGCPs, gdal.Warp whole array"]
    Branch -->|"too big for RAM"| C["chunk bands by MAX_RAM_BYTES:<br/>warp each chunk, write incrementally"]
    C --> M["driver.Create GTiff,<br/>SetGeoTransform + SetSpatialRef"]
    G --> F["copy_metadata_to_gdal_dataset, FlushCache"]
    N --> F
    M --> F
```

`_warp_kwargs` carries everything GDAL needs:

| Key | Value | Notes |
|-----|-------|-------|
| `copyMetadata` | `True` | Preserve source metadata |
| `resampleAlg` | a `GRA_*` constant | From the interpolation chooser |
| `dstSRS` | output `osr.SpatialReference` | The chosen output CRS |
| `polynomialOrder` / `tps` | `1`/`2`/`3` or `True` | Set per transform type |
| `transformerOptions` | `["METHOD=…", "MAX_GCP_ORDER=…"]` | Mirrors the transform type |

The available resampling algorithms are discovered dynamically:
`RESAMPLE_ALGORITHMS = {name: getattr(gdal, name) for name in dir(gdal) if name.startswith("GRA_")}`
— i.e. all of GDAL's `GRA_NearestNeighbour`, `GRA_Bilinear`, `GRA_Cubic`,
`GRA_CubicSpline`, `GRA_Lanczos`, etc.

In every branch the GCPs are attached to a temporary GDAL dataset via
`SetGCPs(gcps, ref_projection)` *before* warping, so GDAL derives the geometric
transform from the GCPs rather than from any pre-existing geotransform on the source.

---

## GCP Persistence

GCPs can be saved and reloaded in two formats, chosen by file extension in
`_on_save_gcps_clicked` / `_on_load_gcps_clicked`:

| Format | Extension | CRS header | Writer / reader |
|--------|-----------|------------|-----------------|
| QGIS points | `.points` | `# CRS, EPSG:code` row (CSV) | `_write_qgis_points_file` / `_read_qgis_points_file` |
| ENVI points | `.pts` | `; projection info = {auth, code, units=Degrees}` comment | `_write_envi_pts_file` / `_read_envi_pts_file` |

On load, `_read_gcp_file` dispatches by extension; `load_gcps_and_srs` rebuilds the
`GeoRefTableEntry` rows and the associated `GeneralCRS`. Both readers fall back to an
embedded WKT line if the authority header is missing. `compare_srs_lenient` is used to
reconcile a loaded file's CRS against the current reference CRS.

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
- **Entry point** — `App.show_geo_reference_dialog` (`src/wiser/gui/app.py`) constructs
  the dialog lazily with the shared `ApplicationState` and the main view, then
  `exec_()`s it.
