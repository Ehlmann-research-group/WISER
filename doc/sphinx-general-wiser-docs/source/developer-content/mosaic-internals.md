# Seamless Mosaic Internals

This page documents the internal architecture of WISER's **Seamless Mosaic** feature —
combining several overlapping georeferenced scenes onto one shared output grid. It is
intended for developers reading, debugging, or extending the tool.

```{note}
This is an in-progress feature (EPIC #629). As of this writing, scene ingestion,
materialization, common-grid/CRS resolution, and the **vector overlay** (footprints,
bounding box, overlap highlight) are implemented and wired into the GUI. **Pixel
compositing (the preview) is not yet implemented** — see [What Isn't Built
Yet](#what-isnt-built-yet).
```

For the design rationale and full child-issue breakdown, see `EPIC_seamless_mosaic.md`
in the repo root. There is no user-facing guide yet since the feature is not
preview/export complete.

## Overview

The feature is built from three cooperating layers:

- **The UI / control layer** — `SeamlessMosaicDialog` is a non-modal, cached top-level
  window (like the other tool dialogs) that hosts a `MosaicPane`. The pane owns an
  "Add Scene" picker, a scene stack (z-order + visibility), and a target-CRS control,
  and drives the ingestion pipeline on a background thread.

- **The non-GUI model** — `MosaicController` is the single source of truth for the
  scene list, z-order, resolution mode, target CRS, and the computed `CommonGrid`. It
  is deliberately Qt-free (may use `osgeo.gdal/ogr/osr`) so it is unit-testable without
  a running application.

- **The rendering layer** — `MosaicView` is a `QWidget` sibling of `RasterView`
  (not a subclass — a mosaic is N scenes on one shared world grid, not one dataset
  zoomed). It draws the **vector overlay** (footprints, bounding box, overlap
  highlight) on a QGIS-style unbounded canvas via a world→screen camera
  (`MosaicViewTransform`); the pixel-compositing layer beneath it is still a stubbed
  seam (#637). See [The Geometry Overlay](#the-geometry-overlay-vector-layer).

Every scene, regardless of its original backing (GDAL file, NumPy array, PDR, etc.),
is first turned into a warpable, disk-backed tiled GeoTIFF by a `SceneMaterializer` —
the `RasterDataSet` is the source of truth for metadata, so materialization stamps
SRS/geotransform/nodata/band metadata from the dataset object, not from its `_impl`.

**Core files:**

| File | Responsibility |
|------|----------------|
| `src/wiser/gui/mosaic_dialog.py` | `SeamlessMosaicDialog` — top-level dialog shell, owns the session `SceneMaterializer` |
| `src/wiser/gui/mosaic_pane.py` | `MosaicPane` — control panel; ingestion orchestration, scene list, CRS controls |
| `src/wiser/gui/mosaic_view.py` | `MosaicView` — the two-layer paint surface (pixel + vector), currently a stub |
| `src/wiser/gui/mosaic_crs_dialog.py` | `ReprojectPromptDialog` — modal target-CRS chooser |
| `src/wiser/raster/mosaic_controller.py` | `MosaicController`, `MosaicScene`, `CommonGrid`, `ResolutionMode` — the non-GUI model |
| `src/wiser/raster/mosaic_ingestion.py` | `validate_scene`, `build_overviews`, `compute_footprint_wkt` — Qt-free ingestion gates |
| `src/wiser/raster/mosaic_materialize.py` | `SceneMaterializer`, `materialize_to_tiled_geotiff` — the object-model adapter |
| `src/wiser/utils/progress.py` | `ProgressReporter` — Qt-free progress plumbing shared with any scheduler task |
| `src/wiser/gui/progress_task.py` | `run_with_progress` — reusable modal + Activity Monitor bridge for scheduler tasks |

---

## Architecture

The GUI shell (dialog, pane, view, ingestion adapter) and the non-GUI data model
(`MosaicController` and what it owns) are split into two diagrams below — combining
them into one made several unrelated ownership edges visually cross through unrelated
boxes (e.g. the "constructs on demand" edge into `ReprojectPromptDialog` appeared to
originate from `SceneMaterializer` just because of where the layout engine placed the
boxes). `MosaicPane` and `MosaicView` both hold a reference to the **same**
`MosaicController` instance; that instance is expanded in the second diagram.

```{mermaid}
classDiagram
    direction TB

    class QDialog["QDialog (Qt)"]
    class QWidget["QWidget (Qt)"]
    class MosaicController["MosaicController (see the data-model diagram below)"]

    class SeamlessMosaicDialog {
        mosaic_dialog.py
        -_materializer : SceneMaterializer
        +get_mosaic_pane()
    }

    class MosaicPane {
        mosaic_pane.py
        -_controller : MosaicController
        -_materializer : SceneMaterializer
        +_on_add_scene_clicked()
        +_on_scene_ingested(scene)
        +_ensure_common_grid()
        +_on_choose_target_crs()
    }

    class MosaicView {
        mosaic_view.py
        -_controller : MosaicController
        -_composite_pixmap : QPixmap
        -_transform : MosaicViewTransform
        -_footprint_paths : List~QPainterPath~
        -_hidden_paths : List~QPainterPath~
        +invalidate_overlay()
        +composite(layers, order) QImage
        +paintEvent(event)
    }

    class SceneMaterializer {
        mosaic_materialize.py
        -_tmp : TemporaryDirectory
        -_cache : dict
        +gdal_source(dataset, progress) str
        +close()
    }

    class ReprojectPromptDialog {
        mosaic_crs_dialog.py
        +selected_target_wkt() str
        +accept()
    }

    QDialog <|-- SeamlessMosaicDialog
    QDialog <|-- ReprojectPromptDialog
    QWidget <|-- MosaicPane
    QWidget <|-- MosaicView

    SeamlessMosaicDialog --> MosaicPane : owns
    SeamlessMosaicDialog --> SceneMaterializer : owns (session-scoped)
    MosaicPane --> MosaicView : owns
    MosaicPane --> ReprojectPromptDialog : constructs on demand
    MosaicPane --> MosaicController : owns, shares with view
    MosaicView --> MosaicController : reads
```

`SceneMaterializer` is injected into `MosaicPane`'s constructor by
`SeamlessMosaicDialog` (see [Entry Point and Dialog
Lifecycle](#entry-point-and-dialog-lifecycle) above) rather than owned separately by
the pane, so it appears only once above.

```{mermaid}
classDiagram
    direction TB

    class RasterDataSet["RasterDataSet (see wiser.raster.dataset)"]

    class MosaicController {
        mosaic_controller.py
        -_scenes : List~MosaicScene~
        -_target_crs_wkt : str
        -_resolution_mode : ResolutionMode
        -_common_grid : CommonGrid
        +add_scene(scene)
        +build_common_grid() CommonGrid
        +scene_crs_summary()
        +scene_crs_choices()
        +validate_target_crs(wkt)
    }

    class MosaicScene {
        <<dataclass>>
        +dataset : RasterDataSet
        +visible : bool
        +gdal_path : str
        +footprint_wkt : str
        +has_overviews : bool
    }

    class CommonGrid {
        <<dataclass>>
        +geotransform
        +extent
        +width
        +height
    }

    MosaicController --> MosaicScene : owns list (bottom-to-top)
    MosaicController --> CommonGrid : owns cached result
    MosaicScene --> RasterDataSet : wraps
```

The controller's scene list is **bottom-to-top**: index 0 renders first (bottom),
the last index is the top scene and wins z-order overlap. This convention is used
consistently by `scene_crs_summary`/`scene_crs_choices` (bottom-to-top) and the
`Scenes` list in `MosaicPane` (which displays top-most first, i.e. reversed).

---

## Entry Point and Dialog Lifecycle

`App.show_seamless_mosaic_dialog` (`src/wiser/gui/app.py`) lazily constructs a single
`SeamlessMosaicDialog(app_state, app_services, parent=self)` and reuses it across
open/close — the dialog is **non-modal** because a mosaic is a long-lived, multi-step
workflow (add scenes, reorder, export), so it stays open alongside the main window.

`SeamlessMosaicDialog` owns one `SceneMaterializer` for the dialog's lifetime. Because
the dialog is cached and reopened, the materializer's temp files must survive
`close()` — cleaning them up on `closeEvent` would orphan every added scene's
`gdal_path` on the next open. Instead the temp directory is torn down only when the
dialog itself is destroyed (`self.destroyed.connect(lambda *_: materializer.close())`),
with `TemporaryDirectory`'s own finalizer as a backstop on GC/interpreter exit.

---

## Scene Ingestion Pipeline

Adding a scene runs three gated phases on a background thread, orchestrated by
`_ingest_scene()` (`mosaic_pane.py`):

```{mermaid}
sequenceDiagram
    participant User
    participant Pane as MosaicPane
    participant Sched as WorkScheduler
    participant Ingest as _ingest_scene (bg thread)
    participant Mat as SceneMaterializer
    participant Ctrl as MosaicController

    User->>Pane: click "Add Scene…"
    Pane->>Pane: validate_scene(dataset, existing_scenes)
    alt validation fails
        Pane-->>User: QMessageBox.warning
    else validation passes
        Pane->>Sched: run_with_progress(_ingest_scene, dataset, materializer)
        Note over Pane: ProgressDialog shown, block_window disabled
        Sched->>Ingest: run on scheduler thread
        Ingest->>Mat: gdal_source(dataset) → materialize/warp-ready GeoTIFF
        Ingest->>Ingest: build_overviews(gdal_path)
        Ingest->>Ingest: compute_footprint_wkt(gdal_path)
        Ingest-->>Sched: MosaicScene(dataset, gdal_path, footprint_wkt, has_overviews=True)
        Sched-->>Pane: on_success(scene) [GUI thread]
        Pane->>Ctrl: add_scene(scene)
        Pane->>Pane: _refresh_scene_list()
        Pane->>Pane: _ensure_common_grid()
        Pane->>Pane: _mosaic_view.update()
    end
```

### Validation (`validate_scene`, `mosaic_ingestion.py`)

Runs synchronously on the **main thread** before any background work starts, so
rejection is immediate (no spinner churn). Rejects:

- **Ungeoreferenced scenes** — geotransform is GDAL's identity sentinel.
- **No SRS** — empty/missing spatial reference.
- **Duplicate** — the dataset is already in the mosaic (by dataset id).
- **Band-count mismatch** — differs from the first existing scene's band count.

Deliberately **not** rejected: missing nodata (a coarser, full-rectangle footprint is
still valid) and dtype mismatches across scenes (the future compositor promotes to the
widest common type at warp time). See `# TODO(#640)` in the source for a planned
warn-but-allow path for band-count mismatches.

### Materialization (`SceneMaterializer.gdal_source`, `mosaic_materialize.py`)

Every scene — GDAL-backed or not — is materialized to a disk-backed, tiled GeoTIFF
(`TILED=YES`, 256×256 blocks) under a per-session temp directory
(`wiser.utils.primitives.temp_dir()`), **not** `/vsimem` (RAM-backed), so GDAL reads
only the requested windows. Metadata (geotransform, SRS, nodata, per-band wavelength,
bad bands, default display bands) is stamped from the `RasterDataSet` object, mirroring
the georeferencer's numpy→GDAL-dataset trick but disk-backed. A per-scene dedup cache
(keyed by dataset id, or `id(dataset)` as a fallback) keeps this to one write per scene
per session — re-adding the same dataset is a cache hit.

The user's original dataset is **never modified**; the materialized copy is a
WISER-owned temp artifact.

### Overviews (`build_overviews`, `mosaic_ingestion.py`)

Builds internal pyramid overviews (`NEAREST`, levels `[2, 4, 8, 16]`) directly inside
the materialized temp GeoTIFF (`GA_Update` + `BuildOverviews`), so preview rendering
has no first-paint stutter. Because materialization already produces a WISER-owned
temp copy, overviews can be written internally rather than as external `.ovr`
sidecars.

### Footprint (`compute_footprint_wkt`, `mosaic_ingestion.py`)

Derives the valid-pixel outline via `gdal.Footprint(..., format="WKT")`, in the
scene's **own** CRS (no reprojection here — that happens later, in the grid builder).
With a nodata value set, this traces the true valid-pixel boundary; without one, it
falls back to the full raster rectangle.

### Progress reporting

All three phases share one `ProgressReporter` (`wiser/utils/progress.py`), split by
weight — materialize 0.5, overviews 0.35, footprint 0.15 — so the overall bar advances
smoothly across phases that report in very different native units (per-band count vs.
GDAL's own `0..1` callbacks). The reporter is Qt-free; `run_with_progress`
(`wiser/gui/progress_task.py`) is the reusable GUI bridge that:

- shows a non-blocking-to-the-rest-of-WISER `ProgressDialog` (disables only the mosaic
  dialog window, not all of WISER, unlike `Qt.WindowModal`),
- mirrors progress into an Activity Monitor row,
- marshals the worker thread's Qt signal emits back onto the GUI thread,
- supports cancellation (dialog close/Escape or the Activity Monitor row), which sets
  a shared `threading.Event` that the worker checks at `progress.raise_if_cancelled()`
  checkpoints between phases.

`_ingest_scene` is intentionally plain — it prints nothing and does not use the
logging module; progress and errors are the only channel out of a scheduler worker.

---

## The Common Grid and CRS Resolution

`MosaicController.build_common_grid()` computes the shared, north-up output grid all
scenes are placed onto. It is cached (`_grid_dirty`) and invalidated by any change to
the scene list, z-order, visibility, resolution mode, custom resolution, or target CRS
(`_invalidate_grid()`, called from `add_scene`, `remove_scene`, `move_scene`,
`set_visibility`, `set_resolution_mode`, `set_custom_resolution`, `set_target_crs`).

```{mermaid}
flowchart TD
    A[build_common_grid] --> B{grid_dirty?}
    B -->|no| C[return cached CommonGrid]
    B -->|yes| D{any visible scenes?}
    D -->|no| E[return empty CommonGrid]
    D -->|yes| F["target = target_crs_wkt or common_scene_crs_wkt()"]
    F --> G{target resolved?}
    G -->|no| H[raise TargetCrsRequired]
    G -->|yes| I["persist target_crs_wkt<br/>validate_target_crs(target)"]
    I --> J["per scene: reproject footprint envelope<br/>into target CRS → union extent"]
    J --> K["per scene: SuggestedWarpOutput resolution<br/>in target CRS (AutoCreateWarpedVRT)"]
    K --> L["pick xres/yres from ResolutionMode"]
    L --> M["north-up geotransform:<br/>(min_x, xres, 0, max_y, 0, -yres)<br/>width/height = ceil(extent / res)"]
    M --> N[cache + return CommonGrid]
```

### Resolution modes (`ResolutionMode`)

| Mode | Pixel size chosen |
|------|--------------------|
| `TOP` (default) | The **top** scene's resolution (last index in z-order) |
| `HIGHEST` | The smallest pixel size across visible scenes (finest detail) |
| `LOWEST` | The largest pixel size across visible scenes (coarsest) |
| `AVERAGE` | Mean pixel size across visible scenes |
| `CUSTOM` | User-specified `(xres, yres)` via `set_custom_resolution()`; raises `ValueError` if unset |

Per-scene resolution is computed once per scene, in the **target CRS**, via
`gdal.AutoCreateWarpedVRT` (equivalent to `SuggestedWarpOutput`) — this makes
cross-CRS resolution comparisons apples-to-apples rather than comparing native pixel
sizes across differing units/projections.

### CRS resolution and the auto-lock behavior

This is the part most likely to surprise a new reader, so it is called out explicitly
(and is covered by `src/tests/test_mosaic_crs_gui.py`):

> **The first scene added to an empty mosaic always auto-resolves and *permanently
> locks* the target CRS.** `build_common_grid()` treats a single-scene mosaic as
> trivially "all scenes share a CRS" (`common_scene_crs_wkt()` returns that one
> scene's CRS), builds successfully, and — critically — **persists** the result into
> `_target_crs_wkt`. Because `_invalidate_grid()` never clears `_target_crs_wkt`
> (only the cached grid), every later `add_scene()` sees a non-`None` target already
> set. A second scene with a *different* CRS therefore reprojects silently onto the
> already-locked target instead of raising `TargetCrsRequired` — **the reproject
> dialog never fires during normal incremental ingestion**, same-CRS or not.

In practice this means:

- `TargetCrsRequired` is only ever raised from `_on_scene_ingested`'s path in the
  degenerate case where `_target_crs_wkt` is still `None` — which does not happen
  once any scene has ever been added, since the very first `add_scene` call always
  locks it.
- The **only** reachable path to `ReprojectPromptDialog` in the current UI is the
  manual **"Choose Target CRS…"** button (`MosaicPane._on_choose_target_crs`), which
  calls `_prompt_for_target_crs()` unconditionally — not gated on
  `TargetCrsRequired` — so the user can override the auto-locked target at any time.
- `validate_target_crs()` still runs on every `build_common_grid()` call, so an
  incoming scene that is genuinely unmappable to the locked target (e.g. no CRS at
  all) still raises `UnmappableCrsError`, which `_ensure_common_grid` surfaces as a
  `QMessageBox.warning` rather than a dialog.

`_ensure_common_grid()` (`mosaic_pane.py`) is the call site, and stays a standalone
method (not inlined into `_on_scene_ingested`) so a future resolution-mode/CRS control
panel can re-run it:

```{mermaid}
flowchart TD
    A["_ensure_common_grid()"] --> B["controller.build_common_grid()"]
    B -->|succeeds| Z[refresh target-CRS label]
    B -->|TargetCrsRequired| C["_prompt_for_target_crs()"]
    C -->|dialog cancelled| Z
    C -->|accepted + valid| D["controller.build_common_grid() again"]
    D -->|UnmappableCrsError| W1[QMessageBox.warning]
    D -->|succeeds| Z
    B -->|UnmappableCrsError| W2[QMessageBox.warning]
    W1 --> Z
    W2 --> Z
```

As noted above, the `TargetCrsRequired` branch is effectively dead in the ingestion
path today but is kept because it is the correct, defensive behavior if the locking
assumption ever changes (e.g. a future "clear target CRS" control).

### Removal and visibility changes

Removing a scene or toggling visibility can only *relax* the CRS constraint (fewer
scenes to satisfy), so `MosaicPane` calls `_rebuild_grid_quietly()` instead of
`_ensure_common_grid()` — it rebuilds if possible and silently leaves the grid
unresolved on `TargetCrsRequired`/`UnmappableCrsError` rather than popping a modal,
since a prompt triggered by a *removal* would be a surprising, unrelated interruption.

### Other controller helpers

- `common_scene_crs_wkt()` — the shared CRS (WKT) if every visible scene's SRS is
  `IsSame`, else `None`.
- `scene_crs_summary()` — `(dataset_name, crs_display_name)` per scene, bottom-to-top;
  feeds the reproject dialog's read-only table.
- `scene_crs_choices()` — `(crs_display_name, crs_wkt)` for each **distinct**
  visible-scene CRS, deduped by `IsSame`, ordered so the **last** entry is the top
  scene's CRS; seeds the dialog's target-CRS combo and its default selection.
- `validate_target_crs(target_wkt)` — raises `UnmappableCrsError` naming any scene
  (by name) that cannot be transformed to the target, via
  `wiser.raster.utils.can_transform_between_srs`.

All SRS objects used for transforms are built with
`OAMS_TRADITIONAL_GIS_ORDER` (long/lat, x/y axis order), matching the geotransform
convention used throughout WISER (see the georeferencer's identical convention in
[Georeferencer Internals](georeferencer-internals.md)).

---

## The Geometry Overlay (Vector Layer)

**Files:** `src/wiser/gui/mosaic_view.py` (rendering) and the Qt-free geometry helpers
in `src/wiser/raster/mosaic_controller.py`.

`MosaicView` draws the vector overlay (issue #636): each visible scene's footprint
outline (green), the union bounding box (dashed), and the **overlap highlight**
(magenta/purple) marking where a scene is hidden by anything above it in z-order. It
matches the ENVI reference and is used **purely for on-screen rendering** — it never
decides which pixel wins (that is z-order in the compositor/export, #637/#639).

```{mermaid}
flowchart TD
    A[paintEvent] --> B{grid.extent is None?}
    B -->|yes, mosaic empty| C["_has_fitted = False<br/>(re-arm the initial fit)"]
    B -->|no| D{"not _has_fitted and<br/>viewport has real size?"}
    D -->|yes| E["transform.fit_to_extent(grid.extent)<br/>_has_fitted = True"]
    D -->|no, already fitted| F
    C --> F{geometry_dirty?}
    E --> F
    F -->|yes| G["_rebuild_overlay_geometry()<br/>geometry_dirty = False"]
    F -->|no, cache is fresh| H
    G --> H["painter.setWorldTransform(world_to_screen)"]
    H --> I["draw pixel layer (stub, #637)"]
    I --> J["draw bounding box (_bbox_extent)"]
    J --> K["per scene: clip to hidden_path (if any),<br/>fill + outline in the highlight color"]
    K --> L["draw all footprint outlines on top"]
```

The `_has_fitted` re-arm on an empty grid matters in practice: without it, removing
every scene and then adding a *different* one left the camera parked on the removed
scene's extent, so the new scene's footprints rendered off-screen (fixed as a
regression once observed — see the camera-reframe test in
`test_mosaic_view_gui.py`).

### The camera: `MosaicViewTransform`

The view is a **QGIS-style unbounded canvas**, not `RasterView`'s
`QScrollArea`-around-a-fixed-`QPixmap` (which is bounded by the pixmap's pixel size).
There is no backing store sized to the data; the only state that persists between
paints is a **camera** — three floats:

- `center_x`, `center_y` — a point in **world (target-CRS) coordinates** that always
  maps to the center of the widget. Panning moves it; nothing clamps it, so panning
  far from every footprint simply shows blank canvas (what makes it "unbounded").
- `world_units_per_pixel` — a single scale; zooming changes it.

There is no invented `(0, 0)` origin: world coordinates come from the common CRS and
screen coordinates are Qt's usual top-left widget space. The visible world rectangle
is *derived* from `center + scale + widget size` each paint (never stored — storing it
would distort the aspect ratio on resize). `world_to_screen(viewport_size)` returns a
`QTransform` (with a y-flip, since world y increases north but screen y increases
downward); `screen_to_world` is its inverse. The viewport size is passed in rather than
cached, since `QWidget` already tracks it. The same camera is shared with the pixel
layer (#637), which is why #636 builds it.

Interaction: **middle-button drag** pans, the **mouse wheel** zooms (anchored at the
cursor). The mosaic extent is framed once — the first time a common grid is available
*and* the widget has a real size (done in `paintEvent`, not on grid-build, to avoid a
0×0 not-yet-shown viewport) — and never auto-refit afterward, so a user's pan/zoom is
never yanked out from under them.

### Footprint reprojection and the overlap computation (Qt-free)

The raw `footprint_wkt` on each `MosaicScene` is in the scene's **own** CRS. Two
Qt-free helpers in `mosaic_controller.py` turn those into common-CRS geometry (so the
overlay stays unit-testable without Qt, mirroring the controller's no-Qt rule):

- `reprojected_footprint_geometry(scene, src_srs, target_srs)` — the footprint polygon
  transformed *whole* into the target CRS (`_footprint_envelope` now delegates to this,
  so the reprojection is defined once).
- `MosaicController.visible_scene_footprints_in_common_crs()` — `(scene, geometry)` for
  each visible scene, bottom-to-top, in the resolved target CRS; returns `[]` when no
  target CRS is resolved yet (the view then draws nothing).
- `compute_union_overlaps(footprints)` — each scene's **hidden region** = its footprint
  ∩ the union of everything above it. Walking top-to-bottom with a running union, this
  is one `Intersection` + one `Union` per scene (`O(n)`), not all-pairs `O(n²)`. The
  topmost scene is never hidden.

### Rendering and the geometry cache

`MosaicView` caches the overlay as world-space `QPainterPath`s (`_footprint_paths` and
the parallel `_hidden_paths`, plus `_bbox_extent`) converted from those `ogr.Geometry`
objects by `_geometry_to_qpainterpath` (handles polygons-with-holes via the odd-even
fill rule, and skips degenerate non-polygon `Intersection` results).

This split is deliberate: **geometry** is recomputed only when scenes, z-order, or the
target CRS change; **the transform** is applied fresh every paint. So pan/zoom is a
cheap repaint (`paintEvent` just installs a new `QTransform` and redraws cached paths)
with no GDAL/OSR work, satisfying the "redraws on pan/zoom without recomputing pixels"
requirement.

`MosaicPane` calls `MosaicView.invalidate_overlay()` (which sets a dirty flag and
schedules a repaint) after every controller mutation that changes the overlay — scene
add/remove, visibility toggle, and target-CRS change. The actual rebuild runs lazily
in `paintEvent` when the flag is set (coalescing repeated invalidations into one
rebuild) and is fully guarded: any OGR/OSR failure clears the cache and logs rather
than letting an exception escape a paint. Pens are **cosmetic**, so outline width stays
constant in screen pixels regardless of zoom.

`paintEvent` draws, in order: the pixel layer (still stubbed, #637), then the bounding
box, then per scene a clip to its hidden region filled + outlined in the highlight
color, then all footprint outlines on top in green (so every boundary stays visible
even where it crosses an overlap region).

---

## `ReprojectPromptDialog`

**File:** `src/wiser/gui/mosaic_crs_dialog.py`

A modal, data-driven dialog: it takes plain lists (`scene_summary`,
`scene_crs_choices`), not the controller, so it can be constructed and tested without
GDAL/OSR objects in hand — the caller passes `controller.scene_crs_summary()` and
`controller.scene_crs_choices()`.

- **Top:** a read-only two-column table (*Dataset* / *CRS*) built from
  `scene_summary`, so a mismatch (or the current per-scene CRS state) is visible.
- **Bottom:** a target-CRS combo box seeded, in order, with:
  1. the distinct scene CRSs from `scene_crs_choices` (default selection = the
     **last** entry, i.e. the top scene's CRS),
  2. the built-in `COMMON_SRS` presets (WGS84, Web Mercator, NAD83/UTM 15N),
  3. any CRS the user has created via the
     [CRS Creator](crs-creator-internals.md) (`app_state.get_user_created_crs()`).
  4. An **authority + code** lookup row (e.g. `EPSG` + `4326`) can add and select a
     fully custom `AuthorityCodeCRS` entry on demand.

Every combo entry stores a `GeneralCRS` subclass (`WktGeneratedCRS`,
`AuthorityCodeCRS`, `UserGeneratedCRS` — the same hierarchy documented in
[Georeferencer Internals](georeferencer-internals.md#the-crs-model)) as `userData`, so
`selected_target_wkt()` reads uniformly across all four sources via
`GeneralCRS.get_osr_crs().ExportToWkt()`. `accept()` refuses to close if nothing is
selected.

---

## What Isn't Built Yet

`MosaicView.paintEvent` now draws the vector overlay (#636, see [The Geometry
Overlay](#the-geometry-overlay-vector-layer)); the pixel layer beneath it is still a
stubbed seam with a `TODO` marker, alongside the control-panel and export gaps:

- **Pixel layer (compositing, issue #637)** — `MosaicView.composite(layers, order)` is
  a stub that returns `None`. The design calls for per-scene ARGB `QImage` caches at
  screen resolution (alpha = validity, from the GDAL mask band), stacked bottom-to-top
  by `QPainter` and drawn beneath the vector overlay via the same
  `MosaicViewTransform` camera; z-order reorders would only re-stack cached layers (no
  GDAL reads), while pan/zoom would re-read from overviews (debounced).
- **Control panel additions (issue #638)** — drag-to-reorder z-order, resampling
  method selector, and a band-metadata chooser are not yet in `MosaicPane`; today's
  panel only has Add Scene, the scene list (visibility toggle + remove), and the
  target-CRS controls.
- **Export (issue #639)** — writing the mosaic via `gdal raster mosaic` (or the
  GDAL < 3.11 `BuildVRT`/`Translate` fallback), ordered by z-order, resolved against
  the common grid, and loaded back into WISER as a new dataset.

None of these gaps affect the ingestion/CRS-resolution logic documented above — they
consume the same `MosaicController` state once implemented.

---

## Integration with WISER

- **`ApplicationState`** — `MosaicPane` reads `get_datasets()` to populate the
  Add-Scene combo (kept in sync via the `dataset_added`/`dataset_removed` signals),
  and `get_user_created_crs()` feeds the reproject dialog's target-CRS chooser.
- **`AppServices` / `WorkScheduler`** — scene ingestion runs via
  `run_with_progress(app_services, ...)`, which submits to
  `app_services.scheduler.submit_thread(...)` and registers an Activity Monitor row;
  see the [System Design](system-design.md) page for how scheduler and Activity
  Monitor updates flow end to end.
- **Entry point** — `App.show_seamless_mosaic_dialog` (`src/wiser/gui/app.py`)
  lazily constructs and caches the dialog, reusing it across open/close.

## Testing

- `src/tests/test_mosaic_controller.py` — grid math per `ResolutionMode`, CRS
  resolution/validation, cache invalidation, and the Qt-free geometry helpers
  (`reprojected_footprint_geometry`, `compute_union_overlaps`,
  `visible_scene_footprints_in_common_crs`) — all pure `MosaicController`, no Qt.
- `src/tests/test_mosaic_view_transform.py` — `MosaicViewTransform` camera math:
  world↔screen round-trip, y-flip orientation, zoom-anchor invariance, aspect-ratio
  preservation (pure `QTransform` math, no widget shown).
- `src/tests/test_mosaic_view_gui.py` — the overlay wiring end to end: geometry cache
  populates for overlapping scenes and re-invalidates on scene removal / target-CRS
  change, driven through the real `MosaicPane` ingestion path.
- `src/tests/test_mosaic_ingestion.py` — `validate_scene`, `build_overviews`,
  `compute_footprint_wkt`, and progress-reporting behavior.
- `src/tests/test_mosaic_crs_dialog.py` — `ReprojectPromptDialog` in isolation
  (offscreen Qt), constructed directly from plain lists.
- `src/tests/test_mosaic_crs_gui.py` — end-to-end through the real
  `MosaicPane._on_scene_ingested` path via the `WiserTestModel` harness; documents
  the auto-lock behavior described above with `mock.patch` on
  `wiser.gui.mosaic_pane.ReprojectPromptDialog` so nothing blocks the test.
- `src/tests/test_mosaic_dialog_gui.py` — dialog shell, Add Scene ingestion flow,
  progress modal.

GUI tests use `@pytest.mark.functional`/`@pytest.mark.smoke` with the
`WiserTestModel` harness (`src/test_utils/test_model.py`), not pytest-qt/qtbot. Run
with `QT_QPA_PLATFORM=offscreen` and `PYTHONPATH=src`.
