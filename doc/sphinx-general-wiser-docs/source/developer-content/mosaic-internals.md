# Seamless Mosaic Internals

This page documents the internal architecture of WISER's **Seamless Mosaic** feature —
combining several overlapping georeferenced scenes onto one shared output grid. It is
intended for developers reading, debugging, or extending the tool.

```{note}
This is an in-progress feature (EPIC #629). As of this writing, scene ingestion,
materialization, common-grid/CRS resolution, the **vector overlay** (footprints,
bounding box, overlap highlight), the **static-scene pixel compositor** (the composited
preview, with an off-thread debounced per-scene cache), the **control panel**
(drag-to-reorder z-order, resolution mode, resampling method, and the band-metadata
chooser), the **full-resolution export path** (streaming ENVI compositor on the
common grid — see [The Export Path](#the-export-path)), and **in-place scene
re-georeferencing** (right-click → warp → swap; see [Re-georeferencing a Scene In
Place](#re-georeferencing-a-scene-in-place)) are implemented and wired into the GUI.
```

For the design rationale and full child-issue breakdown, see `EPIC_seamless_mosaic.md`
in the repo root. There is no user-facing guide yet since the feature is not
preview/export complete.

## Overview

The feature is built from three cooperating layers:

- **The UI / control layer** — `SeamlessMosaicDialog` is a non-modal, cached top-level
  window (like the other tool dialogs) that hosts a `MosaicPane`. The pane owns an
  "Add Scene" picker, a scene stack (drag-to-reorder z-order + visibility), and the
  resolution / target-CRS / resampling / band-metadata controls (see [The Control
  Panel](#the-control-panel)), and drives the ingestion pipeline on a background thread.

- **The non-GUI model** — `MosaicController` is the single source of truth for the
  scene list, z-order, resolution mode, target CRS, and the computed `CommonGrid`. It
  is deliberately Qt-free (may use `osgeo.gdal/ogr/osr`) so it is unit-testable without
  a running application.

- **The rendering layer** — `MosaicView` is a `QWidget` sibling of `RasterView`
  (not a subclass — a mosaic is N scenes on one shared world grid, not one dataset
  zoomed). It draws two layers on a QGIS-style unbounded canvas via a world→screen
  camera (`MosaicViewTransform`): the **pixel layer** (the composited scenes, from a
  per-scene ARGB cache — see [The Pixel Layer](#the-pixel-layer-static-scene-compositor))
  and, on top, the **vector overlay** (footprints, bounding box, overlap highlight — see
  [The Geometry Overlay](#the-geometry-overlay-vector-layer)).

Every scene, regardless of its original backing (GDAL file, NumPy array, PDR, etc.),
is first turned into a warpable, disk-backed tiled GeoTIFF by a `SceneMaterializer` —
the `RasterDataSet` is the source of truth for metadata, so materialization stamps
SRS/geotransform/nodata/band metadata from the dataset object, not from its `_impl`.

**Core files:**

| File | Responsibility |
|------|----------------|
| `src/wiser/gui/mosaic_dialog.py` | `SeamlessMosaicDialog` — top-level dialog shell, owns the session `SceneMaterializer` |
| `src/wiser/gui/mosaic_pane.py` | `MosaicPane` — control panel; ingestion orchestration, scene list (drag-reorder + visibility), resolution / CRS / resampling / band-metadata controls |
| `src/wiser/gui/mosaic_view.py` | `MosaicView` — the two-layer paint surface (pixel + vector), currently a stub |
| `src/wiser/gui/mosaic_crs_dialog.py` | `ReprojectPromptDialog` — modal target-CRS chooser |
| `src/wiser/raster/mosaic_controller.py` | `MosaicController`, `MosaicScene`, `CommonGrid`, `ResolutionMode` — the non-GUI model |
| `src/wiser/raster/mosaic_ingestion.py` | `validate_scene`, `build_overviews`, `compute_footprint_wkt` — Qt-free ingestion gates |
| `src/wiser/raster/mosaic_compositor.py` | `render_scene_argb` — Qt-free per-scene warp→RGBA renderer (alpha = validity) for the pixel layer |
| `src/wiser/raster/mosaic_export.py` | `export_mosaic` — warps each scene onto the common grid and streams the mosaic to ENVI |
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
        -_scene_layers : Dict~int, QImage~
        -_render_signature : tuple
        -_transform : MosaicViewTransform
        -_footprint_paths : List~QPainterPath~
        -_hidden_paths : List~QPainterPath~
        +invalidate_overlay()
        +invalidate_pixels()
        +recomposite_only()
        +composite(layers) QImage
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

## The Pixel Layer (Static-Scene Compositor)

**Files:** `src/wiser/raster/mosaic_compositor.py` (Qt-free per-scene renderer) and
`src/wiser/gui/mosaic_view.py` (cache, compositing, drawing, threading).

Beneath the vector overlay, `MosaicView` draws the actual composited scenes (issue
#637): each visible scene is read at **screen resolution** into an **ARGB `QImage`**
whose alpha is its validity mask, and the scenes are stacked bottom-to-top honoring
z-order so lower scenes show through upper scenes' nodata holes.

### The per-scene renderer (`render_scene_argb`, Qt-free)

`mosaic_compositor.render_scene_argb(scene, target_wkt, world_extent, w, h)` warps one
scene onto the current viewport rectangle at output size `w×h` via
`gdal.Warp(..., dstSRS=target_wkt, outputBounds=world_extent, dstAlpha=True)` and returns
an `(h, w, 4)` uint8 RGBA array. A single `gdal.Warp` does four jobs at once:

- **reprojection** onto the target CRS,
- **downsampling** — because the output is far coarser than the source, GDAL reads from
  the internal **overviews** built in #634 (the whole point of building them),
- **alignment** to the visible world rectangle, and
- **the validity mask** — `dstAlpha=True` yields an alpha band that is `0` on nodata /
  outside-coverage pixels, so the alpha channel *is* the validity mask (no manual
  mask-band read). This mirrors the warp seam already used by `_warped_resolution`.

RGB comes from the dataset's `get_default_display_bands()` (1 band → grayscale, 3 →
RGB), contrast-stretched per band over the valid pixels (2–98 percentile). Invalid
pixels are forced fully clear `(0,0,0,0)` so nothing bleeds under a transparent alpha.
It is Qt-free (produces a NumPy array) so it is unit-testable without a running app and
can run on a background thread; the view wraps the array into a `QImage` on the GUI
thread.

### The per-scene cache and `composite()`

`MosaicView` holds one ARGB `QImage` per visible scene in `_scene_layers` (keyed by
`id(scene)`), built for the **current viewport** — the *render signature*
`(center_x, center_y, world_units_per_pixel, width, height)`. `composite(layers)` stacks
those layers bottom-to-top with `QPainter` `SourceOver` into a single image; `layers` is
already in render order (index 0 = bottom), so z-order is encoded in the list and hidden
scenes are simply `None`. `composite()` is the **single indirection point** where
deferred seamline/feathering work will later re-implement the stacking (by territory
mask) without touching callers.

This split gives the caching tiers the design calls for:

- **Z-order reorder / visibility toggle** (`recomposite_only()`) — a pure restack of the
  cached layers, **no GDAL reads**. Hiding a scene just drops it from the composite (its
  `visible` flag), and unhiding at the same viewport reuses its still-cached layer.
  `recomposite_only()` falls back to a read only when *revealing* a scene that was hidden
  at the last read (so it was never cached at this viewport).
- **Pan / zoom** — changes the render signature, so every layer is re-read. This is the
  cache's deliberate trade-off: it does **not** survive pan/zoom (the composite is
  re-read), but it makes reorder/visibility at a fixed viewport free.
- **Add / remove scene, target-CRS change** (`invalidate_pixels()`) — marks the cache
  stale so the next paint re-reads.

### Off-thread, debounced reads

Reads run **off the UI thread** and pan/zoom re-reads are **debounced** so the view stays
responsive. A `paintEvent` whose render signature changed (re)starts a single-shot
`QTimer` (`_PIXEL_READ_DEBOUNCE_MS`, ~120 ms); a gesture's burst of paints collapses into
one read once the camera settles. On fire, `_start_pixel_read` snapshots the viewport on
the GUI thread (resolving footprints and the in-view intersect test — cheap OSR work),
then hands only the heavy per-scene warps to `app_services.scheduler.submit_thread`. The
worker returns NumPy arrays (no Qt off-thread); the future's done-callback emits the
`_read_ready` signal, whose queued connection wraps them into `QImage`s and restacks on
the GUI thread. `_reading_signature` tracks the in-flight read so a superseded (or
out-of-order) result is discarded and never clobbers newer pixels. In the interim the
prior composite keeps drawing, scaled by the camera (it is drawn mapped from its own
`_composite_world_extent` through the world→screen affine), so pan/zoom shows a scaled
preview until the sharp re-read lands. With no scheduler (e.g. a bare view in a unit
test) the read falls back to synchronous.

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
    G --> P{"render signature<br/>changed / pixels dirty?"}
    P -->|yes| Q["_schedule_pixel_read()<br/>(debounced, off-thread)"]
    P -->|no| H
    Q --> H["draw pixel layer:<br/>_composite_pixmap mapped by world_to_screen"]
    H --> H2["painter.setWorldTransform(world_to_screen)"]
    H2 --> J["draw bounding box (_bbox_extent)"]
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

## The Control Panel

**File:** `src/wiser/gui/mosaic_pane.py` (issue #638)

The right-hand controls area of `MosaicPane` (hosted in a vertically-scrolling
`QScrollArea` so the stack stays reachable on short windows) exposes the model's
handles. Every control mutates the shared `MosaicController` and then invalidates the
view following the same tiered contract the compositor defines — a **pure restack**
(`recomposite_only()`, no GDAL reads) for z-order/visibility, a **re-read**
(`invalidate_pixels()`) when the pixels themselves change, and a **grid rebuild** when
the output geometry changes:

- **Scene list — drag-to-reorder (z-order).** The `QListWidget` is
  `InternalMove`-enabled; items are checkable and draggable but *not* drop targets
  (`~ItemIsDropEnabled`) so a dragged row lands between rows. On drop, `model().rowsMoved`
  fires `_on_scene_rows_moved`, which translates the **visual** move into a controller
  z-order move. Two index subtleties are handled: the list is shown **top-first** while
  the controller stores scenes **bottom-to-top** (visual row `v` ↔ controller index
  `n-1-v`), and Qt's destination `row` is indexed *before* the dragged row is removed
  (shifted down by one on a downward move). It then `move_scene`s, rebuilds the grid
  quietly (a reorder can change the top scene, hence the `TOP`-mode pixel size, but never
  the CRS constraint), defers a `_refresh_scene_list()` to rewrite the now-stale
  `Qt.UserRole` indices safely after the drop, and does the pure-restack invalidation.

- **Resolution mode.** A combo over `ResolutionMode` (Top / Highest / Lowest / Average /
  Custom); Custom reveals two positive-only pixel-size spin boxes (target-CRS units),
  seeded from the current grid on first use so `build_common_grid` never sees `CUSTOM`
  without a size. A change rebuilds the grid quietly. Note the resolution only sizes the
  **export** grid — the preview always warps at *viewport* resolution (see [The Pixel
  Layer](#the-pixel-layer-static-scene-compositor)), so changing it has no on-screen
  effect.

- **Resampling method.** A curated combo — Nearest Neighbor (default), Bilinear, Cubic
  Convolution — whose `userData` is the `gdal.GRA_*` constant. Any non-Nearest-Neighbor
  choice **always** warns (interpolation invents pixel values; there is no product-type
  detection), then `set_resample_alg` + `invalidate_pixels`. The chosen algorithm is
  snapshotted on the GUI thread in `_start_pixel_read` and threaded through the per-scene
  `jobs` into `render_scene_argb`'s `resample_alg`.

- **Band-metadata chooser.** A revisitable combo picking which scene's band metadata
  (wavelengths / names) becomes the **canonical** metadata stamped on the exported output
  (consumed by #639). It is **metadata/labeling only** — it never changes the output band
  count, an invariant guaranteed for free by the ingestion band-count gate. The controller
  stores the choice as a `MosaicScene` reference (stable across reorder/remove) and
  `get_band_metadata_source()` falls back to the top visible scene when nothing is chosen
  or the chosen scene is removed/hidden. Neither this nor the resampling method invalidates
  the grid (both are output-content, not grid geometry).

- **Export / Finish** runs the full-resolution export path (see [The Export
  Path](#the-export-path)); the v1 preview toggle is intentionally deferred.

`ResolutionMode` change, band-metadata selection, resampling warning, and
reorder-without-reads are covered by `src/tests/test_mosaic_pane_gui.py`, with the
controller-side surface (resample round-trip, band-metadata fallback, no-grid-invalidation)
in `src/tests/test_mosaic_controller.py`.

---

## Re-georeferencing a Scene In Place

**File:** `src/wiser/gui/mosaic_pane.py` (issue #685)

A scene may be **mis-registered** — its spatial information places it wrong relative to
the other scenes. Rather than sending the user out to Tools → Georeferencer and back
through a file round-trip, `MosaicPane` lets them fix it **in place**: right-click a
scene → **"Georeference…"**, warp it, and the corrected result is swapped into the
mosaic live, with a clean revert on cancel. This reuses the configurable
`GeoReferencerDialog` (see [Programmatic Configuration &
Locking](georeferencer-internals.md#programmatic-configuration-locking)) — no new warp,
GCP, or CRS logic is added here.

### Entry point: the scene-list context menu

`_build_scene_list_group` gives `self._scene_list` a `Qt.CustomContextMenu` policy;
`_on_scene_context_menu(pos)` resolves the clicked row to a **controller** index via
`item.data(Qt.UserRole)` (the list is shown top-first but each item carries its real
bottom-to-top index) and pops a one-action `QMenu` → `_on_georeference_scene(index)`. The
menu is suppressed when the scheduler or the session materializer is absent, since the
swap reingests (the same requirement Add Scene guards on).

### Opening the dialog (locked target + locked save path)

`_on_georeference_scene(index)` builds a `GeoReferencerConfig` that locks the two things
the mosaic owns and leaves the reference user-chosen:

- `target_dataset = scene.dataset` (the **original** dataset, not a copy) with
  `allow_change_target=False`,
- `save_path` = a fresh temp path with `allow_change_save_path=False`,
- `reference_dataset=None` (unlocked — the user picks a reference or a manual CRS),
- `accept_button_text="Save to Mosaic"`.

It constructs a **fresh, task-scoped** `GeoReferencerDialog` (not the Tools-menu
singleton, so the lock state can never leak back into that flow), connects
`warp_completed` → `_on_scene_rewarped` and `finished` → `_on_geodialog_finished`,
stashes the in-flight state in a `_RegeorefContext` (`orig_scene`, `orig_index`,
`dialog`, `warped_scene`), and `show(config)`s it **non-modally** (the mosaic dialog is
non-modal, so the georeferencer must not block it).

> **The save path is a *fresh* name per warp** (`regeoref_{id(scene)}_{uuid4}.tif` under
> `wiser.utils.primitives.temp_dir()`), not one reused path. The previous warped
> `RasterDataSet` keeps its GeoTIFF open, and on Windows an open file cannot be
> overwritten — so a repeated "Run Warp" onto one fixed path would fail. The file is a
> throwaway regardless: reingestion re-materializes it into the mosaic's own copy.

### Two threaded phases: warp, then reingest-and-swap

```{mermaid}
sequenceDiagram
    participant User
    participant Pane as MosaicPane
    participant Dlg as GeoReferencerDialog
    participant Sched as WorkScheduler
    participant Ctrl as MosaicController
    participant View as MosaicView

    User->>Pane: right-click scene → "Georeference…"
    Pane->>Dlg: show(GeoReferencerConfig, locked target+save)
    User->>Dlg: place GCPs, click "Run Warp"
    Dlg->>Sched: warp_dataset_to_path (bg thread, own progress modal)
    Sched-->>Dlg: written path
    Dlg-->>Pane: warp_completed(path)
    Pane->>Pane: load path → RasterDataSet (NOT registered)
    Pane->>Sched: _ingest_scene(dataset, materializer) [blocks the dialog]
    Sched-->>Pane: MosaicScene (materialized + overviews + footprint)
    Pane->>Ctrl: remove current occupant, add_scene, move_scene(→ orig slot)
    Pane->>Pane: _ensure_common_grid + _refresh_scene_list
    Pane->>View: invalidate_overlay + invalidate_pixels
    User->>Dlg: "Save to Mosaic" (accept) or Cancel (revert)
    Dlg-->>Pane: finished(result) → finalize or _revert_regeoref
```

**Run Warp** is phase one: the dialog threads the full-resolution GDAL warp (PR 1) and
emits `warp_completed(path)`. `_on_scene_rewarped(path)` then wraps that output into a
`RasterDataSet` via the shared loader
(`app_state.get_loader().load_from_file(path, data_cache=app_state.get_cache())[0]`) but
**does not register it** in `ApplicationState` — it is a mosaic-owned throwaway, so it
must never pollute the global dataset list or the Add-Scene combo. Phase two runs
`_ingest_scene` on the scheduler (the **same** materialize → overviews → footprint
pipeline as Add Scene) with its own progress modal, **blocking the georeferencer dialog**
(not the mosaic window) so the user cannot re-run the warp mid-reingest.

`_on_rewarp_ingested(scene)` does the swap on the GUI thread. It replaces the slot's
**current occupant** — the original on the first warp, or the previous warped scene on a
repeat, so warps never stack — located by **object identity** (robust to a user reorder
while the non-modal dialog is open), falling back to the recorded `orig_index`:
`remove_scene(slot)` → `add_scene(scene)` (appends to the top) → `move_scene(scene_count-1,
slot)` to restore the z-order slot. It then runs the normal ingest epilogue
(`_ensure_common_grid`, `_refresh_scene_list`, `invalidate_overlay`, `invalidate_pixels`),
because the warp changed the geotransform, footprint, and extent/resolution, so the whole
derived state must rebuild. The original `MosaicScene` is held aside in the context and
**never mutated**.

### Save vs. revert

`_on_geodialog_finished(result)` centralizes the outcome. On **accept** ("Save to
Mosaic") the warped scene is already live, so it just drops the revert handle. On
**reject / close** it calls `_revert_regeoref`, which removes the swapped-in warped scene
(by identity) and re-adds the original at its slot, rebuilding **quietly**
(`_rebuild_grid_quietly` — a revert restores a previously-valid state, so a reproject
prompt would be a surprising interruption); it is a no-op if no warp ever ran. Either way
the task-scoped dialog is `deleteLater()`d.

### Why reingest, and how the fix reaches the export

This is the part most likely to confuse a new reader, so it is called out explicitly:

> **The georeferencer's "Run Warp" produces a real, full-resolution, all-bands raster
> whose pixels are physically resampled onto the new georeferencing — not merely a
> re-tagged preview.** The mosaic then makes *that file* the scene's backing data: after
> the swap, `scene.dataset` is the warped dataset and `scene.gdal_path` is its
> materialized tiled copy; the original (unwarped) dataset is removed from the mosaic
> entirely. So there is no "unwarped full dataset" left behind to reconcile.

Both the preview compositor
([`render_scene_argb`](mosaic-internals.md#the-per-scene-renderer-render_scene_argb-qt-free))
and the export compositor
([`_warp_scene_to_grid_vrt`](#the-compositor-export_mosaic-qt-free)) read pixels **and**
the geotransform from `scene.gdal_path`. They differ only in output resolution (screen
vs. full), so they are always consistent. The re-georeferenced scene therefore reaches
export the same way any scene does — **twice-warped in sequence**: warp #1 (the
georeferencer) baked the correction into `scene.gdal_path`, and export's per-scene warp #2
reprojects that corrected file onto the shared common grid. The common grid itself
rebuilds off the corrected `footprint_wkt`, so extent and overlap follow too. This is
exactly why the design reingests rather than patching caches in place — one coherent
rebuild, and the export path picks it up for free. (The correction lives only for the
session; the mosaic export re-warps from the temp at full resolution, so the on-disk
product is correct without persisting the single corrected scene separately.)

---

## The Export Path

**Files:** `src/wiser/raster/mosaic_export.py` (Qt-free) and the
`_on_export_clicked` / `_export_mosaic_task` handlers in `src/wiser/gui/mosaic_pane.py`
(issue #639).

Export is the "Finish" half of the mosaic: where the pixel layer renders a
screen-resolution *preview* per scene, export runs **once** at full resolution and writes
a real, value-preserving raster to disk. It composites the visible scenes by **z-order +
per-source nodata** (last valid source wins, never blends) onto the resolved common grid,
and streams the result block-by-block so a mosaic larger than RAM is never buffered whole.
It consumes the same `MosaicController` state as everything above — no new ingestion or
CRS logic.

### The compositor (`export_mosaic`, Qt-free)

`export_mosaic(scenes, grid, target_wkt, resample_alg, output_nodata,
band_metadata_dataset, out_path, progress)` is two GDAL stages, both lazy until the final
streamed write:

1. **Warp each visible scene onto the common grid as a warped VRT**
   (`_warp_scene_to_grid_vrt`). `gdal raster mosaic` does **not** reproject, so this
   per-scene `gdal.Warp(..., format="VRT")` is what applies the CRS transform + resample +
   grid alignment — mirroring the preview warp in `render_scene_argb`, but keeping the raw
   band values (no percentile stretch, no `dstAlpha`). Each scene's Data Ignore Value
   becomes the warp `srcNodata` and `output_nodata` becomes the `dstNodata`, so invalid /
   outside-footprint pixels are marked for the compositor. VRTs are lazy — no pixels are
   materialized here. A scene whose window is disjoint from the grid warps to `None` and
   is simply skipped.

2. **Composite + stream to ENVI** (`_run_raster_mosaic`) via
   `gdal.Run("raster", "mosaic", input=[...], output=..., output_format="ENVI", ...)`
   (GDAL ≥ 3.11, guaranteed by #631; the doc's `BuildVRT`/`Translate` fallback is not
   needed). Inputs are passed **bottom→top** (the top scene last) so the algorithm's
   "last valid source wins" composites z-order correctly, and it writes ENVI
   block-by-block, so no full-image buffer is held in RAM. GDAL's own progress callback is
   forwarded through the `ProgressReporter`.

The VRTs live in a `TemporaryDirectory` scoped around the mosaic call — they are only
referenced until `gdal.Run` finishes the streamed write, then discarded.

### Band-metadata stamping and the ENVI-header patch

`gdal raster mosaic` writes correct pixels, geotransform, and `map info`, but no spectral
metadata, so `_patch_envi_band_metadata` reopens the GDAL-written `.hdr` and rewrites it
with WISER's own ENVI header helpers (`envi.load_envi_header` / `write_envi_header`),
stamping the canonical band metadata from `controller.get_band_metadata_source()`:
wavelengths / units, band names, the bad-band list (`bbl`), and the output nodata. This
keeps the exported file self-describing so it re-opens in WISER with the right spectral
labels.

One subtlety is called out because it caused a crash-on-open regression:

> **GDAL's ENVI writer stamps a *1-based* `default bands` placeholder** (e.g. `{1, 2, 3}`
> for RGB inputs). WISER's ENVI reader reads `default bands` **verbatim** — no
> 1-based→0-based conversion — and `find_display_bands` does **no** bounds check, so
> leaving that placeholder (or copying an out-of-range source value) makes the exported
> file raise `GDALDataset::GetRasterBand(4) - Illegal band #` when opened in a raster
> view. `_resolve_default_bands` therefore always resolves the field explicitly: it keeps
> the canonical source's default display bands only when every index is in range for the
> output, and otherwise **clears** it so WISER derives display bands from the stamped
> wavelengths (truecolor) or the first bands.

The result is **not** loaded back into the running session — export writes the file and
the user opens it manually via **File → Open** (a deliberate product decision), which is
why the on-disk header must be fully self-describing.

### GUI wiring and threading

`MosaicPane._on_export_clicked` (GUI thread) requires ≥1 visible scene, resolves the grid
via `_ensure_common_grid()` (surfacing `TargetCrsRequired`/`UnmappableCrsError` as a
warning), asks for an output path with `QFileDialog.getSaveFileName`, resolves the output
nodata (`_resolve_output_nodata`: the band-metadata dataset's Data Ignore Value, else the
top-most visible scene that has one), snapshots the controller state, and hands the heavy
work to a scheduler thread via the same `run_with_progress` bridge the ingestion path uses
(progress modal + Activity Monitor row + cancellation, blocking only the mosaic dialog).
The module-level worker `_export_mosaic_task` just calls `export_mosaic` and returns the
written path; `_on_export_finished` shows a confirmation dialog. Like `_ingest_scene`, the
worker never logs — progress and raised exceptions are its only channels out.

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
- `src/tests/test_mosaic_compositor.py` — the Qt-free `render_scene_argb`: alpha is 0
  exactly on the nodata collar and 255 on the valid interior, RGBA shape/dtype, valid
  pixels get color, a disjoint viewport comes back fully transparent (no Qt).
- `src/tests/test_mosaic_view_gui.py` — the overlay **and** pixel-layer wiring end to
  end, driven through the real `MosaicPane` ingestion path: geometry cache populates /
  re-invalidates; `composite()` stacks known ARGB layers (top opaque wins, holes reveal
  below); the per-scene cache populates; z-order reorder / visibility toggle trigger **no
  GDAL reads** (spy on `render_scene_argb`); and a pan burst **coalesces into a single
  debounced background read**.
- `src/tests/test_mosaic_pane_gui.py` — the control panel (#638) end to end through the
  real ingestion path: drag-reorder updates controller z-order and triggers **no GDAL
  reads**; a resolution-mode change rebuilds the grid; a band-metadata selection sets the
  canonical source without changing band count; and a non-Nearest-Neighbor resampling
  choice surfaces the warning and invalidates the pixel cache.
- `src/tests/test_mosaic_export.py` — the Qt-free `export_mosaic` (#639): two overlapping
  fixtures give the correct z-order winner per pixel (and the reversed order flips it),
  nodata passthrough (a hole in the top scene reveals the lower one; outside every
  footprint is the output nodata), Nearest-Neighbor value preservation (exact source
  values), band count + canonical metadata round-tripped through the written ENVI, and the
  `default bands` guards (GDAL's 1-based RGB placeholder is cleared; an out-of-range source
  default is dropped so the file opens without an `Illegal band #` crash).
- `src/tests/test_mosaic_export_gui.py` — the Export button end to end via `WiserTestModel`:
  ingest two scenes, patch the save dialog, click Export, and assert the ENVI `.img`/`.hdr`
  are written on the common grid with nodata carried through while **nothing is loaded back
  into WISER** (dataset count unchanged); plus the no-scenes guard (informs and never
  reaches the file picker).
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
- `src/tests/test_mosaic_georeference_gui.py` — re-georeferencing a scene in place
  (#685) end to end via `WiserTestModel`: the context menu opens a `GeoReferencerDialog`
  locked onto the scene (target + save path); a simulated `warp_completed` reingests and
  swaps the corrected scene in at its original z-order slot (blocking the georeferencer
  dialog, not the mosaic window); a repeated warp **replaces** rather than stacks; the
  warp output is **not** registered as a global dataset; Cancel restores the original at
  its slot while "Save to Mosaic" keeps the warped one.

GUI tests use `@pytest.mark.functional`/`@pytest.mark.smoke` with the
`WiserTestModel` harness (`src/test_utils/test_model.py`), not pytest-qt/qtbot. Run
with `QT_QPA_PLATFORM=offscreen` and `PYTHONPATH=src`.
