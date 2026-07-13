# Seamless Mosaic Internals

This page documents the internal architecture of WISER's **Seamless Mosaic** feature —
combining several overlapping georeferenced scenes onto one shared output grid. It is
intended for developers reading, debugging, or extending the tool.

```{note}
This is an in-progress feature (EPIC #629). As of this writing, scene ingestion,
materialization (a **display-only** preview artifact baked eagerly at ingest with the
full-band cube materialized **lazily at export** — see [Display-Only Preview and Frozen
Metadata](#display-only-preview-and-frozen-metadata)), common-grid/CRS resolution, the
**vector overlay** (footprints, bounding box, overlap highlight), the **static-scene
pixel compositor** (the composited
preview, with an off-thread debounced tiled cache (#674) and a stable, zoom-independent
per-scene contrast stretch cached at ingest — see [Stretch
bounds](#stretch-bounds-compute_stretch_bounds-mosaic_ingestionpy)), the **control panel** (drag-to-reorder z-order, resolution mode, resampling
method, and the band-metadata chooser), the **full-resolution export path**
(streaming ENVI compositor on the common grid — see [The Export
Path](#the-export-path)), and **in-place scene
re-georeferencing** (right-click → warp → swap; see [Re-georeferencing a Scene In
Place](#re-georeferencing-a-scene-in-place)), **pending (disabled) scenes**
(non-georeferenced or CRS-incompatible scenes carried in the list until georeferenced;
see [Pending Scenes](#pending-scenes-disabled-scenes)), and **zoom-to-scene**
(right-click a live scene → frame the camera on its footprint; see [Zoom to
Scene](#zoom-to-scene)) are implemented and wired into the GUI.
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
  tiled ARGB cache — see [The Pixel Layer](#the-pixel-layer-static-scene-compositor))
  and, on top, the **vector overlay** (footprints, bounding box, overlap highlight — see
  [The Geometry Overlay](#the-geometry-overlay-vector-layer)).

Every scene, regardless of its original backing (GDAL file, NumPy array, PDR, etc.),
is turned into a warpable, disk-backed tiled GeoTIFF by a `SceneMaterializer` — the
`RasterDataSet` is the source of truth for metadata, so materialization stamps
SRS/geotransform/nodata/band metadata from the dataset object, not from its `_impl`.
As of #677 there are **two** materialization moments: at ingest the pane bakes a
small **display-only** GeoTIFF (just the 1 or 3 display bands) that the preview reads,
and at export the full-band cube is materialized **lazily**, stamped from a metadata
snapshot **frozen at ingest**. See [Display-Only Preview and Frozen
Metadata](#display-only-preview-and-frozen-metadata).

**Core files:**

| File | Responsibility |
|------|----------------|
| `src/wiser/gui/mosaic_dialog.py` | `SeamlessMosaicDialog` — top-level dialog shell, owns the session `SceneMaterializer` |
| `src/wiser/gui/mosaic_pane.py` | `MosaicPane` — control panel; ingestion orchestration (display-band resolution + metadata freeze), scene list (drag-reorder + visibility), resolution / CRS / resampling / band-metadata controls, export (drift warning) |
| `src/wiser/gui/mosaic_view.py` | `MosaicView` — the two-layer paint surface (pixel + vector), currently a stub |
| `src/wiser/gui/mosaic_crs_dialog.py` | `ReprojectPromptDialog` — modal target-CRS chooser |
| `src/wiser/gui/app.py` | `App.get_current_display_bands` — narrow accessor for the main view's live per-dataset band selection (the display-band resolver injected into the mosaic) |
| `src/wiser/raster/mosaic_controller.py` | `MosaicController`, `MosaicScene`, `SceneMetadataSnapshot`, `CommonGrid`, `ResolutionMode`, `ScenePendingReason` — the non-GUI model (incl. the live/pending classifier) |
| `src/wiser/raster/mosaic_ingestion.py` | `is_dataset_georeferenced`, `validate_scene_addable`, `validate_scene`, `build_overviews`, `compute_stretch_bounds`, `compute_footprint_wkt` — Qt-free ingestion gates |
| `src/wiser/raster/mosaic_compositor.py` | `render_scene_argb` — Qt-free per-scene warp→RGBA renderer (alpha = validity) for the pixel layer; reads the display-only artifact's bands in order |
| `src/wiser/raster/mosaic_export.py` | `export_mosaic` — lazily materializes each scene's full-band cube (from live pixels + frozen snapshot), warps onto the common grid, streams to ENVI |
| `src/wiser/raster/mosaic_materialize.py` | `SceneMaterializer` (`build_display_source`), `materialize_to_tiled_geotiff`, `materialize_display_only_to_tiled_geotiff`, `materialize_full_band_from_snapshot` — the object-model adapter |
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
        -_tile_cache : OrderedDict~TileKey, QImage~
        -_inflight_tiles : Set~TileKey~
        -_read_epoch : int
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
        -_display_cache : dict
        +build_display_source(dataset, display_bands, progress) str
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
        +is_scene_pending(scene) bool
        +scene_pending_reason(scene) ScenePendingReason
        +live_scenes() List~MosaicScene~
        +has_pending_scenes() bool
    }

    class MosaicScene {
        <<dataclass>>
        +dataset : RasterDataSet
        +visible : bool
        +gdal_path : str
        +footprint_wkt : str
        +has_overviews : bool
        +snapshot : SceneMetadataSnapshot
    }

    class SceneMetadataSnapshot {
        <<dataclass>>
        +spatial : SpatialMetadata
        +spectral : SpectralMetadata
        +display_bands : tuple
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
    MosaicScene --> SceneMetadataSnapshot : frozen at ingest
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

Adding a scene runs four gated phases on a background thread, orchestrated by
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
    Pane->>Pane: validate_scene_addable(dataset, existing_scenes)
    alt duplicate / band-count mismatch
        Pane-->>User: QMessageBox.warning
    else not georeferenced
        Pane->>Ctrl: add_scene(MosaicScene(dataset)) [pending placeholder]
        Pane->>Pane: _refresh_scene_list() [warning icon + tooltip]
    else georeferenced (may still land pending)
        Pane->>Pane: resolve display bands (main-view selection → find_display_bands)
        Pane->>Pane: SceneMetadataSnapshot.from_dataset(dataset, display_bands)
        Pane->>Sched: run_with_progress(_ingest_scene, dataset, materializer, snapshot)
        Note over Pane: ProgressDialog shown, block_window disabled
        Sched->>Ingest: run on scheduler thread
        Ingest->>Mat: build_display_source(dataset, snapshot.display_bands) → display-only GeoTIFF
        Ingest->>Ingest: build_overviews(gdal_path)
        Ingest->>Ingest: compute_stretch_bounds(gdal_path, dataset)
        Ingest->>Ingest: compute_footprint_wkt(gdal_path)
        Ingest-->>Sched: MosaicScene(dataset, gdal_path, footprint_wkt, has_overviews=True, stretch_bounds)
        Sched-->>Pane: on_success(scene) [GUI thread]
        Pane->>Ctrl: add_scene(scene)
        Pane->>Pane: _refresh_scene_list()
        Pane->>Pane: _ensure_common_grid()
        Pane->>Pane: _mosaic_view.update()
    end
```

Note the display-band resolution and the metadata snapshot are computed on the **GUI
thread** before the worker starts (both must read live main-view / dataset state at
add-time); only the display-only materialization, overviews, and footprint run on the
background thread. See [Display-Only Preview and Frozen
Metadata](#display-only-preview-and-frozen-metadata).

### Validation (`validate_scene`, `mosaic_ingestion.py`)

Runs synchronously on the **main thread** before any background work starts, so
rejection is immediate (no spinner churn). The pane calls **`validate_scene_addable`**,
which enforces only the two gates that make a scene *fundamentally un-addable*
regardless of georeferencing:

- **Duplicate** — the dataset is already in the mosaic (by dataset id).
- **Band-count mismatch** — differs from the first existing scene's band count.

Being ungeoreferenced or CRS-incompatible is **no longer a rejection**: such scenes are
added as disabled **pending** scenes (see [Pending
Scenes](#pending-scenes-disabled-scenes)). The pane branches on
`is_dataset_georeferenced(dataset)` (a non-identity geotransform **and** a non-empty
SRS): a georeferenced dataset runs the full ingest pipeline below (it may still land
pending if its CRS can't reach the target), while a non-georeferenced one is appended
immediately as a bare `MosaicScene(dataset=...)` placeholder with **no** background work.

Deliberately **not** rejected either: missing nodata (a coarser, full-rectangle
footprint is still valid) and dtype mismatches across scenes (the compositor promotes to
the widest common type at warp time). See `# TODO(#640)` in the source for a planned
warn-but-allow path for band-count mismatches.

```{note}
`validate_scene` (the strict "must be immediately placeable" gate that also rejects
ungeoreferenced / no-SRS inputs) is retained in `mosaic_ingestion.py` and now delegates
to `is_dataset_georeferenced` + `validate_scene_addable`. The pane no longer calls it —
it is kept as the canonical full check and for its unit tests.
```

### Materialization (`SceneMaterializer.gdal_source`, `mosaic_materialize.py`)

At ingest each scene — GDAL-backed or not — is materialized to a **display-only**
disk-backed, tiled GeoTIFF (`TILED=YES`, 256×256 blocks) under a per-session temp
directory (`wiser.utils.primitives.temp_dir()`), **not** `/vsimem` (RAM-backed), so
GDAL reads only the requested windows. It contains only the frozen **display bands**
(1 for grayscale, 3 for RGB — bands 1/2/3 *are* R/G/B), so every warp and overview
level touches 1–3 bands instead of all N — the whole point of #677 (see [Display-Only
Preview and Frozen Metadata](#display-only-preview-and-frozen-metadata)). Metadata
(geotransform, SRS, nodata, per-band wavelength) is stamped from the `RasterDataSet`
object band-remapped onto the kept bands, with nodata preserved so the preview alpha
mask still works. A per-scene dedup cache keyed by `(dataset id, display bands)` keeps
this to one write per scene/band-choice per session.

The full-band GeoTIFF (`materialize_to_tiled_geotiff` /
`materialize_full_band_from_snapshot`) is materialized **lazily at export**, not here.
The user's original dataset is **never modified**; the materialized copies are all
WISER-owned temp artifacts.

### Overviews (`build_overviews`, `mosaic_ingestion.py`)

Builds internal pyramid overviews (`NEAREST`, levels `[2, 4, 8, 16]`) directly inside
the materialized **display-only** temp GeoTIFF (`GA_Update` + `BuildOverviews`), so
preview rendering has no first-paint stutter. Because materialization already produces
a WISER-owned temp copy, overviews can be written internally rather than as external
`.ovr` sidecars — and with only 1–3 bands the pyramid is a fraction of the full-cube
cost.

### Stretch bounds (`compute_stretch_bounds`, `mosaic_ingestion.py`)

Computes stable, extent-independent 2-98 percentile contrast-stretch bounds per
display band (issue #675), stored on `MosaicScene.stretch_bounds` (source band index
→ `(lo, hi)`). Runs right after overviews, since it samples one: reads the
level-4 internal overview (falling back to the coarsest available, or the full band
if no overviews exist) rather than the full-resolution data, so a whole-scene
percentile pass stays cheap regardless of the source's size — level 4 rather than the
coarsest level trades a little ingest time for a larger, more representative sample
(the coarsest overview's sparser `NEAREST`-resampled pixels risk missing a small
bright/dark feature). A band with no valid (non-nodata) pixels in the sample is simply
omitted from the result.

This is what fixes the bug the pixel layer's per-scene renderer used to have: see
[The per-scene renderer](#the-per-scene-renderer-render_scene_argb-qt-free) below for
how the compositor consumes these bounds instead of recomputing a stretch from
whatever happens to be in the current viewport.

### Footprint (`compute_footprint_wkt`, `mosaic_ingestion.py`)

Derives the valid-pixel outline via `gdal.Footprint(..., format="WKT")` on the
display-only artifact, in the scene's **own** CRS (no reprojection here — that happens
later, in the grid builder). The nodata is preserved on the display-only file, so with
a nodata value set this traces the true valid-pixel boundary; without one, it falls
back to the full raster rectangle.

### Progress reporting

All four phases share one `ProgressReporter` (`wiser/utils/progress.py`), split by
weight — materialize 0.45, overviews 0.30, stretch bounds 0.10, footprint 0.15 — so the
overall bar advances smoothly across phases that report in very different native units
(per-band count vs. GDAL's own `0..1` callbacks). The reporter is Qt-free; `run_with_progress`
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

## Display-Only Preview and Frozen Metadata

Issue #677 splits scene materialization into two artifacts and freezes the dataset
metadata at ingest, so the preview is fast and the whole session is deterministic
against concurrent edits in main WISER.

### Two artifacts, two moments

The preview compositor only ever reads the **display bands** (1 for grayscale, 3 for
RGB), so carrying every non-display band through each `gdal.Warp` and every overview
level was pure overhead — a large multiplier for hyperspectral scenes. So:

- **At ingest** the pane bakes a **display-only** GeoTIFF containing just the resolved
  display bands (`SceneMaterializer.build_display_source` →
  `materialize_display_only_to_tiled_geotiff`). Its bands 1/2/3 *are* R/G/B, nodata is
  preserved for the alpha mask, and overviews + footprint are built on it. This is what
  `scene.gdal_path` points at and what the pixel layer reads. The session dedup cache
  holds **only** these small artifacts.
- **At export** the full-band cube is materialized **lazily and uncached**
  (`materialize_full_band_from_snapshot`), the only moment all bands are actually needed
  — see [The Export Path](#the-export-path).

Band selection therefore moves from **read time to materialize time**: the compositor no
longer picks 3-of-N, it reads the small file's bands in order.

### Which display bands get baked in

The display-only file uses the bands **currently shown in the main view** for that
dataset, not `dataset.get_default_display_bands()` — a dataset may have no defaults, and
even when it does the user may have changed the displayed bands via the band chooser, a
choice the mosaic should honor. Those bands are **GUI state** (`RasterPane._display_bands`),
so at ingest `MosaicPane._resolve_display_bands` resolves them in order:

1. the main view's current selection for the dataset, via the injected resolver
   (`App.get_current_display_bands` — a narrow public accessor, so the mosaic does not
   reach into `RasterPane` internals);
2. otherwise `find_display_bands(dataset)` (`dataset.py`) — the defaults → human-eye
   wavelength → first 1/3 bands fallback chain (also covers "never shown in the main
   view").

### Freezing the metadata (`SceneMetadataSnapshot`)

`RasterDataSet` metadata (data-ignore, wavelengths, default display bands) can be edited
in main WISER while the mosaic dialog is open (via `dataset_editor_dialog.py`), and
`RasterDataSet` is **not** a `QObject` — it emits no change signal, only sets a dirty
flag. So at ingest the pane snapshots the relevant metadata onto the scene:
`SceneMetadataSnapshot.from_dataset(dataset, display_bands)` **deep-copies** the dataset's
`SpatialMetadata` and `SpectralMetadata` (a shallow copy would share the dataset's
internal `band_info` list) and stores the resolved `display_bands` alongside. Both the
display-only materialization and the lazy full-band export stamp **from this snapshot**,
never the live dataset, so a mid-session edit cannot silently change the mosaic. The
display-band resolution and the snapshot are both built on the **GUI thread** before the
worker starts, since both read live main-view / dataset state at add-time.

Data-ignore is **not** cosmetic here: changing it moves the footprint, the alpha validity
mask, and potentially the common grid — which is exactly why freezing (rather than
consuming a live value at export) is the safe default.

### Drift warning on export

Because the snapshot may diverge from a later live edit, `_on_export_clicked` compares
each visible scene's frozen `snapshot.spectral` against a freshly-read
`dataset.get_spectral_metadata()` (`_scene_metadata_drifted`, using `SpectralMetadata`'s
own `__eq__`) and, if any drifted, shows a **warn-and-proceed** dialog naming them: the
export will use the frozen values, so the user can proceed knowingly or cancel and re-add
the scene to pick up the edit.

### Out of scope (follow-up)

The nicer end-state is to *listen* for changes and re-run the affected ingest phases
(a main-view display-band change already emits `RasterPane.display_bands_change`; the
dataset-model edits do not, so they need new signal infrastructure). The frozen snapshot
is a prerequisite either way — it is how "did it change?" is detected — but the reactive
re-ingest is deliberately deferred.

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
    B -->|yes| D{"any live scenes?<br/>(visible, non-pending)"}
    D -->|no| E[return empty CommonGrid]
    D -->|yes| F["target = target_crs_wkt or<br/>_common_crs_wkt(live scenes)"]
    F --> G{target resolved?}
    G -->|no| H[raise TargetCrsRequired]
    G -->|yes| I["persist target_crs_wkt"]
    I --> J["per live scene: reproject footprint envelope<br/>into target CRS → union extent"]
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
- **Changing the target CRS no longer rejects incompatible scenes.** `build_common_grid`
  builds from **live** scenes only, and `_prompt_for_target_crs` sets the chosen target
  unconditionally (it no longer calls `validate_target_crs` as a blocker). A scene the new
  target can't reach simply becomes [pending](#pending-scenes-disabled-scenes) instead of
  failing the change; if the choice leaves *no* live scenes, `_on_choose_target_crs` warns
  that the preview is empty but still commits the CRS.

`validate_target_crs()` and `validate_new_scene_crs()` remain on the controller (and are
still unit-tested) but are now **advisory** — no GUI path uses them as hard gates.

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
assumption ever changes (e.g. a future "clear target CRS" control). The
`UnmappableCrsError` branches are likewise defensive now: since the pending-scene work,
`build_common_grid` builds from `_live_scenes()` — scenes that are *by construction*
mappable to the target — and no longer calls `validate_target_crs`, so an incompatible
scene becomes [pending](#pending-scenes-disabled-scenes) instead of raising here. The
`except` clauses are retained as a safety net.

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
  `wiser.raster.utils.can_transform_between_srs`. Now **advisory** — the GUI classifies
  incompatible scenes as pending rather than calling this as a hard gate (see [Pending
  Scenes](#pending-scenes-disabled-scenes)).
- `is_scene_pending(scene)` / `scene_pending_reason(scene)` — classify a scene as
  live (`None`) or pending (`ScenePendingReason.NO_CRS` / `INCOMPATIBLE_CRS`).
- `live_scenes()` / `has_live_scenes()` / `has_pending_scenes()` — the live/pending
  partition used by the grid builder, overlay, compositor, and export.

All SRS objects used for transforms are built with
`OAMS_TRADITIONAL_GIS_ORDER` (long/lat, x/y axis order), matching the geotransform
convention used throughout WISER (see the georeferencer's identical convention in
[Georeferencer Internals](georeferencer-internals.md)).

---

## Pending Scenes (Disabled Scenes)

**Files:** `src/wiser/raster/mosaic_controller.py` (the classifier),
`src/wiser/raster/mosaic_ingestion.py` (the add gate), and `src/wiser/gui/mosaic_pane.py`
(add path, list decoration, CRS change, export guard).

A scene can now be added to the mosaic even when it cannot yet be placed on the common
grid — because it is **not georeferenced**, or because its CRS **cannot be transformed**
into the mosaic's locked target CRS. Rather than rejecting it at Add-Scene time, the
mosaic carries it as a **pending** (disabled) scene: it stays in the list, greyed out with
a warning icon, and is excluded from the grid, the overlay, the pixel compositor, and
export until it is georeferenced into a compatible CRS. This lets a user assemble a
working set — including scenes they still intend to register — without the add being a
dead end.

### Live vs. pending is a derived classification, not stored state

There is no `pending` flag on `MosaicScene`. A scene's status is **computed on demand**
from the controller's current state by `scene_pending_reason(scene)`, so it re-evaluates
for free whenever the target CRS or the scene's own georeferencing changes — no bookkeeping
to keep in sync. A scene is pending when:

- **`ScenePendingReason.NO_CRS`** — it never went through ingestion (no `gdal_path` /
  `footprint_wkt`). `_is_ingested` is the controller's Qt-free proxy for "georeferenced":
  the pane only runs the ingest pipeline on georeferenced datasets, so the presence of
  ingestion artifacts *is* the georeferenced signal (and it works with the lightweight
  dataset stand-ins used in tests, without re-reading a geotransform).
- **`ScenePendingReason.INCOMPATIBLE_CRS`** — it is ingested, but its SRS cannot be
  transformed (`can_transform_between_srs`) into the **resolved target CRS**.

`_resolved_target_wkt()` defines what "compatible" is measured against: the explicit
`_target_crs_wkt` if one is set (or was auto-locked by the first grid build), otherwise the
shared CRS of the *georeferenced* visible scenes. When it returns `None` (empty mosaic,
all-pending mosaic, or georeferenced scenes that disagree with no explicit target), an
ingested scene is treated as **live** — there is nothing yet to be incompatible with, and a
multi-CRS disagreement is still the existing `TargetCrsRequired` path, unchanged.

### The live-scene partition drives everything downstream

`_live_scenes()` = visible scenes that are not pending, and it is the input to every
grid-consuming operation:

- `build_common_grid()` unions extents and picks resolution from `_live_scenes()` only, so
  a pending scene never perturbs the output grid.
- `visible_scene_footprints_in_common_crs()` (the overlay) and the pixel compositor iterate
  live scenes, so a pending scene draws nothing.
- Export composites `live_scenes()` (see [The Export Path](#the-export-path)).

Because the grid is built from live scenes, an **all-pending mosaic yields an empty grid**
(not an error): `has_live_scenes()` is `False`, the view shows blank canvas, and the CRS
stays unlocked until the first live scene resolves it.

### Add path (`MosaicPane._on_add_scene_clicked`)

Add-Scene calls `validate_scene_addable` (duplicate + band-count only) and then branches on
`is_dataset_georeferenced`:

```{mermaid}
flowchart TD
    A["Add Scene…"] --> B["validate_scene_addable<br/>(duplicate / band count)"]
    B -->|fails| W[QMessageBox.warning, stop]
    B -->|ok| C{is_dataset_georeferenced?}
    C -->|no| D["add_scene(MosaicScene(dataset))<br/>bare pending placeholder"]
    C -->|yes| E["run ingest pipeline<br/>(materialize/overviews/stretch/footprint)"]
    D --> F["_refresh_scene_list()<br/>(warning icon + tooltip)"]
    E --> G["_on_scene_ingested → add_scene,<br/>_ensure_common_grid"]
    G --> H{scene compatible<br/>with target?}
    H -->|yes| I[live: grid rebuilds, preview shows it]
    H -->|no| J["INCOMPATIBLE_CRS pending<br/>(ingested but off-grid)"]
```

A non-georeferenced dataset is appended as a bare `MosaicScene(dataset=...)` with **no**
background work — there is nothing to materialize until it has real georeferencing. A
georeferenced dataset still runs the full pipeline (its artifacts are what a later
compatibility check reads), and may end up live or `INCOMPATIBLE_CRS`-pending depending on
the locked target.

### List decoration (`_refresh_scene_list`)

For each pending scene the list item gets a warning icon (`SP_MessageBoxWarning`), a greyed
foreground, and a hover tooltip explaining *why* it is disabled (no georeferencing vs.
incompatible CRS) and how to fix it. The visibility checkbox is deliberately left
**enabled** — pending is an independent axis from user-hidden, and a scene's classification
is what excludes it from the grid, not its checkbox.

### Promotion: georeference in place

A pending scene is fixed through the **same** right-click → "Georeference…" flow documented
in [Re-georeferencing a Scene In Place](#re-georeferencing-a-scene-in-place). The warp bakes
a real CRS + geotransform into the scene and reingests it, so it gains `gdal_path` /
`footprint_wkt` in a CRS that (if compatible) makes `scene_pending_reason` return `None` on
the next `_ensure_common_grid` — the scene goes live and the grid rebuilds to include it.
No separate "promote" code path exists; promotion is just "it is no longer pending after the
re-georeference reingest."

### Target-CRS change re-evaluates every scene

Because status is derived, `_on_choose_target_crs` doesn't reject an incompatible target —
it commits the new target and lets the classifier re-partition: scenes that can't reach the
new CRS become `INCOMPATIBLE_CRS`-pending, and previously-pending scenes that *can* reach it
go live. If the change leaves no live scenes at all, the pane warns that the preview is
empty but still keeps the chosen CRS. Removal / visibility changes rebuild quietly as before
(see [Removal and visibility changes](#removal-and-visibility-changes)).

A CRS change also moves the world grid without moving the camera, so `_on_choose_target_crs`
then calls `MosaicView.ensure_scenes_in_view()` to reframe the preview when the mosaic
landed off-screen — see [Reframing after a target-CRS
change](#reframing-after-a-target-crs-change).

### Export guard

Export operates on `live_scenes()`. If `has_pending_scenes()` is `True`,
`_on_export_clicked` warns and asks for confirmation before proceeding, then **skips** the
pending scenes (they are silently absent from the output rather than blocking the export).

---

## The Pixel Layer (Static-Scene Compositor)

**Files:** `src/wiser/raster/mosaic_compositor.py` (Qt-free per-scene renderer) and
`src/wiser/gui/mosaic_view.py` (cache, compositing, drawing, threading).

Beneath the vector overlay, `MosaicView` draws the actual composited scenes (issue
#637): each visible scene is warped, one **tile** at a time, into an **ARGB `QImage`**
whose alpha is its validity mask, and the scenes are stacked bottom-to-top honoring
z-order so lower scenes show through upper scenes' nodata holes. The tiling and caching
(#674) live in `mosaic_view.py`; the renderer below is unchanged — it just receives a
tile's world rectangle rather than the whole viewport.

### The per-scene renderer (`render_scene_argb`, Qt-free)

`mosaic_compositor.render_scene_argb(scene, target_wkt, world_extent, w, h)` warps one
scene onto the requested `world_extent` at output size `w×h` via
`gdal.Warp(..., dstSRS=target_wkt, outputBounds=world_extent, dstAlpha=True)` and returns
an `(h, w, 4)` uint8 RGBA array. Callers pass one tile's world rectangle at `256×256`
(see [The tiled cache](#the-tiled-cache-and-composite)), but the renderer neither knows
nor cares that it is a tile. A single `gdal.Warp` does four jobs at once:

- **reprojection** onto the target CRS,
- **downsampling** — because the output is far coarser than the source, GDAL reads from
  the internal **overviews** built in #634 (the whole point of building them),
- **alignment** to the visible world rectangle, and
- **the validity mask** — `dstAlpha=True` yields an alpha band that is `0` on nodata /
  outside-coverage pixels, so the alpha channel *is* the validity mask (no manual
  mask-band read). This mirrors the warp seam already used by `_warped_resolution`.

RGB comes from the dataset's `get_default_display_bands()` (1 band → grayscale, 3 →
RGB), contrast-stretched per band (2–98 percentile). Invalid pixels are forced fully
clear `(0,0,0,0)` so nothing bleeds under a transparent alpha. It is Qt-free (produces
a NumPy array) so it is unit-testable without a running app and can run on a
background thread; the view wraps the array into a `QImage` on the GUI thread.

> **The stretch is against `scene.stretch_bounds`, not the viewport's pixels** (issue
> #675). Originally the 2-98 percentile was computed from whatever pixels happened to
> land in the current warp output — so zooming in fed a smaller, different pixel
> population into the percentile on every paint, and a scene's on-screen contrast
> visibly drifted as the user zoomed, even though nothing about the data changed.
> `render_scene_argb` now looks up `scene.stretch_bounds[band_idx]` — precomputed once
> at ingest by
> [`compute_stretch_bounds`](#stretch-bounds-compute_stretch_bounds-mosaic_ingestionpy)
> from a whole-scene overview sample — and stretches every render against those fixed
> bounds, so contrast is identical whether the viewport shows the full scene or a
> single zoomed-in pixel. `_stretch_band` still falls back to the old viewport
> percentile when `bounds` is `None` (a `MosaicScene` built directly, e.g. in a unit
> test, without running ingestion).
>
> The bounds are **per scene**, not combined across the mosaic's visible scenes, even
> though a shared min/max across all visible scenes would look more consistent when
> scenes have very different value ranges. That was a deliberate trade-off: combined
> bounds would have to change (and force a re-render) whenever visibility changes, which
> breaks the `recomposite_only()` **zero-GDAL-reads** guarantee for visibility/z-order
> changes described in [The tiled cache and
> `composite()`](#the-tiled-cache-and-composite) below — a guarantee
> `test_mosaic_view_gui.py` asserts by spying on `render_scene_argb`. Independent
> per-scene bounds fully fix the zoom-instability bug (the actual report) without
> touching that contract.

### The tiled cache and `composite()`

`MosaicView` caches the pixel layer as a grid of **tiles**, not one viewport-sized image
per scene (issue #674). World space is quantized, **per discrete zoom bucket**, into a
fixed grid of `_TILE_PX`-square (256×256) cells; one cached ARGB `QImage` is one scene's
warp of one cell, keyed by `_TileKey = (id(scene), bucket, col, row)`. `_tile_cache` is an
insertion-ordered LRU bounded to `_TILE_CACHE_MAX` tiles; `_inflight_tiles` holds the keys
currently being warped so a tile is never submitted twice.

The bucket is `round(log2(world_units_per_pixel))`, and a bucket's tiles are rendered at
`bucket_wupp = 2**bucket` world-units-per-pixel — a power-of-two "pyramid level" (aligned
with the #634 overviews) chosen so a whole *range* of continuous zooms reuses one set of
tiles: the world→screen affine scales those tiles to the exact zoom (the standard
slippy-map look) instead of re-reading on every scale change. Tiles are anchored at the
CRS's real `(0, 0)`, so the same world area always maps to the same cell — which is why a
pan is a cache hit on the tiles it slides back over.

`_render_bucket()` **floors** that bucket at the common grid's resolution so the preview
is never rendered *finer* than the mosaic's real data. Rendering finer would be pure
waste — `gdal.Warp` would upsample the grid onto a denser output, producing no new detail
(and, several multiples below native, getting slow). Past native zoom the render bucket
is therefore pinned at the grid-resolution level: the same grid-resolution tiles are
reused and the affine scales them up (one source pixel drawn as a screen block), so a deep
zoom-in is **all cache hits, zero reads**. The floor is one view-wide value (the grid's
finest pixel size), so every scene still renders at one shared bucket and the per-cell
`composite()` stays pixel-aligned — a per-scene floor would put mixed-resolution scenes on
different grids and break that alignment.

`composite(layers)` is unchanged and remains the **single indirection point**: it stacks a
list of ARGB layers bottom-to-top with `QPainter` `SourceOver`. It is now called **per
cell** — every visible scene's tile at one `(bucket, col, row)` is the same size and
pixel-aligned, so it stacks them exactly as it stacked full-viewport layers before, and
deferred seamline/feathering still re-implements only its internals (by territory mask)
without touching callers.

This gives the same three caching tiers, now in tile terms:

- **Z-order reorder / visibility toggle** (`recomposite_only()`) — a pure repaint from the
  cache, **no GDAL reads**. Draw order in `paintEvent` applies z-order; hidden scenes are
  skipped; re-showing a scene reuses its still-cached tiles (they are keyed per scene and
  never evicted on hide). The only case that reads is *revealing* a scene hidden at every
  prior read of this viewport — `_start_pixel_read` finds its cells missing and warps just
  them, so an all-cached restack submits **zero** warp jobs (asserted in
  `test_mosaic_view_gui.py` by spying on `render_scene_argb`).
- **Pan / zoom** — reuses every tile the new viewport shares with the old and warps **only
  the newly-exposed cells**, plus a one-tile prefetch ring (`_PREFETCH_RING`) so the loaded
  margin travels ahead of the viewport. This is the #674 fix: the per-read cost scales with
  the *newly-exposed* strip, not the whole viewport, so a pan no longer re-warps everything
  and the black edge around the viewport is gone. Zoom reuses tiles too, because a range of
  zooms shares one bucket.
- **Add / remove scene, target-CRS change** (`invalidate_pixels()`) — clears the whole
  cache and bumps `_read_epoch`, so a read already in flight drops its result instead of
  inserting now-stale tiles; the next paint refills.

### Off-thread, debounced reads, and the two-pass paint

Reads run **off the UI thread** and pan/zoom re-reads are **debounced** so the view stays
responsive. A `paintEvent` whose camera moved — plus `invalidate_pixels()` /
`recomposite_only()` — (re)starts a single-shot `QTimer` (`_PIXEL_READ_DEBOUNCE_MS`,
~80 ms); a gesture's burst of paints collapses into one read once the camera settles. On
fire, `_start_pixel_read` snapshots the viewport on the GUI thread (target CRS, resampling
method, footprints and the per-tile intersect test — cheap OSR work) and builds a warp job
**only for each cell missing from `_tile_cache` and not already in `_inflight_tiles`**,
then hands the heavy per-tile warps to `app_services.scheduler.submit_thread`. The worker
returns NumPy arrays (no Qt off-thread); the done-callback emits the `_read_ready` signal,
whose queued connection wraps them into `QImage`s and inserts them (evicting LRU) on the
GUI thread. Reads are **additive** — a late or superseded result is still a valid tile for
its cell — so there is no per-read "discard superseded" step; clearing the submitted keys
from `_inflight_tiles` (and the `_read_epoch` check for invalidations) is all the
bookkeeping needed. With no scheduler (a bare view in a unit test) the read runs
synchronously.

`paintEvent` draws the pixel layer in **two passes**, both mapping each tile's world rect
through the world→screen affine (so off-screen tiles never land on the widget):

1. **Fallback pass** — under-draws the nearest *other* populated zoom bucket's cached tiles
   that intersect the viewport, drawn scaled by the camera. This keeps the frame filled
   during a zoom into a not-yet-read bucket (coarser/finer tiles stretch until the sharp
   bucket lands); a pan's leading edge is already covered by the prefetch ring, so together
   they eliminate the black frame between a gesture and its read.
2. **Sharp pass** — for each current-bucket cell covering the viewport, gathers each
   visible scene's cached tile, `composite()`s them into one cell image, and blits it at the
   cell's world rect. Cells still missing are exactly what the debounced read above is
   fetching.

> **The render bucket is floored at the grid resolution (`_render_bucket()`), so the
> preview never warps finer than the mosaic's real data.** Beyond native zoom, tiles are
> reused and scaled by the affine rather than re-warped — which both matches what the
> viewer would actually see (upsampling adds no detail) and avoids the deep-upsample warp
> cost, since `gdal.Warp` past native would read full-resolution source (no coarser
> overview to shortcut through) for every tile. `test_mosaic_view_gui.py` asserts a deep
> zoom-in past native triggers **zero** `render_scene_argb` calls.

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
    G --> P{"camera moved<br/>or pixels dirty?"}
    P -->|yes| Q["_schedule_pixel_read()<br/>(debounced, off-thread, missing tiles only)"]
    P -->|no| H
    Q --> H["draw pixel layer:<br/>fallback bucket + per-cell composite tiles,<br/>each mapped by world_to_screen"]
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

`paintEvent` draws, in order: the tiled pixel layer (#637/#674), then the bounding
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
  the CRS constraint),   defers a `_refresh_scene_list()` to rewrite the now-stale
  `Qt.UserRole` indices safely after the drop, and does the pure-restack invalidation.
  `_refresh_scene_list()` also decorates any [pending](#pending-scenes-disabled-scenes)
  scene with a warning icon, greyed text, and a why-disabled tooltip (visibility checkbox
  left enabled).

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
reorder-without-reads are covered by `src/tests/test_mosaic_control_panel_gui.py`, with
the controller-side surface (resample round-trip, band-metadata fallback, no-grid-invalidation)
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
bottom-to-top index) and pops a `QMenu` whose actions are built **per-scene**:

- **"Georeference…"** → `_on_georeference_scene(index)`. Offered only when the scheduler
  **and** the session materializer are present, since the swap reingests (the same
  requirement Add Scene guards on).
- **"Zoom to Scene"** → `_on_zoom_to_scene(index)`. Offered only for **live** scenes
  (`not controller.is_scene_pending(scene)`) — a pending scene has no placeable footprint
  on the common canvas, so there is nothing to frame. See [Zoom to
  Scene](#zoom-to-scene) below.

The menu gates are **per-action**, not on the whole menu: the georeference guard no
longer suppresses the entire menu, so a live scene on a materializer-less pane still gets
"Zoom to Scene". If no action qualifies the menu is simply not shown (`menu.isEmpty()`).

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

## Zoom to Scene

**Files:** `src/wiser/raster/mosaic_controller.py` (the extent helper),
`src/wiser/gui/mosaic_view.py` (the camera move), and `src/wiser/gui/mosaic_pane.py`
(the menu action).

Right-clicking a **live** scene offers **"Zoom to Scene"**, which frames the map camera
on just that scene's footprint — the per-scene analogue of the one-time whole-mosaic
auto-fit. It is a thin glue layer over three pieces that already existed; no new
geometry, reprojection, or camera math is added.

- **`MosaicController.scene_extent_in_common_crs(scene)`** (Qt-free) returns
  `(min_x, min_y, max_x, max_y)` of the scene's footprint in the **resolved target CRS**,
  reusing `_footprint_envelope` / `reprojected_footprint_geometry` (the same reprojection
  the grid builder and overlay use). It returns `None` — so the action is a safe no-op —
  when the scene is pending, has no footprint, or no target CRS is resolved yet. This
  mirrors `build_common_grid`'s union extent, but for a single scene.
- **`MosaicView.zoom_to_extent(extent, margin=0.05)`** pads the extent by a small margin
  (so the footprint isn't flush against the viewport edges), calls the camera's existing
  `MosaicViewTransform.fit_to_extent`, and repaints. It is a no-op until the widget has a
  real size. It sets `_has_fitted = True` so the initial-fit branch in `paintEvent` can
  never later snap the camera back to the whole-grid extent — a deliberate zoom is never
  yanked away. A plain `self.update()` is enough: the moved camera makes `paintEvent`
  schedule the debounced, off-thread tile read for the newly-visible cells (the same path
  a pan/zoom gesture takes), so no `invalidate_pixels` full-cache clear is needed.
- **`MosaicPane._on_zoom_to_scene(index)`** resolves the scene, asks the controller for
  its extent, and (if not `None`) calls `zoom_to_extent`. The menu only shows the action
  for live scenes, but the handler re-checks for `None` in case the scene went pending
  between the menu opening and the click.

The margin is symmetric, so the camera **center** lands exactly on the footprint's
midpoint — which is what the GUI test asserts.

### Reframing after a target-CRS change

The camera holds raw world coordinates (`center_x` / `center_y`) that are **not**
transformed when the target CRS changes, so after a manual **"Choose Target CRS…"**
switch the parked camera often points at a region the mosaic no longer occupies (e.g.
switching UTM → geographic moves every footprint from eastings ~4×10⁵ to lon/lat ~−117/34)
— the preview goes blank even though the scenes are fine.

`MosaicView.ensure_scenes_in_view()`, called from `_on_choose_target_crs` right after the
grid rebuild + view invalidation, fixes this **conservatively**: it reprojects the live
footprints into the new CRS (`visible_scene_footprints_in_common_crs()`), tests each
against the current viewport (`_visible_world_extent` + `_envelope_intersects`), and
reframes to the union extent (`get_common_grid().extent`, via `zoom_to_extent`) **only
when none is visible**. If any scene still intersects the viewport the user's pan/zoom is
left untouched — the reframe rescues the off-screen case without yanking the camera on a
CRS change that happened to keep the mosaic in view. It reuses the same `zoom_to_extent`
(hence the same margin) as [Zoom to Scene](#zoom-to-scene).

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
band_metadata_snapshot, out_path, progress)` is now three lazy stages per scene, all
lazy until the final streamed write:

0. **Materialize the full-band cube lazily** (#677, `_warp_scene_to_grid_vrt` is fed
   from `materialize_full_band_from_snapshot`). The preview reads a display-only
   artifact, so export is the only moment all bands are actually needed. Each scene's
   full-band GeoTIFF is materialized here into the export temp dir from the **live**
   `scene.dataset` pixels but stamped with the scene's **frozen** `snapshot` metadata
   (geotransform, SRS, nodata, band metadata) — so a mid-session metadata edit never
   silently alters the product and the export stays aligned with the ingest-frozen
   footprint / common grid. This full-band artifact is **not** cached — it is discarded
   after the streamed write, keeping the session's temp footprint to the small
   display-only files.

1. **Warp each full-band scene onto the common grid as a warped VRT**
   (`_warp_scene_to_grid_vrt`). `gdal raster mosaic` does **not** reproject, so this
   per-scene `gdal.Warp(..., format="VRT")` is what applies the CRS transform + resample +
   grid alignment — mirroring the preview warp in `render_scene_argb`, but keeping the raw
   band values (no percentile stretch, no `dstAlpha`). The scene's **frozen** Data Ignore
   Value (from the snapshot) becomes the warp `srcNodata` and `output_nodata` becomes the
   `dstNodata`, so invalid / outside-footprint pixels are marked for the compositor. VRTs
   are lazy — no pixels are materialized here. A scene whose window is disjoint from the
   grid warps to `None` and is simply skipped.

2. **Composite + stream to ENVI** (`_run_raster_mosaic`) via
   `gdal.Run("raster", "mosaic", input=[...], output=..., output_format="ENVI", ...)`
   (GDAL ≥ 3.11, guaranteed by #631; the doc's `BuildVRT`/`Translate` fallback is not
   needed). Inputs are passed **bottom→top** (the top scene last) so the algorithm's
   "last valid source wins" composites z-order correctly, and it writes ENVI
   block-by-block, so no full-image buffer is held in RAM. GDAL's own progress callback is
   forwarded through the `ProgressReporter`.

Both the lazily-materialized full-band tiffs and the warped VRTs live in a
`TemporaryDirectory` scoped around the mosaic call — they are only referenced until
`gdal.Run` finishes the streamed write, then discarded.

### Band-metadata stamping and the ENVI-header patch

`gdal raster mosaic` writes correct pixels, geotransform, and `map info`, but no spectral
metadata, so `_patch_envi_band_metadata` reopens the GDAL-written `.hdr` and rewrites it
with WISER's own ENVI header helpers (`envi.load_envi_header` / `write_envi_header`),
stamping the canonical band metadata from the **frozen snapshot** of the scene chosen by
`controller.get_band_metadata_source()` (its `band_metadata_snapshot`): wavelengths /
units, band names, the bad-band list (`bbl`), and the output nodata. Reading from the
snapshot rather than the live dataset keeps the export deterministic against concurrent
edits. This keeps the exported file self-describing so it re-opens in WISER with the
right spectral labels.

One subtlety is called out because it caused a crash-on-open regression:

> **GDAL's ENVI writer stamps a *1-based* `default bands` placeholder** (e.g. `{1, 2, 3}`
> for RGB inputs). WISER's ENVI reader reads `default bands` **verbatim** — no
> 1-based→0-based conversion — and `find_display_bands` does **no** bounds check, so
> leaving that placeholder (or copying an out-of-range source value) makes the exported
> file raise `GDALDataset::GetRasterBand(4) - Illegal band #` when opened in a raster
> view. `_resolve_default_bands` therefore always resolves the field explicitly: it keeps
> the snapshot's frozen resolved `display_bands` (the canonical display choice) only when
> every index is in range for the output, and otherwise **clears** it so WISER derives
> display bands from the stamped wavelengths (truecolor) or the first bands.

The result is **not** loaded back into the running session — export writes the file and
the user opens it manually via **File → Open** (a deliberate product decision), which is
why the on-disk header must be fully self-describing.

### GUI wiring and threading

`MosaicPane._on_export_clicked` (GUI thread) requires ≥1 **live** scene
(`live_scenes()` — pending scenes can't be placed and are excluded), and if any pending
scenes exist it **warns and asks for confirmation** before continuing (they are then
silently skipped — see [Pending Scenes](#pending-scenes-disabled-scenes)). Any visible scene whose live dataset
spectral metadata differs from its frozen snapshot — a warn-and-proceed prompt, since
the export will use the frozen values; see [Display-Only Preview and Frozen
Metadata](#display-only-preview-and-frozen-metadata). It resolves the
grid via `_ensure_common_grid()` (surfacing `TargetCrsRequired`/`UnmappableCrsError` as a
warning), asks for an output path with `QFileDialog.getSaveFileName`, resolves the output
nodata (`_resolve_output_nodata`: the band-metadata scene's **frozen** Data Ignore Value,
else the top-most visible scene that has one), snapshots the controller state, and hands
the heavy work to a scheduler thread via the same `run_with_progress` bridge the ingestion
path uses (progress modal + Activity Monitor row + cancellation, blocking only the mosaic
dialog). The module-level worker `_export_mosaic_task` just calls `export_mosaic` and
returns the written path; `_on_export_finished` shows a confirmation dialog. Like
`_ingest_scene`, the worker never logs — progress and raised exceptions are its only
channels out.

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
  Pending scenes: a non-georeferenced scene is `NO_CRS`-pending and excluded from the
  grid, an all-pending mosaic yields an empty grid, and an ingested-but-incompatible
  scene (built incrementally so the first scene auto-locks the target) is
  `INCOMPATIBLE_CRS`-pending — asserting `is_scene_pending`, `scene_pending_reason`,
  `has_live_scenes`, `has_pending_scenes`, and `_live_scenes`. Zoom to Scene:
  `scene_extent_in_common_crs` returns a live scene's footprint envelope (matching
  `_footprint_envelope`, ordered `min_x/min_y/max_x/max_y`) and `None` for a pending
  scene.
- `src/tests/test_mosaic_view_transform.py` — `MosaicViewTransform` camera math:
  world↔screen round-trip, y-flip orientation, zoom-anchor invariance, aspect-ratio
  preservation (pure `QTransform` math, no widget shown).
- `src/tests/test_mosaic_compositor.py` — the Qt-free `render_scene_argb`: alpha is 0
  exactly on the nodata collar and 255 on the valid interior, RGBA shape/dtype, valid
  pixels get color, a disjoint viewport comes back fully transparent (no Qt). Issue
  #675: with explicit `stretch_bounds` set on the scene, a gradient fixture renders
  identically whether read at the full extent or a further-zoomed-in crop; a paired
  test with `stretch_bounds=None` proves the same fixture *does* drift under the old
  viewport-percentile fallback, confirming the regression coverage is meaningful.
- `src/tests/test_mosaic_materialize.py` — the materialization adapter: full-band
  round-trip metadata + tiled layout, window reads, `RasterDataSet` overriding its GDAL
  backing, dedup, temp-dir lifecycle; plus (#677) `build_display_source` bakes only the
  chosen bands in order with nodata preserved, is cached by `(scene, bands)`, rejects
  non-1/3 band counts, and `materialize_full_band_from_snapshot` takes live pixels but
  **frozen** nodata/metadata.
- `src/tests/test_mosaic_view_gui.py` — the overlay **and** pixel-layer wiring end to
  end, driven through the real `MosaicPane` ingestion path: geometry cache populates /
  re-invalidates; `composite()` stacks known ARGB layers (top opaque wins, holes reveal
  below); the tile cache populates; z-order reorder / visibility toggle trigger **no GDAL
  reads** (spy on `render_scene_argb`); a pan burst **coalesces into a single debounced
  background read**; and, for #674, a pan **reuses cached tiles and warps only the
  newly-exposed cells**, a zoom bucket is **read once then reused** on return, a deep
  zoom **past native reads nothing** (the render bucket is floored at the grid
  resolution), and the cache is **LRU-bounded** by `_TILE_CACHE_MAX`.
- `src/tests/test_mosaic_control_panel_gui.py` — the control panel (#638) end to end
  through the real ingestion path: drag-reorder updates controller z-order and triggers
  **no GDAL reads**; a resolution-mode change rebuilds the grid; a band-metadata
  selection sets the canonical source without changing band count; and a
  non-Nearest-Neighbor resampling choice surfaces the warning and invalidates the pixel
  cache.
- `src/tests/test_mosaic_export.py` — the Qt-free `export_mosaic` (#639): two overlapping
  fixtures give the correct z-order winner per pixel (and the reversed order flips it),
  nodata passthrough (a hole in the top scene reveals the lower one; outside every
  footprint is the output nodata), Nearest-Neighbor value preservation (exact source
  values), band count + canonical metadata round-tripped through the written ENVI, and the
  `default bands` guards (GDAL's 1-based RGB placeholder is cleared; an out-of-range source
  default is dropped so the file opens without an `Illegal band #` crash); plus (#677)
  export stamps the **frozen** default bands even after the live dataset is edited.
- `src/tests/test_mosaic_export_gui.py` — the Export button end to end via `WiserTestModel`:
  ingest two scenes, patch the save dialog, click Export, and assert the ENVI `.img`/`.hdr`
  are written on the common grid with nodata carried through while **nothing is loaded back
  into WISER** (dataset count unchanged); the no-scenes guard (informs and never reaches the
  file picker); and (#677) the metadata-drift warning — a post-ingest edit surfaces the
  warn-and-proceed prompt, where declining aborts before the file picker and accepting
  proceeds past it.
- `src/tests/test_mosaic_ingestion.py` — `validate_scene`, `build_overviews`,
  `compute_footprint_wkt`, and progress-reporting behavior, plus the pending-scene add
  split: `is_dataset_georeferenced` (real geotransform + SRS vs. identity/no-SRS) and
  `validate_scene_addable` (duplicate + band-count only; does **not** reject
  ungeoreferenced input). `compute_stretch_bounds`
  (#675): nodata is excluded from the sampled bounds, a value gradient produces a
  meaningfully wide (not collapsed) range, an all-nodata band is omitted from the
  result rather than raising, and the no-overviews fallback (reads the full band).
- `src/tests/test_mosaic_crs_dialog.py` — `ReprojectPromptDialog` in isolation
  (offscreen Qt), constructed directly from plain lists.
- `src/tests/test_mosaic_crs_gui.py` — end-to-end through the real
  `MosaicPane._on_scene_ingested` path via the `WiserTestModel` harness; documents
  the auto-lock behavior described above with `mock.patch` on
  `wiser.gui.mosaic_pane.ReprojectPromptDialog` so nothing blocks the test. Also
  `test_manual_choose_target_crs_incompatible_marks_pending`: choosing an incompatible
  target CRS **commits** the choice, leaves no live scenes, marks the scenes pending, and
  warns (rather than rejecting the change). Reframing: switching UTM → geographic pushes
  the mosaic off the parked camera, and `_on_choose_target_crs` reframes to the new union
  extent (`test_choose_target_crs_reframes_when_mosaic_off_screen`); when a scene is still
  in view the camera is left untouched (`test_ensure_scenes_in_view_is_noop_when_a_scene_is_visible`).
- `src/tests/test_mosaic_dialog_gui.py` — dialog shell, Add Scene ingestion flow,
  progress modal; asserts the ingested `MosaicScene.stretch_bounds` is populated
  (#675).
- `src/tests/test_mosaic_georeference_gui.py` — re-georeferencing a scene in place
  (#685) end to end via `WiserTestModel`: the context menu opens a `GeoReferencerDialog`
  locked onto the scene (target + save path); a simulated `warp_completed` reingests and
  swaps the corrected scene in at its original z-order slot (blocking the georeferencer
  dialog, not the mosaic window); a repeated warp **replaces** rather than stacks; the
  warp output is **not** registered as a global dataset; Cancel restores the original at
  its slot while "Save to Mosaic" keeps the warped one. `test_add_non_georeferenced_scene_is_pending`
  loads a truly ungeoreferenced TIFF, adds it, and asserts it lands as a `NO_CRS`-pending
  placeholder with the warning icon/tooltip and excluded from the grid — and that its
  context menu offers "Georeference…" but **not** "Zoom to Scene". Zoom to Scene: a live
  scene's menu offers **both** actions and each routes to the right handler/index;
  `_on_zoom_to_scene` moves the camera so its center lands on the scene footprint's
  midpoint.

GUI tests use `@pytest.mark.functional`/`@pytest.mark.smoke` with the
`WiserTestModel` harness (`src/test_utils/test_model.py`), not pytest-qt/qtbot. Run
with `QT_QPA_PLATFORM=offscreen` and `PYTHONPATH=src`.
