# Plan: Seamless Mosaic Scene Ingestion (issue #634)

## Context

When a user adds a scene to the mosaic, the raw `RasterDataSet` object must pass
through three gates before it can be displayed or composited:
1. **Validation** — reject only scenes that are *ungeoreferenced* (no CRS) or have a
   band-count mismatch with existing scenes. Missing nodata and dtype mismatches are
   **not** rejected (see the design decisions in Commit 1) — they degrade gracefully
   or are resolved later by the compositor.
2. **Overview building** — pre-build pyramid overviews *inside* the materialized temp
   file (internal overviews, no sidecar) so preview rendering (#637) is fast without a
   first-paint stutter.
3. **Footprint computation** — derive the valid-pixel outline via `gdal.Footprint` so
   geometry rendering (#636) has the actual shape. With no nodata value this
   gracefully falls back to the full raster rectangle.

This issue depends on #632 (materialization adapter, **already done** in
`src/wiser/raster/mosaic_materialize.py`) and #633 (scaffolding, **already done**:
`MosaicController`, `MosaicScene`, `MosaicPane`, `SeamlessMosaicDialog`).

---

## Existing code to reuse (do not reinvent)

| What | Where |
|---|---|
| `SceneMaterializer.gdal_source(dataset)` — always materializes to a writable temp GeoTIFF | `src/wiser/raster/mosaic_materialize.py` |
| `MosaicScene` dataclass (currently: `dataset`, `visible`) | `src/wiser/raster/mosaic_controller.py` |
| `MosaicController.add_scene(dataset)` scaffold stub | same |
| `LoadingOverlay(target_widget)` — `.start()` / `.stop()` | `src/wiser/gui/loading_overlay.py` |
| `app_state.get_datasets()` → `List[RasterDataSet]` | `src/wiser/gui/app_state.py` |
| `app_services.scheduler.submit_thread(PriorityClass, fn, ...)` → `Future` | `src/wiser/utils/work_scheduler.py` |
| `gdal.Footprint(None, ds, format="WKT")` — WKT polygon from valid pixels | stdlib GDAL ≥ 3.x |
| `gdal.Dataset.BuildOverviews("NEAREST", levels)` — pyramid overview build | stdlib GDAL |

---

## Files

**New:**
- `src/wiser/raster/mosaic_ingestion.py` — non-GUI pipeline: validation, overview building, footprint
- `src/tests/test_mosaic_ingestion.py` — unit/integration tests (no Qt)

**Edited:**
- `src/wiser/raster/mosaic_controller.py` — extend `MosaicScene`; update `add_scene` signature
- `src/wiser/gui/mosaic_pane.py` — placeholder "Add Dataset…" button + ingestion orchestration
- `src/wiser/gui/mosaic_dialog.py` — own `SceneMaterializer`; close/cleanup lifecycle
- `src/tests/test_mosaic_dialog_gui.py` — extend smoke test to exercise Add-a-scene GUI path
- `src/tests/test_mosaic_controller.py` — update `add_scene` call sites for new signature

---

## Commit 1 — Ingestion pipeline + extended `MosaicScene` + tests

### `MosaicScene` extension in `mosaic_controller.py`

Add three optional fields to the dataclass (all default to `None`/`False`):

```python
@dataclass
class MosaicScene:
    dataset: "RasterDataSet"
    visible: bool = True
    gdal_path: Optional[str] = None        # path to warpable materialized GeoTIFF
    footprint_wkt: Optional[str] = None    # valid-pixel polygon (WKT, source CRS)
    has_overviews: bool = False
```

**On a wrapper class for artifact paths (asked during review):** not needed in v1.
A scene has exactly **one** on-disk artifact — `gdal_path` (the materialized `.tif`).
Overviews live *inside* that file (internal, see Commit 1), and `footprint_wkt` is an
in-memory string, not a file. So the fields are flat, not "many sidecar paths." Keep
`MosaicScene` flat. If a future issue adds an actual second artifact file (a persisted
footprint, or an external `.ovr`), that is the moment to introduce a small
`SceneArtifacts` grouping — leave a `# TODO` seam noting this, but do not build the
wrapper speculatively.

Change `MosaicController.add_scene` to accept a pre-built `MosaicScene` instead of
a bare dataset (callers now construct the dataclass before calling):

```python
def add_scene(self, scene: MosaicScene) -> MosaicScene:
    self._scenes.append(scene)
    return scene
```

Update existing unit tests in `test_mosaic_controller.py` to pass
`MosaicScene(dataset=fake_ds)` at the call sites.

### New `src/wiser/raster/mosaic_ingestion.py`

Three public functions — no Qt import, GDAL + stdlib only:

**`SceneValidationError(ValueError)`** — carries a `.reason` string for UI display.

**`validate_scene(dataset: RasterDataSet, existing_scenes: List[MosaicScene]) -> None`**

Raises `SceneValidationError` with a specific message for each rejected case:
- **No CRS** (the georeferencing gate): `dataset.get_wkt_spatial_reference()` is
  empty/None. This is the *only* reliable georef check — do **not** test
  `get_geo_transform() is None`. Per `dataset.py`, `get_geo_transform()` never
  returns `None`; it returns an identity transform when the source is
  ungeoreferenced, so that check is dead code. (An optional stricter gate could
  additionally reject an identity/degenerate geotransform, but a missing SRS is
  sufficient for v1.)
- Band count mismatch vs first existing scene: `dataset.num_bands() != existing_scenes[0].dataset.num_bands()`.
  Cross-reference this error message with `# TODO(#640)` for the future
  warn-but-allow path.

**Deliberately NOT rejected** (see design decisions below):
- **No nodata value.** `get_data_ignore_value()` legitimately returns `None` for
  many valid datasets. nodata is used only to carve the footprint; `gdal.Footprint`
  with no nodata just returns the full raster rectangle — a valid, coarser
  footprint. Missing nodata degrades gracefully; it is not an error.
- **Data type mismatch across scenes.** No dtype consistency is required at
  ingestion. Each materialized scene keeps its source dtype; the compositor
  (#635/#637) unifies dtype at warp time by promoting to the widest common type
  (`np.promote_types`, e.g. float32 + float64 → float64). Enforcing equal dtypes
  here would only produce false rejections.

**`build_overviews(gdal_path: str) -> None`**

Builds pyramid overviews (levels `[2, 4, 8, 16]`) on the warpable GDAL source.

**How overviews are stored (design note):** opening the materialized GeoTIFF with
`gdal.GA_Update` and calling `BuildOverviews` writes **internal** overviews
*embedded inside the same `.tif`* — no sidecar file is produced. (External
`.tif.ovr` sidecars only occur when the file is opened read-only or forced via
config; not our case, since `SceneMaterializer` always yields a writable temp
file.) The overviews are **not** a separate array or `gdal.Dataset` we manage:
they are read back via `band.GetOverview(i)`, and GDAL owns all reduced-resolution
geotransform/pixel-size math internally. The scene therefore only records
`has_overviews: bool` — there is no per-level metadata to track and no pixel-size
to recompute ourselves.

All sources in v1 are materialized temp GeoTIFFs (writable), so open with
`gdal.GA_Update` and build internal overviews. 

```python
def build_overviews(gdal_path: str) -> None:
    gdal.UseExceptions()
    ds = gdal.Open(gdal_path, gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"Cannot open {gdal_path} for overview building")
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
    ds = None  # flush/close
```

**`compute_footprint_wkt(gdal_path: str) -> str`**

Returns a WKT polygon of the valid-pixel footprint in the dataset's own CRS:

```python
def compute_footprint_wkt(gdal_path: str) -> str:
    wkt = gdal.Footprint(None, gdal_path, format="WKT")
    if not wkt:
        raise RuntimeError(f"gdal.Footprint returned empty result for {gdal_path}")
    return wkt
```

### Tests in `src/tests/test_mosaic_ingestion.py`

`pytestmark = [pytest.mark.unit, pytest.mark.smoke]`

Helper: `_make_georeffed_tiff(tmp_path, nodata=None, bands=3, dtype=np.float32)` —
writes a small tiled GeoTIFF using `gdal.GetDriverByName("GTiff").Create(...)` with
a real geotransform + WKT SRS; optionally stamps nodata and a nodata-collar border.

Tests (all non-GUI):
- `test_validate_rejects_no_crs` — dataset with empty/None SRS → `SceneValidationError`
- `test_validate_rejects_band_mismatch` — 3-band scene vs 1-band candidate → error
- `test_validate_accepts_no_nodata` — dataset with `data_ignore_value=None` passes
  (no rejection); confirms the nodata gate was removed
- `test_validate_accepts_dtype_mismatch` — float32 first scene, uint16 candidate
  passes (no rejection); confirms dtype consistency is not enforced
- `test_validate_accepts_valid_scene` — passes a clean dataset, no exception
- `test_footprint_excludes_nodata_collar` — build a 20×20 GeoTIFF with a 2-px
  nodata border, check the WKT footprint does not encompass the full raster extent
- `test_footprint_full_rect_without_nodata` — GeoTIFF with no nodata value; assert
  `compute_footprint_wkt` still returns a valid polygon (the full raster rectangle)
- `test_overviews_written_to_file` — call `build_overviews` on a
  temp GeoTIFF, open it back with `gdal.Open`, assert `GetRasterBand(1).GetOverviewCount() > 0`

---

## Commit 2 — GUI wiring + integration test

### `SeamlessMosaicDialog` owns `SceneMaterializer`

Create and own a `SceneMaterializer` (session-scoped, per-dialog-instance) and pass
it into `MosaicPane`:

```python
# mosaic_dialog.py
from wiser.raster.mosaic_materialize import SceneMaterializer

def __init__(self, ...):
    ...
    self._materializer = SceneMaterializer()
    self._mosaic_pane = MosaicPane(..., materializer=self._materializer)

def closeEvent(self, event):
    self._materializer.close()
    super().closeEvent(event)
```

### `MosaicPane` minimal "Add Dataset…" action

Add to the placeholder `_controls` area: a `QComboBox` populated from
`app_state.get_datasets()` + an "Add Scene…" `QPushButton`. Connect
`app_state.dataset_added` to refresh the combo on new loads.

A module-level `_IngestionBridge(QObject)` with two signals handles thread-to-main-thread
marshaling safely (PySide6 cross-thread signal emission is thread-safe):

```python
class _IngestionBridge(QObject):
    succeeded = Signal(object)   # emits the built MosaicScene
    failed    = Signal(str)      # emits error message
```

**Progress UX — Activity Monitor row, not a bare spinner.** Instead of a
`LoadingOverlay`, register the ingestion as a row in the Activity Monitor so it gets
a progress bar, a cancel affordance, and error display — the same surface a
`SemanticTask` would give — without the `TaskStage`/chunking/`DataRef` machinery,
which does not fit a sequential single-file I/O job (see Notes). Use the existing
lightweight API: `activity_monitor.register_task(title, meta, cancel_callback) ->
activity_id`, then `set_task_finished` / `set_task_failed(activity_id, msg)` on
completion. The three steps are indeterminate/short, so drive a coarse
`register_task` row (title `"Adding scene: <name>"`) rather than fine per-step
progress.

`_on_add_scene_clicked()` orchestration:
1. Get the chosen `RasterDataSet` from the combo; bail if nothing selected.
2. Call `validate_scene(dataset, controller.get_scenes())`; on `SceneValidationError`
   show `QMessageBox.warning(self, "Cannot add scene", str(e))` and return.
3. Register the Activity Monitor row:
   ```python
   activity_id = app_services.activity_monitor.register_task(
       title=f"Adding scene: {dataset.get_name()}",
       meta={"bands": str(dataset.num_bands())},
       cancel_callback=None,  # v1: short job, cancellation deferred
   )
   ```
4. Submit background work via the thread pool:
   ```python
   def _bg(dataset, materializer):
       path = materializer.gdal_source(dataset)
       build_overviews(path)
       footprint = compute_footprint_wkt(path)
       return MosaicScene(dataset=dataset, gdal_path=path,
                          footprint_wkt=footprint, has_overviews=True)

   future = app_services.scheduler.submit_thread(
       PriorityClass.BACKGROUND, _bg, dataset, materializer
   )
   ```
5. Done callback emits through the bridge (runs on thread pool thread — the bridge
   marshals back to the main thread; the Activity Monitor must only be touched on the
   main thread, so the `set_task_*` calls happen in the slots, not here):
   ```python
   def _done(f, bridge=self._bridge):
       try:
           bridge.succeeded.emit(f.result())
       except Exception as exc:
           bridge.failed.emit(str(exc))
   future.add_done_callback(_done)
   ```
   Carry `activity_id` into the slots (e.g. bundle it into the emitted payload, or
   keep it on `self` keyed by dataset) so they can close the right row.
6. `_on_ingestion_succeeded(scene)`: `controller.add_scene(scene)`,
   `activity_monitor.set_task_finished(activity_id)`, `self._mosaic_view.update()`.
7. `_on_ingestion_failed(msg)`: `activity_monitor.set_task_failed(activity_id, msg)`,
   `QMessageBox.critical(self, "Add scene failed", msg)`.

### Integration test extension in `test_mosaic_dialog_gui.py`

Add `test_add_valid_scene`:
- Synthesize a small georef'd `RasterDataSet` (from a temp GeoTIFF) and load it
  into `WiserTestModel`'s app state via `app_state.add_dataset(...)`.
- Open the mosaic dialog, wait for it to appear.
- Programmatically select the dataset in the combo and click "Add Scene…"
  (`QTest.mouseClick`); poll with `QTest.qWait` until `controller.scene_count() == 1`.
- Assert `has_overviews=True` and `footprint_wkt is not None`.

Add `test_add_invalid_scene_rejected`:
- Load an **ungeoreferenced** dataset (empty/None SRS) — the only hard-rejected case now.
- Click Add; assert `controller.scene_count()` is still 0 (no scene added on rejection).
- (Note: a no-nodata dataset must *not* be used here — it is now accepted.)

---

## Verification

```bash
# Non-GUI unit/integration tests (no Qt required)
cd src/tests && python -m pytest -s test_mosaic_ingestion.py

# GUI integration tests
cd src/tests && python -m pytest -s test_mosaic_dialog_gui.py

# All mosaic tests by marker
cd src/tests && python -m pytest -s -m smoke -k mosaic
```

Manual: launch WISER, load a georeferenced scene, open **Tools → Seamless Mosaic…**,
pick the dataset from the "Add Scene" combo, click the button, watch the Activity
Monitor row appear and finish, and confirm `scene_count == 1`. The overviews are
internal to the materialized `.tif` (verify with `gdalinfo` showing "Overviews:" on
the temp file — there is no separate `.ovr` sidecar).

---

## Notes / non-goals

- No CRS reprojection of footprints — that is #635/#636.
- The full "Add from file" picker is deferred to #638. Here the combo reads from
  datasets already loaded in `app_state`.
- Overviews are **internal** (embedded in the materialized `.tif`); no external
  `.ovr` sidecar is written or tracked. There is exactly one artifact file per scene
  (`gdal_path`), so no `SceneArtifacts` wrapper is introduced in v1.
- **No dtype consistency enforced.** Scenes may mix dtypes; the compositor
  (#635/#637) promotes to a common widest type (`np.promote_types`) at warp time.
- **No nodata requirement.** Missing nodata is accepted; the footprint falls back to
  the full raster rectangle.
- **Not using the full `SemanticTask`/`TaskStage` pipeline.** That framework is built
  for chunked dataset compute (tile/band chunking, allocated output `DataRef`s,
  per-region `WriteSpec`s) — a poor fit for a sequential three-step single-file I/O
  job whose outputs are a WKT string and in-file side effects. We instead register a
  lightweight Activity Monitor row directly (`register_task` / `set_task_finished` /
  `set_task_failed`), which delivers the same user-visible progress/cancel/error
  surface without the pipeline ceremony. Revisit only if ingestion later needs
  chunked, resumable, storage-backed stages.
