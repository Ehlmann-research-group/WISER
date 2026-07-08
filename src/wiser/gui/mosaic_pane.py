"""
Control panel for the Seamless Mosaic feature (EPIC #629).

Hosts the :class:`MosaicView` alongside a controls area and owns the non-GUI
:class:`MosaicController` that both share. The controls area offers: an "Add Scene"
action (a dataset picker plus a button that ingests the chosen dataset -- materialize
-> build overviews -> compute footprint -- on a background thread and appends it to the
controller); the scene stack with **drag-to-reorder** z-order and per-scene visibility
(#638); resolution-mode, target-CRS, resampling-method, and canonical band-metadata
controls (#638); and a disabled Export button (the export path lands in #639).

Each control mutates the shared controller and then invalidates the view following the
compositor's tiered contract -- a pure restack (``recomposite_only``) for
z-order/visibility, a re-read (``invalidate_pixels``) when the pixels change, and a grid
rebuild when the output geometry changes.
"""

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

from osgeo import gdal, osr

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .app_services import AppServices
from .mosaic_crs_dialog import ReprojectPromptDialog
from .mosaic_view import MosaicView
from .progress_task import run_with_progress

from wiser.raster.dataset import find_display_bands
from wiser.raster.mosaic_controller import (
    CommonGrid,
    MosaicController,
    MosaicScene,
    ResolutionMode,
    SceneMetadataSnapshot,
    TargetCrsRequired,
    UnmappableCrsError,
)
from wiser.raster.mosaic_export import export_mosaic
from wiser.raster.mosaic_ingestion import (
    SceneValidationError,
    build_overviews,
    compute_footprint_wkt,
    validate_scene,
)
from wiser.utils.progress import ProgressReporter

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.mosaic_materialize import SceneMaterializer


def _ingest_scene(
    dataset: "RasterDataSet",
    materializer: "SceneMaterializer",
    snapshot: Optional[SceneMetadataSnapshot] = None,
    progress: Optional[ProgressReporter] = None,
) -> MosaicScene:
    """
    Background I/O for one scene: materialize to a warpable temp GeoTIFF, build
    internal overviews on it, and compute the valid-pixel footprint.

    Runs on a scheduler thread (no Qt here). ``progress`` is split across the three
    phases (weighted by their rough cost) so the overall bar advances smoothly;
    each phase reports fine-grained progress internally. Returns the fully-populated
    :class:`MosaicScene` for the main thread to append to the controller.

    ``snapshot`` is the dataset metadata **frozen at ingest** (#677), built on the GUI
    thread in :meth:`MosaicPane._on_add_scene_clicked` (the display-band resolution and
    the deep-copy of the dataset's metadata must both happen against the live main-view
    / dataset state at add-time). It is carried through and stamped onto the returned
    :class:`MosaicScene`; later steps materialize the display-only artifact and export
    from this snapshot rather than the live dataset.
    """
    progress = progress or ProgressReporter()
    materialize_progress, overview_progress, footprint_progress = progress.split(
        (0.5, "Materializing scene"),
        (0.35, "Building overviews"),
        (0.15, "Computing footprint"),
    )
    gdal_path = materializer.gdal_source(dataset, progress=materialize_progress)
    progress.raise_if_cancelled()
    build_overviews(gdal_path, progress=overview_progress)
    progress.raise_if_cancelled()
    footprint_wkt = compute_footprint_wkt(gdal_path, progress=footprint_progress)
    progress.report_fraction(1.0, "Done")
    return MosaicScene(
        dataset=dataset,
        gdal_path=gdal_path,
        footprint_wkt=footprint_wkt,
        has_overviews=True,
        snapshot=snapshot,
    )


def _export_mosaic_task(
    scenes: Sequence[MosaicScene],
    grid: CommonGrid,
    target_wkt: str,
    resample_alg: int,
    output_nodata: Optional[float],
    band_metadata_dataset: Optional["RasterDataSet"],
    out_path: str,
    progress: Optional[ProgressReporter] = None,
) -> str:
    """
    Background full-resolution export: composite the visible scenes onto the common
    grid and stream the mosaic to ``out_path`` as ENVI.

    Runs on a scheduler thread (no Qt here). Delegates to the Qt-free
    :func:`wiser.raster.mosaic_export.export_mosaic` and returns the written path as
    a string for the main thread's success callback. The result is intentionally
    *not* loaded back into WISER — the user opens the file manually.
    """
    result = export_mosaic(
        scenes,
        grid,
        target_wkt,
        resample_alg,
        output_nodata,
        band_metadata_dataset,
        Path(out_path),
        progress=progress,
    )
    return str(result)


class MosaicPane(QWidget):
    """
    Hosts a :class:`MosaicView` plus a controls area with the Add-Scene action.

    Creates the shared :class:`MosaicController` and wires it into the view so the
    model is the single source of truth for both rendering (#636/#637) and the
    control panel (#638).
    """

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: Optional[AppServices] = None,
        materializer: Optional["SceneMaterializer"] = None,
        display_bands_resolver: Optional[Callable[["RasterDataSet"], Optional[Tuple[int, ...]]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._materializer = materializer
        # Resolves a dataset to the bands currently shown for it in the main view, or
        # None if not shown there. Used at ingest to pick the display bands baked into
        # a scene's display-only preview (#677); falls back to find_display_bands when
        # this is None (e.g. a bare pane in a unit test) or returns None.
        self._display_bands_resolver = display_bands_resolver
        self._controller = MosaicController()
        # Holds the in-flight ingestion task (progress modal + background work) so it
        # is not garbage-collected mid-run; overwritten on the next add.
        self._active_progress_task = None

        self._init_ui()

    def _init_ui(self) -> None:
        # View on the left, controls on the right, split so the panel can grow later.
        self._splitter = QSplitter(Qt.Horizontal, self)

        self._mosaic_view = MosaicView(
            parent=self._splitter,
            app_state=self._app_state,
            controller=self._controller,
            mosaicpane=self,
            app_services=self._app_services,
        )
        self._splitter.addWidget(self._mosaic_view)

        # Controls area: a slim side panel with stacked sections — add a scene, the
        # scene stack (drag-reorder z-order + visibility + remove), resolution mode,
        # target CRS, resampling method, band-metadata source, and a (disabled) export.
        # Built parentless; the scroll area below takes ownership via setWidget().
        self._controls = QWidget()
        self._controls_layout = QVBoxLayout(self._controls)
        self._controls_layout.setContentsMargins(8, 8, 8, 8)
        self._controls_layout.setSpacing(10)

        self._controls_layout.addWidget(self._build_add_scene_group())
        self._controls_layout.addWidget(self._build_scene_list_group(), 1)
        self._controls_layout.addWidget(self._build_resolution_group())
        self._controls_layout.addWidget(self._build_target_crs_group())
        self._controls_layout.addWidget(self._build_resampling_group())
        self._controls_layout.addWidget(self._build_band_metadata_group())

        # Preview toggle deferred for v1 (#638): intentionally not added yet.
        # self._controls_layout.addWidget(self._build_preview_toggle())

        # Export / Finish streams the full-resolution mosaic to an ENVI file (#639).
        self._export_button = QPushButton(self.tr("Export / Finish…"), self._controls)
        self._export_button.setToolTip(
            self.tr("Composite the visible scenes at full resolution and write an ENVI file.")
        )
        self._export_button.clicked.connect(self._on_export_clicked)
        self._controls_layout.addWidget(self._export_button)

        # The controls stack can grow taller than a short window, so host it in a
        # vertically-scrolling area; the fixed minimum width keeps groups from cramping.
        self._controls_scroll = QScrollArea(self._splitter)
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setWidget(self._controls)
        self._controls_scroll.setMinimumWidth(280)
        self._splitter.addWidget(self._controls_scroll)

        # Give the view the bulk of the width; controls stay a slim side panel.
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        # Populate now and keep in sync as datasets are loaded/removed.
        self._refresh_dataset_combo()
        self._app_state.dataset_added.connect(self._on_datasets_changed)
        self._app_state.dataset_removed.connect(self._on_datasets_changed)

        self._refresh_scene_list()
        self._refresh_target_crs_label()

    # -- controls construction ------------------------------------------------

    def _build_add_scene_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Add scene"), self._controls)
        layout = QVBoxLayout(group)

        self._dataset_combo = QComboBox(group)
        self._add_scene_button = QPushButton(self.tr("Add Scene…"), group)
        self._add_scene_button.clicked.connect(self._on_add_scene_clicked)
        # Ingestion needs both a scheduler (to run on) and a materializer; without
        # them the pane is display-only, so disable the action.
        if self._app_services is None or self._materializer is None:
            self._add_scene_button.setEnabled(False)
            self._add_scene_button.setToolTip(self.tr("Scene ingestion is unavailable in this context."))

        layout.addWidget(self._dataset_combo)
        layout.addWidget(self._add_scene_button)
        return group

    def _build_scene_list_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Scenes"), self._controls)
        layout = QVBoxLayout(group)

        self._scene_list = QListWidget(group)
        self._scene_list.setSelectionMode(QAbstractItemView.SingleSelection)
        # Drag-to-reorder == z-order: dropping a row to a new position restacks the
        # scenes (top of the list = top scene). The move is applied to the controller
        # in _on_scene_rows_moved; the widget's own reorder is then overwritten by the
        # authoritative rebuild in _refresh_scene_list.
        self._scene_list.setDragDropMode(QAbstractItemView.InternalMove)
        self._scene_list.setToolTip(
            self.tr("Drag to reorder (top = top of the stack). Uncheck to hide a scene.")
        )
        self._scene_list.itemSelectionChanged.connect(self._on_scene_selection_changed)
        self._scene_list.itemChanged.connect(self._on_scene_item_changed)
        self._scene_list.model().rowsMoved.connect(self._on_scene_rows_moved)
        layout.addWidget(self._scene_list)

        self._remove_scene_button = QPushButton(self.tr("Remove Selected"), group)
        self._remove_scene_button.setEnabled(False)
        self._remove_scene_button.clicked.connect(self._on_remove_scene_clicked)
        layout.addWidget(self._remove_scene_button)
        return group

    def _build_resolution_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Output Spatial Resolution"), self._controls)
        layout = QVBoxLayout(group)

        self._resolution_combo = QComboBox(group)
        # userData is the ResolutionMode member itself, read back in the handler.
        self._resolution_combo.addItem(self.tr("Top scene"), ResolutionMode.TOP)
        self._resolution_combo.addItem(self.tr("Highest (finest)"), ResolutionMode.HIGHEST)
        self._resolution_combo.addItem(self.tr("Lowest (coarsest)"), ResolutionMode.LOWEST)
        self._resolution_combo.addItem(self.tr("Average"), ResolutionMode.AVERAGE)
        self._resolution_combo.addItem(self.tr("Custom…"), ResolutionMode.CUSTOM)
        current = self._controller.get_resolution_mode()
        restored = self._resolution_combo.findData(current)
        if restored >= 0:
            self._resolution_combo.setCurrentIndex(restored)
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_mode_changed)
        layout.addWidget(self._resolution_combo)

        # Custom pixel-size inputs (in target-CRS units), shown only in Custom mode.
        self._custom_res_widget = QWidget(group)
        custom_layout = QFormLayout(self._custom_res_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        self._custom_xres_spin = self._make_resolution_spinbox()
        self._custom_yres_spin = self._make_resolution_spinbox()
        custom_layout.addRow(self.tr("X size:"), self._custom_xres_spin)
        custom_layout.addRow(self.tr("Y size:"), self._custom_yres_spin)
        self._custom_res_widget.setVisible(current is ResolutionMode.CUSTOM)
        layout.addWidget(self._custom_res_widget)
        return group

    def _make_resolution_spinbox(self) -> QDoubleSpinBox:
        """A positive-only pixel-size spinbox (target-CRS units, so a wide range)."""
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        # Minimum is a tiny positive so value() is always > 0 (set_custom_resolution
        # rejects non-positive sizes); the range spans degrees to metres.
        spin.setRange(1e-6, 1e9)
        spin.setValue(1.0)
        spin.valueChanged.connect(self._on_custom_resolution_changed)
        return spin

    def _build_target_crs_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Target CRS"), self._controls)
        layout = QVBoxLayout(group)

        self._target_crs_label = QLabel(group)
        self._target_crs_label.setWordWrap(True)
        layout.addWidget(self._target_crs_label)

        self._choose_crs_button = QPushButton(self.tr("Choose Target CRS…"), group)
        self._choose_crs_button.clicked.connect(self._on_choose_target_crs)
        layout.addWidget(self._choose_crs_button)
        return group

    def _build_resampling_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Resampling"), self._controls)
        layout = QVBoxLayout(group)

        self._resample_combo = QComboBox(group)
        # userData is the GDAL GRA_* constant passed straight to gdal.Warp.
        self._resample_combo.addItem(self.tr("Nearest Neighbor"), gdal.GRA_NearestNeighbour)
        self._resample_combo.addItem(self.tr("Bilinear"), gdal.GRA_Bilinear)
        self._resample_combo.addItem(self.tr("Cubic Convolution"), gdal.GRA_Cubic)
        restored = self._resample_combo.findData(self._controller.get_resample_alg())
        if restored >= 0:
            self._resample_combo.setCurrentIndex(restored)
        # Connect after seeding so the initial selection doesn't fire the warning.
        self._resample_combo.currentIndexChanged.connect(self._on_resample_changed)
        layout.addWidget(self._resample_combo)
        return group

    def _build_band_metadata_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Band metadata"), self._controls)
        layout = QVBoxLayout(group)

        blurb = QLabel(
            self.tr("Which scene's band metadata (wavelengths, names) labels the output."),
            group,
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        self._band_metadata_combo = QComboBox(group)
        self._band_metadata_combo.setToolTip(
            self.tr(
                "Metadata/labeling only — does not change the output band count or which "
                "bands are included."
            )
        )
        # Populated by _refresh_band_metadata_combo (called from _refresh_scene_list);
        # userData is the MosaicScene, or None for the "top scene" default.
        self._band_metadata_combo.currentIndexChanged.connect(self._on_band_metadata_changed)
        layout.addWidget(self._band_metadata_combo)
        return group

    # -- dataset picker -------------------------------------------------------

    def _on_datasets_changed(self, *_args) -> None:
        self._refresh_dataset_combo()

    def _refresh_dataset_combo(self) -> None:
        """Rebuild the combo from ``app_state``, preserving the current selection."""
        previous_id = self._dataset_combo.currentData()
        self._dataset_combo.clear()
        for dataset in self._app_state.get_datasets():
            ds_id = dataset.get_id()
            label = dataset.get_name() or f"Dataset {ds_id}"
            self._dataset_combo.addItem(label, ds_id)
        if previous_id is not None:
            restored = self._dataset_combo.findData(previous_id)
            if restored >= 0:
                self._dataset_combo.setCurrentIndex(restored)

    # -- add scene ------------------------------------------------------------

    def _on_add_scene_clicked(self) -> None:
        if self._app_services is None or self._materializer is None:
            return
        ds_id = self._dataset_combo.currentData()
        if ds_id is None:
            return
        dataset = self._app_state.get_dataset(ds_id)
        name = dataset.get_name() or f"Dataset {ds_id}"

        # Validate on the main thread so rejection is immediate (no spinner churn)
        # and so a scene that can't join the mosaic never pays for the
        # materialize/build-overviews/footprint pipeline it would just be rejected
        # after anyway.
        try:
            validate_scene(dataset, self._controller.get_scenes())
            self._controller.validate_new_scene_crs(name, dataset.get_spatial_ref())
        except (SceneValidationError, UnmappableCrsError) as exc:
            QMessageBox.warning(self, self.tr("Cannot add scene"), str(exc))
            return

        # Freeze the dataset metadata at ingest (#677) on the GUI thread: the display
        # bands are resolved from the *live* main-view selection (GUI state, unsafe to
        # read off-thread), and the dataset's spatial/spectral metadata is deep-copied
        # so a mid-session edit in main WISER cannot silently alter this mosaic. The
        # snapshot rides along to the background worker and both the display-only
        # materialization and the lazy export stamp from it.
        display_bands = self._resolve_display_bands(dataset)
        snapshot = SceneMetadataSnapshot.from_dataset(dataset, display_bands)

        # Run the ingestion on the scheduler with a progress dialog and a mirrored
        # Activity Monitor row. Pass the window (the SeamlessMosaicDialog) as the block
        # target so only it is disabled while ingesting; the rest of WISER stays live.
        self._active_progress_task = run_with_progress(
            self._app_services,
            self.window(),
            self.tr("Adding scene: {0}").format(name),
            _ingest_scene,
            dataset,
            self._materializer,
            snapshot,
            on_success=self._on_scene_ingested,
            on_error=self._on_scene_failed,
            description=self.tr("Materializing scene…"),
            meta={"bands": str(dataset.num_bands())},
        )

    def _resolve_display_bands(self, dataset: "RasterDataSet") -> Tuple[int, ...]:
        """
        Resolve which display bands to bake into ``dataset``'s display-only preview,
        per issue #677's ordering:

          1. the main view's current selection for the dataset (honoring a
             band-chooser choice), via the injected ``display_bands_resolver``;
          2. otherwise ``find_display_bands(dataset)`` -- the robust defaults ->
             human-eye wavelength -> first 1/3 bands fallback chain -- which also
             covers "never shown in the main view" and "no defaults".

        Runs on the GUI thread (the resolver reads live main-view state).
        """
        if self._display_bands_resolver is not None:
            bands = self._display_bands_resolver(dataset)
            if bands:
                return tuple(int(b) for b in bands)
        return tuple(int(b) for b in find_display_bands(dataset))

    def _on_scene_ingested(self, scene: MosaicScene) -> None:
        self._controller.add_scene(scene)
        if not self._ensure_common_grid():
            # The scene could not be placed on the common grid (unmappable CRS, or
            # the user cancelled the reproject prompt) so remove the addition
            self._controller.remove_scene(self._controller.scene_count() - 1)
            return
        self._refresh_scene_list()
        self._mosaic_view.invalidate_overlay()
        self._mosaic_view.invalidate_pixels()

    def _on_scene_failed(self, message: str) -> None:
        QMessageBox.critical(self, self.tr("Add scene failed"), message)

    # -- scene list -----------------------------------------------------------

    def _refresh_scene_list(self) -> None:
        """Rebuild the scene list from the controller (top-most scene shown first)."""
        previous = self._selected_scene_index()
        scenes = self._controller.get_scenes()
        summary = self._controller.scene_crs_summary()

        # blockSignals so repopulating doesn't fire itemChanged (visibility) per row.
        self._scene_list.blockSignals(True)
        self._scene_list.clear()
        # Controller order is bottom-to-top; show the top-most first so the list reads
        # like a layer stack. Each item carries its real controller index in UserRole.
        for index in reversed(range(len(scenes))):
            name, crs_display = summary[index]
            item = QListWidgetItem(f"{name}   ·   {crs_display}")
            item.setData(Qt.UserRole, index)
            # Checkable + draggable, but not a drop target itself, so a dragged row
            # lands *between* rows (reorder) rather than "onto" another row.
            item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsDropEnabled)
            item.setCheckState(Qt.Checked if scenes[index].visible else Qt.Unchecked)
            self._scene_list.addItem(item)
            if index == previous:
                self._scene_list.setCurrentItem(item)
        self._scene_list.blockSignals(False)
        self._on_scene_selection_changed()
        self._refresh_band_metadata_combo()

    def _refresh_band_metadata_combo(self) -> None:
        """
        Rebuild the band-metadata source combo from the controller, preserving the
        current selection by object identity so it survives add/remove/reorder.
        """
        combo = self._band_metadata_combo
        previous = combo.currentData()  # a MosaicScene, or None for the default
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self.tr("Top scene (default)"), None)
        scenes = self._controller.get_scenes()
        # Top-most first, matching the scene list's visual order.
        for index in reversed(range(len(scenes))):
            scene = scenes[index]
            name = scene.dataset.get_name() or f"Scene {index}"
            try:
                label = self.tr("{0} ({1} bands)").format(name, scene.dataset.num_bands())
            except Exception:  # noqa: BLE001 — metadata access must never break the UI
                label = name
            combo.addItem(label, scene)
        restored = 0
        if previous is not None:
            for i in range(combo.count()):
                if combo.itemData(i) is previous:
                    restored = i
                    break
        combo.setCurrentIndex(restored)
        combo.blockSignals(False)

    def _on_band_metadata_changed(self, *_args) -> None:
        # Metadata/labeling only — no grid or view invalidation, and the ingest
        # band-count gate guarantees the output band count is unaffected.
        self._controller.set_band_metadata_source(self._band_metadata_combo.currentData())

    def _selected_scene_index(self) -> Optional[int]:
        item = self._scene_list.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def _on_scene_selection_changed(self) -> None:
        self._remove_scene_button.setEnabled(self._scene_list.currentItem() is not None)

    def _on_scene_item_changed(self, item: QListWidgetItem) -> None:
        """Per-scene visibility toggle from the list checkbox."""
        index = item.data(Qt.UserRole)
        if index is None:
            return
        self._controller.set_visibility(index, item.checkState() == Qt.Checked)
        self._rebuild_grid_quietly()
        self._mosaic_view.invalidate_overlay()
        # Visibility is a restack at the same viewport — no GDAL reads (recompose only
        # falls back to a read if a revealed scene was never cached at this viewport).
        self._mosaic_view.recomposite_only()

    def _on_scene_rows_moved(self, parent, start, end, destination, row) -> None:
        """
        Drag-to-reorder handler: translate the visual move into a controller z-order
        move and restack.

        The list is shown top-first while the controller stores scenes bottom-to-top,
        so a visual row ``v`` maps to controller index ``n - 1 - v``. Qt's ``row``
        (the destination) is indexed *before* the dragged row is removed, so it is
        shifted down by one when a row moves downward. A reorder is a pure restack (no
        GDAL reads), mirroring the visibility toggle's invalidation path.
        """
        count = self._controller.scene_count()
        if count < 2:
            return
        src_visual = start
        dst_visual = row - 1 if row > src_visual else row
        if dst_visual == src_visual:
            return  # dropped back onto its own position — no move
        from_index = (count - 1) - src_visual
        to_index = (count - 1) - dst_visual
        self._controller.move_scene(from_index, to_index)
        # Reorder can change the top scene, so the TOP resolution mode's grid may shift;
        # rebuild quietly (z-order never changes the CRS constraint, so no prompt).
        self._rebuild_grid_quietly()
        # Re-sync the now-stale Qt.UserRole indices after Qt's own reorder. Defer so the
        # rebuild runs after the drop finishes rather than mid-signal.
        QTimer.singleShot(0, self._refresh_scene_list)
        self._mosaic_view.invalidate_overlay()
        self._mosaic_view.recomposite_only()

    def _on_remove_scene_clicked(self) -> None:
        index = self._selected_scene_index()
        if index is None:
            return
        self._controller.remove_scene(index)
        self._refresh_scene_list()
        # A removal can only relax the CRS constraint, so rebuild silently rather than
        # popping the reproject dialog.
        self._rebuild_grid_quietly()
        self._mosaic_view.invalidate_overlay()
        self._mosaic_view.invalidate_pixels()

    # -- resolution -----------------------------------------------------------

    def _on_resolution_mode_changed(self, *_args) -> None:
        mode = self._resolution_combo.currentData()
        is_custom = mode is ResolutionMode.CUSTOM
        self._custom_res_widget.setVisible(is_custom)
        if is_custom:
            # Seed the spinboxes from the current grid (a sensible starting size) the
            # first time Custom is chosen, then hand the controller a size so
            # build_common_grid never sees CUSTOM without one.
            grid = self._controller.get_common_grid()
            if grid.geotransform is not None and self._controller.get_custom_resolution() is None:
                self._custom_xres_spin.blockSignals(True)
                self._custom_yres_spin.blockSignals(True)
                self._custom_xres_spin.setValue(abs(grid.geotransform[1]))
                self._custom_yres_spin.setValue(abs(grid.geotransform[5]))
                self._custom_xres_spin.blockSignals(False)
                self._custom_yres_spin.blockSignals(False)
            self._controller.set_custom_resolution(
                self._custom_xres_spin.value(), self._custom_yres_spin.value()
            )
        self._controller.set_resolution_mode(mode)
        # Resolution changes the output grid's pixel size, not the on-screen preview
        # (the compositor warps at viewport resolution), so a grid rebuild is the whole
        # effect; the CRS constraint is unchanged, so rebuild quietly (no prompt).
        self._rebuild_grid_quietly()

    def _on_custom_resolution_changed(self, *_args) -> None:
        if self._resolution_combo.currentData() is not ResolutionMode.CUSTOM:
            return
        self._controller.set_custom_resolution(self._custom_xres_spin.value(), self._custom_yres_spin.value())
        self._rebuild_grid_quietly()

    # -- resampling -----------------------------------------------------------

    def _on_resample_changed(self, *_args) -> None:
        alg = self._resample_combo.currentData()
        # Always warn on a non-Nearest-Neighbor choice: interpolation invents new pixel
        # values, which distorts quantitative products (there is no product-type
        # detection — we warn unconditionally).
        if alg != gdal.GRA_NearestNeighbour:
            QMessageBox.warning(
                self,
                self.tr("Resampling may alter pixel values"),
                self.tr(
                    "Non-Nearest-Neighbor resampling interpolates new pixel values, "
                    "which can distort quantitative data. Use Nearest Neighbor to "
                    "preserve the original values."
                ),
            )
        self._controller.set_resample_alg(alg)
        # The warp algorithm changes the rendered pixels, so force a fresh read.
        self._mosaic_view.invalidate_pixels()

    # -- export ---------------------------------------------------------------

    def _on_export_clicked(self) -> None:
        """
        Composite the visible scenes at full resolution and stream the result to an
        ENVI file the user picks. Runs on a scheduler thread with a progress modal +
        Activity Monitor row (mirroring the ingestion path); the written file is *not*
        loaded back into WISER — the user opens it manually.
        """
        if self._app_services is None:
            return

        visible = [scene for scene in self._controller.get_scenes() if scene.visible]
        if not visible:
            QMessageBox.information(
                self,
                self.tr("Nothing to export"),
                self.tr("Add at least one visible scene before exporting."),
            )
            return

        # Resolve the output grid first (may prompt for a target CRS), then confirm it
        # actually produced a usable extent before asking for an output path.
        if not self._ensure_common_grid():
            return
        grid = self._controller.get_common_grid()
        if grid.extent is None or not grid.width or not grid.height:
            QMessageBox.warning(
                self,
                self.tr("Cannot export"),
                self.tr("The common output grid is not resolved yet."),
            )
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            self.tr("Export mosaic as ENVI"),
            "",
            self.tr("ENVI raster (*.img);;All files (*)"),
        )
        if not path:
            return

        band_source = self._controller.get_band_metadata_source()
        band_metadata_dataset = band_source.dataset if band_source is not None else None
        output_nodata = self._resolve_output_nodata(band_metadata_dataset, visible)

        # Snapshot the controller state on the GUI thread and hand the heavy work to a
        # scheduler thread; block only the mosaic dialog while it runs.
        self._active_progress_task = run_with_progress(
            app_services=self._app_services,
            block_window=self.window(),
            title=self.tr("Exporting mosaic"),
            fn=_export_mosaic_task,
            scenes=visible,
            grid=grid,
            target_wkt=self._controller.get_target_crs(),
            resample_alg=self._controller.get_resample_alg(),
            output_nodata=output_nodata,
            band_metadata_dataset=band_metadata_dataset,
            out_path=path,
            on_success=self._on_export_finished,
            on_error=self._on_export_failed,
            description=self.tr("Compositing mosaic…"),
            meta={"scenes": str(len(visible))},
        )

    @staticmethod
    def _resolve_output_nodata(
        band_metadata_dataset: Optional["RasterDataSet"],
        visible_scenes: Sequence[MosaicScene],
    ) -> Optional[float]:
        """
        Pick the output Data Ignore Value: the canonical band-metadata dataset's if it
        has one, else the top-most visible scene that has one (top → bottom), else
        ``None`` (no scene defines a nodata, so nodata compositing is a no-op).
        """
        if band_metadata_dataset is not None:
            nodata = band_metadata_dataset.get_data_ignore_value()
            if nodata is not None:
                return nodata
        for scene in reversed(list(visible_scenes)):
            nodata = scene.dataset.get_data_ignore_value()
            if nodata is not None:
                return nodata
        return None

    def _on_export_finished(self, out_path: str) -> None:
        QMessageBox.information(
            self,
            self.tr("Export complete"),
            self.tr("Mosaic written to:\n{0}").format(out_path),
        )

    def _on_export_failed(self, message: str) -> None:
        QMessageBox.warning(self, self.tr("Export failed"), message)

    # -- common grid / target CRS ---------------------------------------------

    def _ensure_common_grid(self) -> bool:
        """
        Resolve the shared output grid, prompting for a target CRS if needed.

        Same-CRS mosaics auto-resolve (``build_common_grid`` picks the shared scene
        CRS), so the dialog only appears on a real CRS mismatch. Standalone (not
        inlined into ``_on_scene_ingested``) so #638 can re-run it when the
        resolution mode or CRS changes.

        Returns ``True`` once the grid is resolved, ``False`` if it is left
        unresolved (unmappable CRS, or the user cancelled the reproject prompt) so
        a caller that just added a scene can decide to roll that addition back.
        """
        resolved = True
        try:
            self._controller.build_common_grid()
        except TargetCrsRequired:
            if self._prompt_for_target_crs():
                try:
                    self._controller.build_common_grid()
                except UnmappableCrsError as exc:
                    QMessageBox.warning(self, self.tr("Cannot reproject"), str(exc))
                    resolved = False
            else:
                resolved = False
        except UnmappableCrsError as exc:
            QMessageBox.warning(self, self.tr("Cannot reproject"), str(exc))
            resolved = False
        self._refresh_target_crs_label()
        return resolved

    def _on_choose_target_crs(self) -> None:
        """Let the user pick / override the target CRS at any time (manual button)."""
        if not self._prompt_for_target_crs():
            return
        try:
            self._controller.build_common_grid()
        except UnmappableCrsError as exc:
            QMessageBox.warning(self, self.tr("Cannot reproject"), str(exc))
        self._refresh_target_crs_label()
        self._mosaic_view.invalidate_overlay()
        self._mosaic_view.invalidate_pixels()

    def _prompt_for_target_crs(self) -> bool:
        """
        Show the reproject dialog and, on accept, validate + set the chosen target
        CRS. Returns ``True`` when a target CRS was set, ``False`` otherwise (no
        scenes, cancelled, or the choice was unmappable).
        """
        if self._controller.scene_count() == 0:
            QMessageBox.information(
                self,
                self.tr("No scenes"),
                self.tr("Add at least one scene before choosing a target CRS."),
            )
            return False

        dlg = ReprojectPromptDialog(
            self._controller.scene_crs_summary(),
            self._controller.scene_crs_choices(),
            self._app_state,
            self,
        )
        if dlg.exec() != QDialog.Accepted:
            return False
        target = dlg.selected_target_wkt()
        try:
            self._controller.validate_target_crs(target)
        except UnmappableCrsError as exc:
            QMessageBox.warning(self, self.tr("Cannot reproject"), str(exc))
            return False
        self._controller.set_target_crs(target)
        return True

    def _rebuild_grid_quietly(self) -> None:
        """
        Rebuild the common grid if it resolves; stay silent if it can't. Used after
        removals / visibility changes, where a modal prompt would be surprising.
        """
        try:
            self._controller.build_common_grid()
        except (TargetCrsRequired, UnmappableCrsError):
            pass
        self._refresh_target_crs_label()

    def _refresh_target_crs_label(self) -> None:
        wkt = self._controller.get_target_crs()
        if not wkt:
            self._target_crs_label.setText(
                self.tr("Not set — resolved automatically when scenes share a CRS.")
            )
            return
        srs = osr.SpatialReference()
        name = srs.GetName() if srs.ImportFromWkt(wkt) == 0 else None
        self._target_crs_label.setText(self.tr("Current: {0}").format(name or self.tr("(custom)")))

    # -- accessors ------------------------------------------------------------

    def get_controller(self) -> MosaicController:
        return self._controller

    def get_mosaic_view(self) -> MosaicView:
        return self._mosaic_view
