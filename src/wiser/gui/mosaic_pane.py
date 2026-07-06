"""
Control panel for the Seamless Mosaic feature (EPIC #629).

Hosts the :class:`MosaicView` alongside a controls area and owns the non-GUI
:class:`MosaicController` that both share. In this issue (#634) the controls area
gains a minimal "Add Scene" action: a dataset picker plus a button that ingests the
chosen dataset (materialize -> build overviews -> compute footprint) on a background
thread and appends it to the controller.

The richer control panel (scene list with drag-to-reorder, per-scene visibility,
resolution / CRS / resampling selectors, export) still lands in #638. The full
"add from file" picker is also #638; here the combo reads datasets already loaded in
``app_state``.
"""

from typing import Optional, TYPE_CHECKING

from osgeo import osr

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .app_services import AppServices
from .mosaic_crs_dialog import ReprojectPromptDialog
from .mosaic_view import MosaicView
from .progress_task import run_with_progress

from wiser.raster.mosaic_controller import (
    MosaicController,
    MosaicScene,
    TargetCrsRequired,
    UnmappableCrsError,
)
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
    progress: Optional[ProgressReporter] = None,
) -> MosaicScene:
    """
    Background I/O for one scene: materialize to a warpable temp GeoTIFF, build
    internal overviews on it, and compute the valid-pixel footprint.

    Runs on a scheduler thread (no Qt here). ``progress`` is split across the three
    phases (weighted by their rough cost) so the overall bar advances smoothly;
    each phase reports fine-grained progress internally. Returns the fully-populated
    :class:`MosaicScene` for the main thread to append to the controller.
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
    )


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
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._materializer = materializer
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
        # current scene stack (per-scene visibility + remove), and target-CRS
        # resolution. #638 extends this further (drag-to-reorder, resampling, export).
        self._controls = QWidget(self._splitter)
        self._controls.setMinimumWidth(280)
        self._controls_layout = QVBoxLayout(self._controls)
        self._controls_layout.setContentsMargins(8, 8, 8, 8)
        self._controls_layout.setSpacing(10)

        self._controls_layout.addWidget(self._build_add_scene_group())
        self._controls_layout.addWidget(self._build_scene_list_group(), 1)
        self._controls_layout.addWidget(self._build_target_crs_group())
        self._splitter.addWidget(self._controls)

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
        self._scene_list.setToolTip(
            self.tr("Scenes in top-to-bottom stacking order. Uncheck to hide a scene.")
        )
        self._scene_list.itemSelectionChanged.connect(self._on_scene_selection_changed)
        self._scene_list.itemChanged.connect(self._on_scene_item_changed)
        layout.addWidget(self._scene_list)

        self._remove_scene_button = QPushButton(self.tr("Remove Selected"), group)
        self._remove_scene_button.setEnabled(False)
        self._remove_scene_button.clicked.connect(self._on_remove_scene_clicked)
        layout.addWidget(self._remove_scene_button)
        return group

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
            on_success=self._on_scene_ingested,
            on_error=self._on_scene_failed,
            description=self.tr("Materializing scene…"),
            meta={"bands": str(dataset.num_bands())},
        )

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
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if scenes[index].visible else Qt.Unchecked)
            self._scene_list.addItem(item)
            if index == previous:
                self._scene_list.setCurrentItem(item)
        self._scene_list.blockSignals(False)
        self._on_scene_selection_changed()

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
