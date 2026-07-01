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

from concurrent.futures import Future
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .app_state import ApplicationState
from .app_services import AppServices
from .mosaic_view import MosaicView

from wiser.raster.mosaic_controller import MosaicController, MosaicScene
from wiser.raster.mosaic_ingestion import (
    SceneValidationError,
    build_overviews,
    compute_footprint_wkt,
    validate_scene,
)
from wiser.utils.primitives import PriorityClass

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.mosaic_materialize import SceneMaterializer


def _ingest_scene(dataset: "RasterDataSet", materializer: "SceneMaterializer") -> MosaicScene:
    """
    Background I/O for one scene: materialize to a warpable temp GeoTIFF, build
    internal overviews on it, and compute the valid-pixel footprint.

    Runs on a scheduler thread (no Qt here). Returns the fully-populated
    :class:`MosaicScene` for the main thread to append to the controller.
    """
    gdal_path = materializer.gdal_source(dataset)
    build_overviews(gdal_path)
    footprint_wkt = compute_footprint_wkt(gdal_path)
    return MosaicScene(
        dataset=dataset,
        gdal_path=gdal_path,
        footprint_wkt=footprint_wkt,
        has_overviews=True,
    )


class _IngestionBridge(QObject):
    """
    Marshals ingestion results from the scheduler thread back to the main thread.

    The done-callback runs on a pool thread; emitting these signals (a thread-safe
    op in PySide6) hops to the GUI thread via Qt's queued connections, so the slots
    touch widgets / the Activity Monitor only on the main thread. Each payload is
    ``(activity_id, result)`` so the slot can close the matching Activity row.
    """

    # (activity_id: int, scene: MosaicScene)
    succeeded = Signal(object)
    # (activity_id: int, message: str)
    failed = Signal(object)


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

        # Thread -> main-thread marshaling for background ingestion results.
        self._bridge = _IngestionBridge()
        self._bridge.succeeded.connect(self._on_ingestion_succeeded)
        self._bridge.failed.connect(self._on_ingestion_failed)

        self._init_ui()

    def _init_ui(self) -> None:
        # View on the left, controls on the right, split so the panel can grow later.
        self._splitter = QSplitter(Qt.Horizontal, self)

        self._mosaic_view = MosaicView(
            parent=self._splitter,
            app_state=self._app_state,
            controller=self._controller,
            mosaicpane=self,
        )
        self._splitter.addWidget(self._mosaic_view)

        # Controls area. #638 fills in the full scene list / selectors; for now it
        # carries the Add-Scene picker + button.
        self._controls = QWidget(self._splitter)
        self._controls_layout = QVBoxLayout(self._controls)

        self._dataset_combo = QComboBox(self._controls)
        self._add_scene_button = QPushButton(self.tr("Add Scene…"), self._controls)
        self._add_scene_button.clicked.connect(self._on_add_scene_clicked)
        # Ingestion needs both a scheduler (to run on) and a materializer; without
        # them the pane is display-only, so disable the action.
        if self._app_services is None or self._materializer is None:
            self._add_scene_button.setEnabled(False)

        self._controls_layout.addWidget(self._dataset_combo)
        self._controls_layout.addWidget(self._add_scene_button)
        self._controls_layout.addStretch(1)
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

        # Validate on the main thread so rejection is immediate (no spinner churn).
        try:
            validate_scene(dataset, self._controller.get_scenes())
        except SceneValidationError as exc:
            QMessageBox.warning(self, self.tr("Cannot add scene"), str(exc))
            return

        activity_id = self._app_services.activity_monitor.register_task(
            title=self.tr("Adding scene: {0}").format(dataset.get_name() or f"Dataset {ds_id}"),
            meta={"bands": str(dataset.num_bands())},
            cancel_callback=None,  # v1: short job; cancellation deferred to a follow-up.
        )

        future = self._app_services.scheduler.submit_thread(
            PriorityClass.BACKGROUND, _ingest_scene, dataset, self._materializer
        )

        def _done(finished: Future, activity_id: int = activity_id) -> None:
            # Runs on the pool thread; hop to the GUI thread via the bridge.
            try:
                self._bridge.succeeded.emit((activity_id, finished.result()))
            except Exception as exc:  # noqa: BLE001 - reported to the user via the bridge
                self._bridge.failed.emit((activity_id, str(exc)))

        future.add_done_callback(_done)

    def _on_ingestion_succeeded(self, payload: object) -> None:
        activity_id, scene = payload
        self._controller.add_scene(scene)
        if self._app_services is not None:
            self._app_services.activity_monitor.set_task_finished(activity_id)
        self._mosaic_view.update()

    def _on_ingestion_failed(self, payload: object) -> None:
        activity_id, message = payload
        if self._app_services is not None:
            self._app_services.activity_monitor.set_task_failed(activity_id, message)
        QMessageBox.critical(self, self.tr("Add scene failed"), message)

    # -- accessors ------------------------------------------------------------

    def get_controller(self) -> MosaicController:
        return self._controller

    def get_mosaic_view(self) -> MosaicView:
        return self._mosaic_view
