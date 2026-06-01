from __future__ import division

import datetime
from concurrent.futures import Future
from dataclasses import dataclass
import logging
from enum import Enum
from typing import Callable, Dict, TYPE_CHECKING, Optional

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot
from PySide2.QtWidgets import QDialog, QMessageBox
from sklearn.decomposition import PCA

from wiser import plugins
from wiser.gui.run_history import EigenScreeRunHistoryDialog, RunHistoryManagerBase
from wiser.gui.scree_plot import build_scree_plot_figure
from wiser.gui.generated.pca_dialog_ui import Ui_PCA_Dialog
from wiser.raster import RasterDataLoader, RasterDataSet
from wiser.raster.utils import compute_PCA_on_image, create_pca_metadata_widget
from wiser.utils.primitives import DataRef, DatasetRegionRef, PriorityClass
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.task_stage_utils import get_pca_pipeline
from wiser.utils.task_system import SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.app_services import AppServices
    from wiser.gui.app_state import ApplicationState


@dataclass(frozen=True)
class PCARunRecord:
    """Immutable record of one completed PCA run.

    Snapshots the eigenvalue array (sklearn's ``pca.explained_variance_``)
    so the scree plot is still viewable after the input dataset is closed
    or the sklearn ``PCA`` object is gone.

    ``eigenvalues`` has length ``num_components_chosen`` — sklearn only
    retains the top-N components it was asked for, so this is necessarily
    a subset of the full spectrum.  That is acceptable for the scree plot.
    """

    run_id: int
    timestamp: datetime.datetime
    input_dataset_id: int
    input_dataset_name_snapshot: str
    num_components_chosen: int
    max_components_available: int
    eigenvalues: np.ndarray


class PCAHistoryManager(RunHistoryManagerBase[PCARunRecord]):
    """Owns the in-memory list of completed PCA runs."""


class PCAHistoryDialog(EigenScreeRunHistoryDialog[PCARunRecord]):
    """Non-modal viewer for past PCA runs.  See base class for behavior."""

    task_label = "PCA"


def compute_max_pca_components(dataset: RasterDataSet) -> int:
    """Maximum number of principal components meaningfully extractable.

    PCA is fit on the *good* bands only (bad bands are dropped before the
    covariance is built — see :func:`compute_PCA_on_image`), so the ceiling
    is the count of good bands.  Used both to seed the dialog's spin-box
    bounds/default and to populate ``max_components_available`` on the
    :class:`PCARunRecord` written when a run completes.
    """
    return int(sum(dataset.get_bad_bands()))


class ESTIMATOR_TYPES(Enum):
    COVARIANCE = "Covariance"


class PCAPluginTask(QObject, SemanticTask):
    result_ready = Signal(object, object)
    # Emitted from _load_result_into_wiser once the run record is built.
    # Payload is a PCARunRecord; the plugin wires this to
    # app_state.get_pca_history().add_record so the past-runs viewer sees it.
    run_recorded = Signal(object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: RasterDataSet,
        input_ref: DataRef,
        num_components: int,
        max_components_available: int,
        output_ref_name: str = "pca_image",
        pca_json_ref_name: str = "pca_model",
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=get_pca_pipeline(
                dataset_ref=input_ref,
                num_components=num_components,
                output_ref_name=output_ref_name,
                pca_json_ref_name=pca_json_ref_name,
            ),
            task_title="Principal Component Analysis",
            task_variables={
                "Num Components": num_components,
                "Dataset": source_dataset.get_name(),
            },
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._output_ref_name = output_ref_name
        self._pca_json_ref_name = pca_json_ref_name
        # Snapshotted for the run record so the past-runs table can show how
        # many components were chosen out of how many were available.
        self._num_components_chosen = num_components
        self._max_components_available = max_components_available
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing PCA output binding: {self._output_ref_name}")
        pca_ref = bindings.get(self._pca_json_ref_name)
        if pca_ref is None:
            raise KeyError(f"Missing PCA model binding: {self._pca_json_ref_name}")

        storage_client = get_process_storage_client()
        data_meta = storage_client.get_meta(output_ref)
        height, width, bands = data_meta.shape
        output_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        reduced_data, _ = storage_client.read_region(output_ref, output_region)
        pca_payload = storage_client.read_json_value(pca_ref)
        self.result_ready.emit(np.asarray(reduced_data), pca_payload)

    @Slot(object, object)
    def _load_result_into_wiser(self, reduced_data: object, pca_payload: object) -> None:
        reduced_array = np.asarray(reduced_data)
        reduced_array_by_band = reduced_array.transpose(2, 0, 1)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        reduced_dataset = loader.dataset_from_numpy_array(reduced_array_by_band, cache)

        source_name = self._source_dataset.get_name() or "Dataset"
        reduced_dataset.set_name(self._app_state.unique_dataset_name(f"PCA on {source_name}"))
        reduced_dataset.set_description(self._source_dataset.get_description())
        reduced_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())
        reduced_dataset.set_data_ignore_value(self._source_dataset.get_data_ignore_value())

        pca: PCA = pca_payload["pca"]
        self._pca_widget = create_pca_metadata_widget(pca=pca, dataset=reduced_dataset)
        self._pca_widget.show()

        self._app_state.add_dataset(reduced_dataset, view_dataset=False)

        # Snapshot the eigenvalues (sklearn calls these explained_variance_)
        # into a fresh array so the record is independent of sklearn's
        # internal storage.
        eigenvalues = np.asarray(pca.explained_variance_, dtype=np.float64).copy()

        # Pop the scree plot up alongside the PCA metadata widget so the user
        # immediately sees the eigenvalue spectrum that justifies (or doesn't)
        # the component count they picked.  The same plot is reachable later
        # via the past-runs dialog — this is just the on-completion surface.
        scree_title = f"PCA Run {self.id} — Scree Plot ({source_name})"
        scree_description = (
            f"PCA run {self.id} on '{source_name}' — "
            f"{self._num_components_chosen} of {self._max_components_available} components kept."
        )
        figure, axes = build_scree_plot_figure(eigenvalues, title=scree_title)
        self._app_state.show_matplotlib_display_widget(
            figure=figure,
            axes=axes,
            window_title=scree_title,
            description=scree_description,
        )

        self.run_recorded.emit(
            PCARunRecord(
                run_id=self.id,
                timestamp=datetime.datetime.now(),
                input_dataset_id=self._source_dataset.get_id(),
                input_dataset_name_snapshot=self._source_dataset.get_name() or "Dataset",
                num_components_chosen=self._num_components_chosen,
                max_components_available=self._max_components_available,
                eigenvalues=eigenvalues,
            )
        )


class PCADialog(QDialog):
    """Non-modal dialog for configuring and launching a PCA task.

    A persistent instance is owned by :class:`PCAPlugin` and reused across
    every "PCA" right-click invocation; we just re-bind it to whichever
    dataset the user clicked on via :meth:`configure_for_dataset`.

    Being non-modal matters: while a PCA run is being configured the user
    may want to open the past-runs viewer and inspect a scree plot, which
    appears as its own top-level matplotlib window.  An ``exec()``-style
    modal dialog would block input to that scree plot.
    """

    def __init__(
        self,
        app_state: "ApplicationState",
        run_callback: Callable[[RasterDataSet, int, "ESTIMATOR_TYPES", int], None],
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setModal(False)
        self._app_state = app_state
        self._run_callback = run_callback
        # Tracked as a plain id (not a dataset object) so we can detect when
        # the underlying dataset is removed from the app while we're still
        # showing — see :meth:`_on_dataset_removed`.
        self._dataset_id: Optional[int] = None

        self._ui = Ui_PCA_Dialog()
        self._ui.setupUi(self)

        self._ui.sbox_num_components.setMinimum(1)
        for est in ESTIMATOR_TYPES:
            self._ui.cbox_estimator.addItem(est.value, est)
        if self._ui.cbox_estimator.count() == 1:
            self._ui.cbox_estimator.setEnabled(False)

        # Past-runs viewer — lazily created on first click and kept alive on
        # this dialog so reopening preserves scroll position / closed-runs
        # toggle state.  The records themselves live on app_state.
        self._past_runs_dialog: Optional[PCAHistoryDialog] = None
        self._ui.btn_past_results.clicked.connect(self._on_view_past_runs)

        app_state.dataset_removed.connect(self._on_dataset_removed)

    def configure_for_dataset(self, dataset: RasterDataSet) -> None:
        """Bind this dialog to a specific dataset and reset the form."""
        self._dataset_id = dataset.get_id()
        self.setWindowTitle(self.tr(f"PCA: {dataset.get_name() or 'Dataset'}"))
        # max() guards against datasets where every band is bad (0 good
        # bands); spin-box max must be ≥ min (1) or Qt rejects the setter.
        max_components = max(1, compute_max_pca_components(dataset))
        self._ui.sbox_num_components.setMaximum(max_components)
        self._ui.sbox_num_components.setValue(max_components)

    def accept(self) -> None:
        # The selected dataset may have been removed between the context-menu
        # click and the user pressing OK.  Bail with a warning rather than
        # submit a task against a stale id.
        if self._dataset_id is None:
            return
        try:
            dataset = self._app_state.get_dataset(self._dataset_id)
        except KeyError:
            QMessageBox.warning(
                self,
                self.tr("PCA"),
                self.tr("The dataset this PCA dialog was bound to is no longer available."),
            )
            return

        num_components = self._ui.sbox_num_components.value()
        estimator: "ESTIMATOR_TYPES" = self._ui.cbox_estimator.currentData()
        # The spin-box maximum was set in configure_for_dataset to the true
        # max — reuse it here so the run record reports the same value the
        # user saw in the dialog.
        max_components = self._ui.sbox_num_components.maximum()

        self._run_callback(dataset, num_components, estimator, max_components)
        super().accept()

    def _on_view_past_runs(self) -> None:
        if self._past_runs_dialog is None:
            self._past_runs_dialog = PCAHistoryDialog(
                app_state=self._app_state,
                manager=self._app_state.get_pca_history(),
                parent=self,
            )
        self._past_runs_dialog.show()
        self._past_runs_dialog.raise_()
        self._past_runs_dialog.activateWindow()

    def _on_dataset_removed(self, removed_id: int) -> None:
        # If the dataset we were configured against goes away, close quietly.
        # Submitting PCA against a deleted dataset would fail downstream, and
        # leaving the dialog open with a stale binding is confusing.
        if self._dataset_id is not None and int(removed_id) == int(self._dataset_id):
            self.close()


class PCAPlugin(plugins.ContextMenuPlugin):
    def __init__(self):
        logging.info("PCA Initializing")
        # Persistent dialog reused across context-menu invocations.  Created
        # lazily because the plugin is constructed before app_state /
        # app_services exist.
        self._dialog: Optional[PCADialog] = None

    def add_context_menu_items(self, context_type: plugins.types.ContextMenuType, context_menu, context):
        if context_type == plugins.ContextMenuType.RASTER_VIEW:
            act1 = context_menu.addAction(context_menu.tr("PCA"))
            act1.triggered.connect(lambda checked=False: self.show_pca(context=context))

    def show_pca(self, context: Dict):
        dataset: RasterDataSet = context["dataset"]
        app_state: "ApplicationState" = context["wiser"]
        app_services: "AppServices" = context.get("app_services")

        if self._dialog is None:
            # Capture app_state + app_services in the callback closure so the
            # dialog stays decoupled from the plugin; both are singletons
            # within a WISER session so capturing once is safe.
            def _submit(
                ds: RasterDataSet,
                num_components: int,
                estimator: "ESTIMATOR_TYPES",
                max_components_available: int,
            ) -> None:
                self.run_pca(
                    dataset=ds,
                    num_components=num_components,
                    estimator=estimator,
                    app_state=app_state,
                    app_services=app_services,
                    max_components_available=max_components_available,
                )

            self._dialog = PCADialog(app_state=app_state, run_callback=_submit)

        self._dialog.configure_for_dataset(dataset)
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def run_pca(
        self,
        dataset: RasterDataSet,
        num_components: int,
        estimator: ESTIMATOR_TYPES,
        app_state: "ApplicationState",
        app_services: "AppServices" = None,
        test_mode: bool = False,
        max_components_available: Optional[int] = None,
    ) -> Optional[Future]:
        _ = estimator
        if app_services is None:
            app = getattr(app_state, "_app", None)
            app_services = getattr(app, "_app_services", None)

        if test_mode or app_services is None:
            image_arr = dataset.get_image_data()
            masked_arr, pca = compute_PCA_on_image(
                image_arr=image_arr,
                num_components=num_components,
                bad_bands=dataset.get_bad_bands(),
                data_ignore=dataset.get_data_ignore_value(),
            )
            masked_arr = masked_arr.transpose(2, 0, 1).copy(order="C")
            if test_mode:
                return masked_arr

            data_loader = RasterDataLoader()
            new_dataset = data_loader.dataset_from_numpy_array(masked_arr)
            new_dataset.set_name(f"PCA on {dataset.get_name()}")
            new_dataset.set_description(dataset.get_description())
            new_dataset.copy_spatial_metadata(dataset.get_spatial_metadata())
            new_dataset.set_data_ignore_value(dataset.get_data_ignore_value())
            self._pca_widget = create_pca_metadata_widget(pca=pca, dataset=new_dataset)
            self._pca_widget.show()
            app_state.add_dataset(new_dataset)
            return None

        dataset_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=dataset)
        )
        # If the caller didn't tell us the dataset's max-components ceiling
        # (e.g. an external script calling run_pca directly), fall back to
        # the canonical value — the same one the dialog uses.
        if max_components_available is None:
            max_components_available = compute_max_pca_components(dataset)
        pca_task = PCAPluginTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            num_components=num_components,
            max_components_available=max_components_available,
        )
        # Record the run in the application-level history once the task
        # finishes loading its result into the app (mirrors linear_unmixing.py:393).
        pca_task.run_recorded.connect(app_state.get_pca_history().add_record)
        self._last_pca_task = pca_task
        task_plan = app_services.task_planner.plan_semantic_task(pca_task)
        return app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
