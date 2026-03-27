from __future__ import division

from concurrent.futures import Future
import logging
from enum import Enum
from typing import Dict, TYPE_CHECKING, Optional

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot
from PySide2.QtWidgets import QDialog

from wiser import plugins
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


class ESTIMATOR_TYPES(Enum):
    COVARIANCE = "Covariance"


class PCAPluginTask(QObject, SemanticTask):
    result_ready = Signal(object, object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: RasterDataSet,
        input_ref: DataRef,
        num_components: int,
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

        self._pca_widget = create_pca_metadata_widget(pca=pca_payload["pca"], dataset=reduced_dataset)
        self._pca_widget.show()

        self._app_state.add_dataset(reduced_dataset, view_dataset=False)


class PCAPlugin(plugins.ContextMenuPlugin):
    def __init__(self):
        logging.info("PCA Initializing")

    def add_context_menu_items(self, context_type: plugins.types.ContextMenuType, context_menu, context):
        if context_type == plugins.ContextMenuType.RASTER_VIEW:
            act1 = context_menu.addAction(context_menu.tr("PCA"))
            act1.triggered.connect(lambda checked=False: self.show_pca(context=context))

    def show_pca(self, context: Dict):
        pca_dialog = QDialog()
        pca_dialog._ui = Ui_PCA_Dialog()
        pca_dialog._ui.setupUi(pca_dialog)

        dataset: RasterDataSet = context["dataset"]
        num_valid_bands = sum(dataset.get_bad_bands())

        pca_dialog._ui.sbox_num_components.setMinimum(1)
        pca_dialog._ui.sbox_num_components.setMaximum(num_valid_bands)

        for est in ESTIMATOR_TYPES:
            pca_dialog._ui.cbox_estimator.addItem(est.value, est)

        if pca_dialog._ui.cbox_estimator.count() == 1:
            pca_dialog._ui.cbox_estimator.setEnabled(False)

        if pca_dialog.exec() == QDialog.Accepted:
            num_components = pca_dialog._ui.sbox_num_components.value()
            estimator: ESTIMATOR_TYPES = pca_dialog._ui.cbox_estimator.currentData()
            self.run_pca(
                dataset=dataset,
                num_components=num_components,
                estimator=estimator,
                app_state=context["wiser"],
                app_services=context.get("app_services"),
            )

    def run_pca(
        self,
        dataset: RasterDataSet,
        num_components: int,
        estimator: ESTIMATOR_TYPES,
        app_state: "ApplicationState",
        app_services: "AppServices" = None,
        test_mode: bool = False,
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
        pca_task = PCAPluginTask(
            app_state=app_state,
            source_dataset=dataset,
            input_ref=dataset_ref,
            num_components=num_components,
        )
        self._last_pca_task = pca_task
        task_plan = app_services.task_planner.plan_semantic_task(pca_task)
        return app_services.task_manager.register_and_submit_task_plan(app_services.scheduler, task_plan)
