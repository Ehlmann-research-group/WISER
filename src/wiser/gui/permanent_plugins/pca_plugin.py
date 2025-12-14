from __future__ import division
import numpy as np
import sys
import logging
from timeit import default_timer as timer
from sklearn.decomposition import PCA
from typing import Dict
from enum import Enum
from typing import TYPE_CHECKING

from PySide2.QtWidgets import QDialog

from wiser import plugins
from wiser.raster import RasterDataLoader
from wiser.raster import RasterDataSet
from wiser.raster.utils import compute_PCA_on_image, create_pca_metadata_widget
from wiser.gui.generated.pca_dialog_ui import Ui_PCA_Dialog

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


class ESTIMATOR_TYPES(Enum):
    COVARIANCE = "Covariance"


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
        # The list from get_bad_bands() has 1's for the good bands
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
            )

    def run_pca(
        self,
        dataset: RasterDataSet,
        num_components: int,
        estimator: ESTIMATOR_TYPES,
        app_state: "ApplicationState",
        test_mode: bool = False,
    ):
        image_arr = dataset.get_image_data()

        # Returns in format [y][x][num_components]
        masked_arr, pca = compute_PCA_on_image(
            image_arr=image_arr,
            num_components=num_components,
            bad_bands=dataset.get_bad_bands(),
            data_ignore=dataset.get_data_ignore_value(),
        )

        # [y][x][num_components] --> [num_components][y][x] (WISER uses this format)
        masked_arr = masked_arr.transpose(2, 0, 1).copy(order="C")
        if test_mode:
            return masked_arr
        data_loader = RasterDataLoader()
        new_dataset = data_loader.dataset_from_numpy_array(masked_arr)
        new_dataset.set_name(f"PCA on {dataset.get_name()}")
        new_dataset.set_description(dataset.get_description())
        new_dataset.copy_spatial_metadata(dataset.get_spatial_metadata())

        self._pca_widget = create_pca_metadata_widget(pca=pca, dataset=new_dataset)

        self._pca_widget.show()

        app_state.add_dataset(new_dataset)
