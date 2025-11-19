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
from wiser.raster.utils import operation_on_all_spectra
from wiser.gui.generated.pca_dialog_ui import Ui_PCA_Dialog

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


class ESTIMATOR_TYPES(Enum):
    COVARIANCE = "Covariance"
    CORRELATION = "Correlation"


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
        num_bands = dataset.num_bands()

        pca_dialog._ui.sbox_num_components.setMinimum(1)
        pca_dialog._ui.sbox_num_components.setMaximum(num_bands)

        for est in ESTIMATOR_TYPES:
            pca_dialog._ui.cbox_estimator.addItem(est.value, est)

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
    ):
        num_cols = dataset.get_width()
        num_rows = dataset.get_height()
        num_bands = dataset.num_bands()
        image_arr = dataset.get_image_data()
        print(f"num_cols: {num_cols}, num_rows: {num_rows}, num_bands: {num_bands}")
        # if isinstance(image_arr, np.ma.masked_array) or True:
        if estimator == ESTIMATOR_TYPES.COVARIANCE:
            # print(f"masked array")
            # masked_2d = np.all(image_arr.mask == True, axis = 0)
            # print(f"masked_2d.shape: {masked_2d.shape}")
            # print(f"masked_2d: {masked_2d}")
            # arr = image_arr.data[~image_arr.mask]
            # print(f"image_arr.data.shape: {image_arr.data.shape}")
            # print(f"image_arr.flags: {image_arr.flags}")
            # # print(f"image_arr.data: {image_arr.data}")
            # print(f"image_arr.mask.shape: {image_arr.mask.shape}")
            # # print(f"image_arr.mask: {image_arr.mask}")
            # print(f"Arr.shape: {arr.shape}")

            # Match each spectra with its location in the image.

            # Remove the bad bands and be able to reconstruct the bad bands

            # Returns to use in [y][x][num_components]
            masked_arr = operation_on_all_spectra(
                image_arr=image_arr,
                num_components=num_components,
                bad_bands=dataset.get_bad_bands(),
                data_ignore=dataset.get_data_ignore_value(),
            )

            # [y][x][num_components] --> [num_components][y][x]
            masked_arr = np.copy(masked_arr.transpose(2, 0, 1), order="C")
            data_loader = RasterDataLoader()
            new_dataset = data_loader.dataset_from_numpy_array(masked_arr)
            new_dataset.set_name(f"PCA on {dataset.get_name()}")
            new_dataset.set_description(dataset.get_description())
            app_state.add_dataset(new_dataset)
            return
            # print(f"100x100 spec: {image_arr.data[:, 100, 100]}")
            # image_arr = image_arr.transpose(1, 2, 0)  # [b][y][x] --> [y][x][b]
            # print(f"image_arr.flags['OWNDATA'] 1: {image_arr.flags['OWNDATA']}")
            # image_arr: np.ma.masked_array = image_arr.reshape(
            #     (image_arr.shape[0] * image_arr.shape[1], image_arr.shape[2])
            # )
            # print(f"image_arr.flags['OWNDATA'] 2: {image_arr.flags['OWNDATA']}")
            # print(f"type(image_arr): {type(image_arr)}")
            # mask_1d = ~np.all(image_arr.mask == True, axis=1)
            # print(f"mask_1d.shape: {mask_1d.shape}")
            # print(f"mask_1d.count: {mask_1d.sum()}")
            # # arr = np.take(image_arr.data, mask_1d, axis=0)
            # arr = image_arr.data[mask_1d, :]
            # print(f"Arr.shape after take: {arr.shape}")
        else:
            print("non-masked array")
            image_arr = np.copy(image_arr.transpose(1, 2, 0), order="C")
            arr = image_arr.reshape((num_rows * num_cols, num_bands))
            print(f"image_arr.shape: {image_arr.shape}")
            print(f"Arr.shape: {arr.shape}")

        # elif isinstance(image_arr, np.ndarray):
        #     print(f"non-masked array")
        #     arr = image_arr.reshape((num_rows * num_cols, num_bands))
        #     print(f"image_arr.shape: {image_arr.shape}")
        #     print(f"Arr.shape: {arr.shape}")
        # else:
        #     raise ValueError(
        #         "Array returned from get_image_data is "
        #         + "neither np.ma.masked_array or np.ndarray.\n"
        #         + f"Instead it's: {type(image_arr)}."
        #     )

        pca = PCA(n_components=num_components)
        PCA_result = pca.fit_transform(arr)
        print(f"result type: {type(PCA_result)}")
        print(f"result shape: {PCA_result.shape}")
        print(f"PCA_result shape: {PCA_result.shape}")
        PCA_result = PCA_result.reshape((num_rows, num_cols, num_components))
        new_image_data = PCA_result.copy().transpose(2, 0, 1)  # [y][x][b] -> [b][y][x]
        raster_data = RasterDataLoader()
        new_data = raster_data.dataset_from_numpy_array(new_image_data)
        new_data.set_name(f"PCA on {dataset.get_name()}")
        new_data.set_description(dataset.get_description())
        app_state.add_dataset(new_data)

    # return the image PCAed
    def image(self, context):
        start = timer()
        datasets = context["wiser"].get_datasets()
        # A numpy array such that the pixel (x, y) values (spectrum value) of
        # band b are at element array[b][y][x]
        image_data = datasets[0].get_image_data()
        filename = datasets[0].get_name()
        description = datasets[0].get_description()
        # default_bands = datasets[0].default_display_bands()
        wavelengths = datasets[0].band_list()
        wavelengths = np.array([float(i["wavelength_str"]) for i in wavelengths])
        # wavelengths = np.array([i.value for i in wavelengths])

        # cols = len(image_data[0][0])
        # rows = len(image_data[0])
        # bands = len(image_data)
        # image_transposed = image_data.copy().transpose(1,2,0) #[b][y][x] -> [y][x][b]
        image_transposed = np.copy(image_data.transpose(1, 2, 0), order="C")  # [b][y][x] -> [y][x][b]
        # image_transposed = np.delete(image_transposed, 319, axis=1)
        # cols = 319
        # image_2d = image_transposed.reshape((rows * cols, bands))  # [y][x][b] -> [y*x][b]
        # use = np.all(image2D > -9999, axis = 1) #ignore locations with -9999 in the data
        # spectra = image_2d[use, :]
        pca_ = PCA(n_components=8)
        PCA_result = pca_.fit_transform(image_transposed)
        end = timer()
        logging.info(end - start)
        PCA_result = np.array(PCA_result.image)
        new_image_data = PCA_result.copy().transpose(2, 0, 1)  # [y][x][b] -> [b][y][x]
        raster_data = RasterDataLoader()
        new_data = raster_data.dataset_from_numpy_array(new_image_data)
        new_data.set_name(f"PCA on {filename}")
        new_data.set_description(description)
        context["wiser"].add_dataset(new_data)
        logging.info("done")

    # helper function: find all NaN values
    def bad_values(self, context):
        datasets = context["wiser"].get_datasets()
        imageData = datasets[0].get_image_data()
        nanList = np.argwhere(np.isnan(imageData)).tolist()  # list consisting NaN values [b,x,y]
        badBandIdx = []
        for i in nanList:
            if i[0] not in badBandIdx:
                badBandIdx.append(i[0])
        logging.info(badBandIdx)
        return imageData
