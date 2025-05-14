import os
from typing import List, Optional, Dict, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from wiser.gui.rasterpane import RasterPane
from wiser.gui.similarity_transform_pane import SimilarityTransformPane
from wiser.gui.app_state import ApplicationState

from wiser.raster.dataset import RasterDataSet, pixel_coord_to_geo_coord
from wiser.raster.dataset_impl import GDALRasterDataImpl

from .generated.similarity_transform_dialog_ui import Ui_SimilarityTransform

from .util import pillow_rotate_scale_expand, cv2_rotate_scale_expand

from osgeo import gdal, gdal_array

from wiser.bandmath.builtins.constants import MAX_RAM_BYTES
from wiser.bandmath.utils import write_raster_to_dataset

from wiser.raster.utils import copy_metadata_to_gdal_dataset

class SimilarityTransformDialog(QDialog):

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)

        self._app_state = app_state

        self._ui = Ui_SimilarityTransform()
        self._ui.setupUi(self)

        # --- Image renderings ----------------------------------------------------
        self._rotate_scale_pane = SimilarityTransformPane(app_state)
        self._translate_pane = SimilarityTransformPane(app_state, translation=True)
        self._translate_pane.pixel_selected_for_translation.connect(self._on_translation_pixel_selected)
        self._translate_pane.dataset_changed.connect(self._on_translation_dataset_changed)
        self._rotate_scale_pane.dataset_changed.connect(self._on_rotate_scale_dataset_changed)
        self._init_rasterpanes()

        # --- Internal state ----------------------------------------------------
        self._image_rotation: float = 0.0  # degrees CCW
        self._image_scale: float = 1.0     # isotropic scale factor
        self._lat_north_translate: float = 0.0
        self._lon_east_translate: float = 0.0

        # --- Validators --------------------------------------------------------
        rot_validator = QDoubleValidator(0.0, 360.0, 4, self)
        rot_validator.setNotation(QDoubleValidator.StandardNotation)
        self._ui.ledit_rotation.setValidator(rot_validator)

        scale_validator = QDoubleValidator(0.0, 100.0, 6, self)
        scale_validator.setNotation(QDoubleValidator.StandardNotation)
        self._ui.ledit_scale.setValidator(scale_validator)

        # --- Defaults ----------------------------------------------------------
        self._ui.ledit_rotation.setText("0.0")
        self._ui.ledit_scale.setText("1.0")

        self._ui.slider_rotation.setRange(0, 360)
        self._ui.slider_rotation.setSingleStep(1)
        self._ui.slider_rotation.setValue(0)

        # --- Signal ↔ slot wiring ----------------------------------------------
        # 1. Rotation – keep QLineEdit and QSlider in sync, but avoid infinite loops.
        self._ui.ledit_rotation.textEdited.connect(self._on_rotation_edited)
        self._ui.slider_rotation.valueChanged.connect(self._on_slider_rotation_changed)

        # 2. Scale.
        self._ui.ledit_scale.textEdited.connect(self._on_scale_edited)

        # 3. Buttons.
        self._ui.btn_rotate_scale.clicked.connect(self._on_run_rotate_scale)
        self._ui.btn_create_translation.clicked.connect(self._on_create_translation)

        # 4. Translation edits.
        self._ui.ledit_lat_north.textChanged.connect(self._on_lat_north_changed)
        self._ui.ledit_lon_east.textChanged.connect(self._on_lon_east_changed)

        self._translation_dataset: Optional[RasterDataSet] = None
        self._rotate_scale_dataset: Optional[RasterDataSet] = None
        self._selected_point: Optional[Tuple[int, int]] = None

        self._init_file_saver_rotate_scale()
        self._init_file_saver_translate()


    # -------------------------------------------------------------------------
    # Initializers
    # -------------------------------------------------------------------------

    def _init_rotate_scale_button(self):
        self._ui.btn_rotate_scale.clicked.connect(self._on_create_rotated_scaled_dataset)

    def _init_translation_button(self):
        self._ui.btn_create_translation.clicked.connect(self._on_create_translated_dataset)

    def _init_rasterpanes(self):
        rotate_scale_layout = QVBoxLayout(self._ui.widget_rotate_scale)
        self._ui.widget_rotate_scale.setLayout(rotate_scale_layout)
        rotate_scale_layout.addWidget(self._rotate_scale_pane)

        translate_layout = QVBoxLayout(self._ui.widget_translate)
        self._ui.widget_translate.setLayout(translate_layout)
        translate_layout.addWidget(self._translate_pane)

    def _init_file_saver_rotate_scale(self):
        self._ui.btn_save_path_rs.clicked.connect(self._on_choose_save_filename_rs)

    def _init_file_saver_translate(self):
        self._ui.btn_save_path_translate.clicked.connect(self._on_choose_save_filename_translate)

    def _get_save_file_path_rs(self) -> str:
        path = self._ui.ledit_save_path_rs.text()
        if len(path) > 0:
            abs_path = os.path.abspath(path)
            return abs_path
        return None

    def _get_save_file_path_translate(self) -> str:
        path = self._ui.ledit_save_path_translate.text()
        if len(path) > 0:
            abs_path = os.path.abspath(path)
            return abs_path
        return None

    # -------------------------------------------------------------------------
    # Rotation handlers
    # -------------------------------------------------------------------------

    @Slot(str)
    def _on_rotation_edited(self, text: str) -> None:
        """User typed a rotation value – sync slider and store state."""
        if not text:
            return  # ignore empty edits
        try:
            value = float(text)
        except ValueError:
            return  # validator should prevent this

        # Clamp to [0, 360]
        value = max(0.0, min(360.0, value))
        self._image_rotation = value

        # Update slider without triggering its signal.
        self._ui.slider_rotation.blockSignals(True)
        self._ui.slider_rotation.setValue(int(round(value)))
        self._ui.slider_rotation.blockSignals(False)
        print(f"self._image_rotation: {self._image_rotation}")
        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    @Slot(int)
    def _on_slider_rotation_changed(self, value: int) -> None:
        """Slider moved by user – sync line‑edit and store integer rotation."""
        # Update line‑edit without re‑entering _on_rotation_edited.
        self._ui.ledit_rotation.blockSignals(True)
        self._ui.ledit_rotation.setText(f"{value}")
        self._ui.ledit_rotation.blockSignals(False)

        self._image_rotation = float(value)
        print(f"self._image_rotation: {self._image_rotation}")

        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    # -------------------------------------------------------------------------
    # Scale handlers
    # -------------------------------------------------------------------------

    @Slot(str)
    def _on_scale_edited(self, text: str) -> None:
        """Scale edited – store value between 0 and 100 (float)."""
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            return

        # Clamp to [0, 100]
        value = max(0.0, min(100.0, value))
        self._image_scale = value
        print(f"self._image_scale: {self._image_scale}")
        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    # -------------------------------------------------------------------------
    # Translation handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_lat_north_changed(self) -> None:
        text = self._ui.ledit_lat_north.text()
        print(f"_on_lat_north_changed")
        try:
            self._lat_north_translate = float(text)
            self._update_upper_left_coord_labels()
        except ValueError:
            pass
        print(f"self._lat_north_translate: {self._lat_north_translate}")
        self._update_prev_and_new_coords()


    @Slot()
    def _on_lon_east_changed(self) -> None:
        text = self._ui.ledit_lon_east.text()
        print(f"_on_lon_east_changed")
        try:
            self._lon_east_translate = float(text)
            self._update_upper_left_coord_labels()
        except ValueError:
            pass
        print(f"self._lon_east_translate: {self._lon_east_translate}")
        self._update_prev_and_new_coords()
        

    def _make_point_to_text(self, point):
        return f"({point[0]}, {point[1]})"

    @Slot()
    def _on_rotate_scale_dataset_changed(self, ds_id):
        # We want to do many things here. But for now we just set the CRS
        self._rotate_scale_dataset = self._app_state.get_dataset(ds_id)
        print(f"rotation dataset changed to: {self._rotate_scale_dataset.get_name()}")
        self._check_rotate_scale_save_path()
    
    def _check_rotate_scale_save_path(self):
        if self._rotate_scale_dataset is None:
            return
        rotate_scale_ds_paths = self._rotate_scale_dataset.get_filepaths()
        rotate_scale_save_filepath = self._ui.ledit_save_path_rs.text()
        if rotate_scale_save_filepath in rotate_scale_ds_paths:
            self._ui.ledit_save_path_rs.clear()

    def _check_translation_save_path(self):
        if self._translation_dataset is None:
            return
        translation_ds_paths = self._translation_dataset.get_filepaths()
        translation_save_filepath = self._ui.ledit_save_path_translate.text()
        if translation_save_filepath in translation_ds_paths:
            self._ui.ledit_save_path_translate.clear()

    @Slot()
    def _on_translation_dataset_changed(self, ds_id):
        # We want to do many things here. But for now we just set the CRS
        self._translation_dataset = self._app_state.get_dataset(ds_id)
        print(f"translation dataset changed to: {self._translation_dataset.get_name()}")
        srs = self._translation_dataset.get_spatial_ref()
        if srs is None:
            return
        name = srs.GetName()
        self.set_crs_text(name)
        self._update_upper_left_coord_labels()
        self._check_translation_save_path()
    
    def _update_upper_left_coord_labels(self):
        origin_lon_east, pixel_w, rot_x, origin_lat_north, rot_y, pixel_h = self._translation_dataset.get_geo_transform()
        self.set_lon_east_ul_text(str(origin_lon_east + self._lon_east_translate))
        self.set_lat_north_ul_text(str(origin_lat_north + self._lat_north_translate))

    @Slot()
    def _on_translation_pixel_selected(self, dataset: RasterDataSet, point: QPoint) -> None:
        print(f"_on_translation_pixel_selected")
        print(f"dataset: {dataset}")
        print(f"point: {point}")
        assert dataset == self._translation_dataset, ("Dataset clicked is not equal to Translation dataset."
                                                    f"Clicked: {dataset.get_name()} | Translation Dataset: {self._translation_dataset.get_name()} ")
        self._selected_point = (point.x(), point.y())
        self._update_prev_and_new_coords()
    
    def _update_prev_and_new_coords(self):
        if self._selected_point is None:
            return
        # Get current point's dataset
        orig_geo_coords = self._translation_dataset.to_geographic_coords(self._selected_point)
        if orig_geo_coords is None:
            print(f"Dataset has no geo transform!")
            return
        origin_lon_east, pixel_w, rot_x, origin_lat_north, rot_y, pixel_h = self._translation_dataset.get_geo_transform()
        new_lon_east = origin_lon_east + self._lon_east_translate
        new_lat_north = origin_lat_north + self._lat_north_translate

        new_gt = (new_lon_east, pixel_w, rot_x, new_lat_north, rot_y, pixel_h)
        new_geo_coord = pixel_coord_to_geo_coord(self._selected_point, new_gt)

        print(f"orig_geo_coords: {orig_geo_coords}")
        print(f"new_geo_coord: {new_geo_coord}")

        self.set_orig_coord_text(self._make_point_to_text(orig_geo_coords))
        self.set_new_coord_text(self._make_point_to_text(new_geo_coord))


    # -------------------------------------------------------------------------
    # Button handlers – currently stubs
    # -------------------------------------------------------------------------

    @Slot()
    def _on_run_rotate_scale(self) -> None:
        print("Running rotate and scale")
        # Placeholder – real implementation will apply transform.

    @Slot()
    def _on_create_translation(self) -> None:
        print("Creating translation")
        self._on_create_translated_dataset()
        # Placeholder – real implementation will apply translation.

    def _on_choose_save_filename_rs(self, checked=False):
        # TODO (Joshua G-K): Allow this to also save as an .hdr
        file_dialog = QFileDialog(parent=self,
            caption=self.tr('Save raster dataset'))

        # Restrict selection to only .tif files.
        file_dialog.setNameFilter("TIFF files (*.tif)")
        # Optionally, set a default suffix to ensure the saved file gets a .tif extension.
        file_dialog.setDefaultSuffix("tif")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_save_path_rs.text().strip()
        if len(initial_filename) > 0:
            base, ext = os.path.splitext(initial_filename)
            if ext.lower() != ".tif":
                initial_filename = f"{base}.tif"
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            selected_ds = self._rotate_scale_dataset
            if selected_ds is not None:
                selected_ds_filepaths = selected_ds.get_filepaths()
                if filename in selected_ds_filepaths:
                    QMessageBox.information(self, self.tr("Wrong Save Path"), \
                                            self.tr("The save path you chose matches either the target\n" + 
                                                    "or reference dataset's save path. Please change.\n\n"
                                                    f"Chosen save path:\n{filename}"))
                    return
            self._ui.ledit_save_path_rs.setText(filename)
    
    
    def _on_choose_save_filename_translate(self, checked=False):
        # TODO (Joshua G-K): Allow this to also save as an .hdr
        file_dialog = QFileDialog(parent=self,
            caption=self.tr('Save raster dataset'))

        # Restrict selection to only .tif files.
        file_dialog.setNameFilter("TIFF files (*.tif)")
        # Optionally, set a default suffix to ensure the saved file gets a .tif extension.
        file_dialog.setDefaultSuffix("tif")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_save_path_translate.text().strip()
        if len(initial_filename) > 0:
            base, ext = os.path.splitext(initial_filename)
            if ext.lower() != ".tif":
                initial_filename = f"{base}.tif"
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            selected_ds = self._translation_dataset
            if selected_ds is not None:
                selected_ds_filepaths = selected_ds.get_filepaths()
                print(f"selected_ds_filepaths: {selected_ds_filepaths}")
                print(f"filename: {filename}")
                if filename in selected_ds_filepaths:
                    QMessageBox.information(self, self.tr("Wrong Save Path"), \
                                            self.tr("The save path you chose matches either the target\n" + 
                                                    "or reference dataset's save path. Please change.\n\n"
                                                    f"Chosen save path:\n{filename}"))
                    return
            self._ui.ledit_save_path_translate.setText(filename)


    # -------------------------------------------------------------------------
    # Public helper methods for external callers
    # -------------------------------------------------------------------------

    def set_orig_coord_text(self, text: str) -> None:
        """Update the previous coordinate label."""
        self._ui.lbl_orig_coord_input.setText(text)

    def set_new_coord_text(self, text: str) -> None:
        """Update the current coordinate label."""
        self._ui.lbl_new_coord_input.setText(text)

    def set_lat_north_ul_text(self, text: str) -> None:
        """Update the current coordinate label."""
        self._ui.ledit_lat_north_ul.setText(text)

    def set_lon_east_ul_text(self, text: str) -> None:
        """Update the previous coordinate label."""
        self._ui.ledit_lon_east_ul.setText(text)

    def set_crs_text(self, text: str) -> None:
        """Update (read-only) CRS line-edit."""
        self._ui.ledit_crs.setText(text)

    # -------------------------------------------------------------------------
    # Convenience getters – optional but handy
    # -------------------------------------------------------------------------

    def image_rotation(self) -> float:
        return self._image_rotation

    def image_scale(self) -> float:
        return self._image_scale

    def _get_translated_geotransform(self) -> tuple[float, float]:
        origin_lon_east, pixel_w, rot_x, origin_lat_north, rot_y, pixel_h = self._translation_dataset.get_geo_transform()
        new_lon_east = origin_lon_east + self._lon_east_translate
        new_lat_north = origin_lat_north + self._lat_north_translate

        assert new_lon_east == float(self._ui.ledit_lon_east_ul.text())
        assert new_lat_north == float(self._ui.ledit_lat_north_ul.text())
        return (new_lon_east, pixel_w, rot_x, new_lat_north, rot_y, pixel_h)
    
    def _on_create_rotated_scaled_dataset(self):
        pass

    def _on_create_translated_dataset(self):
        print(f"in _on_create_translated_dataset")
        driver: gdal.Driver = gdal.GetDriverByName('GTiff')
        new_geo_transform = self._get_translated_geotransform()
        save_path = self._get_save_file_path_translate()
        print(f"save_path: \n{save_path}")
        if isinstance(self._translation_dataset.get_impl(), GDALRasterDataImpl):
        # if isinstance(self._translation_dataset, GDALRasterDataImpl):
            impl = self._translation_dataset.get_impl()
            translation_gdal_dataset = impl.gdal_dataset
            if driver is None:
                raise RuntimeError("GDAL driver not available")
            print(f"saving at save_path: {save_path}")
            new_dataset: gdal.Dataset = driver.CreateCopy(save_path, translation_gdal_dataset, 0)
            print("finished creating copy!")
            new_dataset.SetGeoTransform(new_geo_transform)
            new_dataset.FlushCache()
            print(f"finished flushing cache")
            new_dataset = None
            print(f"finished setting none!")
        else:
            print(f"in non gdal part!")
            # Create a gdal dataset with height as self._translation_dataset.get_height(), width as 
            # .get_width(), and number of bands as self._translation_dataset.num_bands(). 
            # The datatype should be self._translation_dataset.get_elem_type (which returns a np.dtype)
            # that may need to be changed into a gdal type. The save path will be obtained from
            # self._get_save_file_path_translate()
            # Example usage based on your description
            height = self._translation_dataset.get_height()
            width = self._translation_dataset.get_width()
            num_bands = self._translation_dataset.num_bands()
            np_dtype = self._translation_dataset.get_elem_type()  # Returns a numpy dtype
            gdal_data_type = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)  # Convert numpy dtype to GDAL type

            output_bytes = width * height * num_bands * np_dtype.itemsize

            
            
            ratio = MAX_RAM_BYTES / output_bytes
            print(f"about to enter if statements")
            # if ratio > 1.0:
            #     print(f" in ratio > 1.0")
            #     dataset_arr = self._translation_dataset.get_image_data() 
            #     print(f"about to make new dataset, got dataset_arr!")
            #     new_dataset: gdal.Dataset = gdal_array.OpenNumPyArray(dataset_arr, True, ['GTiff:' + save_path])
            #     new_dataset.SetGeoTransform(new_geo_transform)
            #     print(f"made new_dataset successfully!")
            # else:
            #     print(f"chunking!")

            # Create the GDAL dataset
            new_dataset = driver.Create(save_path, width, height, num_bands, gdal_data_type)
            print(f"just created new dataset")
            if new_dataset is None:
                raise RuntimeError("Failed to create the output dataset")
            num_bands_per = int(ratio * num_bands)
            for band_index in range(0, num_bands, num_bands_per):
                band_list_index = [band for band in range(band_index, band_index+num_bands_per) if band < num_bands]
                band_arr = self._translation_dataset.get_multiple_band_data(band_list_index)
                print(f"about to write {band_list_index} to raster")
                write_raster_to_dataset(new_dataset, band_list_index, band_arr, gdal_data_type)
                print(f"finished writing {band_list_index} to raster")
            copy_metadata_to_gdal_dataset(new_dataset, self._translation_dataset)
            new_dataset.FlushCache()
            new_dataset.SetSpatialRef(self._translation_dataset.get_spatial_ref())
            new_dataset.SetGeoTransform(new_geo_transform)
            new_dataset = None
            print(f"Done translating!!!")


            #  We will then use self._translation_dataset.get_image_data()
            # (which returns the full numpy array). Here is the doc string for it 
            '''
            Returns a numpy 3D array of the entire image cube.

            The numpy array is configured such that the pixel (x, y) values of band
            b are at element array[b][y][x].

            If the data-set has a "data ignore value" and filter_data_ignore_value
            is also set to True, the array will be filtered such that any element
            with the "data ignore value" will be filtered to NaN.  Note that this
            filtering will impact performance.
            '''
            '''
            We will use get_image
            '''


