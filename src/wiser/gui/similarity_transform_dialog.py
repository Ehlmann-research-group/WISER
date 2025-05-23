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

from .util import (pillow_rotate_scale_expand, cv2_rotate_scale_expand, rotate_scale_geotransform, 
        make_into_help_button, rotate_scale_geotransform_v2)

from osgeo import gdal, gdal_array

from wiser.bandmath.builtins.constants import MAX_RAM_BYTES
from wiser.bandmath.utils import write_raster_to_dataset

from wiser.raster.utils import copy_metadata_to_gdal_dataset

import numpy as np

import cv2

# from enum import Enum

# class InterpolationOptions(Enum):

INTERPOLATION_TYPES = {
    "Nearest":           cv2.INTER_NEAREST,
    "Nearest Exact":     cv2.INTER_NEAREST_EXACT,
    "Linear":            cv2.INTER_LINEAR,
    "Linear Exact":      cv2.INTER_LINEAR_EXACT,
    "Cubic":             cv2.INTER_CUBIC,
    "Area":              cv2.INTER_AREA,
    "Lanczos4":          cv2.INTER_LANCZOS4,
}

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

        self._init_interpolation_cbox()

        make_into_help_button(self._ui.tbtn_help,
                              'https://ehlmann-research-group.github.io/WISER-UserManual/Similarity_Transform/#translating-coordinate-system',
                              'Get help on translating coordinate systems')


    # -------------------------------------------------------------------------
    # Initializers
    # -------------------------------------------------------------------------

    def _init_interpolation_cbox(self):
        cbox = self._ui.cbox_interpolation
        cbox.clear()
        # populate combo with user‐friendly labels
        for label, interp_type in INTERPOLATION_TYPES.items():
            cbox.addItem(label, interp_type)
        # optionally set a default, e.g. “Linear”
        default = "Linear"
        idx = cbox.findText(default)
        if idx != -1:
            cbox.setCurrentIndex(idx)

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
    
    def _get_interpolation_type(self) -> int:
        return self._ui.cbox_interpolation.currentData()

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
        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    @Slot(int)
    def _on_slider_rotation_changed(self, value: int) -> None:
        """Slider moved by user – sync line‑edit and store integer rotation."""
        # Update line‑edit without re‑entering _on_rotation_edited.
        self._ui.ledit_rotation.blockSignals(True)
        self._ui.ledit_rotation.setText(f"{value}")
        self._ui.ledit_rotation.blockSignals(False)

        self._image_rotation = float(value)

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
        self._rotate_scale_pane.rotate_and_scale_rasterview(self._image_rotation, self._image_scale)

    # -------------------------------------------------------------------------
    # Translation handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_lat_north_changed(self) -> None:
        text = self._ui.ledit_lat_north.text()
        try:
            self._lat_north_translate = float(text)
            self._update_upper_left_coord_labels()
        except ValueError:
            pass
        self._update_prev_and_new_coords()


    @Slot()
    def _on_lon_east_changed(self) -> None:
        text = self._ui.ledit_lon_east.text()
        try:
            self._lon_east_translate = float(text)
            self._update_upper_left_coord_labels()
        except ValueError:
            pass
        self._update_prev_and_new_coords()
        

    def _make_point_to_text(self, point):
        return f"({point[0]}, {point[1]})"

    @Slot()
    def _on_rotate_scale_dataset_changed(self, ds_id):
        # We want to do many things here. But for now we just set the CRS
        self._rotate_scale_dataset = self._app_state.get_dataset(ds_id)
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
    # Button handlers
    # -------------------------------------------------------------------------

    @Slot()
    def _on_run_rotate_scale(self) -> None:
        print("Running rotate and scale")
        self._on_create_rotated_scaled_dataset()
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

    # region Convenience getters


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
    
    def _get_rotated_scaled_dataset_spatial_center(self) -> Tuple[int, int]:
        gt = self._rotate_scale_dataset.get_geo_transform()
        half_width = self._rotate_scale_dataset.get_width() / 2
        half_height = self._rotate_scale_dataset.get_height() / 2

        spatial_center_x = gt[0] + gt[1] * half_width + gt[2] * half_height
        spatial_center_y = gt[3] + gt[4] * half_width + gt[5] * half_height

        return (spatial_center_x, spatial_center_y)

    def set_translate_message_text(self, text: str):
        self._ui.lbl_translate_message.setText(text)

    def set_rotate_scale_message_text(self, text: str):
        self._ui.lbl_rotate_scale_message.setText(text)

    def _on_create_rotated_scaled_dataset(self):
        print(f"in non gdal part!")
        print(f"interp type: {self._get_interpolation_type()}")
        print(f"interp name: {self._ui.cbox_interpolation.currentText()}")
        if self._rotate_scale_dataset is None:
            QMessageBox.warning(self,
                                self.tr("Rotate/Scale Dataset Not Selected"),
                                self.tr("You have no rotate/scale dataset selected.\n" \
                                        "Please select a rotate/scale dataset."))
            return
        driver: gdal.Driver = gdal.GetDriverByName('GTiff')
        save_path = self._get_save_file_path_rs()
        if save_path is None:
            QMessageBox.warning(self,
                                self.tr("Save Path Is Empty"),
                                self.tr("The save path is empty. Please enter a save path.")
                                )
            return
        if save_path in self._rotate_scale_dataset.get_filepaths():
            QMessageBox.warning(self,
                                self.tr("Save Path Equals Dataset Path"),
                                self.tr("The save path and the dataset path are the same, so\n" \
                                        "rotating/scaling can not occur. Please fix this and\n" \
                                        "try again."))
            return
        print(f"Save_path: {save_path}")

        try:
            self.set_rotate_scale_message_text("Starting Rotate and Scale.")
            pixmap = self._rotate_scale_pane.get_rasterview().get_unscaled_pixmap()
            pixmap_height = pixmap.height()
            pixmap_width = pixmap.width()
            orig_height = self._rotate_scale_dataset.get_height()
            orig_width = self._rotate_scale_dataset.get_width()
            num_bands = self._rotate_scale_dataset.num_bands()
            np_dtype = self._rotate_scale_dataset.get_elem_type()  # Returns a numpy dtype
            gdal_data_type = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)  # Convert numpy dtype to GDAL type

            output_bytes = pixmap_width * pixmap_height * num_bands * np_dtype.itemsize
            
            ratio = MAX_RAM_BYTES / output_bytes
            print(f"about to enter if statements")
            new_dataset = driver.Create(save_path, pixmap_width, pixmap_height, num_bands, gdal_data_type)
            print(f"just created new dataset")
            if new_dataset is None:
                raise RuntimeError("Failed to create the output dataset")
            num_bands_per = int(ratio * num_bands)
            ds_data_ignore = self._rotate_scale_dataset.get_data_ignore_value()
            data_ignore = ds_data_ignore if ds_data_ignore is not None else 0
            for band_index in range(0, num_bands, num_bands_per):
                band_list_index = [band for band in range(band_index, band_index+num_bands_per) if band < num_bands]
                band_arr = self._rotate_scale_dataset.get_multiple_band_data(band_list_index)
                print(f"about to write {band_list_index} to raster")
                print(f"shape of band_arr: {band_arr.shape}")
                # np_corrected_band_arr = band_arr.reshape(orig_height, orig_width, len(band_list_index))
                print(f"before np_corrected_band_arr before: {band_arr.shape}")
                np_corrected_band_arr = np.transpose(band_arr, (1, 2, 0))
                print(f"before np_corrected_band_arr after: {np_corrected_band_arr.shape}")
                rotated_scaled_band_arr = cv2_rotate_scale_expand(np_corrected_band_arr, self._image_rotation, self._image_scale,
                                                                interp=self._get_interpolation_type(),
                                                                mask_fill_value=0)
                print(f"type of rotated_scaled_band_arr before transpose: {type(rotated_scaled_band_arr)}")
                # rotated_scaled_band_arr = rotated_scaled_band_arr.reshape(len(band_list_index), pixmap_height, pixmap_width)
                print(f"after rotated_scaled_band_arr before: {rotated_scaled_band_arr.shape}")
                rotated_scaled_band_arr = np.transpose(rotated_scaled_band_arr, (2, 0, 1))
                print(f"type of rotated_scaled_band_arr after transpose: {type(rotated_scaled_band_arr)}")
                print(f"after rotated_scaled_band_arr after: {rotated_scaled_band_arr.shape}")
                print(f"shape of rotated_scaled: {rotated_scaled_band_arr.shape}")
                if isinstance(rotated_scaled_band_arr, np.ma.masked_array):
                    print(f"WE ARE A MASKED ARRAY!!!")
                    rotated_scaled_band_arr = rotated_scaled_band_arr.filled(data_ignore)
                write_raster_to_dataset(new_dataset, band_list_index, rotated_scaled_band_arr, gdal_data_type)
                print(f"finished writing {band_list_index} to raster")
            copy_metadata_to_gdal_dataset(new_dataset, self._rotate_scale_dataset)
            new_dataset.FlushCache()
            new_dataset.SetSpatialRef(self._rotate_scale_dataset.get_spatial_ref())
            gt = self._rotate_scale_dataset.get_geo_transform()
            rotation = self._image_rotation
            scale = self._image_scale
            pivot = self._get_rotated_scaled_dataset_spatial_center()
            width = self._rotate_scale_dataset.get_width()
            height = self._rotate_scale_dataset.get_height()
            print(f"gt: {gt}")
            print(f"rotation: {rotation}")
            print(f"scale: {scale}")
            print(f"pivot: {pivot}")
            print(f"width: {width}")
            print(f"height: {height}")
            rotated_scaled_gt = rotate_scale_geotransform(gt, -rotation, scale, width, height)
            new_dataset.SetGeoTransform(rotated_scaled_gt)
            new_dataset = None
            print(f"Done rotating and scaling!!!")
            self.set_rotate_scale_message_text("Finished Rotate and Scale.")
        except BaseException as e:
            QMessageBox.critical(self,
                                 self.tr("Error While Translating Dataset"),
                                 self.tr(f"Error:\n\n{e}"))
            return
        finally:
            new_dataset = None

    def _on_create_translated_dataset(self):
        if self._translation_dataset is None:
            QMessageBox.warning(self,
                                self.tr("Translation Dataset Not Selected"),
                                self.tr("You have no translation dataset selected.\n" \
                                        "Please select a translation dataset."))
            return
        driver: gdal.Driver = gdal.GetDriverByName('GTiff')
        new_geo_transform = self._get_translated_geotransform()
        save_path = self._get_save_file_path_translate()
        if save_path is None:
            QMessageBox.warning(self,
                                self.tr("Save Path Is Empty"),
                                self.tr("The save path is empty. Please enter a save path.")
                                )
            return
        if save_path in self._translation_dataset.get_filepaths():
            QMessageBox.warning(self,
                                self.tr("Save Path Equals Dataset Path"),
                                self.tr("The save path and the dataset path are the same, so\n" \
                                        "translating can not occur. Please fix this and try again."))
            return

        try:
            self.set_translate_message_text("Starting Translation.")
            if isinstance(self._translation_dataset.get_impl(), GDALRasterDataImpl):
                impl = self._translation_dataset.get_impl()
                translation_gdal_dataset = impl.gdal_dataset
                if driver is None:
                    raise RuntimeError("GDAL driver not available")
                new_dataset: gdal.Dataset = driver.CreateCopy(save_path, translation_gdal_dataset, 0)
                new_dataset.SetGeoTransform(new_geo_transform)
                new_dataset.FlushCache()
                new_dataset = None
            else:
                height = self._translation_dataset.get_height()
                width = self._translation_dataset.get_width()
                num_bands = self._translation_dataset.num_bands()
                np_dtype = self._translation_dataset.get_elem_type()  # Returns a numpy dtype
                gdal_data_type = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)  # Convert numpy dtype to GDAL type

                output_bytes = width * height * num_bands * np_dtype.itemsize
                ratio = MAX_RAM_BYTES / output_bytes

                # Create the GDAL dataset
                new_dataset = driver.Create(save_path, width, height, num_bands, gdal_data_type)
                if new_dataset is None:
                    raise RuntimeError("Failed to create the output dataset")
                num_bands_per = int(ratio * num_bands)
                for band_index in range(0, num_bands, num_bands_per):
                    band_list_index = [band for band in range(band_index, band_index+num_bands_per) if band < num_bands]
                    band_arr = self._translation_dataset.get_multiple_band_data(band_list_index)
                    write_raster_to_dataset(new_dataset, band_list_index, band_arr, gdal_data_type)
                copy_metadata_to_gdal_dataset(new_dataset, self._translation_dataset)
                new_dataset.FlushCache()
                new_dataset.SetSpatialRef(self._translation_dataset.get_spatial_ref())
                new_dataset.SetGeoTransform(new_geo_transform)
                new_dataset = None
            self.set_translate_message_text("Finished Translation.")
        except BaseException as e:
            QMessageBox.critical(self,
                                 self.tr("Error While Translating Dataset"),
                                 self.tr(f"Error:\n\n{e}"))
            return
        finally:
            new_dataset = None

