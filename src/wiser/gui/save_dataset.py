import os

from typing import Any, Dict, List, Optional, Tuple, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u

from .generated.save_dataset_ui import Ui_SaveDatasetDialog
from .generated.save_dataset_basic_details_ui import Ui_SaveDatasetBasicDetails
from .generated.save_dataset_advanced_details_ui import Ui_SaveDatasetAdvancedDetails

from wiser.raster.dataset import RasterDataSet, find_truecolor_bands
from wiser.raster.utils import spectral_unit_to_string


def get_boolean_tablewidgetitem(value: bool) -> QTableWidgetItem:
    """
    Returns a ``QTableWidgetItem`` object for displaying a Boolean flag.  The
    initial value of the flag is specified as the ``value`` argument.
    """
    twi = QTableWidgetItem()
    twi.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
    if value:
        twi.setCheckState(Qt.Checked)
    else:
        twi.setCheckState(Qt.Unchecked)

    # twi.setStyleSheet('margin-left:50%; margin-right:50%;')
    twi.setTextAlignment(Qt.AlignHCenter)
    return twi


def get_defaultband_tablewidgetitem(
    band_index: int, defaults: Union[Tuple[int], Tuple[int, int, int]]
):
    """
    Returns a ``QTableWidgetItem`` object for displaying the "default display
    bands" information.
    """
    txt = ""
    if defaults is not None:
        if len(defaults) == 1 and band_index == defaults[0]:
            txt = "Grayscale"

        elif len(defaults) == 3:
            try:
                # Look up whether the band-index is red, green, blue - or not in
                # the defaults at all.
                txt = ["Red", "Green", "Blue"][defaults.index(band_index)]
            except ValueError:
                pass

    twi = QTableWidgetItem(txt)
    return twi


class SaveDatasetDetails(QWidget):
    """
    A base-class for both of the save-dataset widgets, since they share some
    basic behavior.
    """

    def __init__(self, ui, app_state, ds_id, parent=None):
        super().__init__(parent=parent)
        self._ui = ui
        self._ui.setupUi(self)

        self._app_state = app_state

        self._ds_id: Optional[int] = ds_id

        self._dataset: Optional[RasterDataSet] = None
        self._choosable_dataset: bool = True

        # Load and show values from the dataset

        if ds_id is not None:
            self._dataset = self._app_state.get_dataset(ds_id)
            self._choosable_dataset = False

        self._configure_ui()

        # Configure the UI components

        self._ui.cbox_save_format.addItem("ENVI")

        self._ui.ledit_filename.editingFinished.connect(self._on_edit_save_filename)
        self._ui.btn_filename.clicked.connect(self._on_choose_save_filename)

        self._load_dataset_details()

    def _configure_ui(self):
        pass

    def _load_dataset_details(self):
        """
        Load configuration from the datset into the UI widgets.  The base
        implementation only populates the common widgets.
        """
        # TODO(donnie):  If user can choose a dataset, populate dataset combo-box
        # TODO(donnie):  If dataset is specified, select it in the combo-box

        if self._choosable_dataset:
            # Add all the datasets to the combobox.
            for dataset in self._app_state.get_datasets():
                self._ui.cbox_dataset.addItem(dataset.get_name(), dataset.get_id())

                # We need to have *some* initial dataset in this dialog.
                if self._dataset is None:
                    self._dataset = dataset

            # Make sure the combobox displays the current dataset being considered.
            index = self._ui.cbox_dataset.findData(self._dataset.get_id())
            if index != -1:
                self._ui.cbox_dataset.setCurrentIndex(index)
        else:
            # Just add the one dataset to the combobox.
            self._ui.cbox_dataset.addItem(
                self._dataset.get_name(), self._dataset.get_id()
            )

        self._update_save_filenames()

    def _on_edit_save_filename(self):
        """
        A handler for when editing is finished on the "save-filename" field.
        """
        self._update_save_filenames()

    def _on_choose_save_filename(self, checked=False):
        """
        A handler for when the file-chooser for the "save-filename" is shown.
        """

        # TODO(donnie):  Do we want a filter on this dialog?
        file_dialog = QFileDialog(parent=self, caption=self.tr("Save raster dataset"))

        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_filename.text().strip()
        if len(initial_filename) > 0:
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            self._ui.ledit_filename.setText(filename)

            self._update_save_filenames()

    def _update_save_filenames(self):
        """
        When the "save-filename" field changes, the "writes to" field is updated
        with the actual files that will be written to.
        """
        self._ui.lbl_filenames_value.clear()

        path = self._ui.ledit_filename.text().strip()
        has_path = len(path) > 0

        self._ui.lbl_writes_to.setVisible(has_path)
        self._ui.lbl_filenames_value.setVisible(has_path)
        if not has_path:
            return

        format = self._ui.cbox_save_format.currentText()

        loader = self._app_state.get_loader()
        filenames = loader.get_save_filenames(path, format)

        display_filenames = "\n".join(
            [os.path.basename(filename) for filename in filenames]
        )
        self._ui.lbl_filenames_value.setText(display_filenames)

    def verify_config(self) -> bool:
        path = self._ui.ledit_filename.text().strip()
        if not path:
            QMessageBox.warning(
                self, self.tr("Missing filename"), self.tr("Filename must be specified")
            )
            return False

        return True

    def set_config(self, config: Dict):
        source_ds_id = config.get("source_ds_id")
        if source_ds_id:
            # Update the dataset display/config.

            if self._ds_id != source_ds_id:
                self._ds_id = source_ds_id
                self._dataset = self._app_state.get_dataset(source_ds_id)

                index = self._ui.cbox_dataset.findData(source_ds_id)
                if index != -1:
                    self._ui.cbox_dataset.setCurrentIndex(index)
                else:
                    # Hmm weird that the dataset wouldn't already be somewhere
                    # in the combobox.  Just add it.
                    self._ui.cbox_dataset.addItem(dataset.get_name(), dataset.get_id())
                    self._ui.cbox_dataset.setCurrentIndex(self._ui.cbox_dataset.count())

        self._ui.ledit_filename.setText(config.get("path"))

        format = config.get("format", "ENVI")
        index = self._ui.cbox_save_format.findText(format)
        if index == -1:
            index = 0
        self._ui.cbox_save_format.setCurrentIndex(index)

        self._update_save_filenames()

    def get_config(self) -> Dict[str, Any]:
        return {
            "source_ds_id": self._ui.cbox_dataset.currentData(),
            "path": self._ui.ledit_filename.text().strip(),
            "format": self._ui.cbox_save_format.currentText(),
        }


class SaveDatasetBasicDetails(SaveDatasetDetails):
    def __init__(self, app_state, ds_id, parent=None):
        super().__init__(Ui_SaveDatasetBasicDetails(), app_state, ds_id, parent=parent)


class SaveDatasetAdvancedDetails(SaveDatasetDetails):
    def __init__(self, app_state, ds_id, parent=None):
        super().__init__(
            Ui_SaveDatasetAdvancedDetails(), app_state, ds_id, parent=parent
        )

    def _configure_ui(self):
        """
        This helper method takes care of GENERAL (i.e. NOT dataset-specific)
        configuration of the user interface, such as populating general data
        values, hooking up event-handlers, etc.

        Dataset-specific configuration should be in the _load_dataset_details()
        method.
        """
        super()._configure_ui()

        # ========================================
        # General tab

        self._ui.ledit_data_ignore_value.setValidator(QDoubleValidator())

        # ========================================
        # Dimensions tab

        # None needed

        # ========================================
        # Bands tab

        # Populate the "wavelength-units" combobox.
        self._ui.cbox_wavelength_units.addItem(self.tr("No units"), None)
        self._ui.cbox_wavelength_units.addItem(self.tr("Meters"), u.m)
        self._ui.cbox_wavelength_units.addItem(self.tr("Centimeters"), u.cm)
        self._ui.cbox_wavelength_units.addItem(self.tr("Millimeters"), u.mm)
        self._ui.cbox_wavelength_units.addItem(self.tr("Micrometers"), u.micrometer)
        self._ui.cbox_wavelength_units.addItem(self.tr("Nanometers"), u.nm)
        self._ui.cbox_wavelength_units.addItem(self.tr("Microns"), u.micron)
        self._ui.cbox_wavelength_units.addItem(self.tr("Angstroms"), u.angstrom)
        self._ui.cbox_wavelength_units.addItem(self.tr("Wavenumber"), u.cm**-1)
        self._ui.cbox_wavelength_units.addItem(self.tr("MHz"), u.MHz)
        self._ui.cbox_wavelength_units.addItem(self.tr("GHz"), u.GHz)

        # TODO(donnie):  Add back in the future
        self._ui.tabWidget.removeTab(
            self._ui.tabWidget.indexOf(self._ui.tab_projection)
        )

        # Hook up event handlers

        self._ui.cbox_dataset.activated.connect(self._on_dataset_changed)

        self._ui.tbtn_include_all_bands.clicked.connect(self._on_include_all_bands)
        self._ui.tbtn_exclude_all_bands.clicked.connect(self._on_exclude_all_bands)

        self._ui.rb_rgb_default_bands.clicked.connect(self._on_rgb_default_bands)
        self._ui.rb_gray_default_bands.clicked.connect(self._on_gray_default_bands)
        self._ui.btn_choose_visible_light_bands.clicked.connect(
            self._on_choose_visible_light_bands
        )

    def _load_dataset_details(self):
        # Do the basic configuration first.
        super()._load_dataset_details()

        # Handle all the advanced configuration now.
        self._show_dataset_in_ui()

    def _on_dataset_changed(self, index):
        ds_id = self._ui.cbox_dataset.currentData()
        self._dataset = self._app_state.get_dataset(ds_id)
        self._show_dataset_in_ui()

    def _on_include_all_bands(self, checked=False):
        for i in range(self._ui.tbl_bands.rowCount()):
            twi = self._ui.tbl_bands.item(i, 0)
            twi.setCheckState(Qt.Checked)

    def _on_exclude_all_bands(self, checked=False):
        for i in range(self._ui.tbl_bands.rowCount()):
            twi = self._ui.tbl_bands.item(i, 0)
            twi.setCheckState(Qt.Unchecked)

    def _on_rgb_default_bands(self, checked=False):
        if checked:
            self._ui.stk_default_bands.setCurrentIndex(0)

    def _on_gray_default_bands(self, checked=False):
        if checked:
            self._ui.stk_default_bands.setCurrentIndex(1)

    def _show_dataset_in_ui(self):
        (bands, height, width) = self._dataset.get_shape()
        defaults = self._dataset.default_display_bands()

        # General information

        self._ui.ledit_description.setText(self._dataset.get_description())
        self._ui.ledit_description.setCursorPosition(0)

        s = ""
        v = self._dataset.get_data_ignore_value()
        if v is not None:
            s = str(v)

        self._ui.ledit_data_ignore_value.setText(s)

        # ========================================
        # Dimensions tab

        self._ui.lbl_src_dims_value.setText(f"{width} x {height}")

        self._ui.sbox_left.setRange(0, width - 1)
        self._ui.sbox_left.setValue(0)

        self._ui.sbox_top.setRange(0, height - 1)
        self._ui.sbox_top.setValue(0)

        self._ui.sbox_width.setRange(1, width)
        self._ui.sbox_width.setValue(width)

        self._ui.sbox_height.setRange(1, height)
        self._ui.sbox_height.setValue(height)

        # Image bands

        self._ui.tbl_bands.clearContents()
        # TODO(donnie) - Needed?
        # self._ui.tbl_bands.setRowCount(0)

        bad_bands = self._dataset.get_bad_bands()
        for band_info in self._dataset.band_list():
            index = band_info["index"]

            self._ui.tbl_bands.insertRow(index)

            # Column 0:  Include the band?
            self._ui.tbl_bands.setItem(index, 0, get_boolean_tablewidgetitem(True))

            # Column 1:  Is it a bad band?
            self._ui.tbl_bands.setItem(
                index, 1, get_boolean_tablewidgetitem(bad_bands[index] == 0)
            )

            # Column 2:  Band name / wavelength
            if self._dataset.has_wavelengths():
                desc = band_info["wavelength_str"]
            else:
                desc = band_info["description"]

            self._ui.tbl_bands.setItem(index, 2, QTableWidgetItem(desc))

        # Band units

        band_units = self._dataset.get_band_unit()
        index = self._ui.cbox_wavelength_units.findData(band_units)
        if index == -1:
            index = 0

        self._ui.cbox_wavelength_units.setCurrentIndex(index)

        # Default display bands

        # Always populate the comboboxes, regardless of whether we have defaults
        # already.
        for cbox in [
            self._ui.cbox_default_red_band,
            self._ui.cbox_default_green_band,
            self._ui.cbox_default_blue_band,
            self._ui.cbox_default_gray_band,
        ]:
            for band_info in self._dataset.band_list():
                cbox.addItem(band_info["description"])

        has_defaults = defaults is not None

        self._ui.gbox_default_bands.setChecked(has_defaults)
        if has_defaults:
            # Configure the widgets based on whether we have RGB defaults or
            # grayscale defaults
            if len(defaults) == 3:
                self._ui.rb_rgb_default_bands.setChecked(True)
                self._ui.stk_default_bands.setCurrentIndex(0)
                self._ui.cbox_default_red_band.setCurrentIndex(defaults[0])
                self._ui.cbox_default_green_band.setCurrentIndex(defaults[1])
                self._ui.cbox_default_blue_band.setCurrentIndex(defaults[2])

            elif len(defaults) == 1:
                self._ui.rb_gray_default_bands.setChecked(True)
                self._ui.stk_default_bands.setCurrentIndex(1)
                self._ui.cbox_default_gray_band.setCurrentIndex(defaults[0])

            else:
                raise ValueError(
                    f"Default-display-bands {defaults} length is unexpected"
                )

        else:
            # Make sure at least one of the radio-buttons is checked.
            self._ui.rb_rgb_default_bands.setChecked(True)
            self._ui.stk_default_bands.setCurrentIndex(0)

        truecolor_bands = find_truecolor_bands(
            self._dataset,
            red=self._app_state.get_config("general.red_wavelength_nm") * u.nm,
            green=self._app_state.get_config("general.green_wavelength_nm") * u.nm,
            blue=self._app_state.get_config("general.blue_wavelength_nm") * u.nm,
        )

        self._ui.btn_choose_visible_light_bands.setEnabled(truecolor_bands is not None)

    def _on_choose_visible_light_bands(self, checked=False):
        truecolor_bands = find_truecolor_bands(
            self._dataset,
            red=self._app_state.get_config("general.red_wavelength_nm") * u.nm,
            green=self._app_state.get_config("general.green_wavelength_nm") * u.nm,
            blue=self._app_state.get_config("general.blue_wavelength_nm") * u.nm,
        )

        self._ui.cbox_default_red_band.setCurrentIndex(truecolor_bands[0])
        self._ui.cbox_default_green_band.setCurrentIndex(truecolor_bands[1])
        self._ui.cbox_default_blue_band.setCurrentIndex(truecolor_bands[2])

    def verify_config(self) -> bool:
        # ========================================
        # General tab

        path = self._ui.ledit_filename.text().strip()
        if not path:
            self._ui.tabWidget.setCurrentWidget(self._ui.tab_general)
            QMessageBox.warning(
                self, self.tr("Missing filename"), self.tr("Filename must be specified")
            )
            return False

        # ========================================
        # Dimensions tab

        if self._ui.gbox_spatial_subset.isChecked():
            # Input dimensions
            (bands, height, width) = self._dataset.get_shape()

            output_left = self._ui.sbox_left.value()
            output_top = self._ui.sbox_top.value()
            output_width = self._ui.sbox_width.value()
            output_height = self._ui.sbox_height.value()

            if not (0 <= output_left < width):
                self._ui.tabWidget.setCurrentWidget(self._ui.tab_dimensions)
                QMessageBox.warning(
                    self,
                    self.tr("Bad Dimensions"),
                    self.tr("Left-value must be between 0 and the image width"),
                )
                return False

            if not (0 <= output_top < height):
                self._ui.tabWidget.setCurrentWidget(self._ui.tab_dimensions)
                QMessageBox.warning(
                    self,
                    self.tr("Bad Dimensions"),
                    self.tr("Top-value must be between 0 and the image height"),
                )
                return False

            if not (0 < output_left + output_width <= width):
                self._ui.tabWidget.setCurrentWidget(self._ui.tab_dimensions)
                QMessageBox.warning(
                    self,
                    self.tr("Bad Dimensions"),
                    self.tr(
                        "Left-value and output-width must sum to less than the image width"
                    ),
                )
                return False

            if not (0 < output_top + output_height <= height):
                self._ui.tabWidget.setCurrentWidget(self._ui.tab_dimensions)
                QMessageBox.warning(
                    self,
                    self.tr("Bad Dimensions"),
                    self.tr(
                        "Top-value and output-height must sum to less than the image height"
                    ),
                )
                return False

        # ========================================
        # Bands tab

        band_units: Optional[u.Unit] = self._ui.cbox_wavelength_units.currentData()

        count_included = 0
        for i in range(self._ui.tbl_bands.rowCount()):
            include_band = self._ui.tbl_bands.item(i, 0).checkState() == Qt.Checked
            if include_band:
                count_included += 1

            if band_units is not None:
                band_name = self._ui.tbl_bands.item(i, 2).text()
                try:
                    wl = float(band_name)
                except ValueError:
                    self._ui.tabWidget.setCurrentWidget(self._ui.tab_bands)
                    self._ui.tbl_bands.showRow(i)
                    # Add 1 to the index, since the Qt table will have 1-based
                    # row numbers
                    QMessageBox.warning(
                        self,
                        self.tr("Bad Band-Wavelength"),
                        self.tr(f"Band {i + 1} name must be a number"),
                    )
                    return False

        if count_included == 0:
            self._ui.tabWidget.setCurrentWidget(self._ui.tab_bands)
            QMessageBox.warning(
                self,
                self.tr("No Bands Included"),
                self.tr(
                    "No bands are included in the output.\n"
                    + "Please mark at least one band to be included."
                ),
            )
            return False

        # ========================================
        # Display Bands tab

        if self._ui.gbox_default_bands.isChecked():
            # Pull out the default display bands
            defaults = None
            if self._ui.rb_rgb_default_bands.isChecked():
                defaults = (
                    self._ui.cbox_default_red_band.currentIndex(),
                    self._ui.cbox_default_green_band.currentIndex(),
                    self._ui.cbox_default_blue_band.currentIndex(),
                )

            else:
                assert self._ui.rb_gray_default_bands.isChecked()
                defaults = (self._ui.cbox_default_gray_band.currentIndex(),)

            # Make sure that every default display-band is actually marked as
            # included in the output.
            for i in defaults:
                include_band = self._ui.tbl_bands.item(i, 0).checkState() == Qt.Checked

                if not include_band:
                    self._ui.tabWidget.setCurrentWidget(self._ui.tab_display_bands)
                    # Add 1 to the index, since the Qt table will have 1-based
                    # row numbers
                    QMessageBox.warning(
                        self,
                        self.tr("Display Band Not Included"),
                        self.tr(
                            f"Band {i + 1} is set as a default display band,\n"
                            + "but is also not being included in the output."
                        ),
                    )
                    return False

        # ========================================
        # Projection tab

        # Nothing to do on this tab.

        return True

    def get_config(self) -> Dict[str, Any]:
        # Let the superclass implementation get the basic info into a dictionary
        config = super().get_config()

        # ========================================
        # General tab

        s = self._ui.ledit_description.text().strip()
        if s:
            config["description"] = s

        s = self._ui.ledit_data_ignore_value.text()
        if s:
            config["data_ignore"] = float(s)

        # ========================================
        # Dimensions tab

        if self._ui.gbox_spatial_subset.isChecked():
            (bands, height, width) = self._dataset.get_shape()

            output_left = self._ui.sbox_left.value()
            output_top = self._ui.sbox_top.value()
            output_width = self._ui.sbox_width.value()
            output_height = self._ui.sbox_height.value()

            config.update(
                {
                    "left": output_left,
                    "top": output_top,
                    "width": output_width,
                    "height": output_height,
                }
            )

        # ========================================
        # Bands tab

        config.update(self._get_ui_band_info())

        # ========================================
        # Display Bands tab

        if self._ui.gbox_default_bands.isChecked():
            # Pull out the default display bands.  Note that the indexes are
            # relative to the original image's bands, not the target image's
            # bands, as some bands may not be included in the target image.
            defaults = None
            if self._ui.rb_rgb_default_bands.isChecked():
                defaults = (
                    self._ui.cbox_default_red_band.currentIndex(),
                    self._ui.cbox_default_green_band.currentIndex(),
                    self._ui.cbox_default_blue_band.currentIndex(),
                )

            else:
                assert self._ui.rb_gray_default_bands.isChecked()
                defaults = (self._ui.cbox_default_gray_band.currentIndex(),)

            config["default_display_bands"] = defaults

        # ========================================
        # Projection tab

        # TODO(donnie):  ?

        return config

    def _get_ui_band_info(self):
        """
        This helper function retrieves the band-configuration information from
        the advanced user interface, and returns it as a dictionary of key-value
        pairs.
        """

        band_config = {}

        band_units: Optional[u.Unit] = self._ui.cbox_wavelength_units.currentData()
        band_units_str: Optional[str] = spectral_unit_to_string(band_units)

        names_key: str = "names"
        if band_units is not None:
            names_key = "wavelengths"
            band_config["wavelength_units"] = band_units_str

        include_bands: List[bool] = []
        bad_bands: List[bool] = []
        band_names: List[str] = []

        need_includes = False
        for i in range(self._ui.tbl_bands.rowCount()):
            include_band = self._ui.tbl_bands.item(i, 0).checkState() == Qt.Checked
            include_bands.append(include_band)
            if not include_band:
                need_includes = True

            bad_band = self._ui.tbl_bands.item(i, 1).checkState() == Qt.Checked
            bad_bands.append(bad_band)

            band_name = self._ui.tbl_bands.item(i, 2).text()

            if band_units is not None:
                # Try to parse the band-name into a number.
                band_name = float(band_name)

            band_names.append(band_name)

        band_config[names_key] = band_names
        band_config["bad_bands"] = bad_bands
        if need_includes:
            band_config["include_bands"] = include_bands

        return band_config


class SaveDatasetDialog(QDialog):
    """
    A dialog for saving a dataset, with optional features for subsetting and
    editing metadata.
    """

    def __init__(self, app_state, ds_id: Optional[int] = None, parent=None):
        super().__init__(parent=parent)
        self._ui = Ui_SaveDatasetDialog()
        self._ui.setupUi(self)

        # Configure UI components

        self._basic_mode: bool = True

        self._basic_details = SaveDatasetBasicDetails(app_state, ds_id)
        self._advanced_details = SaveDatasetAdvancedDetails(app_state, ds_id)

        self._ui.wgt_details.insertWidget(0, self._basic_details)
        self._ui.wgt_details.insertWidget(1, self._advanced_details)
        self._ui.wgt_details.setCurrentIndex(0)
        self._ui.btn_toggle_advanced.setText(self.tr("Show Advanced..."))
        # self.adjustSize()

        self._ui.btn_toggle_advanced.clicked.connect(self._on_toggle_advanced)

    def _on_toggle_advanced(self, checked=False):
        old_widget = self._current_detail_widget()
        config = old_widget.get_config()

        self._ui.wgt_details.setCurrentIndex(1 - self._ui.wgt_details.currentIndex())
        self._basic_mode = not self._basic_mode
        if self._basic_mode:
            self._ui.wgt_details.setCurrentIndex(0)
            self._ui.btn_toggle_advanced.setText(self.tr("Show Advanced..."))

        else:
            self._ui.wgt_details.setCurrentIndex(1)
            self._ui.btn_toggle_advanced.setText(self.tr("Show Basic..."))

        # self.adjustSize()

        new_widget = self._current_detail_widget()
        new_widget.set_config(config)

    def _configure_ui(self):
        # TODO(donnie):  If user can choose a dataset, populate dataset combo-box
        # TODO(donnie):  If dataset is specified, select it in the combo-box

        if self._choosable_dataset:
            # Add all the datasets to the combobox.
            for dataset in self._app_state.get_datasets():
                self._ui.cbox_dataset.addItem(dataset.get_name(), dataset.get_id())

                # We need to have *some* initial dataset in this dialog.
                if self._dataset is None:
                    self._dataset = dataset

            # Make sure the combobox displays the current dataset being considered.
            index = self._ui.cbox_dataset.findData(self._dataset.get_id())
            if index != -1:
                self._ui.cbox_dataset.setCurrentIndex(index)
        else:
            # Just add the one dataset to the combobox.
            self._ui.cbox_dataset.addItem(
                self._dataset.get_name(), self._dataset.get_id()
            )

        self._update_save_filenames()

    def _on_edit_save_filename(self):
        self._update_save_filenames()

    def _on_choose_save_filename(self, checked=False):
        # TODO(donnie):  Do we want a filter on this dialog?
        file_dialog = QFileDialog(parent=self, caption=self.tr("Save raster dataset"))

        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)

        # If there is already an initial filename, select it in the dialog.
        initial_filename = self._ui.ledit_filename.text().strip()
        if len(initial_filename) > 0:
            file_dialog.selectFile(initial_filename)

        result = file_dialog.exec()
        if result == QDialog.Accepted:
            filename = file_dialog.selectedFiles()[0]
            self._ui.ledit_filename.setText(filename)

            self._update_save_filenames()

    def _update_save_filenames(self):
        self._ui.lbl_filenames_value.clear()

        path = self._ui.ledit_filename.text().strip()
        has_path = len(path) > 0

        self._ui.lbl_writes_to.setVisible(has_path)
        self._ui.lbl_filenames_value.setVisible(has_path)
        if not has_path:
            return

        format = self._ui.cbox_save_format.currentText()

        loader = self._app_state.get_loader()
        filenames = loader.get_save_filenames(path, format)

        display_filenames = "\n".join(
            [os.path.basename(filename) for filename in filenames]
        )
        self._ui.lbl_filenames_value.setText(display_filenames)

    def accept(self):
        # If the current detail-widget says the config is good, accept it!
        if self._current_detail_widget().verify_config():
            super().accept()

    def _current_detail_widget(self):
        if self._basic_mode:
            return self._basic_details
        else:
            return self._advanced_details

    def get_config(self) -> Dict[str, Any]:
        return self._current_detail_widget().get_config()

    def get_save_path(self) -> Optional[str]:
        config = self.get_config()
        return config["path"]

    def get_save_format(self) -> str:
        config = self.get_config()
        return config["format"]
