import os

from typing import Any, Dict, List, Optional, Tuple, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.save_dataset_ui import Ui_SaveDatasetDialog
from .generated.save_dataset_basic_details_ui import Ui_SaveDatasetBasicDetails
from .generated.save_dataset_advanced_details_ui import Ui_SaveDatasetAdvancedDetails



def get_boolean_tablewidgetitem(value: bool) -> QTableWidgetItem:
    '''
    Returns a ``QTableWidgetItem`` object for displaying a Boolean flag.  The
    initial value of the flag is specified as the ``value`` argument.
    '''
    twi = QTableWidgetItem()
    twi.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
    if value:
        twi.setCheckState(Qt.Checked)
    else:
        twi.setCheckState(Qt.Unchecked)

    # twi.setStyleSheet('margin-left:50%; margin-right:50%;')
    twi.setTextAlignment(Qt.AlignHCenter)
    return twi


def get_defaultband_tablewidgetitem(band_index: int, defaults: Union[Tuple[int], Tuple[int, int, int]]):
    '''
    Returns a ``QTableWidgetItem`` object for displaying the "default display
    bands" information.
    '''
    txt = ''
    if defaults is not None:
        if len(defaults) == 1 and band_index == defaults[0]:
            txt = 'Grayscale'

        elif len(defaults) == 3:
            try:
                # Look up whether the band-index is red, green, blue - or not in
                # the defaults at all.
                txt = ['Red', 'Green', 'Blue'][defaults.index(band_index)]
            except ValueError:
                pass

    twi = QTableWidgetItem(txt)
    return twi



class SaveDatasetDetails(QWidget):
    '''
    A base-class for both of the save-dataset widgets, since they share some
    basic behavior.
    '''
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

        self._ui.cbox_save_format.addItem('ENVI')

        self._ui.ledit_filename.editingFinished.connect(self._on_edit_save_filename)
        self._ui.btn_filename.clicked.connect(self._on_choose_save_filename)


    def _configure_ui(self):
        '''
        Load configuration from the datset into the UI widgets.  The base
        implementation only populates the common widgets.
        '''
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
            self._ui.cbox_dataset.addItem(self._dataset.get_name(), self._dataset.get_id())

        self._update_save_filenames()


    def _on_edit_save_filename(self):
        '''
        A handler for when editing is finished on the "save-filename" field.
        '''
        self._update_save_filenames()


    def _on_choose_save_filename(self, checked=False):
        '''
        A handler for when the file-chooser for the "save-filename" is shown.
        '''

        # TODO(donnie):  Do we want a filter on this dialog?
        file_dialog = QFileDialog(parent=self,
            caption=self.tr('Save raster dataset'))

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
        '''
        When the "save-filename" field changes, the "writes to" field is updated
        with the actual files that will be written to.
        '''
        self._ui.lbl_filenames_value.clear()

        path = self._ui.ledit_filename.text().strip()
        has_path = (len(path) > 0)

        self._ui.lbl_writes_to.setVisible(has_path)
        self._ui.lbl_filenames_value.setVisible(has_path)
        if not has_path:
            return

        format = self._ui.cbox_save_format.currentText()

        loader = self._app_state.get_loader()
        filenames = loader.get_save_filenames(path, format)

        display_filenames = '\n'.join([os.path.basename(filename) for filename in filenames])
        self._ui.lbl_filenames_value.setText(display_filenames)



class SaveDatasetBasicDetails(SaveDatasetDetails):
    def __init__(self, app_state, ds_id, parent=None):
        super().__init__(Ui_SaveDatasetBasicDetails(),
            app_state, ds_id, parent=parent)

    def set_config(self, config: Dict):
        pass

    def get_config():
        pass


class SaveDatasetAdvancedDetails(SaveDatasetDetails):
    def __init__(self, app_state, ds_id, parent=None):
        super().__init__(Ui_SaveDatasetAdvancedDetails(),
            app_state, ds_id, parent=parent)

        self._ui.cbox_dataset.activated.connect(self._on_dataset_changed)

        self._ui.gbox_default_bands.clicked.connect(self._on_default_bands)
        self._ui.rb_rgb_default_bands.clicked.connect(self._on_rgb_default_bands)
        self._ui.rb_gray_default_bands.clicked.connect(self._on_gray_default_bands)

    def _configure_ui(self):
        # Do the basic configuration first.
        super()._configure_ui()

        # Handle all the advanced configuration now.
        self._show_dataset_in_ui()


    def _on_dataset_changed(self, index):
        ds_id = self._ui.cbox_dataset.currentData()
        self._dataset = self._app_state.get_dataset(ds_id)
        self._show_dataset_in_ui()


    def _on_default_bands(self, checked=False):
        for w in [self._ui.rb_rgb_default_bands,
                  self._ui.rb_gray_default_bands,
                  self._ui.lbl_default_red_band,
                  self._ui.lbl_default_green_band,
                  self._ui.lbl_default_blue_band,
                  self._ui.cbox_default_red_band,
                  self._ui.cbox_default_green_band,
                  self._ui.cbox_default_blue_band,
                  self._ui.lbl_default_gray_band,
                  self._ui.cbox_default_gray_band,
                  self._ui.btn_choose_visible_light_bands]:
            w.setEnabled(checked)


    def _on_rgb_default_bands(self, checked=False):
        if checked:
            self._ui.stk_default_bands.setCurrentIndex(0)


    def _on_gray_default_bands(self, checked=False):
        if checked:
            self._ui.stk_default_bands.setCurrentIndex(1)


    def _show_dataset_in_ui(self):
        (width, height, bands) = self._dataset.get_shape()
        defaults = self._dataset.default_display_bands()

        # General information

        self._ui.ledit_description.setText(self._dataset.get_description())
        self._ui.ledit_description.setCursorPosition(0)

        s = ''
        v = self._dataset.get_data_ignore_value()
        if v is not None:
            s = str(v)

        self._ui.ledit_data_ignore_value.setText(s)

        # Image dimensions

        self._ui.lbl_src_dims_value.setText(f'{width} x {height}')

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
        #self._ui.tbl_bands.setRowCount(0)

        bad_bands = self._dataset.get_bad_bands()
        for band_info in self._dataset.band_list():
            index = band_info['index']

            self._ui.tbl_bands.insertRow(index)

            # Column 0:  Include the band?
            self._ui.tbl_bands.setItem(index, 0,
                get_boolean_tablewidgetitem(True))

            # Column 1:  Is it a bad band?
            self._ui.tbl_bands.setItem(index, 1,
                get_boolean_tablewidgetitem(bad_bands[index] == 0))

            # Column 2:  Is it a default display band?  If so, what kind?
            # self._ui.tbl_bands.setItem(index, 2,
            #     get_defaultband_tablewidgetitem(index, defaults))

            # Column 2:  Band name / wavelength
            self._ui.tbl_bands.setItem(index, 2, QTableWidgetItem(band_info['description']))

        # Default display bands

        # Always populate the comboboxes, regardless of whether we have defaults
        # already.
        for cbox in [self._ui.cbox_default_red_band,
                     self._ui.cbox_default_green_band,
                     self._ui.cbox_default_blue_band,
                     self._ui.cbox_default_gray_band]:

            for band_info in self._dataset.band_list():
                cbox.addItem(band_info['description'])

        has_defaults = (defaults is not None)

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
                raise ValueError(f'Default-display-bands {defaults} length is unexpected')

        else:
            # Make sure at least one of the radio-buttons is checked.
            self._ui.rb_rgb_default_bands.setChecked(True)


class SaveDatasetDialog(QDialog):
    '''
    A dialog for saving a dataset, with optional features for subsetting and
    editing metadata.
    '''

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
        self._ui.btn_toggle_advanced.setText(self.tr('Show Advanced...'))
        # self.adjustSize()

        self._ui.btn_toggle_advanced.clicked.connect(self._on_toggle_advanced)


    def _on_toggle_advanced(self, checked=False):
        self._ui.wgt_details.setCurrentIndex(1 - self._ui.wgt_details.currentIndex())
        self._basic_mode = not self._basic_mode
        if self._basic_mode:
            self._ui.wgt_details.setCurrentIndex(0)
            self._ui.btn_toggle_advanced.setText(self.tr('Show Advanced...'))

        else:
            self._ui.wgt_details.setCurrentIndex(1)
            self._ui.btn_toggle_advanced.setText(self.tr('Show Basic...'))

        # self.adjustSize()


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
            self._ui.cbox_dataset.addItem(self._dataset.get_name(), self._dataset.get_id())

        self._update_save_filenames()


    def _on_edit_save_filename(self):
        self._update_save_filenames()


    def _on_choose_save_filename(self, checked=False):

        # TODO(donnie):  Do we want a filter on this dialog?
        file_dialog = QFileDialog(parent=self,
            caption=self.tr('Save raster dataset'))

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
        has_path = (len(path) > 0)

        self._ui.lbl_writes_to.setVisible(has_path)
        self._ui.lbl_filenames_value.setVisible(has_path)
        if not has_path:
            return

        format = self._ui.cbox_save_format.currentText()

        loader = self._app_state.get_loader()
        filenames = loader.get_save_filenames(path, format)

        display_filenames = '\n'.join([os.path.basename(filename) for filename in filenames])
        self._ui.lbl_filenames_value.setText(display_filenames)


    def accept(self):

        path = self._ui.ledit_filename.text().strip()
        if not path:
            QMessageBox.warning(self, self.tr('Missing filename'),
                self.tr('Filename must be specified'))
            return

        super().accept()


    def get_save_path(self) -> Optional[str]:
        path = self._ui.ledit_filename.text().strip()
        if path:
            return path

        return None


    def get_save_format(self) -> str:
        return self._ui.cbox_save_format.currentText()
