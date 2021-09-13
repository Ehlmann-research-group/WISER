import os

from typing import Any, List, Optional, Tuple, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.save_dataset_advanced_ui import Ui_AdvancedSaveDatasetDialog


class DefaultBandItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem,
                     index: QModelIndex) -> QWidget:
        cbox = QComboBox(parent=parent)
        cbox.addItem('(none)')
        cbox.insertSeparator(cbox.count())
        cbox.addItem('Red')
        cbox.addItem('Green')
        cbox.addItem('Blue')
        cbox.insertSeparator(cbox.count())
        cbox.addItem('Grayscale')
        return cbox

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        pass


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


class AdvancedSaveDatasetDialog(QDialog):
    '''
    A dialog for saving a dataset, allowing the user to specify advanced options
    such as what are the default bands, saving a subset of the image, etc.
    '''

    def __init__(self, app_state, ds_id=None, parent=None):

        super().__init__(parent=parent)
        self._ui = Ui_AdvancedSaveDatasetDialog()
        self._ui.setupUi(self)

        # Configure UI components

        self._ui.cbox_save_format.addItem('ENVI')

        self._ui.ledit_filename.editingFinished.connect(self._on_edit_save_filename)
        self._ui.btn_filename.clicked.connect(self._on_choose_save_filename)

        self._default_band_delegate = DefaultBandItemDelegate()
        self._ui.tbl_bands.setItemDelegateForColumn(2, self._default_band_delegate)

        self._app_state: ApplicationState = app_state
        self._dataset: Optional[RasterDataSet] = None
        self._choosable_dataset: bool = True

        # Load and show values from the dataset

        if ds_id is not None:
            self._dataset = self._app_state.get_dataset(ds_id)
            self._choosable_dataset = False

        self._configure_ui()


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

        self._show_dataset_in_ui()
        self._update_save_filenames()


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
            self._ui.tbl_bands.setItem(index, 2,
                get_defaultband_tablewidgetitem(index, defaults))

            # Column 3:  Band name / wavelength

            self._ui.tbl_bands.setItem(index, 3, QTableWidgetItem(band_info['description']))


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
        self._ui.lbl_filenames_value.setVisible(len(path) > 0)
        if not path:
            return

        format = self._ui.cbox_save_format.currentText()

        loader = self._app_state.get_loader()
        filenames = loader.get_save_filenames(path, format)

        display_filenames = '\n'.join([os.path.basename(filename) for filename in filenames])
        self._ui.lbl_filenames_value.setText(display_filenames)


    def accept(self):

        path = self._ui.ledit_filename.text().strip()
        if not path:
            self._ui.tabWidget.setCurrentWidget(self._ui.tab_general)
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
