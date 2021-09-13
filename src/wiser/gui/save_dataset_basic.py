import os

from typing import Any, List, Optional, Tuple, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.save_dataset_basic_ui import Ui_SaveDatasetDialog
from .generated.save_dataset_advanced_ui import Ui_AdvancedSaveDatasetDialog


class BasicSaveDatasetDialog(QDialog):
    '''
    A dialog for saving a dataset, allowing the user to specify advanced options
    such as what are the default bands, saving a subset of the image, etc.
    '''

    def __init__(self, app_state, ds_id=None, parent=None):

        super().__init__(parent=parent)
        self._ui = Ui_SaveDatasetDialog()
        self._ui.setupUi(self)

        # Configure UI components

        self._ui.cbox_save_format.addItem('ENVI')

        self._ui.ledit_filename.editingFinished.connect(self._on_edit_save_filename)
        self._ui.btn_filename.clicked.connect(self._on_choose_save_filename)

        self._app_state: ApplicationState = app_state
        self._dataset: Optional[RasterDataSet] = None
        self._choosable_dataset: bool = True

        # Load and show values from the dataset

        if ds_id is not None:
            self._dataset = self._app_state.get_dataset(ds_id)
            self._choosable_dataset = False

        self._configure_ui()

        self._ui.btn_advanced.setEnabled(False)
        # self._ui.btn_advanced.clicked.connect(self._on_advanced_options)


    def _on_advanced_options(self, checked=False):
        self._ui = Ui_AdvancedSaveDatasetDialog()
        self._ui.setupUi(self)


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
