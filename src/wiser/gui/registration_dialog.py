import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.registration_dialog_ui import Ui_RegistrationDialog

from wiser.gui.app_state import ApplicationState

class RegistrationDialog(QDialog):

    def __init__(self, app_state: ApplicationState, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state

        self._ui = Ui_RegistrationDialog()
        self._ui.setupUi(self)

        
        self._ui.ledit_filename.editingFinished.connect(self._on_edit_save_filename)
        self._ui.btn_file.clicked.connect(self._on_choose_save_filename)

        self._update_save_filenames()

        # Go through the datasets in app state and populate the combo box with them
        if len(self._app_state._datasets) > 0:
            for ds_id, ds in self._app_state._datasets.items():
                self._ui.src_ds_combo.addItem(ds.get_name(), ds_id)
                self._ui.target_ds_combo.addItem(ds.get_name(), ds_id)
        else:
            self._ui.src_ds_combo.setEnabled(False)
            self._ui.target_ds_combo.setEnabled(False)

        # Set up the button to do file paths

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

        # format = self._ui.cbox_save_format.currentText()

        loader = self._app_state.get_loader()
        filenames = loader.get_save_filenames(path)

        display_filenames = '\n'.join([os.path.basename(filename) for filename in filenames])
        self._ui.lbl_filenames_value.setText(display_filenames)

    def accept(self):
        super().accept()

