from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .about_dialog_ui import Ui_AboutDialog
from .system_info import SystemInfoDialog

from version import VERSION

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_AboutDialog()
        self._ui.setupUi(self)

        self._ui.label_version.setText(self.tr('Version:  {0}').format(VERSION))

        self._ui.btn_system_info.clicked.connect(self._on_system_info)


    def _on_system_info(self, checked):
        dialog = SystemInfoDialog(self)
        dialog.setModal(True)
        dialog.show()
