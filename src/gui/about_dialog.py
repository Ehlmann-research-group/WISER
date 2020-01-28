from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from version import VERSION

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr('About the Imaging Spectroscopy Workbench'))

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(self.tr('Version:  ' + VERSION)))

        # Dialog buttons - hook to built-in QDialog functions
        buttons = QDialogButtonBox(QDialogButtonBox.Ok, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)
