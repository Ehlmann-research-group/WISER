
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.geo_referencer_dialog_ui import Ui_GeoReferencerDialog

class GeoReferencerDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_GeoReferencerDialog()
        self._ui.setupUi(self)

        # Set up dataset choosers 