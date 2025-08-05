
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.batch_processing_wizard_ui import Ui_BatchProcessingWizard

class BatchProcessingWizard(QWizard):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._ui = Ui_BatchProcessingWizard()
        self._ui.setupUi(self)
