from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .generated.split_unsplit_pane_ui import Ui_SplitPaneDialog


class SplitPaneDialog(QDialog):
    '''
    A dialog for editing information about collected spectra and library
    spectra.  Depending on the type of spectrum, different fields will be
    made available to the user.
    '''

    def __init__(self, initial=(1,1), parent=None):

        super().__init__(parent=parent)
        self._ui = Ui_SplitPaneDialog()
        self._ui.setupUi(self)

        self._ui.sb_rows.setValue(initial[0])
        self._ui.sb_cols.setValue(initial[1])


    def get_dimensions(self):
        return (self._ui.sb_rows.value(), self._ui.sb_cols.value())
