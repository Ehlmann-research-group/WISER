import enum

from typing import List, Optional, Set, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import lark

from .generated.band_math_ui import Ui_BandMathDialog

from .app_state import ApplicationState
from .rasterview import RasterView

from raster.dataset import RasterDataSet
from bandmath import parser


class VariableType(enum.Enum):
    IMAGE = 1

    IMAGE_BAND = 2

    # IMAGE_PIXEL_SPECTRUM = 3

    LIBRARY_SPECTRUM = 4


class BandMathDialog(QDialog):

    def __init__(self, app_state: ApplicationState,
            rasterview: Optional[RasterView] = None, parent=None):
        super().__init__(parent=parent)

        self._app_state = app_state

        self._rasterview = rasterview
        # TODO(donnie):  If rasterview is not none, set up some bindings
        # self._dataset = dataset
        # self._display_bands = display_bands

        # Set up the UI state
        self._ui = Ui_BandMathDialog()
        self._ui.setupUi(self)

        # Hook up event handlers

        self._ui.btn_parse.clicked.connect(self._on_parse_expr)


    def _on_parse_expr(self, checked):
        expr = self._ui.txt_math_expr.toPlainText().strip()
        if not expr:
            QMessageBox.critical(self, self.tr('No Expression'),
                self.tr('Please enter a band-math expression'))
            return

        try:
            variables: Set[str] = parser.get_variables(expr)
            print(f'Found variables:  {variables}')

            self._sync_binding_table_with_variables(variables)

        except lark.exceptions.LarkError as err:
            QMessageBox.critical(self, self.tr('Parse Error'),
                self.tr('The expression does not parse correctly.\n' +
                        'Please fix it and try parsing again.'))


    def _sync_binding_table_with_variables(self, variables):
        # Disable sorting while we update the table.
        self._ui.tbl_variables.setSortingEnabled(False)

        # Add new variables to the list of bindings.
        for var in variables:
            index = self._find_variable_in_bindings(var)
            if index == -1:
                new_row = self._ui.tbl_variables.rowCount()
                self._ui.tbl_variables.insertRow(new_row)

                item = QTableWidgetItem(var)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ui.tbl_variables.setItem(new_row, 0, item)

                widget = QComboBox()
                widget.addItem('Image', VariableType.IMAGE)
                widget.addItem('Image band', VariableType.IMAGE_BAND)
                # widget.addItem('Image pixel spectrum', VariableType.IMAGE_PIXEL_SPECTRUM)
                widget.addItem('Library spectrum', VariableType.LIBRARY_SPECTRUM)

                item = QTableWidgetItem('Image band')
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                self._ui.tbl_variables.setCellWidget(new_row, 1, widget)
                self._ui.tbl_variables.setItem(new_row, 1, item)

        # Remove deleted variables from the list of bindings.  Do this in
        # reverse order so we don't have to adjust for rows shifting up.
        for row in range(self._ui.tbl_variables.rowCount() - 1, -1, -1):
            if self._ui.tbl_variables.item(row, 0).text() not in variables:
                self._ui.tbl_variables.removeRow(row)

        # Reenable the sorting on the table.
        self._ui.tbl_variables.sortByColumn(0, Qt.AscendingOrder)
        self._ui.tbl_variables.setSortingEnabled(True)


    def _find_variable_in_bindings(self, variable) -> int:
        '''
        Look for the specified variable in the variable-bindings table.
        If found, the row index is returned.  If not found, -1 is returned.
        '''
        for i in range(self._ui.tbl_variables.rowCount()):
            if self._ui.tbl_variables.item(i, 0).text() == variable:
                return i

        return -1
