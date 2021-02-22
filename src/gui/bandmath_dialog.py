from typing import Any, Dict, List, Optional, Set, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import lark

from .generated.band_math_ui import Ui_BandMathDialog

from .app_state import ApplicationState
from .rasterview import RasterView

from raster.dataset import RasterDataSet, RasterDataBand
import bandmath


def guess_variable_type_from_name(variable: str) -> bandmath.VariableType:
    '''
    Given a variable name, this function guesses the variable's type.  The guess
    is very simple:

    *   If the variable starts with "i" then the guess is IMAGE_CUBE
    *   If the variable starts with "s" then the guess is SPECTRUM
    *   Otherwise, the guess is IMAGE_BAND
    '''
    variable = variable.strip().lower()

    if variable.startswith('i'):
        return bandmath.VariableType.IMAGE_CUBE

    elif variable.startswith('s'):
        return bandmath.VariableType.SPECTRUM

    else:
        # This is the default guess
        return bandmath.VariableType.IMAGE_BAND


def make_dataset_chooser(app_state) -> QComboBox:
    chooser = QComboBox()
    for ds in app_state.get_datasets():
        chooser.addItem(ds.get_filepaths()[0], ds.get_id())

    return chooser


class DatasetBandChooserWidget(QWidget):
    '''
    This class presents a dataset-chooser and a band-chooser in a single widget,
    for the selection of a band in a band-math expression.
    '''
    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)

        self._app_state = app_state

        layout = QGridLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        self.setLayout(layout)

        self._dataset_chooser = make_dataset_chooser(self._app_state)
        layout.addWidget(self._dataset_chooser, 0, 0)

        self._band_chooser = QComboBox()
        self._band_chooser.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._populate_band_chooser()
        layout.addWidget(self._band_chooser, 0, 1)

        self._dataset_chooser.activated.connect(self._on_dataset_changed)


    def keyPressEvent(self, event):
        print(f'Key press event:  {event}')


    def _populate_band_chooser(self):
        self._band_chooser.clear()

        ds_id = self._dataset_chooser.currentData()
        dataset = self._app_state.get_dataset(ds_id)

        for b in dataset.band_list():
            # TODO(donnie):  Generate a band name in some generalized way.
            band_name = f'Band {b["index"]}'
            self._band_chooser.addItem(band_name, b['index'])


    def _on_dataset_changed(self, index):
        self._populate_band_chooser()


    def get_ds_band(self) -> Tuple[int, int]:
        return (self._dataset_chooser.currentData(),
                self._band_chooser.currentData())


class ReturnEventFilter(QObject):

    def eventFilter(obj, evt) -> bool:
        print(f'Object:  {obj}  Event:  {evt}')
        return False


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

        # self._ui.ledit_math_expr.install
        self._ui.ledit_math_expr.installEventFilter(ReturnEventFilter())

        self._ui.ledit_math_expr.editingFinished.connect(lambda: self._on_parse_expr())
        self._ui.btn_parse.clicked.connect(lambda: self._on_parse_expr(manual=True))


    def _on_parse_expr(self, manual=False):
        expr = self._ui.ledit_math_expr.text().strip()
        if not expr:
            QMessageBox.critical(self, self.tr('No Expression'),
                self.tr('Please enter a band-math expression'))
            return

        try:
            variables: Set[str] = bandmath.get_bandmath_variables(expr)
            self._sync_binding_table_with_variables(variables)

        except lark.exceptions.LarkError as err:
            if manual:
                # Only show the error dialog if the user manually invoked the
                # parser
                QMessageBox.critical(self, self.tr('Parse Error'),
                    self.tr('The expression does not parse correctly.\n' +
                            'Please fix it and try parsing again.'))


    def _sync_binding_table_with_variables(self, variables):
        # Disable sorting while we update the table.
        self._ui.tbl_variables.setSortingEnabled(False)

        # Add new variables to the table of bindings.
        for var in variables:
            # Look for the variable in the current table of bindings.
            index = self._find_variable_in_bindings(var)
            if index == -1:
                # Couldn't find variable in the table of bindings.
                # Add a new row for it.
                new_row = self._ui.tbl_variables.rowCount()
                self._ui.tbl_variables.insertRow(new_row)

                # First column is the variable name.  This is not editable in
                # the table.

                item = QTableWidgetItem(var)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._ui.tbl_variables.setItem(new_row, 0, item)

                # Second column is the type of variable.

                type_widget = QComboBox()
                type_widget.addItem(self.tr('Image'), bandmath.VariableType.IMAGE_CUBE)
                type_widget.addItem(self.tr('Image band'), bandmath.VariableType.IMAGE_BAND)
                type_widget.addItem(self.tr('Spectrum'), bandmath.VariableType.SPECTRUM)
                type_widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)

                # Guess the type of the variable based on its name, and choose
                # that as the variable's initial type.
                type_guess = guess_variable_type_from_name(var)
                type_widget.setCurrentIndex(type_widget.findData(type_guess))

                type_widget.activated.connect(
                    lambda index, var_name=var :
                    self._on_variable_type_change(index, var_name))

                # item = QTableWidgetItem('Image band')
                # item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                self._ui.tbl_variables.setCellWidget(new_row, 1, type_widget)
                # self._ui.tbl_variables.setItem(new_row, 1, item)

                # Third column is the actual value bound to the variable.  Both
                # the kind of widget(s) and the value depend on the setting of
                # the second column.

                value_widget = self._make_value_widget(type_guess)
                self._ui.tbl_variables.setCellWidget(new_row, 2, value_widget)


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


    def _make_value_widget(self, variable_type: bandmath.VariableType):
        value_widget = None

        if variable_type == bandmath.VariableType.IMAGE_CUBE:
            value_widget = make_dataset_chooser(self._app_state)

        elif variable_type == bandmath.VariableType.IMAGE_BAND:
            value_widget = DatasetBandChooserWidget(self._app_state)

        elif variable_type == bandmath.VariableType.SPECTRUM:
            value_widget = QComboBox()
            value_widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            active = self._app_state.get_active_spectrum()
            if active:
                # Add active spectrum to list
                name = self.tr('Active:  {name}').format(name=active.get_name())
                value_widget.addItem(name, active.get_id())

            collected = self._app_state.get_collected_spectra()
            if collected:
                # Add collected spectra to list

                if value_widget.count() > 0:
                    value_widget.insertSeparator(value_widget.count())

                for s in collected:
                    name = self.tr('{name}').format(name=s.get_name())
                    value_widget.addItem(name, s.get_id())

            # TODO(donnie):  Add spectral libraries to list

        else:
            raise AssertionError(f'Unrecognized variable type {variable_type}')

        return value_widget


    def _on_variable_type_change(self, type_index: int, var_name: str):
        print(f'Variable {var_name} type change to {type_index}')
        var_row = self._find_variable_in_bindings(var_name)
        if var_row == -1:
            raise AssertionError(f'Received unrecognized variable name {var_name}')

        var_type = self._ui.tbl_variables.cellWidget(var_row, 1).currentData()
        print(f' * New type is {var_type}')
        value_widget = self._make_value_widget(var_type)
        self._ui.tbl_variables.setCellWidget(var_row, 2, value_widget)


    def accept(self):
        # TODO(donnie):  validation
        # Make sure that all variable-bindings are specified, and that there are
        # no obvious errors with the band-math expression.
        '''
        if not self.all_variables_bound():
            QMessageBox.critical(self, self.tr('Binding Error'),
                self.tr('Please specify all variable bindings.'))
            return
        '''

        # TODO(donnie):  Check for obvious issues with the band math?

        super().accept()


    def get_expression(self) -> str:
        '''
        Returns the math expression as entered by the user.
        '''
        return self._ui.ledit_math_expr.text()


    def get_variable_bindings(self) -> Dict[str, Tuple[bandmath.VariableType, Any]]:
        '''
        Returns the variable bindings as specified by the user.  The result is
        in the form that is required by bandmath.evaluator.eval_bandmath_expr().

        Note that this function doesn't guarantee that the variable-bindings
        actually reflect the expression, or that there are no semantic errors
        or mismatched types in the expression.
        '''
        variables = {}
        for row in range(self._ui.tbl_variables.rowCount()):
            var = self._ui.tbl_variables.item(row, 0).text()
            type = self._ui.tbl_variables.cellWidget(row, 1).currentData()

            if type == bandmath.VariableType.IMAGE_CUBE:
                ds_id = self._ui.tbl_variables.cellWidget(row, 2).currentData()
                value = self._app_state.get_dataset(ds_id)

            elif type == bandmath.VariableType.IMAGE_BAND:
                (ds_id, band_index) = self._ui.tbl_variables.cellWidget(row, 2).get_ds_band()
                dataset = self._app_state.get_dataset(ds_id)
                value = RasterDataBand(dataset, band_index)

            elif type == bandmath.VariableType.SPECTRUM:
                spectrum_id = self._ui.tbl_variables.cellWidget(row, 2).currentData()
                value = self._app_state.get_spectrum(spectrum_id)

            else:
                raise AssertionError(
                    f'Unrecognized binding type {type} for variable {var}')

            variables[var] = (type, value)

        return variables
