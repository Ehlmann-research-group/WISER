import logging

from typing import Any, Dict, List, Optional, Set, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import lark

from .generated.band_math_ui import Ui_BandMathDialog

from .app_state import ApplicationState
from .rasterview import RasterView

from wiser.raster.dataset import RasterDataSet, RasterDataBand
from wiser import bandmath
from wiser.bandmath.utils import get_dimensions


logger = logging.getLogger(__name__)


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


def get_memory_size(size_bytes: int) -> str:
    '''
    This helper function takes a size in bytes, and generates a human-readable
    string versino of the size.  The size will be reported using bytes,
    kilobytes (=2**10 bytes), megabytes (=2**20 bytes), gigabytes, or terabytes,
    depending on the most appropriate option for the input size.
    '''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    size = size_bytes
    for i in range(len(suffixes)):
        if size < 1024:
            if i == 0:
                return f'{size}{suffixes[i]}'
            else:
                return f'{size:.1f}{suffixes[i]}'
        size /= 1024.0

    return f'{size:.1f}{suffixes[-1]}'


def all_bindings_specified(bindings: Dict[str, Tuple[bandmath.VariableType, Any]]):
    '''
    This helper function returns True if all variables in the ``bindings``
    dictionary specify usable values, or ``False`` otherwise.  (A missing value
    is indicated by ``None``.)
    '''
    for (name, (type, value)) in bindings.items():
        if value is None:
            return False

    return True


def make_dataset_chooser(app_state) -> QComboBox:
    '''
    This helper function returns a combobox for choosing a dataset from the
    set of currently loaded datasets.
    '''
    chooser = QComboBox()
    for ds in app_state.get_datasets():
        chooser.addItem(ds.get_name(), ds.get_id())

    return chooser


def make_spectrum_chooser(app_state) -> QComboBox:
    '''
    This helper function returns a combobox for choosing a spectrum from the
    set of currently loaded spectra.
    '''
    chooser = QComboBox()
    chooser.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    active = app_state.get_active_spectrum()
    if active:
        # Add active spectrum to list
        name = app_state.tr('Active:  {name}').format(name=active.get_name())
        chooser.addItem(name, active.get_id())

    collected = app_state.get_collected_spectra()
    if collected:
        # Add collected spectra to list

        if chooser.count() > 0:
            chooser.insertSeparator(chooser.count())

        for s in collected:
            name = f'{s.get_name()}'
            chooser.addItem(name, s.get_id())

    # TODO(donnie):  Add spectral libraries to list

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

        self.dataset_chooser = make_dataset_chooser(self._app_state)
        layout.addWidget(self.dataset_chooser, 0, 0)

        self.band_chooser = QComboBox()
        self.band_chooser.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._populate_band_chooser()
        layout.addWidget(self.band_chooser, 0, 1)

        self.dataset_chooser.activated.connect(self._on_dataset_changed)


    def _populate_band_chooser(self):
        '''
        Populate the bands in the band-chooser widget, based on the currently
        selected dataset.
        '''
        self.band_chooser.clear()

        ds_id = self.dataset_chooser.currentData()
        dataset = self._app_state.get_dataset(ds_id)

        for b in dataset.band_list():
            # TODO(donnie):  Generate a band name in some generalized way.

            desc = b['description']
            if desc:
                desc = f'Band {b["index"]}: {desc}'
            else:
                desc = f'Band {b["index"]}'

            self.band_chooser.addItem(desc, b['index'])


    def _on_dataset_changed(self, index):
        '''
        When the dataset is changed by the user, we need to repopulate the list
        of available bands.
        '''
        self._populate_band_chooser()


    def get_ds_band(self) -> Tuple[int, int]:
        '''
        This method returns the currently selected dataset and band.  The
        information is reported as a 2-tuple of this form:  (dataset ID,
        band index).
        '''
        return (self.dataset_chooser.currentData(),
                self.band_chooser.currentData())


class ReturnEventFilter(QObject):
    # TODO(donnie):  This is my attempt at trying to filter carriage-returns on
    #     the band-math expression field.

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

        # Keep track of whether there are unsaved changes to the "saved
        # expressions" list.
        self._saved_exprs_modified: bool = False

        # Set up the UI state
        self._ui = Ui_BandMathDialog()
        self._ui.setupUi(self)

        # Hook up event handlers

        # self._ui.ledit_math_expr.install
        # self._ui.ledit_math_expr.installEventFilter(ReturnEventFilter())

        #==================================
        # "Current expression" UI widgets

        self._ui.btn_toggle_help.clicked.connect(self._on_toggle_help)

        # Always start with the help info hidden.
        # self._ui.tedit_bandmath_help.setVisible(False)

        self._ui.ledit_expression.editingFinished.connect(lambda: self._analyze_expr())
        self._ui.btn_add_to_saved.clicked.connect(self._on_add_expr_to_saved)

        #==================================
        # "Saved expressions" UI widgets

        self._ui.cbox_saved_exprs.activated.connect(self._on_choose_saved_expr)

        self._ui.btn_load_saved_exprs.clicked.connect(self._on_load_saved_exprs)
        self._ui.btn_save_saved_exprs.clicked.connect(self._on_save_saved_exprs)

        # Do this here so that we can use the text-translation facilities.
        self._variable_types_text = {
            bandmath.VariableType.IMAGE_CUBE: self.tr('Image'),
            bandmath.VariableType.IMAGE_BAND: self.tr('Image Band'),
            bandmath.VariableType.SPECTRUM: self.tr('Spectrum'),
            bandmath.VariableType.REGION_OF_INTEREST: self.tr('Region of Interest'),
            bandmath.VariableType.NUMBER: self.tr('Number'),
            bandmath.VariableType.BOOLEAN: self.tr('Boolean'),
            bandmath.VariableType.STRING: self.tr('String'),
        }


    def _on_toggle_help(self, checked=False):
        is_visible = self._ui.tedit_bandmath_help.isVisible()
        dialog_size = self.size()
        help_size = self._ui.tedit_bandmath_help.size()

        self._ui.tedit_bandmath_help.setVisible(not is_visible)

        delta = help_size.width()
        if is_visible:
            # Hiding the help info will shrink the dialog
            dialog_size.setWidth(dialog_size.width() - delta)
        else:
            # Showing the help info will grow the dialog
            dialog_size.setWidth(dialog_size.width() + delta)

        self.resize(dialog_size)


    def _analyze_expr(self):
        '''
        This helper method parses and analyzes the current band-math expression,
        identifying all variables in the expression, and updating the list of
        variable bindings shown in the dialog.  If all variables are bound, the
        method also analyzes the expression to predict the type, shape and size
        of the expression's result, and displays this information in the UI.
        '''
        expr = self.get_expression()
        if not expr:
            return

        # Try to identify details about the expression by parsing and analyzing
        # it.  This could fail, of course.
        try:
            # Try to identify variables in the band-math expression.
            variables: Set[str] = bandmath.get_bandmath_variables(expr)
            self._sync_binding_table_with_variables(variables)

            bindings = self.get_variable_bindings()
            if not all_bindings_specified(bindings):
                self._ui.lbl_result_info.setText(self.tr(
                    'Please specify values for all variables'))
                return

            # Analyze the expression and share info about the result.
            self._ui.lbl_result_info.clear()
            self._ui.lbl_result_info.setStyleSheet('QLabel { color: black; }')
            expr_info = bandmath.get_bandmath_expr_info(expr, bindings, None)

            if expr_info.result_type not in [bandmath.VariableType.IMAGE_CUBE,
                bandmath.VariableType.IMAGE_BAND, bandmath.VariableType.SPECTRUM]:
                self._ui.lbl_result_info.setText(self.tr('Enter an ' +
                    'expression that produces an image cube, band, or spectrum'))
                self._ui.lbl_result_info.setStyleSheet('QLabel { color: red; }')
                return

            type_str = self._variable_types_text.get(expr_info.result_type,
                self.tr('Unrecognized type'))
            dims_str = ''
            mem_size_str = ''

            if expr_info.result_type in [bandmath.VariableType.IMAGE_CUBE,
                bandmath.VariableType.IMAGE_BAND, bandmath.VariableType.SPECTRUM]:
                dims_str = f' {get_dimensions(expr_info.result_type, expr_info.shape)}'
                mem_size_str = f' ({get_memory_size(expr_info.result_size())})'

            s = self.tr('Result: {type}{dimensions}{mem_size}')
            s = s.format(type=type_str, dimensions=dims_str, mem_size=mem_size_str)
            self._ui.lbl_result_info.setText(s)

        except lark.exceptions.VisitError as e:
            # This would be an exception raised by the analysis code to indicate
            # a semantic error in the expression.
            logger.exception(f'Bandmath UI:  Analysis error on expression "{expr}"')
            self._ui.lbl_result_info.setText(self.tr('Error:  {0}').format(e.orig_exc))
            self._ui.lbl_result_info.setStyleSheet('QLabel { color: red; }')

        except lark.exceptions.LarkError as e:
            # This would be an exception raised by the parsing code.
            logger.exception(f'Bandmath UI:  Parse error on expression "{expr}"')
            self._ui.lbl_result_info.setText(self.tr('Parse error!'))
            self._ui.lbl_result_info.setStyleSheet('QLabel { color: red; }')

        except Exception as e:
            # This would likely be an exception generated by an internal WISER
            # bug.
            logger.exception(f'Bandmath UI:  Other error on expression "{expr}"')
            self._ui.lbl_result_info.setText(self.tr('Error:  {0}').format(e))
            self._ui.lbl_result_info.setStyleSheet('QLabel { color: red; }')


    def _sync_binding_table_with_variables(self, variables):
        '''
        This helper function synchronizes the GUI's variable-binding table with
        the variables found in the current expression.  New variables are added
        to the table; existing variables are left untouched; and missing
        variables are removed from the table.
        '''
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
        '''
        Given a specific variable-type (e.g. "image cube," "image band," or
        "spectrum"), this method constructs a Qt widget for letting the user
        specify the value to use.  The widget is populated with the current
        values from the WISER application state.
        '''
        value_widget = None

        if variable_type == bandmath.VariableType.IMAGE_CUBE:
            value_widget = make_dataset_chooser(self._app_state)
            value_widget.activated.connect(self._on_variable_shape_change)

        elif variable_type == bandmath.VariableType.IMAGE_BAND:
            value_widget = DatasetBandChooserWidget(self._app_state)
            value_widget.dataset_chooser.activated.connect(self._on_variable_shape_change)

        elif variable_type == bandmath.VariableType.SPECTRUM:
            value_widget = make_spectrum_chooser(self._app_state)
            value_widget.activated.connect(self._on_variable_shape_change)

        else:
            raise AssertionError(f'Unrecognized variable type {variable_type}')

        return value_widget


    def _on_variable_type_change(self, type_index: int, var_name: str):
        '''
        When a variable's type changes, the dialog must show a new value-chooser
        for that variable.
        '''
        print(f'Variable {var_name} type change to {type_index}')
        var_row = self._find_variable_in_bindings(var_name)
        if var_row == -1:
            raise AssertionError(f'Received unrecognized variable name {var_name}')

        var_type = self._ui.tbl_variables.cellWidget(var_row, 1).currentData()
        print(f' * New type is {var_type}')
        value_widget = self._make_value_widget(var_type)
        self._ui.tbl_variables.setCellWidget(var_row, 2, value_widget)

        # Update the expression analysis
        self._analyze_expr()


    def _on_variable_shape_change(self, index: int):
        '''
        When a variable's shape changes, the dialog must update its analysis of
        the expression's results.
        '''
        self._analyze_expr()


    def _on_add_expr_to_saved(self, checked=False):
        expr = self.get_expression()

        # Verify that we have a parseable expression.
        if not bandmath.bandmath_parses(expr):
            QMessageBox.critical(self, self.tr('Parse Error'),
                self.tr('The current band-math expression does not parse.\n\n' +
                        'Please fix the expression before saving it.'))
            return

        # Verify that the expression doesn't already appear in the saved
        # expressions.
        for i in range(self._ui.cbox_saved_exprs.count()):
            # TODO(donnie):  This comparison doesn't catch situations where
            #     whitespace is the only difference.
            saved_expr = self._ui.cbox_saved_exprs.itemText(i).casefold()
            if expr == saved_expr:
                QMessageBox.critical(self, self.tr('Expression already saved'),
                    self.tr('The current band-math expression is already\n' +
                            'in the saved-expressions list.'))
                return

        # Add the expression to the end of the list, and make sure it is
        # visible/selected in the combo-box.
        self._ui.cbox_saved_exprs.addItem(expr)
        self._ui.cbox_saved_exprs.setCurrentIndex(self._ui.cbox_saved_exprs.count() - 1)
        self._saved_exprs_modified = True


    def _on_load_saved_exprs(self, checked=False):
        '''
        This helper method implements the "load a saved-expressions file".  If
        there are unsaved expressions, the user is prompted about discarding
        them before loading any new expression list.
        '''
        # Are there unsaved changes to the saved-expressions list?
        if self._ui.cbox_saved_exprs.count() > 0 and self._saved_exprs_modified:
            response = QMessageBox.question(self, self.tr('Un-saved Expressions'),
                self.tr('Discard unsaved changes to saved expressions?'))

            if response != QMessageBox.Yes:
                return

        (path, _) = QFileDialog.getOpenFileName(self,
            self.tr('Read Saved-Expressions from File'),
            self._app_state.get_current_dir(),
            self.tr('Text files (*.txt);;All files (*)'))

        if not path:
            return

        try:
            with open(path) as f:
                # Read in all lines of the file
                lines = f.readlines()

        except Exception as e:
            QMessageBox.critical(self, self.tr('Couldn\'t Open File'),
                self.tr('Couldn\'t open file {0}:\n\n{1}').format(path, e))
            return

        # Strip leading and trailing whitespace off every line, and
        # convert to lowercase so everything is normalized.
        lines = [line.strip().casefold() for line in lines]

        # Filter out blank lines.
        lines = [line for line in lines if line]

        self._ui.cbox_saved_exprs.clear()
        for line in lines:
            # TODO(donnie):  Make sure all lines parse?
            self._ui.cbox_saved_exprs.addItem(line)

        self._saved_exprs_modified = False


    def _on_save_saved_exprs(self, checked=False):
        (path, _) = QFileDialog.getSaveFileName(self,
            self.tr('Write Saved-Expressions to File'),
            self._app_state.get_current_dir(),
            self.tr('Text files (*.txt);;All files (*)'))

        if not path:
            return

        try:
            with open(path, 'w') as f:
                for i in range(self._ui.cbox_saved_exprs.count()):
                    expr = self._ui.cbox_saved_exprs.itemText(i)
                    f.write(f'{expr}\n')

        except Exception as e:
            QMessageBox.critical(self, self.tr('Couldn\'t Open File'),
                self.tr('Couldn\'t open file {0}:\n\n{1}').format(path, e))
            return

        self._saved_exprs_modified = False

    def _on_choose_saved_expr(self, index):
        '''
        Handle events where the user chooses one of the saved expressions.
        '''
        expr = self._ui.cbox_saved_exprs.currentText()
        self._ui.ledit_expression.setText(expr)
        self._analyze_expr()


    def _check_saved_expressions(self):
        if self._ui.cbox_saved_exprs.count() > 0 and self._saved_exprs_modified:
            response = QMessageBox.question(self, self.tr('Un-saved Expressions'),
                self.tr('Do you want to save your changes\n' +
                        'to the saved-expressions list?'))

            if response == QMessageBox.Yes:
                self._on_save_saved_exprs()


    def accept(self):
        # Have changes been made to the saved-expressions list?
        self._check_saved_expressions()

        # Make sure that all variable-bindings are specified, and that there are
        # no obvious errors with the band-math expression.

        bindings = self.get_variable_bindings()
        if not all_bindings_specified(bindings):
            QMessageBox.critical(self, self.tr('Binding Error'),
                self.tr('Please specify all variable bindings.'))
            return

        # TODO(donnie):  Check for obvious issues with the band math?

        super().accept()


    def reject(self):
        # Have changes been made to the saved-expressions list?
        self._check_saved_expressions()

        super().reject()


    def get_expression(self) -> str:
        '''
        Returns the math expression as entered by the user, with leading and
        trailing whitespace stripped, and the expression casefolded to
        lowercase.
        '''
        return self._ui.ledit_expression.text().strip().casefold()


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
            value = None

            if type == bandmath.VariableType.IMAGE_CUBE:
                ds_id = self._ui.tbl_variables.cellWidget(row, 2).currentData()
                if ds_id is not None:
                    value = self._app_state.get_dataset(ds_id)

            elif type == bandmath.VariableType.IMAGE_BAND:
                (ds_id, band_index) = self._ui.tbl_variables.cellWidget(row, 2).get_ds_band()
                if ds_id is not None and band_index is not None:
                    dataset = self._app_state.get_dataset(ds_id)
                    value = RasterDataBand(dataset, band_index)

            elif type == bandmath.VariableType.SPECTRUM:
                spectrum_id = self._ui.tbl_variables.cellWidget(row, 2).currentData()
                if spectrum_id is not None:
                    value = self._app_state.get_spectrum(spectrum_id)

            else:
                raise AssertionError(
                    f'Unrecognized binding type {type} for variable {var}')

            variables[var] = (type, value)

        return variables
