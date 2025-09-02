from enum import Enum
import logging
import os

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from astropy import units as u
import numpy as np

import lark

from .generated.band_math_ui import Ui_BandMathDialog

from .app_state import ApplicationState
from .rasterview import RasterView

from wiser.raster.dataset import RasterDataBand, RasterDataSet, RasterDataBatchBand
from wiser.raster.spectrum import Spectrum
from wiser import bandmath
from wiser.bandmath.utils import get_dimensions, bandmath_success_callback, bandmath_progress_callback, bandmath_error_callback
from wiser.bandmath.types import BANDMATH_VALUE_TYPE
from wiser.gui.util import get_plugin_fns
from wiser.gui.subprocessing_manager import ProcessManager
from wiser.gui.parallel_task import ParallelTask

import copy

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
    chooser.setSizeAdjustPolicy(QComboBox.AdjustToContents)

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

    # Add spectral libraries to list
    for lib in app_state.get_spectral_libraries():
        if lib.num_spectra() == 0:
            continue

        if chooser.count() > 0:
            chooser.insertSeparator(chooser.count())

        lib_id = lib.get_id()
        for index in range(lib.num_spectra()):
            name = lib.get_spectrum_name(index)
            chooser.addItem(name, (lib_id, index, ) )

    return chooser

def make_image_cube_batch_chooser(text: str) -> QLabel:
    '''
    This helper function returns a label telling the user that this variable
    uses the input folder path for all image cubes
    '''
    label = QLabel()
    label.setText(text)
    return label

def make_image_band_batch_chooser(text: str) -> QLabel:
    '''
    This helper function returns a label telling the user that this variable
    uses the input folder path for all image bands, a combo box to let the user
    choose the type of way to select the image band (whether by index or wavelength).
    '''
    label = QLabel()
    label.setText(text)
    return label

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
        try:
            dataset = self._app_state.get_dataset(ds_id)
        except KeyError:
            # This probably isn't a serious problem; for example, it can
            # occur when WISER has no datasets loaded.
            logger.info(f'Couldn\'t retrieve dataset with ID {ds_id}')
            return

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


class ImageBandBatchChooserWidget(QWidget):
    """
    Compact widget for choosing an image band for batch processing.

    Layout (by mode):
      - Index:      [Select: ▾] [<band index>]
      - Wavelength: [Select: ▾] [<wavelength>] [units ▾] epsilon [<ε>]

    Notes
    -----
    - Band index input is int-validated.
    - Wavelength and epsilon inputs are float-validated.
    - Units list mirrors astropy.units names/aliases used in your codebase.
    """

    modeChanged = Signal(str)  # "index" or "wavelength"

    class Mode(str, Enum):
        INDEX = "index"
        WAVELENGTH = "wavelength"

    # Visible keys -> astropy units (use keys in the UI; values for computation)
    UNIT_MAP: Dict[str, u.UnitBase] = {
        "nanometers": u.nanometer,
        "centimeters": u.cm,
        "meters": u.m,
        "micrometers": u.micrometer,
        "millimeters": u.millimeter,
        "microns": u.micron,
        "cm": u.centimeter,
        "m": u.meter,
        "mm": u.millimeter,
        "nm": u.nanometer,
        "um": u.micrometer,
        "wavenumber": u.cm ** -1,
        "angstroms": u.angstrom,
        "ghz": u.GHz,
        "mhz": u.MHz,
    }

    def __init__(self, app_state, table_widget: QTableWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._app_state = app_state  # reserved for future use
        self._tbl_wdgt_parent = table_widget
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Layout: single row, roomy and elastic
        row = QHBoxLayout()
        row.setContentsMargins(QMargins(0, 0, 0, 0))
        row.setSpacing(8)
        self.setLayout(row)

        # Mode selector (no separate "Select:" label)
        self._cmb_mode = QComboBox()
        self._cmb_mode.addItem("Band Index", self.Mode.INDEX.value)
        self._cmb_mode.addItem("Wavelength", self.Mode.WAVELENGTH.value)
        self._cmb_mode.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._cmb_mode.setMinimumContentsLength(10)
        self._cmb_mode.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        row.addWidget(self._cmb_mode)

        # Primary value (index or wavelength)
        self._ledit_value = QLineEdit()
        self._ledit_value.setPlaceholderText("Band index")
        self._ledit_value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        row.addWidget(self._ledit_value)

        # Units (wavelength-only)
        self._cmb_units = QComboBox()
        self._cmb_units.addItems(list(self.UNIT_MAP.keys()))
        self._cmb_units.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._cmb_units.setMinimumContentsLength(9)
        self._cmb_units.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        row.addWidget(self._cmb_units)

        # Epsilon (wavelength-only)
        self._lbl_eps = QLabel("epsilon")
        row.addWidget(self._lbl_eps)

        self._ledit_eps = QLineEdit()
        self._ledit_eps.setPlaceholderText("e.g., 1")
        self._ledit_eps.setToolTip(
            "If the exact wavelength is not found, use the closest value within "
            "epsilon. If none are within epsilon, skip the calculation."
        )
        self._ledit_eps.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        row.addWidget(self._ledit_eps)

        # # Spacer lets the row grow and prevents crowding
        # row.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # Validators (switched per mode)
        self._int_validator = QIntValidator(0, 10**9, self)
        self._float_validator_value = QDoubleValidator(0.0, 1e15, 8, self)
        self._float_validator_value.setNotation(QDoubleValidator.StandardNotation)
        self._float_validator_eps = QDoubleValidator(0.0, 1e15, 8, self)
        self._float_validator_eps.setNotation(QDoubleValidator.StandardNotation)

        # Initial state and signals
        self.set_mode(self.Mode.INDEX.value)
        self._cmb_mode.currentIndexChanged.connect(self._on_mode_changed)

    # ---------------- Public API ----------------
    def current_mode(self) -> str:
        """Return current mode: 'index' or 'wavelength'."""
        return self._cmb_mode.currentData()

    def set_mode(self, mode: str) -> None:
        """Programmatically set mode and update UI."""
        idx = self._cmb_mode.findData(mode)
        if idx >= 0:
            self._cmb_mode.setCurrentIndex(idx)
        self._apply_mode(mode)

    def get_settings(self) -> Dict[str, Optional[object]]:
        """
        Return current settings. For wavelength mode, includes both the UI key
        and the resolved astropy unit object.

        Returns
        -------
        dict
            {
              "mode": "index" | "wavelength",
              "index": str or None,
              "wavelength": str or None,
              "units_key": str or None,
              "unit": astropy.units.UnitBase or None,
              "epsilon": str or None,
            }
        """
        mode = self.current_mode()
        if mode == self.Mode.INDEX.value:
            index_text = self._ledit_value.text().strip()
            index = int(index_text) if index_text.isdigit() else None
            return {
                "mode": mode,
                "index": index,
                "wavelength": None,
                "units_key": None,
                "unit": None,
                "epsilon": None,
            }

        key = self._cmb_units.currentText()
        wvl_text = self._ledit_value.text().strip()
        wvl = float(wvl_text) if wvl_text.replace('.', '', 1).isdigit() else None
        epsilon_text = self._ledit_eps.text().strip()
        epsilon = float(epsilon_text) if epsilon_text.replace('.', '', 1).isdigit() else None
        return {
            "mode": mode,
            "index": None,
            "wavelength": wvl,
            "units_key": key,
            "unit": self.UNIT_MAP.get(key),
            "epsilon": epsilon,
        }

    # ---------------- Internals -----------------
    def _on_mode_changed(self, _i: int) -> None:
        self._apply_mode(self.current_mode())
        self.modeChanged.emit(self.current_mode())

    def _apply_mode(self, mode: str) -> None:
        """Update placeholders, validators, and visibility for the mode."""
        is_wavelength = (mode == self.Mode.WAVELENGTH.value)

        # Configure the primary value field
        if is_wavelength:
            self._ledit_value.setPlaceholderText("Wavelength")
            self._ledit_value.setValidator(self._float_validator_value)
            if not self._ledit_eps.text():
                self._ledit_eps.setText("20")
            # Default to a common unit if nothing selected yet
            if self._cmb_units.currentIndex() < 0:
                self._cmb_units.setCurrentIndex(self._cmb_units.findText("nm"))
        else:
            self._ledit_value.setPlaceholderText("Band index")
            self._ledit_value.setValidator(self._int_validator)

        # Epsilon is always float-validated
        self._ledit_eps.setValidator(self._float_validator_eps)

        # Toggle wavelength-only controls
        self._cmb_units.setVisible(is_wavelength)
        self._lbl_eps.setVisible(is_wavelength)
        self._ledit_eps.setVisible(is_wavelength)

        # Nudge layouts to recompute
        self.layout().invalidate()
        self.updateGeometry()
        if self._tbl_wdgt_parent:
            self._tbl_wdgt_parent.resizeColumnsToContents()

''' TODO(donnie):  Coming soon...
class VariableTypeDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def displayText(self, value, locale) -> str:
        pass

    def createEditor(self, parent, option, index):
        type_widget = QComboBox(parent=parent)
        type_widget.addItem(self.tr('Image'), bandmath.VariableType.IMAGE_CUBE)
        type_widget.addItem(self.tr('Image band'), bandmath.VariableType.IMAGE_BAND)
        type_widget.addItem(self.tr('Spectrum'), bandmath.VariableType.SPECTRUM)
        type_widget.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        # Guess the type of the variable based on its name, and choose
        # that as the variable's initial type.
        type_guess = guess_variable_type_from_name(var)
        type_widget.setCurrentIndex(type_widget.findData(type_guess))
'''


class ExpressionReturnEventFilter(QObject):
    '''
    This event-filter helper class is installed on the expression line-edit so
    that pressing Return/Enter after typing an expression will not cause the
    dialog to close; rather, it will cause the expression to be analyzed and the
    variable-list to be updated.  It seems to be natural to press Enter at the
    end of typing an expression.
    '''

    def __init__(self, bandmath_dialog):
        super().__init__()
        self._bandmath_dialog = bandmath_dialog

    def eventFilter(self, obj, evt) -> bool:
        if (evt.type() == QEvent.KeyPress and
            evt.key() in [Qt.Key_Return, Qt.Key_Enter]):
            # Instead of letting the Return/Enter event propagate up (where it
            # would close the band-math dialog), cause the expression to be
            # analyzed instead.
            self._bandmath_dialog._analyze_expr()

            # Since we handled the event, ignore it now.
            return True

        return False

class BandmathBatchJob:
    '''
    A batch job is a single unit of work that is to be performed by the batch
    processing system.  It contains the expression to be evaluated, the variables
    to be used, the input and output folders, and the load-into-wiser flag.

    Once the job is started, it will contain the process manager for that job.
    '''

    def __init__(self, job_id: int, expression: str, expr_info: bandmath.BandMathExprInfo, 
                 variables: Dict[str, Tuple[bandmath.VariableType, Any]], input_folder: str,
                 output_folder: str, load_into_wiser: bool, result_suffix: str):
        self._job_id = job_id
        self._expression = expression
        self._expr_info = copy.deepcopy(expr_info)
        self._variables = copy.deepcopy(variables)
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._load_into_wiser = load_into_wiser
        self._result_suffix = result_suffix
        self._process_manager: Optional[ProcessManager] = None
        self._btn_start: Optional[QPushButton] = None
        self._btn_cancel: Optional[QPushButton] = None
        self._btn_remove: Optional[QPushButton] = None
        self._progress_bar: Optional[QProgressBar] = None
        self._btn_view_errors: Optional[QPushButton] = None
        self._errors: Optional[List[Tuple[str, str, str]]] = []


    def get_process_manager(self) -> Optional[ProcessManager]:
        return self._process_manager

    def set_process_manager(self, process_manager: ProcessManager) -> None:
        self._process_manager = process_manager

    def get_btn_start(self) -> Optional[QPushButton]:
        return self._btn_start
    
    def set_btn_start(self, btn_start: QPushButton) -> None:
        self._btn_start = btn_start
    
    def get_btn_cancel(self) -> Optional[QPushButton]:
        return self._btn_cancel
    
    def set_btn_cancel(self, btn_cancel: QPushButton) -> None:
        self._btn_cancel = btn_cancel
    
    def get_btn_remove(self) -> Optional[QPushButton]:
        return self._btn_remove

    def set_btn_remove(self, btn_remove: QPushButton) -> None:
        self._btn_remove = btn_remove
    
    def get_progress_bar(self) -> Optional[QProgressBar]:
        return self._progress_bar

    def set_progress_bar(self, progress_bar: QProgressBar) -> None:
        self._progress_bar = progress_bar

    def get_btn_view_errors(self) -> Optional[QPushButton]:
        return self._btn_view_errors

    def set_btn_view_errors(self, btn_view_errors: QPushButton) -> None:
        self._btn_view_errors = btn_view_errors

    def get_job_id(self) -> int:
        return self._job_id

    def set_job_id(self, job_id: int) -> None:
        self._job_id = job_id

    def get_expression(self) -> str:
        return self._expression

    def get_expr_info(self) -> bandmath.BandMathExprInfo:
        return self._expr_info

    def get_variables(self) -> Dict[str, Tuple[bandmath.VariableType, Any]]:
        return self._variables

    def get_input_folder(self) -> str:
        return self._input_folder

    def set_input_folder(self, input_folder: str) -> None:
        self._input_folder = input_folder

    def get_output_folder(self) -> str:
        return self._output_folder

    def set_output_folder(self, output_folder: str) -> None:
        self._output_folder = output_folder

    def get_load_into_wiser(self) -> bool:
        return self._load_into_wiser

    def set_load_into_wiser(self, load_into_wiser: bool) -> None:
        self._load_into_wiser = load_into_wiser

    def get_result_suffix(self) -> str:
        return self._result_suffix

    def set_result_suffix(self, result_suffix: str) -> None:
        self._result_suffix = result_suffix
    
    def get_errors(self) -> Optional[List[Tuple[str, str, str]]]:
        return self._errors
    
    def set_errors(self, errors: List[Tuple[str, str, str]]) -> None:
        self._errors = errors
    
    def __eq__(self, other):
        if not isinstance(other, BandmathBatchJob):
            return False
        return (
            self._job_id == other._job_id and
            self._expression == other._expression and
            self._input_folder == other._input_folder and
            self._output_folder == other._output_folder and
            self._load_into_wiser == other._load_into_wiser and
            self._result_suffix == other._result_suffix
        )

def icon_text_label(text: str, icon_path: str, icon_size: int = 16) -> QWidget:
    """Return a QWidget with an icon on the left and text on the right."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)

    # Icon (resource path like ':/icons/wiser.ico' or ':/icons/choose-bands.svg')
    icon_lbl = QLabel()
    icon = QIcon(icon_path)
    if not icon.isNull():
        pm = icon.pixmap(QSize(icon_size, icon_size), QIcon.Normal, QIcon.Off)
        icon_lbl.setPixmap(pm)
    icon_lbl.setFixedSize(icon_size, icon_size)
    h.addWidget(icon_lbl, 0, Qt.AlignVCenter)

    # Text
    text_lbl = QLabel(text)
    text_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    h.addWidget(text_lbl, 0, Qt.AlignVCenter)

    h.addStretch(1)
    return w

class BatchJobInfoWidget(QWidget):
    def __init__(self, expression: str, variables: Dict[str, Tuple[bandmath.VariableType, Any]], 
                 input_folder: str, output_folder: str, load_results_into_wiser: bool,
                 result_name: str, width_hint=150, parent=None):
        super().__init__(parent)
        self._width_hint = width_hint

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Expression (label + read-only line edit)
        layout.addWidget(QLabel("Expression:"))
        le_expr = QLineEdit()
        le_expr.setReadOnly(True)
        le_expr.setText(expression)
        le_expr.setToolTip(expression)
        layout.addWidget(le_expr)

        layout.addWidget(QLabel("Assignments:"))

        lbl_assign = QLabel()
        lbl_assign.setWordWrap(False)  # single line
        disp_text, tip_text = self._build_assignments_summary(variables)
        lbl_assign.setText(disp_text)
        lbl_assign.setToolTip(tip_text)
        layout.addWidget(lbl_assign)

        # Input Folder (label + read-only line edit)
        layout.addWidget(QLabel("Input Folder:"))
        le_input = QLineEdit()
        le_input.setReadOnly(True)
        le_input.setText(input_folder)
        le_input.setCursorPosition(0)
        le_input.setToolTip(input_folder)
        layout.addWidget(le_input)

        # Output Folder (only if exists)
        if output_folder:
            layout.addWidget(QLabel("Output Folder:"))
            le_output = QLineEdit()
            le_output.setReadOnly(True)
            le_output.setText(output_folder)
            le_input.setCursorPosition(0)
            le_output.setToolTip(output_folder)
            layout.addWidget(le_output)

        layout.addWidget(icon_text_label("Load Into WISER",
                                 ":/icons/wiser.ico"))

        # Result Prefix (label + read-only line edit)
        layout.addWidget(QLabel("Result Prefix:"))
        le_result = QLineEdit()
        le_result.setReadOnly(True)
        le_result.setText(result_name)
        le_result.setToolTip(result_name)
        layout.addWidget(le_result)

        self.setLayout(layout)

        # Let the view know we can grow, but prefer the given width hint
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def sizeHint(self):
        # Use the layout’s computed height but our fixed-ish width hint
        h = self.layout().sizeHint().height() if self.layout() else super().sizeHint().height()
        return QSize(self._width_hint, h)
    
    def truncate_text(self, text: str, max_len: int = 15) -> str:
        """Truncate text to max_len characters, appending '...' if needed."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _build_assignments_summary(self, variables: Dict[str, Tuple[bandmath.VariableType, Any]]) -> Tuple[str, str]:
        """Return (display_text_one_line, tooltip_multiline) for the Assignments label."""
        entries: List[str] = []
        for var_name, (var_type, value) in variables.items():
            entries.append(self._format_var_entry(var_name, var_type, value))

        # One-line display: first two; show "..." if more
        first_two = entries[:2]
        display = "\n".join(first_two)
        if len(entries) > 2:
            display = display + "\n..."

        # Tooltip: full list, one per line
        tooltip = "\n".join(entries)
        return display, tooltip

    def _format_var_entry(self, var_name: str, var_type: bandmath.VariableType, value: Any) -> str:
        """Format per your rules."""
        try:
            t = var_type
        except Exception:
            # In case a plain enum value sneaks in
            t = bandmath.VariableType(var_type)

        # --- IMAGE (cube) ---
        if t == bandmath.VariableType.IMAGE_CUBE:
            ds_name = self._safe_name(value)
            return f'{var_name}: Image, {self.truncate_text(ds_name, 15)}'

        # --- IMAGE BAND ---
        if t == bandmath.VariableType.IMAGE_BAND:
            ds = self._band_dataset(value)
            ds_name = self._safe_name(ds) if ds else "?"
            # Prefer wavelength if available, else index
            wvl = self._band_wavelength(value)
            if wvl is not None:
                val, unit = wvl
                return f'{var_name}: Band, {self.truncate_text(ds_name, 10)}; Wvl: {self._fmt_number(val)} {unit}'
            idx = self._band_index(value)
            if idx is not None:
                return f'{var_name}: Band, {self.truncate_text(ds_name, 10)}; Index: {idx}'
            return f'{var_name}: Band, {self.truncate_text(ds_name, 10)}'

        # --- SPECTRUM ---
        if t == bandmath.VariableType.SPECTRUM:
            sname = self._safe_name(value)
            sname = self._strip_spectrum_prefix(sname)
            return f'{var_name}: Spectrum, {self.truncate_text(sname, 10)}'

        # --- BATCH IMAGE ---
        if t == bandmath.VariableType.IMAGE_CUBE_BATCH:
            return f'{var_name}: Batch Image'

        # --- BATCH BAND ---
        if t == bandmath.VariableType.IMAGE_BAND_BATCH:
            wvl = self._band_wavelength(value)
            if wvl is not None:
                val, unit = wvl
                return f'{var_name}: Batch Band; Wvl: {self._fmt_number(val)} {unit}'
            idx = self._band_index(value)
            if idx is not None:
                return f'{var_name}: Batch Band; Index: {idx}'
            return f'{var_name}: Batch Band'

        # Fallback (numbers/strings/etc.)
        return f'{var_name}: {t.name.title()}'

    def _safe_name(self, obj: Any) -> str:
        """Try common getters to retrieve a human name; fallback to str(obj)."""
        if obj is None:
            return "?"
        for attr in ("get_name", "name"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    val = val() if callable(val) else val
                    if isinstance(val, str) and val:
                        return val
                except Exception:
                    pass
        return str(obj)

    def _band_dataset(self, band_obj: Any) -> Optional[Any]:
        """Try to fetch the dataset object from a RasterDataBand-like object."""
        for attr in ("get_dataset", "dataset"):
            if hasattr(band_obj, attr):
                try:
                    val = getattr(band_obj, attr)
                    return val() if callable(val) else val
                except Exception:
                    pass
        return None

    def _band_index(self, band_obj: Any) -> Optional[int]:
        """Fetch the band index if present."""
        for attr in ("get_band_index", "band_index", "index"):
            if hasattr(band_obj, attr):
                try:
                    val = getattr(band_obj, attr)
                    val = val() if callable(val) else val
                    if isinstance(val, (int, np.integer)):
                        return int(val)
                except Exception:
                    pass
        return None

    def _band_wavelength(self, band_obj: Any) -> Optional[Tuple[float, str]]:
        """
        Try to fetch (value, unit_str) for wavelength from either:
        - Quantity attribute (e.g., .wavelength -> Quantity)
        - Numeric+units attributes (e.g., .wavelength_value + .wavelength_units)
        Returns None if not available.
        """
        # Quantity style
        for attr in ("wavelength", "get_wavelength"):
            if hasattr(band_obj, attr):
                try:
                    q = getattr(band_obj, attr)
                    q = q() if callable(q) else q
                    # Quantity from astropy
                    if hasattr(q, "unit") and hasattr(q, "value"):
                        return (float(q.value), str(q.unit))
                except Exception:
                    pass

        # Separate value + units style (used by RasterDataBatchBand in your code)
        val = None
        if hasattr(band_obj, "wavelength_value"):
            try:
                v = getattr(band_obj, "wavelength_value")
                val = v() if callable(v) else v
            except Exception:
                pass
        unit = None
        if hasattr(band_obj, "wavelength_units"):
            try:
                u_ = getattr(band_obj, "wavelength_units")
                unit = u_() if callable(u_) else u_
            except Exception:
                pass
        if val is not None and unit:
            try:
                return (float(val), str(unit))
            except Exception:
                return (val, str(unit))
        return None

    def _strip_spectrum_prefix(self, name: str) -> str:
        n = name.strip()
        low = n.lower()
        for prefix in ("spectrum at ", "spectrum "):
            if low.startswith(prefix):
                return n[len(prefix):]
        return n

    def _fmt_number(self, x: Union[int, float]) -> str:
        """Pretty number: ints as ints; small floats compact."""
        if isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer()):
            return str(int(x))
        return f"{x:.6g}"

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

        # Expression-info from the most recenly completed expression
        self._expr_info: Optional[bandmath.BandMathExprInfo] = None

        # Set up the UI state
        self._ui = Ui_BandMathDialog()
        self._ui.setupUi(self)

        # Hook up event handlers

        self._expr_filter = ExpressionReturnEventFilter(self)
        self._ui.ledit_expression.installEventFilter(self._expr_filter)

        #==================================
        # "Current expression" UI widgets

        self._ui.btn_toggle_help.clicked.connect(self._on_toggle_help)

        # Always start with the help info hidden.
        # TODO(donnie):  This isn't working.
        # self._ui.tedit_bandmath_help.setVisible(False)

        self._ui.ledit_expression.editingFinished.connect(lambda: self._analyze_expr())
        self._ui.btn_add_to_saved.clicked.connect(self._on_add_expr_to_saved)

        
        #==================================
        # Configure the clear inputs button
        reset_button = self._ui.buttonBox.button(QDialogButtonBox.Reset)
        reset_button.setText('Clear Inputs')
        reset_button.clicked.connect(self._on_clear_inputs)


        #==================================
        # Variable-bindings table

        # TODO(donnie):  Coming soon...
        # self._ui.tbl_variables.setItemDelegateForColumn(1, VariableTypeDelegate())

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
            bandmath.VariableType.IMAGE_CUBE_BATCH: self.tr('Batch - Image'),
            bandmath.VariableType.IMAGE_BAND_BATCH: self.tr('Batch - Image Band')
        }

        #==================================
        # Batch processing initialization
        
        # There are other values important to batch processing:
        # If batch processing is enabled, if we load results into
        # WISER, and if there is an output path folder. These
        # are done with getters.

        self._batch_jobs: List[BandmathBatchJob] = []
        self._init_batch_process_ui()

    def _init_batch_process_ui(self):
        # Wire up folder pickers
        self._ui.btn_input_folder.clicked.connect(
            lambda: self._pick_input_folder("Select input folder")
        )
        self._ui.btn_output_folder.clicked.connect(
            lambda: self._pick_output_folder("Select output folder")
        )
        self._ui.chkbox_enable_batch.clicked.connect(self._on_enable_batch_changed)
        self._sync_batch_process_ui()

        self._ui.btn_create_batch_job.clicked.connect(self._on_create_batch_job)

        tbl = self._ui.tbl_wdgt_batch_jobs
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        tbl.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self._ui.btn_run_all.clicked.connect(self._run_all_batch_jobs)

    def _run_all_batch_jobs(self):
        for job in self._batch_jobs:
            self._run_batch_job(job)

    def on_bandmath_job_success(self, job: BandmathBatchJob,
                            results: List[Tuple[bandmath.VariableType, Any]]):
        job.get_btn_start().setEnabled(True)
        job.get_btn_cancel().setEnabled(False)

        bandmath_success_callback(parent=self, app_state=self._app_state,
                                results=results,
                                expression=job.get_expression(),
                                batch_enabled=True, load_into_wiser=job.get_load_into_wiser())

    def on_bandmath_job_started(self, job: BandmathBatchJob, task: ParallelTask):
        try:
            job.get_btn_start().setEnabled(False)
            job.get_btn_cancel().setEnabled(True)
        except RuntimeError as e:
            # There is a point where the process is cancelled and these buttons
            # have been deleted before this callback is called. So we just ignore
            # this runtime error.
            pass

    def on_bandmath_job_cancelled(self, job: BandmathBatchJob, task: ParallelTask):
        try:
            job.get_btn_start().setEnabled(True)
            job.get_btn_cancel().setEnabled(False)
        except RuntimeError as e:
            # There is a point where the process is cancelled and these buttons
            # have been deleted before this callback is called. So we just ignore
            # this runtime error.
            pass

    def on_bandmath_job_status(self, job: BandmathBatchJob, progress: Tuple[str, Dict[str, str]]):
        if progress[0] == "progress":
            progress_dict = progress[1]
            # The dict progress has 3 entries. Numerator, Denominator, and Status message like so:
            # {'Numerator': 1, 'Denominator': 2, 'Status': 'Running'}
            # We will update the progress bar with the Numerator and Denominator and have the status
            # be the text in the progress bar. Access the progress bar with job.get_progress_bar()
            numerator = int(progress_dict.get("Numerator", 0))
            denominator = int(progress_dict.get("Denominator", 1))  # avoid divide-by-zero
            status = progress_dict.get("Status", "")

            progress_bar = job.get_progress_bar()
            progress_bar.setRange(0, denominator)
            progress_bar.setValue(numerator)

            # Display the status text (instead of % completed) inside the bar
            progress_bar.setTextVisible(True)
            progress_bar.setFormat(f"{status} ({numerator}/{denominator})")
        elif progress[0] == "error":
            error_dict = progress[1]
            result_name = error_dict.get("Result Name", None)
            error_message = error_dict.get("Message", None)
            error_traceback = error_dict.get("Traceback", None)

            if result_name and error_message and error_traceback:
                job.set_errors(job.get_errors() + [(result_name, error_message, error_traceback)])
            
            if job.get_errors():
                job.get_btn_view_errors().setEnabled(True)
            else:
                job.get_btn_view_errors().setEnabled(False)
        elif progress[0] == "process_error":
            raise RuntimeError("Process in subprocess:\n" + progress[1]["traceback"])



    def _run_batch_job(self, job: BandmathBatchJob):
        # Run eval bandmath expr
        success_callback = lambda results: self.on_bandmath_job_success(job, results)

        functions = get_plugin_fns(self._app_state)
        started_callback = lambda task: self.on_bandmath_job_started(job, task)
        cancelled_callback = lambda task: self.on_bandmath_job_cancelled(job, task)
        status_callback = lambda progress: self.on_bandmath_job_status(job, progress)
        process_manager = bandmath.eval_bandmath_expr(succeeded_callback=success_callback,
                                                      status_callback=status_callback,
                                                      error_callback=bandmath_error_callback,
                                                      started_callback=started_callback,
                                                      cancelled_callback=cancelled_callback,
                                                      bandmath_expr=job.get_expression(), expr_info=job.get_expr_info(),
                                                      app_state=self._app_state, result_name=job.get_result_suffix(),
                                                      cache=self._app_state.get_cache(), variables=job.get_variables(),
                                                      functions=functions)
        job.set_process_manager(process_manager)

    def _on_create_batch_job(self):
        missing = []

        if not self._expr_info:
            missing.append("Expression Info")
        if not self.get_expression():
            missing.append("Expression")
        if not self._get_input_folder():
            missing.append("Input Folder")
        if not self._get_output_folder() and not self.load_results_into_wiser():
            missing.append("Output Folder or 'Load Results into WISER' checked")
        if not self.get_result_name():
            missing.append("Result Prefix")

        if missing:
            QMessageBox.warning(
                self,
                "Missing Necessary Inputs for Batch Processing",
                "You are missing necessary inputs for batch processing. "
                "The inputs you are missing are:\n- " + "\n- ".join(missing)
            )
            return
        
        job = BandmathBatchJob(
            job_id=self._app_state.get_next_process_pool_id(),
            expression=self.get_expression(),
            expr_info=self._expr_info,
            variables=self.get_variable_bindings(),
            input_folder=self._get_input_folder(),
            output_folder=self._get_output_folder(),
            load_into_wiser=self.load_results_into_wiser(),
            result_suffix=self.get_result_name()
        )

        self._batch_jobs.append(job)
        self._add_batch_job_to_table(job)

    def _add_batch_job_to_table(self, batch_job: BandmathBatchJob):
        t = self._ui.tbl_wdgt_batch_jobs
        new_row = t.rowCount()
        t.insertRow(new_row)

        # Col 0: id item (centered)
        t.setItem(new_row, 0, self._create_job_id_item(batch_job))

        # Col 1: info widget + an item carrying the size hint
        info_widget = self._create_batch_job_info_widget(batch_job)
        info_widget.updateGeometry()  # ensure layout computed before we read sizeHint
        t.setCellWidget(new_row, 1, info_widget)

        size_item = QTableWidgetItem()
        size_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        size_item.setData(Qt.SizeHintRole, info_widget.sizeHint())
        t.setItem(new_row, 1, size_item)

        # Col 2: run controls (start, cancel, remove, progress)
        t.setCellWidget(new_row, 2, self._create_job_run_controls_widget(batch_job))

        # Defer so Qt can polish the embedded widget's layout; then size to contents
        QTimer.singleShot(0, lambda: (t.resizeColumnToContents(1),
                                    t.resizeRowToContents(new_row)))

    def _create_job_id_item(self, batch_job: BandmathBatchJob) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{batch_job.get_job_id()}")
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def _create_job_run_controls_widget(self, batch_job: BandmathBatchJob) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        container.setLayout(layout)

        # Start button (wired up)
        btn_start = QPushButton(self.tr("Start"))
        btn_start.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btn_start.setMaximumWidth(140)
        btn_start.clicked.connect(lambda: self._run_batch_job(batch_job))
        batch_job.set_btn_start(btn_start)
        layout.addWidget(btn_start)

        # Cancel button (no functionality yet)
        btn_cancel = QPushButton(self.tr("Cancel"))
        btn_cancel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btn_cancel.setMaximumWidth(140)
        btn_cancel.clicked.connect(lambda: self._cancel_batch_job(batch_job))
        btn_cancel.setEnabled(False)
        batch_job.set_btn_cancel(btn_cancel)
        layout.addWidget(btn_cancel)

        # Remove button (no functionality yet)
        btn_remove = QPushButton(self.tr("Remove"))
        btn_remove.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btn_remove.setMaximumWidth(140)
        batch_job.set_btn_remove(btn_remove)
        btn_remove.clicked.connect(lambda: self._remove_batch_job(batch_job))
        layout.addWidget(btn_remove)

        # View Errors button (no functionality yet)
        btn_view_errors = QPushButton(self.tr("View Errors"))
        btn_view_errors.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btn_view_errors.setMaximumWidth(140)
        btn_view_errors.clicked.connect(lambda: self._view_batch_job_errors(batch_job))
        btn_view_errors.setEnabled(False)
        batch_job.set_btn_view_errors(btn_view_errors)
        layout.addWidget(btn_view_errors)

        # Progress bar (no functionality yet)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(True)
        progress.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        progress.setMaximumWidth(160)
        batch_job.set_progress_bar(progress)
        layout.addWidget(progress)

        return container

    def _view_batch_job_errors(self, batch_job: BandmathBatchJob):
        '''
        View the errors for the given batch job. Errors are viewed per file.
        '''
        batch_job_errors = batch_job.get_errors()
        if batch_job_errors:
            formatted_errors = "<br><br>".join(
                f"<b>File:</b> {error[0]}<br>&nbsp;&nbsp;<b>Error:</b> {error[1]}"
                for error in batch_job_errors
            )
            QMessageBox.warning(self, "Batch Errors", formatted_errors)
        else:
            QMessageBox.warning(
                self,
                "No Batch Errors",
                "There are no errors for this batch job. Close and reopen the dialog to see new errors."
            )

    def _cancel_batch_job(self, batch_job: BandmathBatchJob):
        process_manager = batch_job.get_process_manager()
        if process_manager is not None:
            process_manager.get_task().cancel()

    def _remove_batch_job(self, batch_job: BandmathBatchJob):
        if batch_job.get_process_manager() is not None:
            batch_job.get_process_manager().get_task().cancel()
        batch_job_row_index = self._get_batch_job_table_row_index(batch_job)
        if batch_job_row_index != -1:
            self._ui.tbl_wdgt_batch_jobs.removeRow(batch_job_row_index)
        self._batch_jobs.remove(batch_job)

    def _get_batch_job_table_row_index(self, batch_job: BandmathBatchJob) -> int:
        # Iterate through the rows in self._ui.tbl_wdgt_batch_jobs
        for i in range(self._ui.tbl_wdgt_batch_jobs.rowCount()):
            if self._ui.tbl_wdgt_batch_jobs.item(i, 0).text() == str(batch_job.get_job_id()):
                return i
        return -1

    def _create_batch_job_info_widget(self, batch_job: BandmathBatchJob) -> QWidget:
        """
        Creates a QWidget with job info in a vertical layout for use inside a QTableWidget.
        Expands vertically as needed, and horizontally up to 150px.
        """
        info_widget = BatchJobInfoWidget(
            expression=batch_job.get_expression(),
            variables=batch_job.get_variables(),
            input_folder=batch_job.get_input_folder(),
            output_folder=batch_job.get_output_folder(),
            load_results_into_wiser=batch_job.get_load_into_wiser(),
            result_name=batch_job.get_result_suffix(),
            width_hint=150
        )

        return info_widget

    def _pick_output_folder(self, title: str) -> None:
        """Open a folder chooser and write the selected path into the given QLineEdit."""
        start_dir = self._ui.ledit_output_folder.text().strip()
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")

        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly

        folder = QFileDialog.getExistingDirectory(
            parent=self._ui.ledit_output_folder.window(),
            caption=title,
            dir=start_dir,
            options=options
        )

        if folder:
            folder = os.path.normpath(folder)
            if folder == os.path.normpath(self._ui.ledit_input_folder.text().strip()):
                QMessageBox.warning(
                    self._ui.ledit_output_folder.window(),
                    "Invalid Output Folder",
                    "The output folder was not selected because it is the same as the input folder."
                )
                return
            self._ui.ledit_output_folder.setText(folder)

    def _pick_input_folder(self, title: str) -> None:
        """Open a folder chooser and write the selected path into the given QLineEdit."""
        start_dir = self._ui.ledit_input_folder.text().strip()
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = os.path.expanduser("~")

        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly

        folder = QFileDialog.getExistingDirectory(
            parent=self._ui.ledit_input_folder.window(),
            caption=title,
            dir=start_dir,
            options=options
        )

        if folder:
            folder = os.path.normpath(folder)
            if folder == os.path.normpath(self._ui.ledit_output_folder.text().strip()):
                QMessageBox.warning(
                    self._ui.ledit_input_folder.window(),
                    "Invalid Input Folder",
                    "The input folder was not selected because it is the same as the output folder."
                )
                return
            self._ui.ledit_input_folder.setText(folder)
            self._analyze_expr()
    
    def _clear_input_folder(self):
        '''
        Clears the batch processing input folder line edit.
        '''
        self._ui.ledit_input_folder.clear()

    def _get_input_folder(self):
        return self._ui.ledit_input_folder.text()

    def _clear_output_folder(self):
        '''
        Clears the batch processing output folder line edit.
        '''
        self._ui.ledit_output_folder.clear()

    def _get_output_folder(self):
        return self._ui.ledit_output_folder.text()

    def _uncheck_load_into_wiser(self):
        '''
        Unchecks the load result into wiser check box.
        '''
        self._ui.chkbox_load_into_wiser.setChecked(False)

    def load_results_into_wiser(self):
        return self._ui.chkbox_load_into_wiser.isChecked()
    
    def is_batch_processing_enabled(self):
        return self._ui.chkbox_enable_batch.isChecked()

    def _get_batch_processing_ui_components(self) -> List[QObject]:
        ui_components = [
            self._ui.hlayout_input_folder,
            self._ui.lbl_input_folder,
            self._ui.ledit_input_folder,
            self._ui.btn_input_folder,
            self._ui.hlayout_output_folder,
            self._ui.lbl_output_folder,
            self._ui.ledit_output_folder,
            self._ui.btn_output_folder,
            self._ui.chkbox_load_into_wiser,
            self._ui.btn_create_batch_job,
            self._ui.lbl_select_batch_output
        ]
        return ui_components

    def _on_enable_batch_changed(self, checked: bool):
        self._sync_batch_process_ui()
        self._analyze_expr()

    def _sync_batch_process_ui(self):
        is_enabled = self._ui.chkbox_enable_batch.isChecked()
        batch_process_ui_elements = self._get_batch_processing_ui_components()
        for element in batch_process_ui_elements:
            if isinstance(element, QWidget):
                element.setVisible(is_enabled)

        dialog_size = self.size()
        batch_job_table_size = self._ui.tbl_wdgt_batch_jobs.size()
        self._ui.tbl_wdgt_batch_jobs.setVisible(is_enabled)
        self._ui.btn_run_all.setVisible(is_enabled)
        self._ui.btn_cancel_all.setVisible(is_enabled)
        if is_enabled:
            dialog_size.setWidth(dialog_size.height() + batch_job_table_size.height())
        else:
            dialog_size.setWidth(dialog_size.width() - batch_job_table_size.width())

        if is_enabled:
            self._ui.lbl_result_name.setText(self.tr('Result suffix (required):'))
        else:
            self._ui.lbl_result_name.setText(self.tr('Result name (optional):'))


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
        self._expr_info = None
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
            functions = get_plugin_fns(self._app_state)
            expr_info = bandmath.get_bandmath_expr_info(expr, bindings, functions)

            if expr_info.result_type not in [bandmath.VariableType.IMAGE_CUBE,
                bandmath.VariableType.IMAGE_BAND, bandmath.VariableType.SPECTRUM,
                bandmath.VariableType.IMAGE_CUBE_BATCH, bandmath.VariableType.IMAGE_BAND_BATCH]:
                self._ui.lbl_result_info.setText(self.tr('Enter an ' +
                    'expression that produces an image cube, band, or spectrum'))
                self._ui.lbl_result_info.setStyleSheet('QLabel { color: red; }')
                return

            self._expr_info = expr_info

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
            # If there is a row for batch processing and batch processing is disabled, remove the row
            # If there is no row for batch processing and batch processing is enabled, add a new row
            batch_proc_mismatch = (self._is_batch_var_row(index) != self.is_batch_processing_enabled())
            # If the current row's dataset got deleted, we need to readd the row
            row_state_changed = not self.is_row_state_unchanged(index)
            if index == -1 or batch_proc_mismatch or row_state_changed:
                if (batch_proc_mismatch and index != -1) or row_state_changed:
                    self._ui.tbl_variables.removeRow(index)

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
                type_widget.addItem(self._variable_types_text[bandmath.VariableType.IMAGE_CUBE], bandmath.VariableType.IMAGE_CUBE)
                type_widget.addItem(self._variable_types_text[bandmath.VariableType.IMAGE_BAND], bandmath.VariableType.IMAGE_BAND)
                type_widget.addItem(self._variable_types_text[bandmath.VariableType.SPECTRUM], bandmath.VariableType.SPECTRUM)
                if self.is_batch_processing_enabled():
                    type_widget.addItem(self._variable_types_text[bandmath.VariableType.IMAGE_CUBE_BATCH], bandmath.VariableType.IMAGE_CUBE_BATCH)
                    type_widget.addItem(self._variable_types_text[bandmath.VariableType.IMAGE_BAND_BATCH], bandmath.VariableType.IMAGE_BAND_BATCH)
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

                # TODO(donnie):  Coming soon...
                # type_guess = guess_variable_type_from_name(var)
                # item = QTableWidgetItem(type_guess)
                # item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
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

        self._ui.tbl_variables.resizeColumnsToContents()


    def _is_batch_var_row(self, row_index: int):
        row_cbox = self._ui.tbl_variables.cellWidget(row_index, 1)
        if row_cbox:
            for i in range(row_cbox.count()):
                text = row_cbox.itemText(i)
                if text == self._variable_types_text[bandmath.VariableType.IMAGE_CUBE_BATCH] or \
                    text == self._variable_types_text[bandmath.VariableType.IMAGE_CUBE_BATCH]:
                    return True
        return False


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

        elif variable_type == bandmath.VariableType.IMAGE_CUBE_BATCH:
            value_widget = make_image_cube_batch_chooser(self.tr('Using Input Folder'))
        elif variable_type == bandmath.VariableType.IMAGE_BAND_BATCH:
            value_widget = ImageBandBatchChooserWidget(self._app_state, self._ui.tbl_variables)
        else:
            raise AssertionError(f'Unrecognized variable type {variable_type}')

        return value_widget


    def _on_variable_type_change(self, type_index: int, var_name: str):
        '''
        When a variable's type changes, the dialog must show a new value-chooser
        for that variable.
        '''
        var_row = self._find_variable_in_bindings(var_name)
        if var_row == -1:
            raise AssertionError(f'Received unrecognized variable name {var_name}')

        var_type = self._ui.tbl_variables.cellWidget(var_row, 1).currentData()
        value_widget = self._make_value_widget(var_type)
        self._ui.tbl_variables.setCellWidget(var_row, 2, value_widget)

        self._ui.tbl_variables.resizeColumnToContents(2)

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

    def _on_clear_inputs(self, checked: bool):
        '''
        Clears all inputs. The inputs to clear are the expression, the variable bindings, the result_name,
        the input_folder, the output folder, and the load result into wiser check box
        '''
        self._clear_expression()
        self._clear_variable_bindings()
        self._clear_result_name()
        if self.is_batch_processing_enabled():
            self._clear_input_folder()
            self._clear_output_folder()
            self._uncheck_load_into_wiser()
        self._analyze_expr()

    def _clear_variable_bindings(self) -> None:
        """
        Removes all rows and associated widgets from the variable-bindings table.
        """
        tbl = self._ui.tbl_variables
        for row in range(tbl.rowCount()):
            for col in range(tbl.columnCount()):
                widget = tbl.cellWidget(row, col)
                if widget is not None:
                    widget.deleteLater()
                    tbl.removeCellWidget(row, col)
        tbl.setRowCount(0)

    def _check_saved_expressions(self):
        if self._ui.cbox_saved_exprs.count() > 0 and self._saved_exprs_modified:
            response = QMessageBox.question(self, self.tr('Un-saved Expressions'),
                self.tr('Do you want to save your changes\n' +
                        'to the saved-expressions list?'))

            if response == QMessageBox.Yes:
                self._on_save_saved_exprs()


    def showEvent(self, event):
        super().showEvent(event)
        # This code is run right before the QDialog is shown
        self._analyze_expr()


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

    def is_row_state_unchanged(self, row: int) -> bool:
        """
        Returns True if the row's currently selected resource still exists in app_state,
        otherwise False. For batch/other types that don't bind to app_state objects,
        returns True.
        """
        tbl = self._ui.tbl_variables
        if row < 0 or row >= tbl.rowCount():
            return True  # out of range -> treat as unchanged

        # Column 1: variable type (QComboBox with data=bandmath.VariableType)
        type_widget = tbl.cellWidget(row, 1)
        if not isinstance(type_widget, QComboBox):
            return True
        var_type = type_widget.currentData()

        # Column 2: value widget (chooser)
        value_widget = tbl.cellWidget(row, 2)
        if value_widget is None:
            return True  # nothing bound yet

        # IMAGE_CUBE -> dataset ID from a QComboBox
        if var_type == bandmath.VariableType.IMAGE_CUBE:
            if isinstance(value_widget, QComboBox):
                ds_id = value_widget.currentData()
                return ds_id is not None and self._app_state.has_dataset(ds_id)
            return False

        # IMAGE_BAND -> dataset ID from DatasetBandChooserWidget
        if var_type == bandmath.VariableType.IMAGE_BAND:
            if isinstance(value_widget, DatasetBandChooserWidget):
                ds_id, _band_idx = value_widget.get_ds_band()
                return ds_id is not None and self._app_state.has_dataset(ds_id)
            return False

        # SPECTRUM -> either spectrum ID (int) or (lib_id, spectrum_index) tuple
        if var_type == bandmath.VariableType.SPECTRUM:
            if isinstance(value_widget, QComboBox):
                info = value_widget.currentData()
                if isinstance(info, int):
                    return self._app_state.has_spectrum(info)
                if isinstance(info, tuple) and len(info) == 2:
                    lib_id, spectrum_index = info
                    if not self._app_state.has_spectral_library(lib_id):
                        return False
                    lib = self._app_state.get_spectral_library(lib_id)
                    return 0 <= spectrum_index < lib.num_spectra()
            return False

        # Batch/other types don't reference app_state entities directly
        return True

    def _clear_expression(self) -> str:
        '''
        Clears the text in the expression editor.
        '''
        self._ui.ledit_expression.clear()

    def get_expression(self) -> str:
        '''
        Returns the math expression as entered by the user, with leading and
        trailing whitespace stripped, and the expression casefolded to
        lowercase.
        '''
        return self._ui.ledit_expression.text().strip().casefold()


    def get_expression_info(self) -> Optional[bandmath.BandMathExprInfo]:
        return self._expr_info


    def get_variable_bindings(self) -> Dict[str, Tuple[bandmath.VariableType, BANDMATH_VALUE_TYPE]]:
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
            var_type = self._ui.tbl_variables.cellWidget(row, 1).currentData()
            value = None

            if var_type == bandmath.VariableType.IMAGE_CUBE:
                ds_id = self._ui.tbl_variables.cellWidget(row, 2).currentData()
                if ds_id is not None and self._app_state.has_dataset(ds_id):
                    value = self._app_state.get_dataset(ds_id)

            elif var_type == bandmath.VariableType.IMAGE_BAND:
                (ds_id, band_index) = self._ui.tbl_variables.cellWidget(row, 2).get_ds_band()
                if ds_id is not None and band_index is not None and self._app_state.has_dataset(ds_id):
                    dataset = self._app_state.get_dataset(ds_id)
                    value = RasterDataBand(dataset, band_index)

            elif var_type == bandmath.VariableType.SPECTRUM:
                spectrum_info = self._ui.tbl_variables.cellWidget(row, 2).currentData()
                if isinstance(spectrum_info, int) and self._app_state.has_spectrum(spectrum_info):
                    value = self._app_state.get_spectrum(spectrum_info)

                elif isinstance(spectrum_info, tuple):
                    (lib_id, spectrum_index) = spectrum_info
                    if self._app_state.has_spectral_library(lib_id):
                        lib = self._app_state.get_spectral_library(lib_id)
                        value = lib.get_spectrum(spectrum_index)

                else:
                    raise TypeError(f'Unrecognized type of spectrum info:  {spectrum_info}')
            elif var_type == bandmath.VariableType.IMAGE_CUBE_BATCH:
                value = self._get_input_folder()
            elif var_type == bandmath.VariableType.IMAGE_BAND_BATCH:
                input_folder = self._get_input_folder()
                band_batch_chooser: ImageBandBatchChooserWidget = self._ui.tbl_variables.cellWidget(row, 2)
                row_mode = band_batch_chooser.get_settings()['mode']
                row_band_index = band_batch_chooser.get_settings()['index']
                row_wavelength_value = band_batch_chooser.get_settings()['wavelength']
                row_wavelength_units = band_batch_chooser.get_settings()['units_key']
                row_epsilon = band_batch_chooser.get_settings()['epsilon']
                if row_mode == ImageBandBatchChooserWidget.Mode.INDEX:  
                    value = RasterDataBatchBand(input_folder, band_index=row_band_index)
                elif row_mode == ImageBandBatchChooserWidget.Mode.WAVELENGTH:
                    value = RasterDataBatchBand(input_folder, wavelength_value=row_wavelength_value, wavelength_units=row_wavelength_units, epsilon=row_epsilon)
                else:
                    raise AssertionError(f'Unrecognized mode: {row_mode}')
            else:
                raise AssertionError(
                    f'Unrecognized binding type {var_type} for variable {var}')

            variables[var] = (var_type, value)

        return variables


    def _clear_result_name(self) -> None:
        '''
        Clears the result name / result suffix line edit
        '''
        self._ui.ledit_result_name.clear()


    def get_result_name(self) -> Optional[str]:
        '''
        Return the optional name of the result.
        '''
        name = self._ui.ledit_result_name.text().strip()
        if not name:
            name = None

        return name
