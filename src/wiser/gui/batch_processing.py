import logging

from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.batch_processing_ui import Ui_BatchProcessing

from wiser import plugins
from wiser.plugins import Plugin, BatchProcessingPlugin

from wiser.plugins.types import BatchProcessingInputType

from wiser.gui.bandmath_dialog import make_dataset_chooser, make_spectrum_chooser, DatasetBandChooserWidget

import itertools

if TYPE_CHECKING:
    from .app_state import ApplicationState

def make_input_widget(app_state, in_type: BatchProcessingInputType) -> QWidget:
    """
    Factory that converts an input-type enum into an editor widget.

    *   IMAGE_CUBE   → dataset combo
    *   IMAGE_BAND   → dataset+band chooser widget
    *   SPECTRUM     → spectrum combo
    *   NUMBER       → QDoubleSpinBox
    """
    if in_type == BatchProcessingInputType.IMAGE_CUBE:
        return make_dataset_chooser(app_state)

    if in_type == BatchProcessingInputType.IMAGE_BAND:
        return DatasetBandChooserWidget(app_state)

    if in_type == BatchProcessingInputType.SPECTRUM:
        return make_spectrum_chooser(app_state)

    if in_type == BatchProcessingInputType.NUMBER:
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e9, 1e9)
        return spin

    raise AssertionError(f"Unsupported input type: {in_type}")

class JobRowWidget(QWidget):
    """
    A single job row:
    | Job<ID> | inputs (grows) | outputs (grows) | [Run] [Cancel] |
    """
    def __init__(self, app_state: 'ApplicationState', plugin: 'BatchProcessingPlugin', job_id: int, parent=None):
        super().__init__(parent)

        self._app_state = app_state
        self._plugin = plugin
        self._input_types = plugin.get_ordered_input_types()
        self._output_types = plugin.get_ordered_output_types()

        h = QHBoxLayout(self)
        h.setContentsMargins(4, 4, 4, 4)
        h.setSpacing(6)

        # Job ID (fixed width)
        lbl_id = QLabel(f"Job {job_id}")
        lbl_id.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        h.addWidget(lbl_id)

        # -------------- INPUTS: dynamic form ----------------------------
        inputs_box   = QWidget()
        inputs_form  = QFormLayout(inputs_box)
        inputs_form.setContentsMargins(0, 0, 0, 0)
        inputs_form.setHorizontalSpacing(4)

        # one row per declared input type
        for idx, in_type in enumerate(self._input_types, 1):
            print(f"in_type: {in_type}")
            editor = make_input_widget(app_state, in_type)
            print(f"editor type: {type(editor)}")
            inputs_form.addRow(f"Arg {idx}:", editor)

        # make the whole panel stretchy
        inputs_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        h.addWidget(inputs_box, 1)

        # ── OUTPUTS FORM ──────────────────────────────────────────────
        outputs_box = QWidget()
        v_out      = QVBoxLayout(outputs_box)
        v_out.setContentsMargins(0,0,0,0)
        v_out.setSpacing(4)

        # Change the static text to a prompt:
        lbl_enter_outputs = QLabel("Enter Output Names:")
        v_out.addWidget(lbl_enter_outputs)

        # Build a little form:  “Output 1:” [ QLineEdit ] …
        form = QFormLayout()
        form.setContentsMargins(0,0,0,0)
        for idx, out_type in enumerate(self._output_types, start=1):
            le = QLineEdit()
            form.addRow(f"Output {idx}:", le)

        v_out.addLayout(form)
        outputs_box.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        h.addWidget(outputs_box, 1)

        # Two stacked buttons
        btn_box = QWidget()
        vbtn = QVBoxLayout(btn_box)
        vbtn.setContentsMargins(0, 0, 0, 0)
        vbtn.setSpacing(4)
        vbtn.addWidget(QPushButton("Run"))
        vbtn.addWidget(QPushButton("Cancel"))
        vbtn.addStretch()
        btn_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        h.addWidget(btn_box)

class PluginRowWidget(QWidget):
    """
    A row that shows a plugin name, + / - buttons, and its own QListWidget
    containing JobRowWidgets.
    """
    _job_id_counter = itertools.count(1)  # demo only

    def __init__(self, app_state: 'ApplicationState', plugin_name: str, plugin: 'BatchProcessingPlugin', parent=None):
        super().__init__(parent)

        self._app_state = app_state
        self._plugin = plugin

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # --- Header bar ----------------------------------------------------
        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)

        display_name = plugin_name.rsplit('.', 1)[-1]
        lbl_name = QLabel(display_name)
        h.addWidget(lbl_name)

        h.addStretch()  # push buttons to the far right

        btn_add = QPushButton("+")
        btn_remove = QPushButton("-")
        btn_add.setFixedWidth(24)
        btn_remove.setFixedWidth(24)
        h.addWidget(btn_add)
        h.addWidget(btn_remove)

        outer.addWidget(header)

        # --- Job list ------------------------------------------------------
        self.jobList = QListWidget()
        self.jobList.setSelectionMode(QAbstractItemView.SingleSelection)
        outer.addWidget(self.jobList)

        # --- Wire up buttons ----------------------------------------------
        btn_add.clicked.connect(self._add_job)
        btn_remove.clicked.connect(self._remove_selected_job)

    # ---------------------------------------------------------------------
    # Slots
    def _add_job(self):
        job_id = next(self._job_id_counter)
        widget = JobRowWidget(self._app_state, self._plugin, job_id)
        item = QListWidgetItem(self.jobList)
        item.setSizeHint(widget.sizeHint())
        self.jobList.addItem(item)
        self.jobList.setItemWidget(item, widget)

    def _remove_selected_job(self):
        row = self.jobList.currentRow()
        if row >= 0:
            self.jobList.takeItem(row)


class BatchProcessing(QWidget):

    def __init__(self, app_state: 'ApplicationState', parent=None):
        super().__init__(parent=parent)

        self._app_state = app_state

        self._ui = Ui_BatchProcessing()
        self._ui.setupUi(self)

        self._batch_processing_plugins: List[Tuple[str, plugins.BatchProcessingPlugin]] = []

        self.load_batch_processing_plugins()


    def load_batch_processing_plugins(self):
        '''
        Loads all batch processing plugins and adds them to the UI.
        '''
        for plugin_name, plugin in self._app_state.get_plugins().items():
            if isinstance(plugin, plugins.BatchProcessingPlugin):
                print(f"plugin_name: {plugin_name}")
                self._batch_processing_plugins.append((plugin_name, plugin))
                self._add_plugins(plugin_name, plugin)
    
    def _add_plugins(self, plugin_name: str, plugin: 'Plugin'):
        plugin_widget = PluginRowWidget(self._app_state, plugin_name, plugin)

        item = QListWidgetItem(self._ui.list_wdgt_plugins)
        item.setSizeHint(plugin_widget.sizeHint())

        self._ui.list_wdgt_plugins.addItem(item)
        self._ui.list_wdgt_plugins.setItemWidget(item, plugin_widget)
