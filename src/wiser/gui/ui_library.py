from typing import TYPE_CHECKING, Any, Optional, List, Dict, Tuple
from enum import IntEnum

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from PySide2.QtWidgets import (
    QDialog,
    QLabel,
    QComboBox,
    QGridLayout,
    QDialogButtonBox,
    QSpinBox,
    QLineEdit,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)

from PySide2.QtGui import QDoubleValidator, QIntValidator
from PySide2.QtCore import Qt, Signal

from .util import populate_combo_box_with_units

from wiser.raster.dataset import RasterDataSet, RasterDataBand

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.spectrum import Spectrum


class SingleItemChooserDialog(QDialog):
    def __init__(
        self,
        label_name: str,
        app_state: "ApplicationState",
        description: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.setWindowTitle(self.tr("Selection Dialog"))

        if app_state is None or label_name is None:
            return

        self._app_state = app_state
        self._accepted = False

        self._lbl = QLabel(self.tr(label_name), self)
        self._cbox = QComboBox(self)  # leave empty, subclass will populate

        layout = QGridLayout(self)
        layout.addWidget(self._lbl, 0, 0)
        layout.addWidget(self._cbox, 0, 1)

        self._lbl_description: QLabel = None
        if description:
            self._lbl_description = QLabel(self.tr(description), self)
            self._lbl_description.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(self._lbl_description, 1, 0, 1, 2)
            self._create_button_box(layout=layout, bbox_row=2)
        else:
            self._create_button_box(layout=layout)

        self.setLayout(layout)
        self.setFixedSize(self.sizeHint())

    def _create_button_box(
        self,
        layout: QGridLayout,
        bbox_row=1,
        bbox_col_span=2,
    ):
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )

        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # Put buttons on next row spanning both columns
        layout.addWidget(self._button_box, bbox_row, 0, 1, bbox_col_span)

    def get_chosen_object(self) -> Optional[Any]:
        """
        Retrieves the chosen object after a user has accepted the dialog.

        If the user didn't accept the dialog, this will return None.
        """
        raise NotImplementedError("This function must be implemented by the subclass.")

    def accept(self):
        self._accepted = True
        super().accept()

    def reject(self):
        self._accepted = False
        return super().reject()


class DynamicInputType(IntEnum):
    COMBO_BOX = 0
    FLOAT_NO_UNITS = 1
    FLOAT_UNITS = 2
    INT_NO_UNITS = 3
    INT_UNITS = 4
    STRING = 5


class DynamicInputDialog(QDialog):
    """
    This class lets plugin creators make a dialog with inputs that they can set!

    You make use of this class by using the function create_input_dialog
    """

    def __init__(
        self,
        dialog_title: Optional[str] = None,
        description: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._dialog_title: Optional[str] = dialog_title
        self._description: Optional[str] = description
        self._return_dict: Dict[str, Any] = {}

    def create_input_dialog(
        self, inputs: List[Tuple[str, str, DynamicInputType, Optional[List[Any]]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Create and execute a dynamic input dialog.

        Args:
            inputs:
                A list of tuples defining the requested inputs:
                [
                    ("<display name>", "<return_key>", DynamicInputType, [<combo_items> or None]),
                    ...
                ]

                - For DynamicInputType.COMBO_BOX:
                    Provide a 4th element: a list of items for the combo box.
                - For DynamicInputType.FLOAT_NO_UNITS:
                    4th element is ignored (can be None).
                - For DynamicInputType.FLOAT_UNITS:
                    4th element is ignored (can be None). A unit selector combo is created
                    using populate_combo_box_with_units.

        Returns:
            Optional[Dict]:
                - On Accepted: { "<return_key>": value, ... }
                - On Rejected: None
        """

        self._return_dict = {}

        if self._dialog_title:
            self.setWindowTitle(self._dialog_title)

        layout = QGridLayout(self)
        row = 0

        # Build user input rows
        for spec in inputs:
            if len(spec) < 3:
                raise ValueError(
                    "Each input spec must have at least 3 elements: "
                    "(display_name, return_key, DynamicInputType[, options])."
                )

            display_name, return_key, input_type = spec[0], spec[1], spec[2]
            options = spec[3] if len(spec) > 3 else None

            label = QLabel(display_name, self)
            layout.addWidget(label, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)

            if input_type == DynamicInputType.COMBO_BOX:
                if options is None:
                    raise ValueError(
                        f"COMBO_BOX input '{display_name}' requires options list as 4th tuple element."
                    )

                combo = QComboBox(self)
                for opt in options:
                    combo.addItem(str(opt))

                # Initialize dict with current value (if any)
                if combo.count() > 0:
                    self._return_dict[return_key] = combo.currentText()
                else:
                    self._return_dict[return_key] = None

                combo.currentIndexChanged.connect(
                    lambda _idx, key=return_key, c=combo: self._on_combo_changed(key, c)
                )

                layout.addWidget(combo, row, 1)

            elif input_type == DynamicInputType.FLOAT_NO_UNITS or input_type == DynamicInputType.INT_NO_UNITS:
                line = QLineEdit(self)
                if input_type == DynamicInputType.FLOAT_NO_UNITS:
                    line.setValidator(QDoubleValidator(line))
                    line.setPlaceholderText("Enter number")
                elif input_type == DynamicInputType.INT_NO_UNITS:
                    line.setValidator(QIntValidator(line))
                    line.setPlaceholderText("Enter integer")

                line.textChanged.connect(
                    lambda text, key=return_key: self._on_float_changed_no_units(key, text)
                )

                self._return_dict[return_key] = None  # start unset
                layout.addWidget(line, row, 1)

            elif input_type == DynamicInputType.FLOAT_UNITS or input_type == DynamicInputType.INT_UNITS:
                line = QLineEdit(self)
                if input_type == DynamicInputType.FLOAT_UNITS:
                    line.setValidator(QDoubleValidator(line))
                    line.setPlaceholderText("Enter number")
                elif input_type == DynamicInputType.INT_UNITS:
                    line.setValidator(QIntValidator(line))
                    line.setPlaceholderText("Enter integer")

                unit_combo = QComboBox(self)
                populate_combo_box_with_units(unit_combo)

                # Container so both appear in a single grid cell
                container = QWidget(self)
                hbox = QHBoxLayout(container)
                hbox.setContentsMargins(0, 0, 0, 0)
                hbox.addWidget(line)
                hbox.addWidget(unit_combo)

                self._return_dict[return_key] = None

                # When value changes, recompute
                line.textChanged.connect(
                    lambda text, key=return_key, le=line, uc=unit_combo: self._on_float_units_changed(
                        key, le, uc
                    )
                )
                # When unit changes, recompute with same numeric value
                unit_combo.currentIndexChanged.connect(
                    lambda _idx, key=return_key, le=line, uc=unit_combo: self._on_float_units_changed(
                        key, le, uc
                    )
                )

                layout.addWidget(container, row, 1)
            elif input_type == DynamicInputType.STRING:
                line = QLineEdit(self)
                line.setPlaceholderText("Enter text")

                line.textChanged.connect(
                    lambda text, key=return_key: self._return_dict.__setitem__(key, text)
                )

                self._return_dict[return_key] = None
                layout.addWidget(line, row, 1)
            else:
                raise ValueError(f"Unsupported DynamicInputType for '{display_name}'.")

            row += 1

        if self._description:
            desc_label = QLabel(self._description, self)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label, row, 0, 1, 2)
            row += 1

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, row, 0, 1, 2)

        # Execute dialog
        result = self.exec()
        if result == QDialog.Accepted:
            return self._return_dict
        return {}

    # Internal update helpers

    def _on_combo_changed(self, key: str, combo: QComboBox) -> None:
        self._return_dict[key] = combo.currentText()

    def _on_float_changed_no_units(self, key: str, text: str) -> None:
        text = text.strip()
        if not text:
            self._return_dict[key] = None
            return
        try:
            self._return_dict[key] = float(text)
        except ValueError:
            # Keep last valid or set None; here we choose None
            self._return_dict[key] = None

    def _on_float_units_changed(self, key: str, line_edit: QLineEdit, unit_combo: QComboBox) -> None:
        text = line_edit.text().strip()
        if not text:
            self._return_dict[key] = None
            return

        try:
            value = float(text)
        except ValueError:
            self._return_dict[key] = None
            return

        unit = unit_combo.currentData()
        if unit is None:
            # "None" selection → plain float
            self._return_dict[key] = value
        else:
            # Assuming astropy.units, store as Quantity
            self._return_dict[key] = value * unit


class SpectrumChooserDialog(SingleItemChooserDialog):
    def __init__(
        self,
        app_state: "ApplicationState",
        description: Optional[str] = None,
        parent=None,
    ):
        super().__init__(
            label_name="Spectrum Chooser",
            app_state=app_state,
            description=description,
            parent=parent,
        )

        spectra = self._app_state.get_all_spectra()

        for spectrum in spectra.values():
            self._cbox.addItem(spectrum.get_name(), spectrum.get_id())

    def get_chosen_object(self) -> Optional["Spectrum"]:
        if not self._accepted:
            return None
        spectrum_id = self._cbox.currentData()
        spectrum = self._app_state.get_spectrum(spectrum_id)
        return spectrum


class ROIChooserDialog(SingleItemChooserDialog):
    def __init__(
        self,
        app_state: "ApplicationState",
        description: Optional[str] = None,
        parent=None,
    ):
        super().__init__(
            label_name="ROI (Region Of Interest) Chooser",
            app_state=app_state,
            description=description,
            parent=parent,
        )

        rois = self._app_state.get_rois()

        for roi in rois:
            self._cbox.addItem(roi.get_name(), roi.get_id())

    def get_chosen_object(self) -> Optional["RasterDataSet"]:
        if not self._accepted:
            return None
        roi_id = self._cbox.currentData()
        kwargs = {"id": roi_id}
        roi = self._app_state.get_roi(**kwargs)
        return roi


class DatasetChooserDialog(SingleItemChooserDialog):
    def __init__(
        self,
        app_state: "ApplicationState",
        description: Optional[str] = None,
        parent=None,
    ):
        super().__init__(
            label_name="Dataset Chooser",
            app_state=app_state,
            description=description,
            parent=parent,
        )

        datasets = self._app_state.get_datasets()

        for dataset in datasets:
            self._cbox.addItem(dataset.get_name(), dataset.get_id())

    def get_chosen_object(self) -> Optional["RasterDataSet"]:
        if not self._accepted:
            return None
        ds_id = self._cbox.currentData()
        dataset = self._app_state.get_dataset(ds_id)
        return dataset


class BandChooserDialog(SingleItemChooserDialog):
    def __init__(
        self,
        app_state: "ApplicationState",
        description: Optional[str] = None,
        parent=None,
    ):
        super().__init__(label_name=None, app_state=None, parent=parent)

        self._app_state = app_state

        # Dataset row
        self._lbl_dataset = QLabel(self.tr("Dataset"), self)
        self._cbox_dataset = QComboBox(self)

        # Band row
        self._lbl_band = QLabel(self.tr("Band #"), self)
        self._sbox_band = QSpinBox(self)
        self._sbox_band.setMinimum(0)  # set max later based on dataset
        self._cbox_band = QComboBox(self)

        layout = QGridLayout(self)
        layout.addWidget(self._lbl_dataset, 0, 0)
        layout.addWidget(self._cbox_dataset, 0, 1, 1, 2)

        layout.addWidget(self._lbl_band, 1, 0)
        layout.addWidget(self._sbox_band, 1, 1)
        layout.addWidget(self._cbox_band, 1, 2)

        self._lbl_description_title: QLabel = None
        self._lbl_description: QLabel = None
        if description:
            self._lbl_description = QLabel(self.tr(description), self)
            self._lbl_description.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            layout.addWidget(self._lbl_description, 2, 1, 1, 3)
            self._create_button_box(layout=layout, bbox_row=3, bbox_col_span=3)
        else:
            self._create_button_box(layout=layout, bbox_row=2, bbox_col_span=3)

        self.setLayout(layout)

        # Populate combo box
        datasets = self._app_state.get_datasets()
        for dataset in datasets:
            self._cbox_dataset.addItem(dataset.get_name(), dataset.get_id())

        self._cbox_dataset.currentIndexChanged.connect(self._on_cbox_dataset_changed)

        # Initial band chooser sync
        self._sync_band_chooser()

        # Keeps the combo box and spin box in sync
        self._cbox_band.currentIndexChanged.connect(self._sync_cbox_to_sbox)
        self._sbox_band.valueChanged.connect(self._sync_sbox_to_cbox)

        self.setFixedSize(self.sizeHint())

    def _on_cbox_dataset_changed(self, checked=False):
        self._sync_band_chooser()

    def _sync_band_chooser(self):
        ds_id = self._cbox_dataset.currentData()
        if self._app_state.has_dataset(ds_id):
            dataset = self._app_state.get_dataset(ds_id)
            bands = dataset.band_list()
            descriptions = list([band["description"] for band in bands])
            if descriptions[0]:
                band_descriptions = list(
                    (
                        f"Band {descriptions.index(descr)}: " + descr,
                        descriptions.index(descr),
                    )
                    for descr in descriptions
                )
            else:
                band_descriptions = list((f"Band {i}", i) for i in range(len(descriptions)))
            self._cbox_band.clear()
            for descr, index in band_descriptions:
                self._cbox_band.addItem(self.tr(f"{descr}"), index)
            self._sbox_band.setRange(0, len(band_descriptions) - 1)
        else:
            self._cbox_band.clear()

    def _sync_cbox_to_sbox(self, checked=False):
        idx = self._cbox_band.currentIndex()
        self._sbox_band.setValue(idx)

    def _sync_sbox_to_cbox(self, checked=False):
        idx = self._sbox_band.value()
        self._cbox_band.setCurrentIndex(idx)

    def get_chosen_object(self) -> RasterDataBand:
        if not self._accepted:
            return None
        dataset = self._app_state.get_dataset(self._cbox_dataset.currentData())
        band_id = self._cbox_band.currentData()
        band = RasterDataBand(dataset=dataset, band_index=band_id)
        return band


class TableDisplayWidget(QWidget):
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._layout = QVBoxLayout(self)

        # Description label
        self._description_label = QLabel(self)
        self._description_label.setWordWrap(True)
        self._description_label.hide()
        self._layout.addWidget(self._description_label)

        # Table widget
        self._table = QTableWidget(self)
        self._table.setAlternatingRowColors(True)
        self._table.setCornerButtonEnabled(False)
        self._table.verticalHeader().setVisible(False)

        # Selectable, but not editable
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setFocusPolicy(Qt.StrongFocus)

        # Allow column expansion
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)

        self._layout.addWidget(self._table)

    def create_table(
        self,
        header: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Creates a GUI item that has a description at the top and a
        QTableWidget with the specified header and the given rows.
        The table widget is only for display, so no interaction
        can be done.

        Args:
            header (List[str]):
                A list of each of the column header names in order

            rows (List[List[Any]]):
                Each of the rows to put into the table in order. The
                elements in the outer list correspond to rows in the
                table. The elements in the inner list correspond to
                columns for that row.

            description (str, optional):
                Optional text placed above the table.

            title (str, optional):
                Optional title displayed above the description.
        """

        if title:
            self.setWindowTitle(title)

        if description:
            self._description_label.setText(description)
            self._description_label.show()
        else:
            self._description_label.hide()

        # Reset the table
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

        # Set headers
        self._table.setColumnCount(len(header))
        self._table.setHorizontalHeaderLabels(header)

        # Insert rows
        self._table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            for c, value in enumerate(row[: len(header)]):
                text = "" if value is None else str(value)
                item = QTableWidgetItem(text)

                # Allow selection but no editing
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

                self._table.setItem(r, c, item)

        # Adjust final sizing
        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()

    def closeEvent(self, event):
        self.closed.emit()
        return super().closeEvent(event)


class MatplotlibDisplayWidget(QWidget):
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def create_plot(
        self,
        figure: Figure,
        axes: Axes,
        window_title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Creates a QWidget that containts the figure and axes passed in. You
        can optionally set the window title and provide a description.
        """
        self._figure = figure
        self._figure_canvas = FigureCanvas(self._figure)
        self._axes = axes

        self._toolbar = NavigationToolbar(self._figure_canvas, self)

        if window_title:
            self.setWindowTitle(self.tr(window_title))

        layout = QVBoxLayout(self)

        # Description label
        if description:
            description_label = QLabel(self)
            description_label.setWordWrap(True)
            description_label.setText(self.tr(description))
            layout.addWidget(description_label)

        layout.addWidget(self._toolbar)
        layout.addWidget(self._figure_canvas)

    def closeEvent(self, event):
        self.closed.emit()
        return super().closeEvent(event)
