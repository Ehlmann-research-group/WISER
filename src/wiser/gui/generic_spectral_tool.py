# Generic shared logic for spectral computations (parent class)
from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING, Union
import os
import warnings
import numpy as np
from numba import njit, prange, types
from wiser.utils.numba_wrapper import numba_njit_wrapper

from PySide2.QtWidgets import (
    QDialog,
    QMessageBox,
    QApplication,
    QFileDialog,
    QTableWidgetItem,
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QDialogButtonBox,
    QLabel,
    QFormLayout,
    QWidget,
    QPushButton,
)
from PySide2.QtCore import Qt, QSettings
from astropy import units as u

from wiser.raster.spectrum import NumPyArraySpectrum, Spectrum
from wiser.raster.envi_spectral_library import ENVISpectralLibrary
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.dataset import RasterDataSet
from wiser.raster.loader import RasterDataLoader
from wiser.gui.import_spectra_text import ImportSpectraTextDialog
from wiser.gui.spectrum_plot import SpectrumPlotGeneric
from wiser.gui.generated.generic_spectral_computation_ui import (
    Ui_GenericSpectralComputation,
)
from .util import (
    populate_combo_box_with_units,
    StateChange,
)

from wiser.config import FLAGS

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

DEFAULT_NO_SPECTRA_TEXT = "No spectra collected"
DEFAULT_NO_DATASETS_TEXT = "No image cubes loaded"


class SpectralComputationInputs:
    def __init__(
        self,
        target: Union[NumPyArraySpectrum, "RasterDataSet"],
        mode: str,
        refs: List[NumPyArraySpectrum],
        thresholds: List[float],
        global_thr: float,
        min_wvl: Optional[u.Quantity],
        max_wvl: Optional[u.Quantity],
        lib_name_by_spec_id: Dict[int, str],
    ):
        assert len(refs) == len(thresholds), "Number of refs and thresholds must me equal!"
        self.target = target
        self.mode = mode  # Either 'Spectrum' or 'Image Cube'
        self.refs = refs
        self.thresholds = thresholds
        self.global_thr = global_thr
        self.min_wvl = min_wvl
        self.max_wvl = max_wvl
        self.lib_name_by_spec_id = lib_name_by_spec_id


class GenericSpectralComputationTool(QDialog):
    """
    Template/parent for spectral computations (SAM, SFF, etc.).
    Child classes must implement:
      - SETTINGS_NAMESPACE (str)           : QSettings namespace
      - RUN_BUTTON_TEXT (str)
      - SCORE_HEADER (str)                 : name for Score column in details/history
      - THRESHOLD_SPIN_CONFIG (dict)       : min, max, decimals, step
      - compute_score(self, target, ref) -> (float, dict)
          returns (score, extras), NaN score to skip
      - details_columns(self) -> List[Tuple[str, str]]
          list of (header, key) for detail table (must include "score" and "threshold")
      - filename_stub(self) -> str         : e.g., "SAM" or "SFF"
    """

    SETTINGS_NAMESPACE = "Wiser/GenericSpectral"
    RUN_BUTTON_TEXT = "Run"
    SCORE_HEADER = "Score"
    THRESHOLD_HEADER = "Initial Threshold"
    THRESHOLD_SPIN_CONFIG = dict(min=0.0, max=1.0, decimals=2, step=0.5)

    # default thresholds: children should set self._method_threshold and configure spin
    def __init__(self, widget_name: str, app_state: ApplicationState, parent: QWidget = None):
        super().__init__(parent)

        self._ui = Ui_GenericSpectralComputation()
        self._ui.setupUi(self)
        self.setWindowTitle(widget_name)

        # configure run button label
        self._ui.addRunBtn.setText(self.RUN_BUTTON_TEXT)

        # units combo
        populate_combo_box_with_units(self._ui.cbox_units, use_none_unit=False)

        # threshold spin config
        cfg = self.THRESHOLD_SPIN_CONFIG
        self._spin = self._ui.method_threshold
        self._spin.setRange(cfg.get("min", 0.0), cfg.get("max", 1.0))
        self._spin.setDecimals(cfg.get("decimals", 2))
        self._spin.setSingleStep(cfg.get("step", 0.1))
        self._spin.setKeyboardTracking(False)

        # interpolation note hook
        self._add_interpolation_note()

        self._app_state = app_state
        self._target: Optional[NumPyArraySpectrum] = None

        self._run_history: Dict[str, List[Dict[str, Any]]] = {}

        self._lib_rows = []
        self._spec_rows = []

        self._ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._setup_connections()
        self._init_target_dropdowns()
        self._init_reference_selection()
        self._init_threshold_header()

        self._sessions_purged_flag = False
        self._purge_old_sessions_once()

        self._load_state()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(
                lambda: QSettings(self.SETTINGS_NAMESPACE).remove(f"session/{os.getpid()}")
            )

    # ----------------- To be overridden/used by children -----------------
    def details_columns(self) -> List[Tuple[str, str]]:
        # Default: Score + Threshold (children can override/add columns)
        return [(self.SCORE_HEADER, "score"), ("Threshold", "threshold")]

    def filename_stub(self) -> str:
        return "GENERIC"

    # ----------------- Public setters/getters -----------------

    def get_target_name(self, target: NumPyArraySpectrum) -> str:
        return target.get_name() if target else "<no target>"

    def get_wavelength_min(self) -> Optional[u.Quantity]:
        try:
            min_v = float(self._ui.lineEdit.text())
            unit = self._get_wavelength_units()
            return u.Quantity(min_v, unit)
        except ValueError:
            raise ValueError("Min wavelength must be a number.")

    def get_wavelength_max(self) -> Optional[u.Quantity]:
        try:
            max_v = float(self._ui.lineEdit_2.text())
            unit = self._get_wavelength_units()
            return u.Quantity(max_v, unit)
        except ValueError:
            raise ValueError("Max wavelength must be a number.")

    def _get_wavelength_units(self) -> u.Unit:
        return self._ui.cbox_units.currentData()

    # ----------------- UI wiring -----------------
    def _setup_connections(self):
        ui = self._ui
        ui.addRunBtn.clicked.connect(self._on_run_clicked)
        ui.addSaveCloseBtn.clicked.connect(self.save_and_close)
        ui.addCancelBtn.clicked.connect(self.cancel)
        ui.addLibBtn.clicked.connect(self._on_add_library_clicked)
        ui.addSpecBtn.clicked.connect(self._on_add_spectrum_clicked)
        ui.btn_add_collected_spec.clicked.connect(self._on_add_collected_spectrum)
        ui.SelectTargetData.currentTextChanged.connect(self._on_target_type_changed)
        ui.clearRunsBtn.clicked.connect(self._on_clear_runs_clicked)

        self._app_state.collected_spectra_changed.connect(self._on_collected_spectra_changed)

    def _add_interpolation_note(self) -> None:
        note = QLabel("Interpolation: linear")
        note.setStyleSheet("color:#666; font-size:11px; font-style: italic;")
        note.setAlignment(Qt.AlignRight)

        root = self._ui.grp_settings.layout()
        thr_layout = None
        # find form layout containing the threshold spin
        for i in range(root.count()):
            item = root.itemAt(i)
            lay = getattr(item, "layout", lambda: None)()
            if isinstance(lay, QFormLayout) and lay.indexOf(self._ui.method_threshold) != -1:
                thr_layout = lay
                break
        if thr_layout is not None:
            thr_layout.addRow("", note)

    def _init_threshold_header(self) -> None:
        self._ui.label_threshold.setText(self.THRESHOLD_HEADER)

    def _init_reference_selection(self) -> None:
        for lbl in (
            self._ui.hdr_lib,
            self._ui.hdr_thresh_lib,
            self._ui.hdr_spec,
            self._ui.hdr_thresh_spec,
        ):
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def _init_target_dropdowns(self) -> None:
        ui = self._ui
        ui.SelectTargetData.blockSignals(True)
        ui.SelectTargetData.clear()
        if FLAGS.sff_sam_image_cube:
            ui.SelectTargetData.addItems(["Spectrum", "Image Cube"])
        else:
            ui.SelectTargetData.addItems(["Spectrum"])
        ui.SelectTargetData.blockSignals(False)
        ui.SelectTargetData.setCurrentText("Spectrum")
        self._on_target_type_changed("Spectrum")

    def _on_add_collected_spectrum(self):
        spec = self._app_state.choose_spectrum_ui()
        if spec is None:
            return

        self.addSpectrumRow()
        row = self._spec_rows[-1]
        spec_name = spec.get_name()
        row["checkbox"].setText(f"{spec_name}")
        row["checkbox"].setChecked(True)
        row["specs"].extend([spec])

    # ----------------- File pickers -----------------
    def _on_add_spectrum_clicked(self):
        start_dir = self._app_state.get_current_dir() or os.path.expanduser("~")
        filedlg = QFileDialog(
            self,
            "Import Spectra from Text File",
            start_dir,
            "Text files (*.txt);;All Files (*)",
        )
        filedlg.setFileMode(QFileDialog.ExistingFile)
        filedlg.setAcceptMode(QFileDialog.AcceptOpen)
        filedlg.setWindowModality(Qt.WindowModal)
        if filedlg.exec_() != QDialog.Accepted:
            return
        path = filedlg.selectedFiles()[0]

        self._app_state.update_cwd_from_path(path)
        dlg = ImportSpectraTextDialog(path, parent=self)
        dlg.setWindowModality(Qt.WindowModal)
        if dlg.exec() != QDialog.Accepted:
            return
        specs = dlg.get_spectra()
        if not specs:
            return

        lib = ListSpectralLibrary(specs, path=path)
        self._app_state.add_spectral_library(lib)

        self.addSpectrumRow()
        row = self._spec_rows[-1]
        filename = os.path.basename(path)
        row["checkbox"].setText(f"{filename} ({len(specs)} spectra)")
        row["checkbox"].setChecked(True)
        row["path"] = path
        row["specs"].extend(specs)

    def _on_add_library_clicked(self):
        self.raise_()
        self.activateWindow()
        dlg = QFileDialog(self, "Select Library File")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilters(["Spectral Library (*.hdr *.csv)", "All Files (*)"])
        dlg.setWindowModality(Qt.WindowModal)
        if dlg.exec_() == QDialog.Accepted:
            path = dlg.selectedFiles()[0]
            filename = os.path.basename(path)
            self.addLibraryRow()
            row = self._lib_rows[-1]
            row["checkbox"].setText(filename)
            row["checkbox"].setChecked(True)
            row["path"] = path

    # ----------------- Add rows -----------------
    def addSpectrumRow(self):
        row = self._ui.specGrid.rowCount()
        cb = QCheckBox(f"Added_spectrum{row}")
        spin = self._make_row_threshold_spin()
        self._ui.specGrid.addWidget(cb, row, 0, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.specGrid.addWidget(spin, row, 1, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.specGrid.setRowStretch(row + 1, 1)
        self._spec_rows.append(
            {
                "checkbox": cb,
                "threshold": spin,
                "row_index": row,
                "path": None,
                "specs": [],
            }
        )

    def addLibraryRow(self):
        row = len(self._lib_rows) + 1
        cb = QCheckBox(f"Library_{row-1}")
        spin = self._make_row_threshold_spin()
        self._ui.libGrid.addWidget(cb, row, 0, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.libGrid.addWidget(spin, row, 1, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.libGrid.setRowStretch(row + 1, 1)
        self._lib_rows.append(
            {
                "checkbox": cb,
                "threshold": spin,
                "row_index": row,
                "path": None,
                "lib_obj": None,
            }
        )

    def _make_row_threshold_spin(self) -> QDoubleSpinBox:
        cfg = self.THRESHOLD_SPIN_CONFIG
        spin = QDoubleSpinBox()
        spin.setRange(cfg.get("min", 0.0), cfg.get("max", 1.0))
        spin.setDecimals(cfg.get("decimals", 2))
        spin.setSingleStep(cfg.get("step", 0.1))
        spin.setKeyboardTracking(False)
        spin.setValue(float(self._ui.method_threshold.value()))
        spin.setProperty("overridden", False)
        spin.editingFinished.connect(lambda sb=spin: sb.setProperty("overridden", True))
        return spin

    def default_library_path(self) -> str | None:
        """Child/apps override to return a .hdr/.csv path, or None to skip."""
        return None

    def _maybe_add_default_library(self):
        path = self.default_library_path()
        if not path:
            return
        if os.path.exists(path):
            self.addLibraryRow()
            row = self._lib_rows[-1]
            row["checkbox"].setText(os.path.basename(path))
            row["checkbox"].setChecked(True)
            row["path"] = path

    # ----------------- Target switching helpers -----------------
    def get_all_non_active_spectra(self) -> List[Spectrum]:
        """
        Retrieves all spectra from the collected spectra and spectral libraries.
        """
        collected_spectra = self._app_state.get_collected_spectra()
        spectral_libraries = self._app_state.get_spectral_libraries()

        # Extract individual spectra from libraries
        all_spectra_in_libraries: List[Spectrum] = []
        for library in list(spectral_libraries):
            for spectrum_index in range(library.num_spectra()):
                spectrum = library.get_spectrum(spectrum_index)
                all_spectra_in_libraries.append(spectrum)
        # Combine and return
        all_non_active_spectra = collected_spectra + all_spectra_in_libraries
        return all_non_active_spectra

    # ----------------- Target switching -----------------
    def _on_target_type_changed(self, text):
        ui = self._ui
        combo = ui.SelectTargetData_2
        combo.clear()
        if text == "Spectrum":
            objs = self.get_all_non_active_spectra()
            placeholder = DEFAULT_NO_SPECTRA_TEXT
        elif text == "Image Cube":
            objs = self._app_state.get_datasets()
            placeholder = DEFAULT_NO_DATASETS_TEXT
        else:
            raise ValueError(f"Unknown text for target type. Text: {text}")
        if objs:
            for obj in objs:
                name = obj.get_name() if hasattr(obj, "get_name") else getattr(obj, "name", str(obj))
                combo.addItem(name, obj)
            combo.setEnabled(True)
            ui.addRunBtn.setEnabled(True)
            combo.setCurrentIndex(0)
        else:
            combo.addItem(placeholder)
            combo.setEnabled(False)
            ui.addRunBtn.setEnabled(False)

    def _on_collected_spectra_changed(self, state_change: StateChange, index: int, _id: int):
        """
        Before, we had spectra with their own name and ID. A spectrum was either added,
        removed, or edited.

        If removed, we have to check whether the selected one was removed, or the one
        after or before it. Basically, we need to rebuild the list of collected spectra,
        and if ours was removed, we move it to the top; if not, we keep it in its
        previous position.
        """
        if state_change != StateChange.ITEM_ADDED and state_change != StateChange.ITEM_REMOVED:
            return
        ui = self._ui
        current_selected_spectrum = self._ui.SelectTargetData_2.currentData()
        # This case happens at the start when there were no spectra collected
        if current_selected_spectrum is None:
            new_cbox_index = 0
        new_cbox_index = None
        if state_change == StateChange.ITEM_REMOVED:
            if current_selected_spectrum.get_id() == _id:
                # The spectrum we currently had selected got removed, so we just set the currentIndex to zero
                new_cbox_index = 0

        if self._ui.SelectTargetData.currentText() == "Spectrum":
            spectra = self.get_all_non_active_spectra()

            cbox = self._ui.SelectTargetData_2
            cbox.clear()
            if len(spectra) > 0:
                for spectrum in spectra:
                    name = (
                        spectrum.get_name()
                        if hasattr(spectrum, "get_name")
                        else getattr(spectrum, "name", str(spectrum))
                    )
                    cbox.addItem(name, spectrum)
                cbox.setEnabled(True)
                ui.addRunBtn.setEnabled(True)
                # We only find the data if it wasn't deleted, else we default to the first item
                if new_cbox_index is None:
                    new_cbox_index = self._ui.SelectTargetData_2.findData(current_selected_spectrum)
                    if new_cbox_index == -1:
                        new_cbox_index = 0

                cbox.setCurrentIndex(new_cbox_index)
            else:
                cbox.addItem(DEFAULT_NO_SPECTRA_TEXT)
                cbox.setEnabled(False)
                ui.addRunBtn.setEnabled(False)

    # ----------------- Shared input resolution -----------------
    def _slice_to_bounds(
        self,
        spectrum: NumPyArraySpectrum,
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[np.ndarray, u.Quantity]:
        """
        Slices the given spectrum to the specified wavelength bounds.
        Returns a numpy array and a list of u.Quantity objects
        """
        wls = spectrum.get_wavelengths()
        if not isinstance(wls, u.Quantity):
            unit = (
                spectrum.get_wavelength_units()
                if spectrum.has_wavelengths()
                else self._get_wavelength_units()
            )
            wls = u.Quantity(wls, unit)
        elif wls.unit == u.dimensionless_unscaled:
            unit = self._get_wavelength_units()
            wls = u.Quantity(wls.value, unit)

        min_wvl = min_wvl.to(unit)
        max_wvl = max_wvl.to(unit)

        arr = spectrum.get_spectrum()
        if arr.ndim != 1 or arr.shape[0] != wls.shape[0]:
            raise ValueError(
                f"Shape mismatch: reflectance has shape {arr.shape}, wavelengths has shape {wls.shape}"
            )

        mask = np.ones(wls.shape, dtype=bool)
        if min_wvl is not None:
            mask &= wls >= min_wvl
        if max_wvl is not None:
            mask &= wls <= max_wvl

        return arr[mask], wls[mask]

    def _set_inputs(self) -> SpectralComputationInputs:
        try:
            global_thr = float(self._ui.method_threshold.value())
        except ValueError:
            raise ValueError("Threshold must be numbers.")

        units = self._get_wavelength_units()
        min_wl = self.get_wavelength_min()
        max_wl = self.get_wavelength_max()
        if min_wl >= max_wl:
            raise ValueError("Min wavelength must be < Max wavelength.")

        mode = self._ui.SelectTargetData.currentText()
        target = self._ui.SelectTargetData_2.currentData()
        if target is None:
            raise ValueError(f"No {mode.lower()} selected.")

        next_id = self._app_state.take_next_id()

        lib_name_by_spec_id: Dict[int, str] = {}
        refs: List[NumPyArraySpectrum] = []
        thresholds: List[float] = []

        # Libraries
        for lib_row in self._lib_rows:
            if lib_row["checkbox"].isChecked() and lib_row.get("path"):
                envilib = ENVISpectralLibrary(lib_row["path"])
                wls = u.Quantity([b["wavelength"] for b in envilib._band_list], units)
                lib_filename = os.path.basename(lib_row["path"])
                row_thr = float(lib_row["threshold"].value())
                for i in range(envilib._num_spectra):
                    arr = envilib._data[i]
                    name = envilib._spectra_names[i] if hasattr(envilib, "_spectra_names") else None
                    spec_from_lib = NumPyArraySpectrum(arr=arr, name=name, wavelengths=wls)
                    spec_from_lib.set_id(next_id)
                    lib_name_by_spec_id[spec_from_lib.get_id()] = lib_filename
                    next_id += 1
                    refs.append(spec_from_lib)
                    thresholds.append(row_thr)

        # Individual spectra
        for row in self._spec_rows:
            if row["checkbox"].isChecked() and row.get("specs"):
                row_thr = float(row["threshold"].value())
                spec_filename = os.path.basename(row.get("path") or "")
                for spec in row["specs"]:
                    spec.set_id(next_id)
                    lib_name_by_spec_id[spec.get_id()] = spec_filename
                    next_id += 1
                    refs.append(spec)
                    thresholds.append(row_thr)

        if not refs:
            raise ValueError("Please check at least one reference file.")

        spectral_inputs = SpectralComputationInputs(
            target=target,
            mode=mode,
            refs=refs,
            thresholds=thresholds,
            global_thr=global_thr,
            min_wvl=min_wl,
            max_wvl=max_wl,
            lib_name_by_spec_id=lib_name_by_spec_id,
        )
        return spectral_inputs

    # ----------------- Match pipeline -----------------
    def compute_score(
        self,
        target: NumPyArraySpectrum,
        ref: NumPyArraySpectrum,
        min_wvl: u.Quantity,
        max_wvl: u.Quantity,
    ) -> Tuple[float, Dict[str, Any]]:
        """Child must implement. Return (score, extras_dict). NaN to skip."""
        raise NotImplementedError

    def compute_score_image(
        self,
        target_image_name: str,
        target_image_arr: np.ndarray,  # float32[:, :, :]
        target_wavelengths: np.ndarray,  # float32[:]
        target_bad_bands: np.ndarray,  # bool[:]
        min_wvl: np.float32,  # float32
        max_wvl: np.float32,  # float32
        reference_spectra: List[NumPyArraySpectrum],
        reference_spectra_arr: np.ndarray,  # float32 [:]
        reference_spectra_wvls: np.ndarray,  # float32[:], in target_image_arr units
        reference_spectra_bad_bands: np.ndarray,  # bool[:]
        reference_spectra_indices: np.ndarray,  # uint32[:]
        thresholds: np.ndarray,  # float32[:]
        python_mode: bool = False,
    ) -> List[int]:
        """Child must implement. Return Nothing. Load dataset into app instead."""
        raise NotImplementedError

    def find_matches(
        self,
        spectral_inputs: SpectralComputationInputs,
        python_mode: bool = False,
    ) -> Union[List[Dict[str, Any]], List[int]]:
        """Find spectral matches for a single spectrum or an image cube.

        This method operates in two modes, driven by ``spectral_inputs.mode``:

        * ``"Spectrum"``: Compute a similarity score between a single target
        spectrum and each reference spectrum. Return a list of match records
        (one per passing reference), including metadata and any extra fields
        from ``compute_score``.
        * ``"Image Cube"``: Treat the target as a raster
        dataset, compute per-pixel scores against all reference spectra
        (via ``compute_score_image``), and attach the resulting products to
        the application. In this mode, the method returns ``None`` and all
        side effects are handled by the callee.
        * Any other mode: Error

        Matching uses a shared convention where **lower scores are better**,
        and a match is accepted if ``score <= threshold``.

        Args:
            spectral_inputs (SpectralComputationInputs):
                Container for all inputs required to perform the
                spectral computation. Expected to provide

                * ``target``: Either a :class:`NumPyArraySpectrum` (Spectrum mode)
                or a :class:`RasterDataSet` (image mode).
                * ``min_wvl`` / ``max_wvl``: Wavelength bounds for the comparison.
                * ``lib_name_by_spec_id``: Mapping from reference spectrum ID to
                library name.
                * ``refs``: Iterable of reference spectra.
                * ``mode``: String flag controlling behavior (e.g. ``"Spectrum"``
                or ``"Image Cube"``).
                * ``thresholds``: Iterable of per-reference score thresholds.

            python_mode (bool):
                Whether to run the compute intensive algorithms in python or not.
                If False, it tries to run in compiled numba code.

        Returns:
            Optional(List(Dict(str, Any))):
                In ``"Spectrum"`` mode, a list of dictionaries describing each
                reference that passes its threshold. Each dictionary includes

                * ``target_name``: Name of the target spectrum.
                * ``reference_data``: Name of the reference spectrum.
                * ``library_name``: Library that the reference belongs to (if any).
                * ``score``: Numeric score returned by ``compute_score``.
                * ``threshold``: Threshold used for acceptance.
                * ``min_wavelength`` / ``max_wavelength``: Bounds used for matching.
                * ``ref_obj``: The reference spectrum object itself.
                * Any additional key/value pairs returned in ``extras`` from
                :meth:`compute_score`.

                In image mode (non-``"Spectrum"``), returns ``None``. In that
                case, the concrete subclass is responsible for consuming the
                output of :meth:`compute_score_image` and attaching datasets
                to the application state.

        Raises:
            AssertionError: If ``spectral_inputs.mode == "Spectrum"`` but
                ``spectral_inputs.target`` is not a :class:`NumPyArraySpectrum`,
                or if image mode is selected but the target is not a
                :class:`RasterDataSet`.
            AssertionError: If the number of thresholds does not match the
                number of reference spectra in image mode.
            ValueError: May be raised indirectly from lower-level routines
                (e.g., wavelength unit conversion, array shape mismatches)
                invoked by :meth:`compute_score` or :meth:`compute_score_image`.

        """
        matches: List[Dict[str, Any]] = []
        target = spectral_inputs.target
        min_wvl = spectral_inputs.min_wvl
        max_wvl = spectral_inputs.max_wvl
        lib_name_by_spec_id = spectral_inputs.lib_name_by_spec_id
        references = spectral_inputs.refs
        mode = spectral_inputs.mode

        if mode == "Spectrum":
            assert isinstance(target, NumPyArraySpectrum)
            for spec, row_thr in zip(spectral_inputs.refs, spectral_inputs.thresholds):
                score, extras = self.compute_score(spectral_inputs.target, spec, min_wvl, max_wvl)
                if not np.isfinite(score):
                    continue
                if score <= row_thr:  # shared convention: lower score is better, pass if <= threshold
                    matches.append(
                        {
                            "target_name": target.get_name(),
                            "reference_data": spec.get_name(),
                            "library_name": lib_name_by_spec_id.get(spec.get_id(), ""),
                            "score": float(score),
                            "threshold": float(row_thr),
                            "min_wavelength": min_wvl,
                            "max_wavelength": max_wvl,
                            "ref_obj": spec,
                            **extras,
                        }
                    )
            return matches
        elif mode == "Image Cube":
            # Image mode: run per-pixel scoring against all reference spectra.
            assert isinstance(target, RasterDataSet)
            target_unit = target.get_band_unit()
            print(f"target_unit: {target_unit}")
            target_image_cube = target.get_image_data()  # [b][y][x]
            print(f"target_image_cube.dtype: {target_image_cube.dtype}")

            # Convert dataset bad-band flags → boolean mask (True = keep).
            target_wavelengths = [b["wavelength"].to(target_unit).value for b in target.get_band_info()]
            target_wavelengths = np.array(target_wavelengths, dtype=np.float32)
            print(f"target_wavelengths: {target_wavelengths}")
            target_bad_bands = np.array(target.get_bad_bands()).astype(
                np.bool_
            )  # 1's correspond for bands we keep, 0's don't

            # Normalize user wavelength bounds to dataset units.
            new_min_wvl = min_wvl.to(target_unit)
            new_min_wvl = np.float32(new_min_wvl.value)
            new_max_wvl = max_wvl.to(target_unit)
            new_max_wvl = np.float32(new_max_wvl.value)
            print(f"new_min_wvl: {new_min_wvl}, new_max_wvl: {new_max_wvl}")

            # Build packed reference buffers (values + wavelengths).
            length_all_references = 0
            ref_offsets = [0]
            for ref in references:
                length_of_ref = ref.get_shape()[0]
                length_all_references += length_of_ref
                ref_offsets.append(ref_offsets[-1] + length_of_ref)

            new_refs_arr = np.full((length_all_references,), fill_value=np.nan, dtype=np.float32)
            new_refs_wvl = np.full((length_all_references,), fill_value=np.nan, dtype=np.float32)
            new_refs_bad_bands = np.ones((length_all_references,), dtype=np.bool_)

            # Copy reference spectra into packed buffers.
            i = 0
            for ref in references:
                ref_unit = ref.get_wavelength_units()
                if ref_unit is None:
                    continue
                new_refs_arr[ref_offsets[i] : ref_offsets[i + 1]] = ref.get_spectrum()
                wvls = [wvl.to(target_unit).value for wvl in ref.get_wavelengths()]
                new_refs_wvl[ref_offsets[i] : ref_offsets[i + 1]] = wvls
                i += 1

            # Per-reference thresholds
            thresholds = np.array(spectral_inputs.thresholds, dtype=np.float32)
            ref_offsets = np.array(ref_offsets, dtype=np.uint32)
            assert thresholds.shape[0] == len(references)

            if isinstance(target_image_cube, np.ma.MaskedArray):
                target_image_arr = target_image_cube.data
            else:
                target_image_arr = target_image_cube

            print(f"!$$ target_image_arr.shape: {target_image_arr.shape}")
            print(f"target_wavelengths.shape: {target_wavelengths.shape}")
            print(f"target_bad_bands.shape: {target_bad_bands.shape}")
            # It's the child class's job to add the output to WISER
            ds_ids = self.compute_score_image(
                target_image_name=target.get_name(),
                target_image_arr=target_image_arr,
                target_wavelengths=target_wavelengths,
                target_bad_bands=target_bad_bands,
                min_wvl=new_min_wvl,
                max_wvl=new_max_wvl,
                reference_spectra=references,
                reference_spectra_arr=new_refs_arr,
                reference_spectra_wvls=new_refs_wvl,
                reference_spectra_bad_bands=new_refs_bad_bands,
                reference_spectra_indices=ref_offsets,
                thresholds=thresholds,
                python_mode=python_mode,
            )
            return ds_ids
        else:
            raise ValueError("Spectral computation mode must be 'Spectrum' or 'Image Cube'.")

    def sort_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(matches, key=lambda rec: rec["score"]) if matches else []

    # ----------------- Run + details -----------------
    def _on_run_clicked(self):
        ui = self._ui
        ui.addRunBtn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            try:
                spectral_inputs = self._set_inputs()
            except Exception as e:
                self._show_message("warning", "Invalid input", str(e))
                raise e

            try:
                matches = self.find_matches(spectral_inputs)
                if spectral_inputs.mode == "Image Cube":
                    return
                sorted_matches = self.sort_matches(matches)
            except Exception as e:
                import traceback

                self._show_message("error", "Error during run", str(e), details=traceback.format_exc())
                raise e

            if sorted_matches:
                self._record_run(spectral_inputs, sorted_matches)
                self._view_details_dialog(sorted_matches, spectral_inputs.target)
            else:
                self._show_message(
                    "info",
                    "No matches found",
                    "No matches were found within the threshold.",
                    informative=f"Threshold: {self._ui.method_threshold.value()}, "
                    f"Range: {spectral_inputs.min_wvl} - {spectral_inputs.max_wvl}",
                )

        finally:
            QApplication.restoreOverrideCursor()
            ui.addRunBtn.setEnabled(True)

    def _record_run(
        self,
        spectral_inputs: SpectralComputationInputs,
        sorted_matches: List[Dict[str, Any]],
    ) -> None:
        if not sorted_matches:
            return
        target = spectral_inputs.target
        key = target.get_name() or "<unnamed target>"
        best = sorted_matches[0]
        run_entry = {
            "target": target,
            "matches": list(sorted_matches),
            "best": best,
            "threshold": float(self._ui.method_threshold.value()),
            "min_wavelength": spectral_inputs.min_wvl,
            "max_wavelength": spectral_inputs.max_wvl,
        }
        self._run_history.setdefault(key, []).append(run_entry)

        table = self._ui.tableWidget
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(key))
        table.setItem(row, 1, QTableWidgetItem(best.get("reference_data", "")))
        table.setItem(row, 2, QTableWidgetItem(f"{best.get('score', float('nan')):.4f}"))

        m = list(sorted_matches)
        t = target
        btn_view = QPushButton("View Details", table)
        btn_view.clicked.connect(lambda _=False, mm=m, tt=t: self._view_details_dialog(mm, tt))
        table.setCellWidget(row, 3, btn_view)

        btn_del = QPushButton("Delete", table)
        btn_del.setProperty("run_key", key)
        btn_del.setProperty("run_entry", run_entry)
        btn_del.clicked.connect(lambda _=False, b=btn_del: self._on_delete_history_clicked(b))
        table.setCellWidget(row, 4, btn_del)
        table.resizeColumnsToContents()

    def _on_delete_history_clicked(self, btn_widget: QWidget) -> None:
        table = self._ui.tableWidget
        row_to_remove = None
        for i in range(table.rowCount()):
            # Prefer the Remove column (4), but also check 3 for robustness
            if table.cellWidget(i, 4) is btn_widget or table.cellWidget(i, 3) is btn_widget:
                row_to_remove = i
                break
        if row_to_remove is None:
            return

        run_key = btn_widget.property("run_key")
        run_entry = btn_widget.property("run_entry")
        if run_key is not None and run_entry is not None:
            runs = self._run_history.get(run_key, [])
            try:
                runs.remove(run_entry)
            except ValueError:
                pass
            if not runs and run_key in self._run_history:
                self._run_history.pop(run_key, None)

        table.removeRow(row_to_remove)
        try:
            self._save_state()
        except Exception:
            pass

    def _view_details_dialog(self, matches: List[Dict[str, Any]], target: NumPyArraySpectrum, parent=None):
        top = matches[:10]
        d = QDialog(parent or self)
        d.setWindowTitle(f"{self.filename_stub()} — Details")
        d.setAttribute(Qt.WA_DeleteOnClose, True)
        d.setWindowFlag(Qt.Tool, True)
        d.setModal(False)
        d.setWindowFlags(
            d.windowFlags() | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint
        )

        lay = self._build_details_layout(d, top, target)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close, parent=d)
        btn_box.button(QDialogButtonBox.Save).setText("Save .txt")
        btn_box.rejected.connect(d.close)
        btn_box.button(QDialogButtonBox.Save).clicked.connect(lambda: self.save_txt(target, matches, d))
        lay.addWidget(btn_box)

        d.resize(1050, 800)
        d.show()
        d.activateWindow()

    def _build_details_layout(self, d: QDialog, rows: List[Dict[str, Any]], target: NumPyArraySpectrum):
        from PySide2.QtWidgets import QVBoxLayout, QTableWidget

        layout = QVBoxLayout(d)

        # Plot
        try:
            target.set_color("#000000")
        except Exception:
            pass
        plot_widget = SpectrumPlotGeneric(self._app_state)
        layout.addWidget(plot_widget)
        plot_widget.add_collected_spectrum(target)
        for rec in rows[:5]:
            plot_widget.add_collected_spectrum(rec["ref_obj"])

        # Table
        table = QTableWidget()
        cols = (
            [("Match", "reference_data"), ("Library", "library_name")]
            + self.details_columns()
            + [
                ("Min WL", "min_wavelength"),
                ("Max WL", "max_wavelength"),
                ("Target", "target_name"),
            ]
        )
        table.setColumnCount(len(cols))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setHorizontalHeaderLabels([c[0] for c in cols])
        table.setRowCount(len(rows))
        for i, rec in enumerate(rows):
            for j, (_, key) in enumerate(cols):
                val = rec.get(key, "")
                if isinstance(val, float):
                    if key in ("score", "threshold"):
                        text = f"{val:.4f}"
                    else:
                        text = f"{val:.4f}"
                else:
                    text = str(val)
                table.setItem(i, j, QTableWidgetItem(text))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        return layout

    # ----------------- Messaging -----------------
    def _show_message(
        self,
        kind: str,
        title: str,
        text: str,
        informative: str = None,
        details: str = None,
        buttons: "QMessageBox.StandardButtons" = QMessageBox.Ok,
        default: "QMessageBox.StandardButton" = QMessageBox.Ok,
    ) -> int:
        icon_map = {
            "info": QMessageBox.Information,
            "warning": QMessageBox.Warning,
            "error": QMessageBox.Critical,
            "question": QMessageBox.Question,
        }
        parent = self.window() or self
        if isinstance(parent, QWidget) and parent.isMinimized():
            parent.showNormal()
        box = QMessageBox(parent)
        box.setIcon(icon_map.get(kind, QMessageBox.Information))
        box.setWindowTitle(title)
        box.setText(text)
        if informative:
            box.setInformativeText(informative)
        if details:
            box.setDetailedText(details)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default)
        box.setWindowModality(Qt.WindowModal)
        box.raise_()
        box.activateWindow()
        return box.exec_()

    # ----------------- Clear / Save State -----------------
    def _on_clear_runs_clicked(self):
        res = self._show_message(
            "question",
            "Clear runs",
            "Clear all run history?",
            buttons=QMessageBox.Yes | QMessageBox.No,
            default=QMessageBox.No,
        )
        if res != QMessageBox.Yes:
            return
        self._run_history.clear()
        self._ui.tableWidget.setRowCount(0)
        try:
            self._save_state()
        except Exception:
            pass

    def cancel(self):
        self.reject()

    def save_and_close(self):
        self._save_state()
        self.accept()

    # ----------------- Settings helpers -----------------
    def _settings(self) -> QSettings:
        s = QSettings(self.SETTINGS_NAMESPACE)
        s.beginGroup(f"session/{os.getpid()}")
        return s

    def _q_to_units(self, q):
        if q is None:
            return None
        try:
            return float(q.to(self._get_wavelength_units()).value)
        except Exception:
            return float(getattr(q, "value", q))

    def _units_to_q(self, v):
        return u.Quantity(v, self._get_wavelength_units()) if v is not None else None

    def _clear_library_rows(self):
        grid = self._ui.libGrid
        for row in self._lib_rows:
            cb, le = row["checkbox"], row["threshold"]
            grid.removeWidget(cb)
            cb.deleteLater()
            grid.removeWidget(le)
            le.deleteLater()
        self._lib_rows.clear()

    def _save_state(self) -> None:
        """
        Saves the current state/laste entered entries to QSettings object.
        This QSettigns object is used to repopulate the dialog when it closes.
        """
        s = self._settings()
        s.setValue("min", self._q_to_units(self.get_wavelength_min()))
        s.setValue("max", self._q_to_units(self.get_wavelength_max()))
        s.setValue("threshold", float(self._ui.method_threshold.value()))
        # target
        mode = self._ui.SelectTargetData.currentText()
        obj = self._ui.SelectTargetData_2.currentData()
        s.setValue("target/mode", mode)
        s.setValue("target/name", getattr(obj, "get_name", lambda: None)())
        s.setValue("target/id", getattr(obj, "get_id", lambda: None)())
        # libraries
        s.beginWriteArray("libraries")
        for i, row in enumerate(self._lib_rows):
            s.setArrayIndex(i)
            s.setValue("path", row.get("path"))
            s.setValue("checked", row["checkbox"].isChecked())
            s.setValue("threshold_text", float(row["threshold"].value()))
            s.setValue("threshold_overridden", bool(row["threshold"].property("overridden")))
        s.endArray()
        # spectra
        s.beginWriteArray("spectra")
        for i, row in enumerate(self._spec_rows):
            s.setArrayIndex(i)
            s.setValue("path", row.get("path"))
            s.setValue("checked", row["checkbox"].isChecked())
            s.setValue("threshold_text", float(row["threshold"].value()))
            s.setValue("threshold_overridden", bool(row["threshold"].property("overridden")))
        s.endArray()
        # history (summary)
        flat = []
        for tname, runs in self._run_history.items():
            for r in runs:
                best = r.get("best", {})
                target = r.get("target")
                flat.append(
                    {
                        "mode": r.get("mode"),
                        "target_name": tname,
                        "target_id": getattr(target, "get_id", lambda: None)(),
                        "min": self._q_to_units(r.get("min_wavelength")),
                        "max": self._q_to_units(r.get("max_wavelength")),
                        "threshold": float(r.get("threshold", float(self._ui.method_threshold.value()))),
                        "best_reference": best.get("reference_data"),
                        "best_library": best.get("library_name"),
                        "best_score": float(best.get("score")) if best.get("score") is not None else None,
                    }
                )
        s.beginWriteArray("history")
        for i, row in enumerate(flat):
            s.setArrayIndex(i)
            for k, v in row.items():
                s.setValue(k, v)
        s.endArray()

    def _restore_target_selection(self, mode, name, sid):
        if mode in ("Spectrum", "Image Cube"):
            self._ui.SelectTargetData.setCurrentText(mode)
            combo = self._ui.SelectTargetData_2
            for idx in range(combo.count()):
                obj = combo.itemData(idx)
                if sid is not None and getattr(obj, "get_id", lambda: None)() == sid:
                    combo.setCurrentIndex(idx)
                    return
            if name:
                for idx in range(combo.count()):
                    if getattr(combo.itemData(idx), "get_name", lambda: None)() == name:
                        combo.setCurrentIndex(idx)
                        return

    def _load_state(self) -> None:
        s = self._settings()
        min_v = s.value("min", type=float)
        max_v = s.value("max", type=float)
        if min_v is not None:
            self._ui.lineEdit.setText(str(min_v))
        if max_v is not None:
            self._ui.lineEdit_2.setText(str(max_v))
        thr = s.value("threshold", None, type=float)
        if thr is not None:
            self._ui.method_threshold.setValue(float(thr))

        count = s.beginReadArray("libraries")
        if count > 0:
            self._clear_library_rows()
            for i in range(count):
                s.setArrayIndex(i)
                self.addLibraryRow()
                row = self._lib_rows[-1]
                row["path"] = s.value("path", type=str)
                row["checkbox"].setText(os.path.basename(row["path"] or "") or "Library")
                row["checkbox"].setChecked(bool(s.value("checked", False, type=bool)))
                saved_v = s.value("threshold_text", None, type=float)
                overridden = bool(s.value("threshold_overridden", False, type=bool))
                row["threshold"].setProperty("overridden", overridden)
                row["threshold"].setValue(
                    float(saved_v)
                    if (overridden and saved_v is not None)
                    else float(self._ui.method_threshold.value())
                )
        s.endArray()

        count = s.beginReadArray("spectra")
        if count > 0:
            for i in range(count):
                s.setArrayIndex(i)
                self.addSpectrumRow()
                row = self._spec_rows[-1]
                row["path"] = s.value("path", type=str)
                row["checkbox"].setChecked(bool(s.value("checked", False, type=bool)))
                saved_v = s.value("threshold_text", None, type=float)
                overridden = bool(s.value("threshold_overridden", False, type=bool))
                row["threshold"].setProperty("overridden", overridden)
                row["threshold"].setValue(
                    float(saved_v)
                    if (overridden and saved_v is not None)
                    else float(self._ui.method_threshold.value())
                )
                filename = os.path.basename(row["path"] or "")
                row["checkbox"].setText(filename or f"Added_spectrum{row['row_index']}")
        s.endArray()

        mode = s.value("target/mode", type=str)
        name = s.value("target/name", type=str)
        sid = s.value("target/id", type=int)
        self._restore_target_selection(mode, name, sid)

        table = self._ui.tableWidget
        count = s.beginReadArray("history")
        for i in range(count):
            s.setArrayIndex(i)
            row_idx = table.rowCount()
            table.insertRow(row_idx)
            tname = s.value("target_name", "", type=str)
            bref = s.value("best_reference", "", type=str)
            bsc = s.value("best_score", None)
            table.setItem(row_idx, 0, QTableWidgetItem(tname))
            table.setItem(row_idx, 1, QTableWidgetItem(bref))
            table.setItem(
                row_idx,
                2,
                QTableWidgetItem(f"{float(bsc):.4f}" if bsc is not None else ""),
            )
            spec = {
                "mode": s.value("mode", "Spectrum", type=str),
                "target_name": tname,
                "target_id": s.value("target_id", None, type=int),
                "min": s.value("min", None, type=float),
                "max": s.value("max", None, type=float),
                "threshold": s.value("threshold", float(self._ui.method_threshold.value()), type=float),
            }
            btn_view = QPushButton("View Details", table)
            btn_view.clicked.connect(lambda _=False, sp=spec: self._replay_saved(sp))
            table.setCellWidget(row_idx, 3, btn_view)

            btn_del = QPushButton("Delete", table)
            btn_del.clicked.connect(lambda _=False, b=btn_del: self._on_delete_history_clicked(b))
            table.setCellWidget(row_idx, 4, btn_del)
        s.endArray()
        table.resizeColumnsToContents()

    def _replay_saved(self, spec: dict) -> None:
        if spec.get("min") is not None:
            self._ui.lineEdit.setText(str(spec["min"]))
        if spec.get("max") is not None:
            self._ui.lineEdit_2.setText(str(spec["max"]))
        if spec.get("threshold") is not None:
            self._ui.method_threshold.setValue(float(spec["threshold"]))
        mode = spec.get("mode", "Spectrum")
        self._ui.SelectTargetData.setCurrentText(mode)
        self._restore_target_selection(mode, spec.get("target_name"), spec.get("target_id"))
        self._on_run_clicked()

    def _purge_old_sessions_once(self):
        if self._sessions_purged_flag:
            return
        s = QSettings(self.SETTINGS_NAMESPACE)
        s.beginGroup("session")
        for g in s.childGroups():
            if g != str(os.getpid()):
                s.remove(g)
        s.endGroup()
        self._sessions_purged_flag = True

    # ----------------- Export -----------------
    def save_txt(self, target, matches, parent=None):
        """
        Saves the last entered input to a .txt file.
        """
        tgt_name = getattr(target, "get_name", lambda: None)() or "target"
        safe_target = tgt_name.replace(" ", "_")
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = getattr(self._app_state, "get_current_dir", lambda: None)() or os.path.expanduser("~")
        default_path = os.path.join(base_dir, f"{safe_target}_{self.filename_stub()}_{stamp}.txt")

        dlg = QFileDialog(parent or self, f"Save {self.filename_stub()} results as .txt")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilters(["Text files (*.txt)", "All Files (*)"])
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.selectFile(os.path.basename(default_path))
        dlg.setDirectory(os.path.dirname(default_path))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        if dlg.exec_() != QDialog.Accepted:
            return
        path = dlg.selectedFiles()[0]

        # header
        try:
            min_v = f"{self.get_wavelength_min().to_value(self._get_wavelength_units()):.1f}"
        except Exception:
            min_v = str(self.get_wavelength_min()) if self.get_wavelength_min() is not None else "—"
        try:
            max_v = f"{self.get_wavelength_max().to_value(self._get_wavelength_units()):.1f}"
        except Exception:
            max_v = str(self.get_wavelength_max()) if self.get_wavelength_max() is not None else "—"

        from datetime import datetime

        unit_str = self._get_wavelength_units().to_string()
        header = [
            f"{self.filename_stub()} — WISER Plugin Export",
            f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target: {tgt_name}",
            f"Wavelength range [{unit_str}]: {min_v} – {max_v}",
            f"Threshold: {float(self._ui.method_threshold.value()):.4f}",
            "Interpolation: linear",
            "",
        ]

        # table
        cols = [
            ("Rank", None),
            ("Reference", "reference_data"),
            ("Library", "library_name"),
            (self.SCORE_HEADER, "score"),
            ("Threshold", "threshold"),
            (f"Min WL [{unit_str}]", "min_wavelength"),
            (f"Max WL [{unit_str}]", "max_wavelength"),
            ("Target", "target_name"),
        ]
        # include child-specific extra columns at the end if present
        extra_cols = [c for c in self.details_columns() if c[1] not in ("score", "threshold")]
        cols[3:3] = extra_cols  # insert after Library

        head = " | ".join([c[0] for c in cols])
        lines = [head, "-" * len(head)]
        for i, rec in enumerate(matches, start=1):
            row = []
            for title, key in cols:
                if key is None:
                    row.append(str(i))
                else:
                    v = rec.get(key, "")
                    if isinstance(v, float):
                        row.append(f"{v:.4f}")
                    else:
                        row.append(str(v))
            lines.append(" | ".join(row))

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(header + lines) + "\n")
            try:
                self._app_state.update_cwd_from_path(path)
            except Exception:
                pass
            msg = QMessageBox(parent or self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Saved")
            msg.setText(f"Results saved to:\n{path}")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setDefaultButton(QMessageBox.Ok)
            msg.setWindowModality(Qt.WindowModal)
            msg.exec_()
        except Exception as e:
            import traceback

            self._show_message("error", "Failed to save", str(e), details=traceback.format_exc())
