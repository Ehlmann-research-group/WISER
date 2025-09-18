# Python SFF logic


"""
Spectral Feature Fitting (SFF) Plugin for WISER
-----------------------------------------------

Provides spectral similarity analysis between a target spectrum (or image cube)
and reference libraries/spectra using the SFF algorithm.

Method:
  1. Continuum removal (convex-hull, piecewise linear)
  2. Depth/contrast scaling to match reference to target
  3. Bandwise least-squares fit → RMS score (lower = better)

Key Features:
- Threshold is a *maximum RMS* cutoff (default 0.03) instead of an angle
- Supports per-row and global RMS thresholds (UI overrides)
- Linear interpolation onto target's wavelength grid
- Works with ENVI spectral libraries or imported spectra
- Session state persisted with QSettings (wavelengths, thresholds, run history)

Notes:
- RMS is dimensionless, typically 0-1 range
- No inheritance or imports from SAM plugin (standalone implementation)
- UI integration via PySide2
"""


from typing import List, Optional, Dict, Any, Tuple, cast
import numpy as np
from PySide2.QtWidgets import *
from PySide2.QtCore import Qt, QSettings
import os
from datetime import datetime

from wiser.gui.app_state import ApplicationState
from astropy import units as u
from wiser.raster.spectrum import NumPyArraySpectrum
from wiser.raster.envi_spectral_library import ENVISpectralLibrary
from scipy.interpolate import interp1d
from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import NumPyRasterDataImpl

from wiser.gui.import_spectra_text import ImportSpectraTextDialog
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.gui.spectrum_plot import SpectrumPlotGeneric

from wiser.gui.generated.spectral_feature_fitting_ui import Ui_SpectralFeatureFitting
import warnings


class SFFTool(QDialog):

    _sessions_purged = False
    MAX_RMS_DEFAULT = 0.03
    RMS_THRESHOLD_LOWER_RANGE = 0.0
    RMS_THRESHOLD_UPPER_RANGE = 1.0
    RMS_DECIMAL_PLC = 4
    RMS_STEP_INCRIMENT = 0.005
    
    def __init__(self, app_state: ApplicationState, parent: QWidget = None):
        super().__init__(parent)

        if not SFFTool._sessions_purged:
            self._purge_old_sessions()
            SFFTool._sessions_purged = True

        self._ui = Ui_SpectralFeatureFitting()
        self._ui.setupUi(self)

        self._app_state = app_state
        self._target: Optional[NumPyArraySpectrum] = None
        self.library: List[NumPyArraySpectrum] = []
        self._lib_name_by_spec_id: Dict[int, str] = {}

        # SFF uses a max RMS threshold (lower is better). Default 0.03 is a reasonable starting point.
        self._max_rms: float = self.MAX_RMS_DEFAULT

        self._min_wavelength: Optional[u.Quantity] = None
        self._max_wavelength: Optional[u.Quantity] = None
        self._run_history: Dict[str, List[Dict[str, Any]]] = {}
        
        self._lib_rows = []
        self._init_reference_selection()
        self._spec_rows = []

        self._ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._setup_connections()
        self._add_interpolation_note()
        self._init_target_dropdowns()

        self._ui.threshSpinBox.setRange(SFFTool.RMS_THRESHOLD_LOWER_RANGE, SFFTool.RMS_THRESHOLD_UPPER_RANGE)
        self._ui.threshSpinBox.setDecimals(SFFTool.RMS_DECIMAL_PLC)
        self._ui.threshSpinBox.setSingleStep(SFFTool.RMS_STEP_INCRIMENT)
        self._ui.threshSpinBox.setValue(self._max_rms)

        self._load_state()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(lambda: QSettings("Wiser","SFFPlugin").remove(f"session/{os.getpid()}"))

    # ----------------- Basic setters/getters -----------------
    def set_target(self, target: NumPyArraySpectrum) -> None:
        self._target = target

    def set_max_rms(self, max_rms: Optional[float]) -> None:
        if max_rms is not None:
            self._max_rms = float(max_rms)

    def set_library(self, spectra: List[NumPyArraySpectrum]) -> None:
        self.library = list(spectra)

    def set_wavelength_min(self, min: Optional[u.Quantity]) -> None:
        self._min_wavelength = min

    def set_wavelength_max(self, max: Optional[u.Quantity]) -> None:
        self._max_wavelength = max

    def get_target_name(self) -> str:
        return self._target.get_name() if self._target else "<no target>"

    def get_max_rms(self) -> Optional[float]:
        return self._max_rms

    def get_wavelength_min(self) -> Optional[u.Quantity]:
        return self._min_wavelength

    def get_wavelength_max(self) -> Optional[u.Quantity]:
        return self._max_wavelength

    # ----------------- Library management -----------------
    def add_spectrum(self, spectrum: NumPyArraySpectrum) -> None:
        """Append a single spectrum and update global wavelength bounds."""
        if not isinstance(spectrum, NumPyArraySpectrum):
            raise TypeError("add_spectrum expects a NumPyArraySpectrum")
        self.library.append(spectrum)

        wls = spectrum.get_wavelengths()
        if isinstance(wls, u.Quantity):
            arr = wls
        else:
            arr = u.Quantity(wls, u.nanometer)
        local_min = arr.min()
        local_max = arr.max()
        if self.get_wavelength_min() is None or local_min < self.get_wavelength_min():
            self._min_wavelength = local_min
        if self.get_wavelength_max() is None or local_max > self.get_wavelength_max():
            self._max_wavelength = local_max

    def add_library(self, spectra: List[NumPyArraySpectrum]) -> None:
        for spec in spectra:
            self.add_spectrum(spec)

    # =========================================================
    # SFF CORE
    # =========================================================
    @staticmethod
    def _resample_to(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        """
        Resample y_src(x_src) onto new grid x_dst.

        - Linear interpolation
        - Values outside overlap → NaN
        """
        if x_src.size < 2 or np.all(~np.isfinite(y_src)):
            return np.full_like(x_dst, np.nan, dtype=float)
        f = interp1d(x_src, y_src, kind="linear", bounds_error=False, fill_value=np.nan)
        return f(x_dst)

    @staticmethod
    def _continuum_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Construct an *upper continuum* (convex hull style) using:
          - endpoints + local maxima as anchors
          - iterative pruning of anchors below line between neighbors

        Returns continuum reflectance values at each x.

        Note: ensures continuum is convex and non-negative.
        """
        n = y.size
        if n < 2:
            return np.maximum(y, 1e-12)

        # indices of candidate anchors: endpoints + local maxima
        idx = [0]
        for i in range(1, n - 1):
            if (y[i] >= y[i - 1]) and (y[i] >= y[i + 1]):
                idx.append(i)
        idx.append(n - 1)
        idx = np.unique(idx)

        # prune any anchor that lies *below* the line between its neighbors (enforce convexity)
        changed = True
        while changed and idx.size > 2:
            changed = False
            keep = [idx[0]]
            for k in range(1, len(idx) - 1):
                i_prev, i, i_next = keep[-1], idx[k], idx[k + 1]
                x0, y0 = x[i_prev], y[i_prev]
                x1, y1 = x[i_next], y[i_next]
                if x1 == x0:
                    # duplicate wavelength — drop the middle point if it's below max of endpoints
                    if y[i] < max(y0, y1):
                        changed = True
                        continue
                else:
                    t = (x[i] - x0) / (x1 - x0)
                    y_line = y0 + t * (y1 - y0)
                    if y[i] < y_line - 1e-12:
                        changed = True
                        continue
                keep.append(i)
            keep.append(idx[-1])
            idx = np.array(keep, dtype=int)

        f = interp1d(x[idx], y[idx], kind="linear", bounds_error=False, fill_value="extrapolate")
        cont = f(x)
        return np.maximum(cont, 1e-12)

    @classmethod
    def _continuum_remove_and_invert(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply continuum removal and inversion:

        CR = y / continuum
        Absorption signal = 1 - CR (so continuum = 0, deeper bands = larger positive values)
        """
        cont = cls._continuum_curve(x, y)
        cr = y / cont
        return 1.0 - cr  # absorption-only signal, continuum at 0

    def _slice_to_bounds(self, spectrum: NumPyArraySpectrum) -> Tuple[np.ndarray, u.Quantity]:
        """Clip a spectrum to global bounds; returns (reflectance_array, wavelength_quantity)."""
        wls = spectrum.get_wavelengths()
        if not isinstance(wls, u.Quantity):
            wls = u.Quantity(wls, u.nanometer)
        elif wls.unit == u.dimensionless_unscaled:
            wls = u.Quantity(wls.value, u.nanometer)

        # Convert to the user's units, if bounds already specify
        if self._min_wavelength is not None:
            wls = wls.to(self._min_wavelength.unit)
        if self._max_wavelength is not None:
            wls = wls.to(self._max_wavelength.unit)

        raw_r = spectrum.get_spectrum()
        arr = raw_r.value if isinstance(raw_r, u.Quantity) else np.asarray(raw_r)
        if arr.ndim != 1 or arr.shape[0] != wls.shape[0]:
            raise ValueError(
                f"Shape mismatch: reflectance has shape {arr.shape}, wavelengths has shape {wls.shape}"
            )

        mask = np.ones(wls.shape, dtype=bool)
        if self._min_wavelength is not None:
            mask &= (wls >= self._min_wavelength)
        if self._max_wavelength is not None:
            mask &= (wls <= self._max_wavelength)

        return arr[mask], wls[mask]

    def compute_sff_fit(self, ref: NumPyArraySpectrum) -> Tuple[float, float]:
        """
        Compute Spectral Feature Fitting (SFF) score between target and reference.

        Steps:
        1. Slice both spectra to global wavelength bounds
        2. Resample reference reflectance onto target's grid (linear, NaN outside overlap)
        3. Continuum removal + inversion (absorption-only signal)
        4. Compute single multiplicative scale factor: scale = (r·t) / (r·r)
           - Negative scales are clamped to 0 (no negative contrast allowed)
        5. RMS = sqrt(mean((A_target - scale*A_ref)^2))

        Returns:
            rms   (float) : root-mean-square misfit (lower = better fit)
            scale (float) : multiplicative factor applied to reference absorption
        """
        if self._target is None:
            raise RuntimeError("compute_sff_fit called without a target set")

        MIN_SAMPLES = 3

        t_reflect, t_wls = self._slice_to_bounds(self._target)
        r_reflect, r_wls = self._slice_to_bounds(ref)

        t_x = t_wls.value
        r_x = r_wls.value

        # Resample reference reflectance onto target grid if needed
        if np.array_equal(r_x, t_x):
            r_reflect_rs = r_reflect
        else:
            r_reflect_rs = self._resample_to(r_x, r_reflect, t_x)

        # Mask invalid
        valid = np.isfinite(t_reflect) & np.isfinite(r_reflect_rs)
        if valid.sum() < MIN_SAMPLES:
            return (np.nan, np.nan)

        t_xv = t_x[valid]
        t_reflect_v = t_reflect[valid]
        r_reflect_v = r_reflect_rs[valid]

        # Continuum removal + inversion (absorption only)
        a_t = self._continuum_remove_and_invert(t_xv, t_reflect_v)
        a_r = self._continuum_remove_and_invert(t_xv, r_reflect_v)

        # If reference absorption is all ~0 (flat), no meaningful fit
        if not np.any(np.isfinite(a_r)) or np.allclose(a_r, 0.0, atol=1e-12):
            return (np.nan, np.nan)

        # Single multiplicative scale factor to match ref to target: s = (r·t)/(r·r)
        num = np.dot(a_r, a_t)
        den = np.dot(a_r, a_r)
        if den <= 0:
            return (np.nan, np.nan)
        scale = max(0.0, float(num / den))  # disallow negative scale

        # Least-squares residuals and RMS
        resid = a_t - scale * a_r
        rms = float(np.sqrt(np.nanmean(resid ** 2)))
        return (rms, scale)

    # =========================================================
    # Matching pipeline
    # =========================================================
    def find_matches(self) -> List[Dict[str, Any]]:
        """
        Compute SFF fit (RMS + scale) between the target and each spectrum in the library;
        keep only those with RMS <= row threshold (or global max RMS). Returns a list of records.
        """
        matches: List[Dict[str, Any]] = []

        for i, spec in enumerate(self.library):
            rms, scale = self.compute_sff_fit(spec)
            if not np.isfinite(rms):
                continue
            
            thr = getattr(spec, "_sff_max_rms", self._max_rms)
            if rms <= float(thr):
                matches.append({
                    "target_name": self.get_target_name(),
                    "fit_rms": rms,
                    "scale": scale,
                    "max_rms": float(thr),
                    "min_wavelength": self.get_wavelength_min(),
                    "max_wavelength": self.get_wavelength_max(),
                    "reference_data": spec.get_name(),
                    "ref_obj": spec,
                    "library_name": self._lib_name_by_spec_id.get(spec.get_id(), "")
                })
        return matches

    @staticmethod
    def sort_matches(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort by *ascending* RMS (best fits first)."""
        return sorted(matches, key=lambda rec: rec["fit_rms"]) if matches else []

    # =========================================================
    # UI actions
    # =========================================================
    def run_sff(self):
        ui = self._ui
        ui.addRunSFFBtn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            try:
                self._set_inputs()
            except Exception as e:
                self._show_message("warning", "Invalid input", str(e))
                return

            try:
                matches = self.find_matches()
                sorted_matches = self.sort_matches(matches)
            except Exception as e:
                import traceback
                self._show_message(
                    "error",
                    "Error during SFF",
                    str(e),
                    details=traceback.format_exc()
                )
                return

            if sorted_matches:
                self._record_run(self._target, sorted_matches)
                self._view_details_dialog(sorted_matches, self._target)
            else:
                self._show_message(
                    "info",
                    "No matches found",
                    "No matches were found within the max RMS threshold.",
                    informative=f"Max RMS: {self._max_rms:.4f}, Range: {self._min_wavelength} – {self._max_wavelength}"
                )
        finally:
            QApplication.restoreOverrideCursor()
            ui.addRunSFFBtn.setEnabled(True)

    def cancel(self):
        self.reject()

    def save_and_close(self):
        self._save_state()
        self.accept()

    # ---------- Rows for Spectrum/Library pickers ----------
    def addSpectrumRow(self):
        row = self._ui.specGrid.rowCount()
        cb = QCheckBox(f"Added_spectrum{row}")

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.005)
        spin.setKeyboardTracking(False)
        self._bind_threshold_spin(spin)

        self._ui.specGrid.addWidget(cb,   row, 0, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.specGrid.addWidget(spin, row, 1, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.specGrid.setRowStretch(row+1, 1)
        self._spec_rows.append({
            "checkbox": cb,
            "threshold": spin,  # per-row *max RMS*
            "row_index": row,
            "path": None,
            "specs": []
        })

    def addLibraryRow(self):
        row = len(self._lib_rows) + 1
        cb = QCheckBox(f"Library_{row-1}")

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.005)
        spin.setKeyboardTracking(False)
        self._bind_threshold_spin(spin)

        self._ui.libGrid.addWidget(cb,   row, 0, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.libGrid.addWidget(spin, row, 1, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.libGrid.setRowStretch(row + 1, 1)
        self._lib_rows.append({
            "checkbox": cb,
            "threshold": spin,  # per-row *max RMS*
            "row_index": row,
            "path": None,
            "lib_obj": None
        })

    def _record_run(self, target: NumPyArraySpectrum, sorted_matches: List[Dict[str, Any]]) -> None:
        if not sorted_matches:
            return
        key = target.get_name() or "<unnamed target>"
        best = sorted_matches[0]

        run_entry = {
            "target": target,
            "matches": list(sorted_matches),
            "best": best,
            "max_rms": self._max_rms,
            "min_wavelength": self._min_wavelength,
            "max_wavelength": self._max_wavelength,
        }
        self._run_history.setdefault(key, []).append(run_entry)

        table = self._ui.tableWidget
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(key))
        table.setItem(row, 1, QTableWidgetItem(best.get("reference_data", "")))
        table.setItem(row, 2, QTableWidgetItem(f"{best['fit_rms']:.4f}"))

        btn = QPushButton("View Details", table)
        m = list(sorted_matches)
        t = target
        btn.clicked.connect(lambda _=False, mm=m, tt=t: self._view_details_dialog(mm, tt))
        table.setCellWidget(row, 3, btn)
        table.resizeColumnsToContents()

    def _setup_connections(self):
        ui = self._ui
        ui.addRunSFFBtn.clicked.connect(self.run_sff)
        ui.addSaveCloseBtn.clicked.connect(self.save_and_close)
        ui.addCancelBtn.clicked.connect(self.cancel)
        ui.addLibBtn.clicked.connect(self._on_add_library_clicked)
        ui.addSpecBtn.clicked.connect(self._on_add_spectrum_clicked)
        ui.SelectTargetData.currentTextChanged.connect(self._on_target_type_changed)
        ui.clearRunsBtn.clicked.connect(self._on_clear_runs_clicked)

    def _add_interpolation_note(self) -> None:
        note = QLabel("Interpolation: linear; Fit metric: RMS (lower=better)")
        note.setStyleSheet("color:#666; font-size:11px; font-style: italic;")
        note.setAlignment(Qt.AlignRight)

        root = self._ui.groupBox_2.layout()
        thr_layout = None
        for i in range(root.count()):
            item = root.itemAt(i)
            lay = item.layout()
            if isinstance(lay, QFormLayout) and lay.indexOf(self._ui.threshSpinBox) != -1:
                thr_layout = lay
                break
        if thr_layout is not None:
            thr_layout.addRow("", note)

    # ---------- Messaging ----------
    def _show_message(
        self,
        kind: str,  # "info" | "warning" | "error" | "question"
        title: str,
        text: str,
        informative: str = None,
        details: str = None,
        buttons: "QMessageBox.StandardButtons" = QMessageBox.Ok,
        default: "QMessageBox.StandardButton" = QMessageBox.Ok
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
        box.raise_(); box.activateWindow()
        return box.exec_()

    # ---------- Targets ----------
    
    def _init_reference_selection(self) -> None:
        self._add_default_library()
        for lbl in (self._ui.hdr_lib, self._ui.hdr_thresh_lib,
            self._ui.hdr_spec, self._ui.hdr_thresh_spec):
            lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def _init_target_dropdowns(self) -> None:
        ui = self._ui
        ui.SelectTargetData.blockSignals(True)
        ui.SelectTargetData.clear()
        ui.SelectTargetData.addItems(["Spectrum", "Image Cube"])  # image-cube path is optional
        ui.SelectTargetData.blockSignals(False)
        ui.SelectTargetData.setCurrentText("Spectrum")
        self._on_target_type_changed("Spectrum")

    def _add_default_library(self) -> None:
        try:
            gui_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(
                gui_dir,
                "../data/"
                "usgs_default_ref_lib",
                "USGS_Mineral_Spectral_Library.hdr",
            )
            if os.path.exists(default_path):
                filename = os.path.basename(default_path)
                self.addLibraryRow()
                row = self._lib_rows[-1]
                row["checkbox"].setText(filename)
                row["checkbox"].setChecked(True)
                row["path"] = default_path
            else:
                warnings.warn(f"Default library not found at: {default_path}", UserWarning)
        except Exception as e:
            warnings.warn(f"Failed to add default library: {e}", UserWarning)

    # ---------- File pickers ----------
    def _on_add_spectrum_clicked(self):
        start_dir = self._app_state.get_current_dir() or os.path.expanduser("~")
        filedlg = QFileDialog(self, "Import Spectra from Text File", start_dir, "Text files (*.txt);;All Files (*)")
        filedlg.setFileMode(QFileDialog.ExistingFile)
        filedlg.setAcceptMode(QFileDialog.AcceptOpen)
        filedlg.setWindowModality(Qt.WindowModal)
        if filedlg.exec_() != QDialog.Accepted:
            return
        path = filedlg.selectedFiles()[0]

        # Parse existing dialog
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
        self.raise_(); self.activateWindow()
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

    def _on_target_type_changed(self, text):
        ui = self._ui
        combo = ui.SelectTargetData_2
        combo.clear()
        if text == "Spectrum":
            objs = self._app_state.get_collected_spectra()
            placeholder = "No spectra loaded"
        else:
            objs = self._app_state.get_datasets()
            placeholder = "No image cubes loaded"
        if objs:
            for obj in objs:
                name = obj.get_name() if hasattr(obj, "get_name") else getattr(obj, "name", str(obj))
                combo.addItem(name, obj)
            combo.setEnabled(True)
            ui.addRunSFFBtn.setEnabled(True)
            combo.setCurrentIndex(0)
        else:
            combo.addItem(placeholder)
            combo.setEnabled(False)
            ui.addRunSFFBtn.setEnabled(False)

    # ---------- Input wiring ----------
    def _set_inputs(self) -> None:
        try:
            min_nm = float(self._ui.lineEdit.text())
            max_nm = float(self._ui.lineEdit_2.text())
            global_thr = float(self._ui.threshSpinBox.value())  # max RMS
        except ValueError:
            raise ValueError("Min, Max, and Threshold must be numbers.")
        if min_nm >= max_nm:
            raise ValueError("Min wavelength must be < Max wavelength.")

        min_wl = u.Quantity(min_nm, "nm")
        max_wl = u.Quantity(max_nm, "nm")

        mode   = self._ui.SelectTargetData.currentText()
        target = self._ui.SelectTargetData_2.currentData()
        if target is None:
            raise ValueError(f"No {mode.lower()} selected.")

        # ----- Assign unique ids for all spectra that might touch -----
        ids: List[int] = []
        try:
            for s in self._app_state.get_collected_spectra():
                try: ids.append(s.get_id())
                except Exception: pass
        except Exception:
            pass
        try:
            for ds in self._app_state.get_datasets():
                try:
                    for s in ds.get_all_spectra():
                        ids.append(s.get_id())
                except Exception:
                    pass
        except Exception:
            pass
        for s in self.library:
            try: ids.append(s.get_id())
            except Exception: pass
        next_id = (max(ids) if ids else 0) + 1
        # -------------------------------------------------------------

        self._lib_name_by_spec_id.clear()
        refs: List[NumPyArraySpectrum] = []

        # Libraries loaded via file paths
        for lib_row in self._lib_rows:
            if lib_row["checkbox"].isChecked() and lib_row.get("path"):
                envilib = ENVISpectralLibrary(lib_row["path"])
                wls = u.Quantity([b["wavelength"] for b in envilib._band_list], u.nanometer)
                lib_filename = os.path.basename(lib_row["path"])
                row_thr = float(lib_row["threshold"].value())
                resolved_thr = row_thr  # per-row override already applied
                for i in range(envilib._num_spectra):
                    arr  = envilib._data[i]
                    name = envilib._spectra_names[i] if hasattr(envilib, "_spectra_names") else None
                    spec_from_lib = NumPyArraySpectrum(arr=arr, name=name, wavelengths=wls)
                    spec_from_lib.set_id(next_id)
                    self._lib_name_by_spec_id[spec_from_lib.get_id()] = lib_filename
                    setattr(spec_from_lib, "_sff_max_rms", resolved_thr)
                    next_id += 1
                    refs.append(spec_from_lib)

        # Individual spectra loaded from text files
        for row in self._spec_rows:
            if row["checkbox"].isChecked() and row.get("specs"):
                row_thr = float(row["threshold"].value())
                resolved_thr = row_thr
                spec_filename = os.path.basename(row.get("path") or "")
                for spec in row["specs"]:
                    spec.set_id(next_id)
                    self._lib_name_by_spec_id[spec.get_id()] = spec_filename
                    setattr(spec, "_sff_max_rms", resolved_thr)
                    next_id += 1
                    refs.append(spec)

        if not refs:
            raise ValueError("Please check at least one reference file.")

        self.set_wavelength_min(min_wl)
        self.set_wavelength_max(max_wl)
        self.set_max_rms(global_thr)
        self.set_target(target)
        self.set_library(refs)

    # ---------- Details dialog ----------
    def _view_details_dialog(self, matches: List[Dict[str, Any]], target: NumPyArraySpectrum, parent=None):
       
        top_ten = matches[:10]

        d = QDialog(parent or self)
        d.setWindowTitle("Spectral Feature Fitting — Details")
        d.setAttribute(Qt.WA_DeleteOnClose, True)
        d.setWindowFlag(Qt.Tool, True)
        d.setModal(False)
        d.setWindowFlags(d.windowFlags() | Qt.WindowTitleHint | Qt.WindowSystemMenuHint | Qt.WindowCloseButtonHint)

        layout = QVBoxLayout(d)

        # Plot target + up to 5 reference spectra
        target.set_color('#000000')
        plot_widget = SpectrumPlotGeneric(self._app_state)
        layout.addWidget(plot_widget)
        plot_widget.add_collected_spectrum(target)
        for rec in top_ten[:5]:
            plot_widget.add_collected_spectrum(rec["ref_obj"])

        # Table
        table = QTableWidget()
        table.setColumnCount(8)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setHorizontalHeaderLabels([
            "Match", "Library", "Fit RMS", "Scale", "Max RMS", "Min WL", "Max WL", "Target"
        ])
        table.setRowCount(len(top_ten))
        for i, rec in enumerate(top_ten):
            table.setItem(i, 0, QTableWidgetItem(rec.get("reference_data", "")))
            table.setItem(i, 1, QTableWidgetItem(rec.get("library_name", "")))
            table.setItem(i, 2, QTableWidgetItem(f"{rec['fit_rms']:.4f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{rec['scale']:.4f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{rec['max_rms']:.4f}"))
            table.setItem(i, 5, QTableWidgetItem(str(rec["min_wavelength"])))
            table.setItem(i, 6, QTableWidgetItem(str(rec["max_wavelength"])))
            table.setItem(i, 7, QTableWidgetItem(rec["target_name"]))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close, parent=d)
        btn_box.button(QDialogButtonBox.Save).setText("Save .txt")
        btn_box.rejected.connect(d.close)
        btn_box.button(QDialogButtonBox.Save).clicked.connect(lambda: self.save_txt(target, matches, d))
        layout.addWidget(btn_box)

        d.resize(1050, 800)
        d.show(); d.activateWindow()

    # ---------- Clear runs ----------
    def _on_clear_runs_clicked(self):
        res = self._show_message(
            "question", "Clear runs", "Clear all run history?", buttons=QMessageBox.Yes | QMessageBox.No, default=QMessageBox.No
        )
        if res != QMessageBox.Yes:
            return
        self._run_history.clear()
        self._ui.tableWidget.setRowCount(0)
        try:
            self._save_state()
        except Exception:
            pass

    # ---------- Settings persistence ----------
    def _settings(self) -> QSettings:
        s = QSettings("Wiser", "SFFPlugin")
        s.beginGroup(f"session/{os.getpid()}")
        return s

    def _q_to_nm(self, q):
        if q is None: return None
        try: return float(q.to(u.nanometer).value)
        except Exception: return float(getattr(q, "value", q))

    def _nm_to_q(self, v):
        return u.Quantity(v, "nm") if v is not None else None

    def _clear_library_rows(self):
        grid = self._ui.libGrid
        for row in self._lib_rows:
            cb, le = row["checkbox"], row["threshold"]
            grid.removeWidget(cb); cb.deleteLater()
            grid.removeWidget(le); le.deleteLater()
        self._lib_rows.clear()

    def _save_state(self) -> None:
        s = self._settings()
        s.setValue("min_nm", self._q_to_nm(self._min_wavelength))
        s.setValue("max_nm", self._q_to_nm(self._max_wavelength))
        s.setValue("threshold", self._max_rms)

        # target
        mode = self._ui.SelectTargetData.currentText()
        obj  = self._ui.SelectTargetData_2.currentData()
        s.setValue("target/mode", mode)
        s.setValue("target/name", getattr(obj, "get_name", lambda: None)())
        s.setValue("target/id",   getattr(obj, "get_id",   lambda: None)())

        # libraries
        s.beginWriteArray("libraries")
        for i, row in enumerate(self._lib_rows):
            s.setArrayIndex(i)
            s.setValue("path", row.get("path"))
            s.setValue("checked", row["checkbox"].isChecked())
            s.setValue("threshold_text", float(row["threshold"].value()))
            s.setValue("threshold_overridden", bool(row["threshold"].property("overridden")))
        s.endArray()

        s.beginWriteArray("spectra")
        for i, row in enumerate(self._spec_rows):
            s.setArrayIndex(i)
            s.setValue("path", row.get("path"))
            s.setValue("checked", row["checkbox"].isChecked())
            s.setValue("threshold_text", float(row["threshold"].value()))
            s.setValue("threshold_overridden", bool(row["threshold"].property("overridden")))
        s.endArray()

        # run history (summary only)
        flat = []
        for tname, runs in self._run_history.items():
            for r in runs:
                best   = r.get("best", {})
                target = r.get("target")
                flat.append({
                    "mode":         r.get("mode"),
                    "target_name":  tname,
                    "target_id":    getattr(target, "get_id", lambda: None)(),
                    "min_nm":       self._q_to_nm(r.get("min_wavelength")),
                    "max_nm":       self._q_to_nm(r.get("max_wavelength")),
                    "threshold":    float(r.get("max_rms", self._max_rms)),
                    "best_reference": best.get("reference_data"),
                    "best_library":   best.get("library_name"),
                    "best_rms":       float(best.get("fit_rms")) if best.get("fit_rms") is not None else None,
                    "best_scale":     float(best.get("scale")) if best.get("scale") is not None else None,
                })

        s.beginWriteArray("history")
        for i, row in enumerate(flat):
            s.setArrayIndex(i)
            s.setValue("mode",           row["mode"])
            s.setValue("target_name",    row["target_name"])
            s.setValue("target_id",      row["target_id"])
            s.setValue("min_nm",         row["min_nm"])
            s.setValue("max_nm",         row["max_nm"])
            s.setValue("threshold",      row["threshold"])
            s.setValue("best_reference", row["best_reference"])
            s.setValue("best_library",   row["best_library"])
            s.setValue("best_rms",       row["best_rms"])
            s.setValue("best_scale",     row["best_scale"])
        s.endArray()

    def _restore_target_selection(self, mode, name, sid):
        if mode in ("Spectrum", "Image Cube"):
            self._ui.SelectTargetData.setCurrentText(mode)
            combo = self._ui.SelectTargetData_2
            for idx in range(combo.count()):
                obj = combo.itemData(idx)
                if sid is not None and getattr(obj, "get_id", lambda: None)() == sid:
                    combo.setCurrentIndex(idx); return
            if name:
                for idx in range(combo.count()):
                    if getattr(combo.itemData(idx), "get_name", lambda: None)() == name:
                        combo.setCurrentIndex(idx); return

    def _load_state(self) -> None:
        s = self._settings()

        # settings
        min_nm = s.value("min_nm", type=float)
        max_nm = s.value("max_nm", type=float)
        self._min_wavelength = self._nm_to_q(min_nm)
        self._max_wavelength = self._nm_to_q(max_nm)
        if self._min_wavelength is not None:
            self._ui.lineEdit.setText(str(self._min_wavelength.to_value(u.nanometer)))
        if self._max_wavelength is not None:
            self._ui.lineEdit_2.setText(str(self._max_wavelength.to_value(u.nanometer)))

        thr = s.value("threshold", 0.03, type=float)
        self._max_rms = float(thr)
        self._ui.threshSpinBox.setValue(self._max_rms)

        # libraries
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
                saved_v   = s.value("threshold_text", None, type=float)
                overridden = bool(s.value("threshold_overridden", False, type=bool))
                row["threshold"].setProperty("overridden", overridden)
                row["threshold"].setValue(float(saved_v) if (overridden and saved_v is not None) else float(self._ui.threshSpinBox.value()))
        s.endArray()

        count = s.beginReadArray("spectra")
        if count > 0:
            if hasattr(self, "_clear_spec_rows"):
                self._clear_spec_rows()
            for i in range(count):
                s.setArrayIndex(i)
                self.addSpectrumRow()
                row = self._spec_rows[-1]
                row["path"] = s.value("path", type=str)
                row["checkbox"].setChecked(bool(s.value("checked", False, type=bool)))
                saved_v    = s.value("threshold_text", None, type=float)
                overridden = bool(s.value("threshold_overridden", False, type=bool))
                row["threshold"].setProperty("overridden", overridden)
                row["threshold"].setValue(float(saved_v) if (overridden and saved_v is not None) else float(self._ui.threshSpinBox.value()))
                filename = os.path.basename(row["path"] or "")
                row["checkbox"].setText(filename or f"Added_spectrum{row['row_index']}")
        s.endArray()

        if not self._lib_rows:
            self._add_default_library()

        # target (after list is populated)
        mode = s.value("target/mode", type=str)
        name = s.value("target/name", type=str)
        sid  = s.value("target/id", type=int)
        self._restore_target_selection(mode, name, sid)

        # history table (summary only)
        table = self._ui.tableWidget
        count = s.beginReadArray("history")
        for i in range(count):
            s.setArrayIndex(i)
            row_idx = table.rowCount()
            table.insertRow(row_idx)
            tname = s.value("target_name", "", type=str)
            bref  = s.value("best_reference", "", type=str)
            brms  = s.value("best_rms", None)
            table.setItem(row_idx, 0, QTableWidgetItem(tname))
            table.setItem(row_idx, 1, QTableWidgetItem(bref))
            table.setItem(row_idx, 2, QTableWidgetItem(f"{float(brms):.4f}" if brms is not None else ""))

            spec = {
                "mode":        s.value("mode", "Spectrum", type=str),
                "target_name": tname,
                "target_id":   s.value("target_id", None, type=int),
                "min_nm":      s.value("min_nm", None, type=float),
                "max_nm":      s.value("max_nm", None, type=float),
                "threshold":   s.value("threshold", self._max_rms, type=float),
            }
            btn = QPushButton("View Details", table)
            btn.clicked.connect(lambda _=False, sp=spec: self._replay_saved(sp))
            table.setCellWidget(row_idx, 3, btn)
        s.endArray()
        table.resizeColumnsToContents()

    def _replay_saved(self, spec: dict) -> None:
        if spec.get("min_nm") is not None:
            self._ui.lineEdit.setText(str(spec["min_nm"]))
        if spec.get("max_nm") is not None:
            self._ui.lineEdit_2.setText(str(spec["max_nm"]))
        if spec.get("threshold") is not None:
            self._ui.threshSpinBox.setValue(float(spec["threshold"]))
        mode = spec.get("mode", "Spectrum")
        self._ui.SelectTargetData.setCurrentText(mode)
        self._restore_target_selection(mode, spec.get("target_name"), spec.get("target_id"))
        self.run_sff()

    def _purge_old_sessions(self):
        s = QSettings("Wiser", "SFFPlugin")
        s.beginGroup("session")
        for g in s.childGroups():
            if g != str(os.getpid()):
                s.remove(g)
        s.endGroup()

    # ---------- Export ----------
    def save_txt(self, target, matches, parent=None):
        tgt_name = getattr(target, "get_name", lambda: None)() or "target"
        safe_target = tgt_name.replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = getattr(self._app_state, "get_current_dir", lambda: None)() or os.path.expanduser("~")
        default_path = os.path.join(base_dir, f"{safe_target}_SFF_{stamp}.txt")

        dlg = QFileDialog(parent or self, "Save SFF results as .txt")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilters(["Text files (*.txt)", "All Files (*)"])
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.selectFile(os.path.basename(default_path))
        dlg.setDirectory(os.path.dirname(default_path))
        dlg.show(); dlg.raise_(); dlg.activateWindow()
        if dlg.exec_() != QDialog.Accepted:
            return
        path = dlg.selectedFiles()[0]

        # header
        max_rms = (matches[0]["max_rms"] if matches else self._max_rms)
        try:
            min_nm = f"{self._min_wavelength.to_value(u.nm):.1f}"
        except Exception:
            min_nm = str(self._min_wavelength) if self._min_wavelength is not None else "—"
        try:
            max_nm = f"{self._max_wavelength.to_value(u.nm):.1f}"
        except Exception:
            max_nm = str(self._max_wavelength) if self._max_wavelength is not None else "—"

        header = [
            "Spectral Feature Fitting (SFF) — WISER Plugin Export",
            f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target: {tgt_name}",
            f"Wavelength range [nm]: {min_nm} – {max_nm}",
            f"Max RMS: {max_rms:.4f}",
            "Interpolation: linear",
            "",
        ]

        head = " | ".join(
            ["Rank", "Reference", "Library", "Fit RMS", "Scale", "Max RMS", "Min WL [nm]", "Max WL [nm]", "Target"]
        )
        lines = [head, "-" * len(head)]
        for i, m in enumerate(matches, start=1):
            lines.append(" | ".join([
                str(i),
                m.get("reference_data", ""),
                m.get("library_name", ""),
                f"{m['fit_rms']:.4f}",
                f"{m['scale']:.4f}",
                f"{m['max_rms']:.4f}",
                str(m.get("min_wavelength", "")),
                str(m.get("max_wavelength", "")),
                m.get("target_name", "")
            ]))

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

    # ---------- Helper to keep rows synced with global ----------
    def _bind_threshold_spin(self, spin: QDoubleSpinBox) -> QDoubleSpinBox:
        spin.setRange(0.0, 1.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.005)
        spin.setKeyboardTracking(False)
        g = float(self._ui.threshSpinBox.value())
        spin.setValue(g)
        spin.setProperty("overridden", False)
        spin.editingFinished.connect(lambda sb=spin: sb.setProperty("overridden", True))
        return spin
