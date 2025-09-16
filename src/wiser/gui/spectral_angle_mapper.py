#Python SAM logic

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

from wiser.gui.generated.spectral_angle_mapper_ui import Ui_SpectralAngleMapper
import warnings

# -------------------------------------------------------------------------
# Spectral Angle Mapper (SAM) Plugin Logic for WISER
#
# Provides tools for running SAM analysis between a target spectrum and
# reference libraries/spectra. Handles:
#   - Loading spectral libraries and user spectra
#   - Managing wavelength bounds and thresholds
#   - Computing spectral angles (per-spectrum or per-image pixel)
#   - Recording run history and exporting results
#   - UI integration with Qt (PySide2)
#
# Typical flow:
#   1. User selects a target (spectrum or image cube) via UI
#   2. Libraries and/or spectra are added to reference list
#   3. User sets wavelength bounds + threshold
#   4. run_sam() executes SAM, saves history, and displays results
# 
# Code originally written by Daphne Nea, UCLA '26
#
# -------------------------------------------------------------------------

class SAMTool(QDialog):
    """
    Core logic for the SAM plugin.

    Responsibilities:
    - Store state (target spectrum, reference libraries, thresholds, run history)
    - Provide setters/getters for UI integration
    - Compute spectral angle (per-spectrum and image cube modes)
    - Connect with UI buttons (run, add library/spectrum, clear runs)
    - Handle persistence via QSettings (save/restore state)

    Key attributes:
    - self._target            : the currently selected target spectrum
    - self.library            : list of reference spectra to compare against
    - self._threshold         : global angle threshold in degrees
    - self._min_wavelength / self._max_wavelength : global wavelength bounds
    - self._run_history       : dictionary of past SAM runs
    """
    THRESHOLD_DEFAULT = 5.0
    _sessions_purged = False
    
    def __init__(self, app_state: ApplicationState, parent: QWidget = None):
        """
        Initialize SAM plugin logic, set up UI, restore prior session state,
        and add a default USGS library if none are loaded.
        """
        super().__init__(parent)
        
        if not SAMTool._sessions_purged:
            self._purge_old_sessions()
            SAMTool._sessions_purged = True
        
        
        self._app_state = app_state
        self._target: NumPyArraySpectrum = None
        self.library: List[NumPyArraySpectrum] = []
        self._lib_name_by_spec_id: Dict[int, str] = {}
        self._threshold: float = None
        self._min_wavelength: Optional[u.Quantity] = None
        self._max_wavelength: Optional[u.Quantity] = None
        self._run_history: Dict[str, List[Dict[str, Any]]] = {}

        self._lib_rows = []
        self._add_default_library()

        self._spec_rows = []

        self._ui = Ui_SpectralAngleMapper()
        self._ui.setupUi(self)
        self._ui.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._setup_connections()
        self._add_interpolation_note()
        self._init_target_dropdowns()
        self._ui.threshSpinBox.setValue(self.THRESHOLD_DEFAULT)
        self._threshold = self.THRESHOLD_DEFAULT
        self._load_state()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(lambda: QSettings("Wiser","SAMPlugin").remove(f"session/{os.getpid()}"))


    # ------------------------- Core Setters/Getters -------------------------

    def set_target(self, target: NumPyArraySpectrum) -> None:
        self._target = target

    def set_threshold(self, threshold: Optional[float]) -> None:
        self._threshold = threshold

    def set_library(self, spectra: List[NumPyArraySpectrum]) -> None:
        self.library = list(spectra)
    
    def set_wavelength_min(self, min: Optional[u.Quantity]) -> None:
        self._min_wavelength = min
    
    def set_wavelength_max(self, max: Optional[u.Quantity]) -> None:
        self._max_wavelength = max
    
    def get_target_name(self) -> str:
        return self._target.get_name() if self._target else "<no target>"

    def get_threshold(self) -> Optional[float]:
        return self._threshold
    
    def get_wavelength_min(self) -> Optional[u.Quantity]:
        return self._min_wavelength
    
    def get_wavelength_max(self) -> Optional[u.Quantity]:
        return self._max_wavelength

    def compute_spectral_angle(self, ref: NumPyArraySpectrum) -> float:
            """
            Compute the spectral angle (in degrees) between self._target and ref:
            θ = arccos( (t·r) / (||t||·||r||) )
            Steps:
            1. slice both to bounds → (arr, wls)
            2. interpolate ref onto target’s wavelength grid
            3. mask out NaNs/infs
            4. enforce MIN_SAMPLES
            5. compute dot/norm safely
            Returns:
            - finite angle in degrees
            - np.nan if too few valid points
            - 90.0° if one vector is zero
            """
            MIN_SAMPLES = 3

            if self._target is None:
                raise RuntimeError("compute_spectral_angle called without a target set")

            t_arr, t_wls = self._slice_to_bounds(self._target)
            r_arr, r_wls = self._slice_to_bounds(ref)

            # pull out raw floats for interp1d—units must be gone here:
            t_x = t_wls.value
            r_x = r_wls.value

            # 2) skip interp if grids are identical
            if np.array_equal(r_x, t_x):
                r_resampled = r_arr
            else:
                interp_fn = interp1d(r_x, r_arr,
                                    bounds_error=False,
                                    fill_value=np.nan)
                r_resampled = interp_fn(t_x)

            # 3) mask out any NaNs or infinities
            valid = np.isfinite(t_arr) & np.isfinite(r_resampled)
            t_vec = t_arr[valid]
            r_vec = r_resampled[valid]

            # 4) check for enough samples
            if t_vec.size < MIN_SAMPLES:
                warnings.warn(
                    f"Only {t_vec.size} valid samples (< {MIN_SAMPLES}) — "
                    "skipping angle computation",
                    UserWarning
                )
                return np.nan

            # 5) compute dot and norms
            numerator = np.dot(t_vec, r_vec)
            denominator = np.linalg.norm(t_vec) * np.linalg.norm(r_vec)
            if denominator == 0:
                return 90.0

            angle_rad = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
            return float(np.degrees(angle_rad))


    #-----------------------IMAGE CUBE--------------------------- (did not implement)

    '''
    def compute_spectral_angle_image(self, dataset: RasterDataSet):
        arr = dataset.get_image_data()

        interate through arr to get each spectrum
        - plug into compute_spectral_angle(ref: NumPyArraySpectrum)
        - put your output values in a 2D array (of size HxW where H and W are from dataset.get_height() and dataset.get_width())
        - Make the 2D array into a NumPyRasterDataImpl (in dataset_impl.py) and then make a RasterDataSet object from this
            - gray_scale_ds = RasterDataSet(NumpyRasterDataImpl(arr), self.app_state.get_cache())
            - gray_scale_ds.set_name(dataset.get_name + "_" )
            - self.app_state.add_dataset(gray_scale_ds)
        things to add:
        - if the dataset's size is too big (get_memory_size, with psutil you can get the available RAM, if memory_size > avilable RAM
        then we'd subset)
    '''

    def compute_spectral_angle_image(self, dataset: RasterDataSet):
        img_cube = dataset.get_image_data()
        height, width = dataset.get_height(), dataset.get_width()
        num_bands = img_cube.shape[2]

        angle_image = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                spectrum_arr = img_cube[y, x, :]
                if np.isnan(spectrum_arr).any():
                    angle = np.nan
                else:
                    spectrum = NumPyArraySpectrum(
                        arr=spectrum_arr,
                        wavelengths=self._target.get_wavelengths(),
                        name=f"pixel_{y}_{x}"
                    )
                    try:
                        angle = self.compute_spectral_angle(spectrum)
                    except Exception as e:
                        angle = np.nan

                angle_image[y, x] = angle

        result_impl = NumPyRasterDataImpl(angle_image)
        result_dataset = RasterDataSet(result_impl, self._app_state.get_cache())
        result_dataset.set_name(dataset.get_name() + "_SAM")

        self._app_state.add_dataset(result_dataset)
 
    #-----------------------IMAGE CUBE---------------------------


    def find_matches(self) -> List[Dict[str, Any]]:
        """
        USED TO POPULATE TABLE OF INFO
        Compute spectral angles between self._target and each spectrum in the library,
        filter by self._threshold, and return a list of dicts which contains in each dict:
            
        - target_name      : name of target spectrum
        - reference_data   : name of reference spectrum
        - library_name     : originating library filename
        - spectral_angle   : computed angle [degrees]
        - threshold_degree : threshold applied to this match [degrees]
        - min_wavelength   : global lower bound [Quantity]
        - max_wavelength   : global upper bound [Quantity]
    
        """

        matches: List[Dict[str, Any]] = []

        for i, spec in enumerate(self.library):
                
            spectral_angle = SAMTool.compute_spectral_angle(self, spec)

            thr = getattr(spec, "_sam_threshold", self._threshold)

            if spectral_angle <= float(thr):
                matches.append({
                    "target_name":           self.get_target_name(),
                    "spectral_angle":          spectral_angle,
                    "threshold_degree":      float(thr),
                    "min_wavelength": self.get_wavelength_min(),
                    "max_wavelength": self.get_wavelength_max(),
                    "reference_data": ((self.library)[i]).get_name(),
                    "ref_obj": spec,
                    "library_name":    self._lib_name_by_spec_id.get(spec.get_id(), "")
                })
                
        return matches

    def sort_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return a sorted list of matches so that the lowest spectral angles (best matches)
        come first. Sorts purely based off of the numeric "angle" value.
        """
        return sorted(matches, key=lambda rec: rec["spectral_angle"])

    def run_sam(self):
        ui = self._ui

        ui.addRunSAMBtn.setEnabled(False)
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
                    "Error during SAM",
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
                    "No matches were found within the threshold.",
                    informative=f"Threshold: {self._threshold}°, "
                                f"Range: {self._min_wavelength} – {self._max_wavelength}"
                )

        finally:
            QApplication.restoreOverrideCursor()
            ui.addRunSAMBtn.setEnabled(True)

    def cancel(self):
       self.reject()
          
    def save_and_close(self):
       """Closes window with saving"""
       self._save_state()
       self.accept()

    def addSpectrumRow(self):
        row = self._ui.specGrid.rowCount()
        cb = QCheckBox(f"Added_spectrum{row}")

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 180.0)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setKeyboardTracking(False)
        self._bind_threshold_spin(spin)

        self._ui.specGrid.addWidget(cb,   row, 0, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.specGrid.addWidget(spin, row, 1, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.specGrid.setRowStretch(row+1, 1)
        self._spec_rows.append({
            "checkbox": cb,
            "threshold": spin,
            "row_index": row,
            "path": None,
            "specs": []
        })
            
    def addLibraryRow(self):
        row = len(self._lib_rows) + 1
        cb = QCheckBox(f"Library_{row-1}")

        spin = QDoubleSpinBox()
        spin.setRange(0.0, 180.0)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setKeyboardTracking(False)
        self._bind_threshold_spin(spin)

        self._ui.libGrid.addWidget(cb,   row, 0, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.libGrid.addWidget(spin, row, 1, Qt.AlignLeft | Qt.AlignBaseline)
        self._ui.libGrid.setRowStretch(row + 1, 1)
        self._lib_rows.append({
            "checkbox": cb,
            "threshold": spin,
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
            "threshold": self._threshold,
            "min_wavelength": self._min_wavelength,
            "max_wavelength": self._max_wavelength,
        }
        self._run_history.setdefault(key, []).append(run_entry)

        table = self._ui.tableWidget
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(key))
        table.setItem(row, 1, QTableWidgetItem(best.get("reference_data", "")))
        table.setItem(row, 2, QTableWidgetItem(f"{best['spectral_angle']:.2f}°"))

        btn = QPushButton("View Details", table)
        m = list(sorted_matches)
        t = target
        btn.clicked.connect(lambda _=False, mm=m, tt=t: self._view_details_dialog(mm, tt))
        table.setCellWidget(row, 3, btn)

        table.resizeColumnsToContents()

    def _setup_connections(self):
       ui = self._ui
       ui.addRunSAMBtn.clicked.connect(self.run_sam)
       ui.addSaveCloseBtn.clicked.connect(self.save_and_close)
       ui.addCancelBtn.clicked.connect(self.cancel)
       ui.addLibBtn.clicked.connect(self._on_add_library_clicked)
       ui.addSpecBtn.clicked.connect(self._on_add_spectrum_clicked)
       ui.SelectTargetData.currentTextChanged.connect(self._on_target_type_changed)
       ui.clearRunsBtn.clicked.connect(self._on_clear_runs_clicked)
       
    def _add_interpolation_note(self) -> None:

        note = QLabel("Interpolation: linear")
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

        thr_layout.addRow("", note)

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

        box.raise_()
        box.activateWindow()
        return box.exec_()


    def _init_target_dropdowns(self) -> None:
        ui = self._ui
        ui.SelectTargetData.blockSignals(True)
        ui.SelectTargetData.clear()
        ui.SelectTargetData.addItems(["Spectrum", "Image Cube"])
        ui.SelectTargetData.blockSignals(False)

        ui.SelectTargetData.setCurrentText("Spectrum")
        self._on_target_type_changed("Spectrum")
        
    def _add_default_library(self) -> None:
        try:
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(
                plugin_dir,
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

    def _slice_to_bounds(self, spectrum: NumPyArraySpectrum) -> Tuple[np.ndarray, u.Quantity]:
    
        """
        Clip a spectrum to global min/max wavelength bounds.

        - Converts input wavelengths to match min/max units.
        - Handles dimensionless → nm conversion.
        - Ensures reflectance array shape matches wavelength array.
        - Returns (reflectance_array, wavelength_array).
        """

        # 1) Get wavelengths as a Quantity
        wls = spectrum.get_wavelengths()
        if not isinstance(wls, u.Quantity):
            wls = u.Quantity(wls, u.nanometer)
        elif wls.unit == u.dimensionless_unscaled:
            wls = u.Quantity(wls.value, u.nanometer)

        # 2) Convert to the user’s units, if needed
        if self._min_wavelength is not None:
            wls = wls.to(self._min_wavelength.unit)
        if self._max_wavelength is not None:
            wls = wls.to(self._max_wavelength.unit)

        # 3) Get reflectances, stripping units if necessary
        raw_r = spectrum.get_spectrum()
        if isinstance(raw_r, u.Quantity):
            arr = raw_r.value
        else:
            arr = np.asarray(raw_r)

        # 4) Sanity-check shapes
        if arr.ndim != 1 or arr.shape[0] != wls.shape[0]:
            raise ValueError(
                f"Shape mismatch: reflectance has shape {arr.shape}, "
                f"wavelengths has shape {wls.shape}"
            )

        # 5) Build a mask of “in‐bounds” wavelengths
        mask = np.ones(wls.shape, dtype=bool)
        if self._min_wavelength is not None:
            mask &= (wls >= self._min_wavelength)
        if self._max_wavelength is not None:
            mask &= (wls <= self._max_wavelength)

        return arr[mask], wls[mask]

    def _on_add_spectrum_clicked(self):
        # 1) open the Qt file-picker as a modal dialog
        start_dir = self._app_state.get_current_dir() or os.path.expanduser("~")

        filedlg = QFileDialog(self, "Import Spectra from Text File", start_dir,
                            "Text files (*.txt);;All Files (*)")
        filedlg.setFileMode(QFileDialog.ExistingFile)
        filedlg.setAcceptMode(QFileDialog.AcceptOpen)
        filedlg.setWindowModality(Qt.WindowModal)

        if filedlg.exec_() != QDialog.Accepted:
            print("ERROR")
            return

        path = filedlg.selectedFiles()[0]

        # 2) update state, run ImportSpectraTextDialog, etc.
        self._app_state.update_cwd_from_path(path)
        dlg = ImportSpectraTextDialog(path, parent=self)
        dlg.setWindowModality(Qt.WindowModal)
        if dlg.exec() != QDialog.Accepted:
            return

        # 3) grab the resulting spectra and register them
        specs = dlg.get_spectra()
        if not specs:
            return

        #  register as a spectral library in app state
        lib = ListSpectralLibrary(specs, path=path)
        self._app_state.add_spectral_library(lib)

        # make one UI row for this file and attach all its spectra
        self.addSpectrumRow()
        row = self._spec_rows[-1]
        filename = os.path.basename(path)
        row["checkbox"].setText(f"{filename} ({len(specs)} spectra)")
        row["checkbox"].setChecked(True)
        row["path"] = path
        row["specs"].extend(specs)   # <-- key: store the parsed spectra here

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
            ui.addRunSAMBtn.setEnabled(True)
            combo.setCurrentIndex(0)
        else:
            combo.addItem(placeholder)
            combo.setEnabled(False)
            ui.addRunSAMBtn.setEnabled(False)

   
    def _set_inputs(self) -> None:
        """
        Saves/sets inputs:
        - min/max wavelength, threshold
        - target selection (mode, name, id)
        - library/spectra rows with thresholds
        - run history summary (not full details)
        """

        try:
            min_nm = float(self._ui.lineEdit.text())
            max_nm = float(self._ui.lineEdit_2.text())
            global_thr = self._ui.threshSpinBox.value()
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
        
# ------------------- id stuff ---------------------------------
        ids = []

        for s in self._app_state.get_collected_spectra():
            try: ids.append(s.get_id())
            except Exception: pass

        for ds in self._app_state.get_datasets():
            try:
                for s in ds.get_all_spectra():
                    ids.append(s.get_id())
            except Exception: pass

        for s in self.library:
            try: ids.append(s.get_id())
            except Exception: pass

        next_id = (max(ids) if ids else 0) + 1

# ------------------- id stuff ---------------------------------
        self._lib_name_by_spec_id.clear()

        refs: List[NumPyArraySpectrum] = []
        
        for lib_row in self._lib_rows:
            if lib_row["checkbox"].isChecked() and lib_row.get("path"):
                envilib = ENVISpectralLibrary(lib_row["path"])
                wls = u.Quantity([b["wavelength"] for b in envilib._band_list], u.nanometer)
                lib_filename = os.path.basename(lib_row["path"])

                val = float(lib_row["threshold"].value())
                row_thr = float(lib_row["threshold"].value())
                resolved_thr = row_thr

                for i in range(envilib._num_spectra):
                    arr  = envilib._data[i]
                    name = envilib._spectra_names[i] if hasattr(envilib, "_spectra_names") else None
                    spec_from_lib = NumPyArraySpectrum(arr=arr, name=name, wavelengths=wls)
                    spec_from_lib.set_id(next_id)
                    self._lib_name_by_spec_id[spec_from_lib.get_id()] = lib_filename

                    setattr(spec_from_lib, "_sam_threshold", resolved_thr)

                    next_id += 1
                    refs.append(spec_from_lib)

        for row in self._spec_rows:
            if row["checkbox"].isChecked() and row.get("specs"):
                val = float(row["threshold"].value())
                row_thr = float(row["threshold"].value())
                resolved_thr = row_thr
                spec_filename = os.path.basename(row.get("path") or "")

                for spec in row["specs"]:
                    spec.set_id(next_id)
                    self._lib_name_by_spec_id[spec.get_id()] = spec_filename

                    setattr(spec, "_sam_threshold", resolved_thr)

                    next_id += 1
                    refs.append(spec)

        if not refs:
            raise ValueError("Please check at least one reference file.")

        self.set_wavelength_min(min_wl)
        self.set_wavelength_max(max_wl)
        self.set_threshold(global_thr)
        self.set_target(target)

        if hasattr(self, "set_library"):
            self.set_library(refs)
        else:
            self.library = refs


    def _view_details_dialog(self, matches, target, parent=None):
        
        top_ten = matches[:10]

        d = QDialog(parent or self)
        d.setWindowTitle("Spectral Angle Details")
        d.setAttribute(Qt.WA_DeleteOnClose, True)
        d.setWindowFlag(Qt.Tool, True)
        d.setModal(False)

        d.setWindowFlags(
            d.windowFlags()
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout(d)

        # Plot
        target.set_color('#000000')
        plot_widget = SpectrumPlotGeneric(self._app_state)
        layout.addWidget(plot_widget)
        plot_widget.add_collected_spectrum(target)
        
        for ref in top_ten[:5]:
            
            plot_widget.add_collected_spectrum(ref["ref_obj"])

        # Table
        table = QTableWidget()
        table.setColumnCount(7)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setHorizontalHeaderLabels([
            "Match", "Library", "Angle (°)", "Threshold (°)", "Min WL", "Max WL", "Target"
        ])
        table.setRowCount(len(top_ten))
        for i, rec in enumerate(top_ten):
            table.setItem(i, 0, QTableWidgetItem(rec.get("reference_data", "")))
            table.setItem(i, 1, QTableWidgetItem(rec.get("library_name", "")))
            table.setItem(i, 2, QTableWidgetItem(f"{rec['spectral_angle']:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{rec['threshold_degree']:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(str(rec["min_wavelength"])))
            table.setItem(i, 5, QTableWidgetItem(str(rec["max_wavelength"])))
            table.setItem(i, 6, QTableWidgetItem(rec["target_name"]))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close, parent=d)
        btn_box.button(QDialogButtonBox.Save).setText("Save .txt")
        btn_box.rejected.connect(d.close)
        btn_box.button(QDialogButtonBox.Save).clicked.connect(lambda: self.save_txt(target, matches, d))
        layout.addWidget(btn_box)

        d.resize(1050, 800)
        d.show()
        d.activateWindow()
        
    def _on_clear_runs_clicked(self):
        res = self._show_message(
            "question",
            "Clear runs",
            "Clear all run history?",
            buttons=QMessageBox.Yes | QMessageBox.No,
            default=QMessageBox.No
        )
        if res != QMessageBox.Yes:
            return
        self._run_history.clear()
        self._ui.tableWidget.setRowCount(0)

        # Clear in-memory history
        self._run_history.clear()

        # Persist the cleared state
        try:
            if hasattr(self, "_save_state"):
                self._save_state()
        except Exception:
            pass


#save session helpers
    def _settings(self) -> QSettings:
        s = QSettings("Wiser", "SAMPlugin")
        s.beginGroup(f"session/{os.getpid()}")  # everything under this session
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
        s.setValue("threshold", self._threshold)

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
                    "threshold":    float(r.get("threshold", self._threshold)),
                    "best_reference": best.get("reference_data"),
                    "best_library":   best.get("library_name"),
                    "best_angle":     float(best.get("spectral_angle")) if best.get("spectral_angle") is not None else None,
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
            s.setValue("best_angle",     row["best_angle"])
        s.endArray()

    def _restore_target_selection(self, mode, name, sid):
        if mode in ("Spectrum", "Image Cube"):
            self._ui.SelectTargetData.setCurrentText(mode)  # repopulates list
            combo = self._ui.SelectTargetData_2
            # try by id, then by name
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

        thr = s.value("threshold", 5.0, type=float)
        self._threshold = float(thr)
        self._ui.threshSpinBox.setValue(self._threshold)

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

                if overridden and saved_v is not None:
                    row["threshold"].setValue(float(saved_v))
                else:
                    # stay synced to current global
                    row["threshold"].setValue(float(self._ui.threshSpinBox.value()))
            
        s.endArray()

        count = s.beginReadArray("spectra")
        if count > 0:
            # remove any placeholder row(s) from a fresh UI
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
                if overridden and saved_v is not None:
                    row["threshold"].setValue(float(saved_v))
                else:
                    row["threshold"].setValue(float(self._ui.threshSpinBox.value()))

                # set label now; we’ll refresh count after lazy parse
                filename = os.path.basename(row["path"] or "")
                row["checkbox"].setText(filename or f"Added_spectrum{row['row_index']}")
        s.endArray()

        # if no libs were restored, add your packaged default
        if not self._lib_rows:
            self._add_default_library()

        # target (after list is populated)
        mode = s.value("target/mode", type=str)
        name = s.value("target/name", type=str)
        sid  = s.value("target/id", type=int)
        self._restore_target_selection(mode, name, sid)

        # history (summary only; details disabled)
        table = self._ui.tableWidget
        count = s.beginReadArray("history")
        for i in range(count):
            s.setArrayIndex(i)
            row_idx = table.rowCount()
            table.insertRow(row_idx)

            tname = s.value("target_name", "", type=str)
            bref  = s.value("best_reference", "", type=str)
            bang  = s.value("best_angle", None)
            table.setItem(row_idx, 0, QTableWidgetItem(tname))
            table.setItem(row_idx, 1, QTableWidgetItem(bref))
            table.setItem(row_idx, 2, QTableWidgetItem(f"{float(bang):.2f}°" if bang is not None else ""))

            spec = {
                "mode":        s.value("mode", "Spectrum", type=str),
                "target_name": tname,
                "target_id":   s.value("target_id", None, type=int),
                "min_nm":      s.value("min_nm", None, type=float),
                "max_nm":      s.value("max_nm", None, type=float),
                "threshold":   s.value("threshold", self._threshold, type=float),
            }

            btn = QPushButton("View Details", table)
            btn.clicked.connect(lambda _=False, sp=spec: self._replay_saved(sp))
            table.setCellWidget(row_idx, 3, btn)
        s.endArray()
        table.resizeColumnsToContents()


    def _replay_saved(self, spec: dict) -> None:
        # 1) set inputs into the UI (so your normal pipeline runs)
        if spec.get("min_nm") is not None:
            self._ui.lineEdit.setText(str(spec["min_nm"]))
        if spec.get("max_nm") is not None:
            self._ui.lineEdit_2.setText(str(spec["max_nm"]))
        if spec.get("threshold") is not None:
            self._ui.threshSpinBox.setValue(float(spec["threshold"]))
            
        mode = spec.get("mode", "Spectrum")
        self._ui.SelectTargetData.setCurrentText(mode)  # repopulates second combo
        self._restore_target_selection(
            mode,
            spec.get("target_name"),
            spec.get("target_id"),
        )

        self.run_sam()

    def _purge_old_sessions(self):
        s = QSettings("Wiser", "SAMPlugin")
        s.beginGroup("session")
        for g in s.childGroups():
            if g != str(os.getpid()):
                s.remove(g)
        s.endGroup()

    def save_txt(self, target, matches, parent=None):
        # default filename
        tgt_name = getattr(target, "get_name", lambda: None)() or "target"
        safe_target = tgt_name.replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = getattr(self._app_state, "get_current_dir", lambda: None)() or os.path.expanduser("~")
        default_path = os.path.join(base_dir, f"{safe_target}_SAM_{stamp}.txt")

        dlg = QFileDialog(parent or self, "Save SAM results as .txt")
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
        thr = f"{(matches[0]['threshold_degree'] if matches else self._threshold):.3f}"
        try: min_nm = f"{self._min_wavelength.to_value(u.nm):.1f}"
        except: min_nm = str(self._min_wavelength) if self._min_wavelength is not None else "—"
        try: max_nm = f"{self._max_wavelength.to_value(u.nm):.1f}"
        except: max_nm = str(self._max_wavelength) if self._max_wavelength is not None else "—"

        header = [
            "Spectral Angle Mapper (SAM) — WISER Plugin Export",
            f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Target: {tgt_name}",
            f"Wavelength range [nm]: {min_nm} – {max_nm}",
            f"Threshold [deg]: {thr}",
            "Interpolation: linear",
            ""
        ]

        # table
        head = " | ".join(["Rank","Reference","Library","Angle [deg]","Thresh [deg]","Min WL [nm]","Max WL [nm]","Target"])
        lines = [head, "-" * len(head)]
        for i, m in enumerate(matches, start=1):
            lines.append(" | ".join([
                str(i),
                m.get("reference_data",""),
                m.get("library_name",""),
                f"{m['spectral_angle']:.3f}",
                f"{m['threshold_degree']:.3f}",
                str(m.get("min_wavelength","")),
                str(m.get("max_wavelength","")),
                m.get("target_name","")
            ]))

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(header + lines) + "\n")
            try: self._app_state.update_cwd_from_path(path)
            except: pass

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

    def _bind_threshold_spin(self, spin: QDoubleSpinBox) -> QDoubleSpinBox:
        """Initialize a row spin to the current global and mark as 'not overridden'.
           Use editingFinished to mark when the user takes control."""
        spin.setRange(0.0, 180.0)        # no sentinel
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setKeyboardTracking(False)

        # initialize to current global
        g = float(self._ui.threshSpinBox.value())
        spin.setValue(g)
        spin.setProperty("overridden", False)

        # mark overridden only on user edit
        spin.editingFinished.connect(lambda sb=spin: sb.setProperty("overridden", True))
        return spin


    #---------------Logic originally used to sort wavelengths to identify default wavelengths-------------------
        #right now wavelength inputs are line edits, might be better to convert to QSpinBox

        # wavelength_list = spectrum.get_wavelengths()
        # local_min = wavelength_list[0]
        # local_max = wavelength_list[0]

        # for wl in wavelength_list:
        #     if wl < local_min:
        #         local_min = wl
        #     if wl > local_max:
        #         local_max = wl

        # if self.get_wavelength_min() is None or local_min < self.get_wavelength_min():
        #     self._min_wavelength = local_min
        # if self.get_wavelength_max() is None or local_max > self.get_wavelength_max():
        #     self._max_wavelength = local_max
