import datetime
import os
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide2.QtCore import QObject, Qt, Signal, Slot
from PySide2.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QTableWidgetItem,
)

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.linear_unmixing_dialog_ui import Ui_LinearUnmixingDialog
from wiser.gui.import_spectra_text import ImportSpectraTextDialog
from wiser.gui.util import StateChange
from wiser.gui.utils import build_trash_button
from wiser.raster.dataset import RasterDataSet
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.spectrum import Spectrum
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataMeta,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    ExternalRasterHandle,
    PriorityClass,
    SpatialTileScheme,
)
from wiser.utils.task_stage_utils import (
    get_good_band_runs,
    split_dataset_tile_by_good_band_runs,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    SemanticTask,
    WriteSpec,
)
from wiser.utils.worker_runtime import get_process_storage_client


_ENDMEMBER_NAME_COL = 0
_ENDMEMBER_REMOVE_COL = 1


class LinearUnmixingDialog(QDialog):
    """Dialog for configuring and launching a linear unmixing task."""

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.setModal(False)
        self._app_state = app_state
        self._app_services = app_services
        # Remembered so that showEvent can re-select the right dataset if the
        # dialog is hidden and shown again after datasets have changed.
        self._selected_dataset_id: Optional[int] = None

        self._ui = Ui_LinearUnmixingDialog()
        self._ui.setupUi(self)

        self._configure_endmember_table()
        self._ui.btn_add_collected_spec.clicked.connect(self._on_add_collected_spec)
        self._ui.btn_import_spec.clicked.connect(self._on_import_spec)

        app_state.dataset_added.connect(self._on_datasets_changed)
        app_state.dataset_removed.connect(self._on_datasets_changed)

        # Keep the endmember table in sync with the app's spectrum store: if a
        # collected spectrum is discarded, or a spectral library is removed,
        # drop any rows referencing them. (No name-change signal exists, so we
        # can't reactively update display names — they'll just stay stale until
        # the row is removed.)
        app_state.collected_spectra_changed.connect(self._on_collected_spectra_changed)
        app_state.spectral_library_removed.connect(self._on_spectral_library_removed)

    def show_linear_unmixing(self, dataset_id: Optional[int] = None) -> None:
        """Rebuild the dataset combo box and pre-select `dataset_id` if given."""
        cbox = self._ui.cbox_input_ds
        cbox.clear()
        # Sentinel item: userData=-1 means "nothing selected". We check for
        # this in get_selected_dataset() to avoid passing junk to the backend.
        cbox.addItem(self.tr("(no data)"), -1)
        for dataset in self._app_state.get_datasets():
            cbox.addItem(dataset.get_name(), dataset.get_id())

        if dataset_id is not None:
            index = cbox.findData(dataset_id)
            # Fall back to index 0 ("no data") if the requested ID disappeared
            # (e.g., the dataset was removed between calls).
            cbox.setCurrentIndex(index if index >= 0 else 0)
        else:
            cbox.setCurrentIndex(0)

    def showEvent(self, event):
        # Re-populate the combo box on every show so it reflects any datasets
        # added or removed while the dialog was hidden.
        self.show_linear_unmixing(dataset_id=self._selected_dataset_id)
        super().showEvent(event)

    def select_dataset(self, dataset_id: Optional[int]) -> None:
        """Called externally (e.g., from the context menu) to pre-select a dataset."""
        self._selected_dataset_id = dataset_id
        self.show_linear_unmixing(dataset_id=dataset_id)

    def _on_datasets_changed(self, *_args) -> None:
        # Try to preserve whatever the user had selected before the change.
        # currentData() returns -1 for the sentinel "no data" item, so we
        # treat anything < 0 as "no selection" and let show_linear_unmixing
        # fall back to index 0.
        current_id = self._ui.cbox_input_ds.currentData()
        preserve_id = current_id if (current_id is not None and int(current_id) >= 0) else None
        self.show_linear_unmixing(dataset_id=preserve_id)

    def get_selected_dataset(self):
        """Return the currently selected dataset, or None if the sentinel is selected."""
        dataset_id = self._ui.cbox_input_ds.currentData()
        if dataset_id is None or int(dataset_id) < 0:
            return None
        return self._app_state.get_dataset(int(dataset_id))

    def _configure_endmember_table(self) -> None:
        table = self._ui.tbl_wdgt_endmembers
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels([self.tr("Spectrum"), self.tr("")])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        header = table.horizontalHeader()
        # Name column takes all available horizontal space; remove button only
        # claims what it needs so it stays compact.
        header.setSectionResizeMode(_ENDMEMBER_NAME_COL, QHeaderView.Stretch)
        header.setSectionResizeMode(_ENDMEMBER_REMOVE_COL, QHeaderView.ResizeToContents)

    def _on_add_collected_spec(self) -> None:
        spec = self._app_state.choose_spectrum_ui()
        if spec is None:
            return
        self._add_endmember_row(spec)

    def _on_import_spec(self) -> None:
        start_dir = self._app_state.get_current_dir() or os.path.expanduser("~")
        filedlg = QFileDialog(
            self,
            self.tr("Import Spectra from Text File"),
            start_dir,
            self.tr("Text files (*.txt);;All Files (*)"),
        )
        filedlg.setFileMode(QFileDialog.ExistingFile)
        filedlg.setAcceptMode(QFileDialog.AcceptOpen)
        # WindowModal blocks interaction with this dialog but not the whole app,
        # which is appropriate for a child file picker.
        filedlg.setWindowModality(Qt.WindowModal)
        if filedlg.exec_() != QDialog.Accepted:
            return
        path = filedlg.selectedFiles()[0]

        # Update the working directory so future file pickers open in the same place.
        self._app_state.update_cwd_from_path(path)
        dlg = ImportSpectraTextDialog(path, parent=self)
        dlg.setWindowModality(Qt.WindowModal)
        if dlg.exec() != QDialog.Accepted:
            return
        specs = dlg.get_spectra()
        if not specs:
            return

        # Wrap the imported spectra in a ListSpectralLibrary so they are
        # registered with the app and appear in the spectrum chooser. This also
        # triggers set_id() on the library, which assigns compound tuple IDs
        # (library_id, index) to each spectrum — see _add_endmember_row.
        lib = ListSpectralLibrary(specs, path=path)
        self._app_state.add_spectral_library(lib)

        for spec in specs:
            self._add_endmember_row(spec)

    def _add_endmember_row(self, spec: Spectrum) -> None:
        # Spectra collected in-app already have an integer ID assigned by
        # take_next_id(). Spectra from a spectral library get a compound
        # (library_id, index) tuple ID when the library is registered — see
        # ListSpectralLibrary.set_id(). The only case where get_id() is None
        # is a brand-new Spectrum that hasn't been added to the app yet.
        if spec.get_id() is None:
            spec.set_id(self._app_state.take_next_id())
        # Do NOT cast to int here, library spectra have tuple IDs, and int()
        # on a tuple raises TypeError. Keep the raw value; tuples are hashable
        # and compare correctly with ==, so everything downstream still works.
        spec_id = spec.get_id()

        # Deduplicate: if this spectrum is already in the table, silently ignore.
        if self._find_endmember_row(spec_id) is not None:
            return

        table = self._ui.tbl_wdgt_endmembers
        row = table.rowCount()
        table.insertRow(row)

        name_item = QTableWidgetItem(spec.get_name() or self.tr("<unnamed>"))
        # Store the ID in UserRole so _find_endmember_row and _remove_endmember
        # can look up the row by identity rather than by display name (which
        # could collide if two spectra share a name).
        name_item.setData(Qt.UserRole, spec_id)
        table.setItem(row, _ENDMEMBER_NAME_COL, name_item)

        remove_button = build_trash_button(
            self,
            # Default argument binding (sid=spec_id) captures the current value
            # of spec_id rather than a closure over the loop variable, which
            # would give every button the same ID by the time it's clicked.
            lambda checked=False, sid=spec_id: self._remove_endmember(sid),
            tooltip=self.tr("Remove endmember"),
            fallback_text=self.tr("Remove"),
        )
        table.setCellWidget(row, _ENDMEMBER_REMOVE_COL, remove_button)

    def _find_endmember_row(self, spec_id) -> Optional[int]:
        """Return the row index for the given spectrum ID, or None if not present."""
        table = self._ui.tbl_wdgt_endmembers
        for row in range(table.rowCount()):
            item = table.item(row, _ENDMEMBER_NAME_COL)
            if item is not None and item.data(Qt.UserRole) == spec_id:
                return row
        return None

    def _remove_endmember(self, spec_id) -> None:
        row = self._find_endmember_row(spec_id)
        if row is not None:
            self._ui.tbl_wdgt_endmembers.removeRow(row)

    def _on_collected_spectra_changed(self, state_change, _index: int, spec_id) -> None:
        # We only care about removals here. Additions don't implicitly add a
        # spectrum to the endmember table — the user has to choose it.
        if state_change != StateChange.ITEM_REMOVED:
            return
        if spec_id == -1:
            # "Remove all collected spectra" sentinel (see app_state.remove_all_collected_spectra).
            # Drop every row whose ID is a plain int — those came from the
            # collected-spectra pool. Library spectra have tuple IDs and are
            # untouched by this signal.
            self._remove_rows_matching(lambda sid: not isinstance(sid, tuple))
        else:
            self._remove_endmember(spec_id)

    def _on_spectral_library_removed(self, lib_id: int) -> None:
        # Library spectra have compound IDs of the form (lib_id, index). Drop
        # every row whose tuple's first element matches the removed library.
        self._remove_rows_matching(lambda sid: isinstance(sid, tuple) and len(sid) >= 1 and sid[0] == lib_id)

    def _remove_rows_matching(self, predicate) -> None:
        # Iterate in reverse so row-index shifting from removeRow() doesn't
        # skip rows we still need to visit.
        table = self._ui.tbl_wdgt_endmembers
        for row in range(table.rowCount() - 1, -1, -1):
            item = table.item(row, _ENDMEMBER_NAME_COL)
            if item is not None and predicate(item.data(Qt.UserRole)):
                table.removeRow(row)

    def _collect_endmember_spectra(self) -> List[Spectrum]:
        """Return the ordered list of endmember spectra from the table widget.

        Raises:
            ValueError: If the table is empty.
            KeyError: If a collected spectrum or spectral library has been
                removed from the app state since it was added to the table.
        """
        table = self._ui.tbl_wdgt_endmembers
        if table.rowCount() == 0:
            raise ValueError(self.tr("Add at least one endmember spectrum before running."))

        spectra: List[Spectrum] = []
        for row in range(table.rowCount()):
            item = table.item(row, _ENDMEMBER_NAME_COL)
            if item is None:
                continue
            spec_id = item.data(Qt.UserRole)
            if isinstance(spec_id, tuple):
                # Library spectrum: (library_id, index)
                lib_id, index = spec_id
                spec = self._app_state.get_spectral_library(lib_id).get_spectrum(index)
            else:
                # Collected spectrum: plain integer ID
                spec = self._app_state.get_spectrum(spec_id)
            spectra.append(spec)
        return spectra

    def _perform_linear_unmixing(self) -> None:
        source_dataset = self.get_selected_dataset()
        if source_dataset is None:
            raise ValueError(self.tr("Select an input dataset before running."))

        endmember_spectra = self._collect_endmember_spectra()

        task = LinearUnmixingSemanticTask(
            app_state=self._app_state,
            app_services=self._app_services,
            source_dataset=source_dataset,
            endmember_spectra=endmember_spectra,
        )
        task_plan = self._app_services.task_planner.plan_semantic_task(task)
        self._app_services.task_manager.register_and_submit_task_plan(
            self._app_services.scheduler,
            task_plan,
        )

    def accept(self) -> None:
        try:
            self._perform_linear_unmixing()
        except (ValueError, KeyError) as exc:
            QMessageBox.warning(self, self.tr("Linear Unmixing"), str(exc))
            return
        QMessageBox.information(
            self,
            self.tr("Linear Unmixing"),
            self.tr("Linear unmixing is running in the background."),
        )


# ---------------------------------------------------------------------------
# Backend: unconstrained linear-unmixing task stage and pipeline builder.
#
# Solve y = X @ a per pixel via the normal equations a = (X^T X)^{-1} X^T y,
# where X is the (L, M) endmember matrix and y is the L-band pixel spectrum.
# Output is (H, W, M+1): M abundance bands (in endmember-list order) followed
# by one RMSE band over good bands.
# ---------------------------------------------------------------------------

_ENDMEMBERS_BROADCAST_KEY = "linear_unmix_endmembers"


# Storage convention: the endmember matrix is stored as ``E`` of shape ``(M, L)``
# (one endmember per row). The math in the GH issue uses ``X`` of shape
# ``(L, M)`` (one endmember per column), so ``X = E.T`` and:
#     C       = X^T X         = E @ E.T          (M, M)
#     X^T y   = E @ y                            (M,)
# For a batch ``Y`` stored as ``(N, L)`` (one pixel per row) the abundances are
#     A_storage = Y @ E.T @ C_inv                (N, M)
# RMSE per pixel is computed over good (unaugmented) bands only:
#     y_recon  = a @ E_good                       (length L_good)
#     rmse     = sqrt(mean((y_good - y_recon)^2))


def _read_endmembers_good(
    endmembers_ref: DataRef,
    good_band_runs: List[tuple],
) -> np.ndarray:
    """Read the (M, L) endmember matrix and keep only columns in good_band_runs."""
    client = get_process_storage_client()
    endmembers_raw, _ = client.read_data(endmembers_ref)
    endmembers = np.asarray(np.ma.getdata(endmembers_raw), dtype=np.float64)
    if endmembers.ndim == 3:
        endmembers = np.squeeze(endmembers, axis=2)
    if endmembers.ndim != 2:
        raise ValueError(f"Endmember matrix must be 2D (M, L); got shape {endmembers.shape}")
    return np.concatenate([endmembers[:, start:end] for start, end in good_band_runs], axis=1)


def _combined_good_band_runs(
    dataset_bad_bands: Optional[np.ndarray],
    endmember_bad_bands: Optional[np.ndarray],
    num_bands: int,
) -> List[tuple]:
    """Intersect the dataset's and endmembers' good-band masks.

    A band is good only if it's good in both contexts: a band flagged bad in
    the dataset has garbage pixel values, and a band flagged bad in the
    endmember registry has an untrusted endmember value — including either
    poisons the least-squares fit. Equivalently, OR the bad-band masks.

    Both arrays (when present) are 1D of length ``num_bands`` and use the
    convention ``0 = bad, 1 = good``. ``None`` means "all bands are good".
    """
    if dataset_bad_bands is None and endmember_bad_bands is None:
        return [(0, num_bands)]

    good = np.ones(num_bands, dtype=np.int8)
    if dataset_bad_bands is not None:
        ds = np.asarray(dataset_bad_bands)
        if ds.shape != (num_bands,):
            raise ValueError(f"Dataset bad_bands shape {ds.shape} does not match band count {num_bands}")
        good &= (ds != 0).astype(np.int8)
    if endmember_bad_bands is not None:
        em = np.asarray(endmember_bad_bands)
        if em.shape != (num_bands,):
            raise ValueError(
                f"Endmember bad_bands shape {em.shape} does not match band count "
                f"{num_bands}; endmembers must be resampled to the dataset's wavelength grid"
            )
        good &= (em != 0).astype(np.int8)

    runs = get_good_band_runs(good)
    if len(runs) == 0:
        raise ValueError(
            "LinearUnmixingStage requires at least one good band, but the union "
            "of dataset and endmember bad-band masks flags every band as bad"
        )
    return runs


def _augment_for_sum_to_unity(endmembers_good: np.ndarray, pixels_good: np.ndarray, weight: float) -> tuple:
    """Append the sum-to-unity weight column to E and weight scalar to each pixel.

    With ``X = E.T`` (so X has shape (L_good, M)), the augmented matrix trick
    appends a row of W to X. In storage that's a column of W on E.
    """
    weight_col = np.full((endmembers_good.shape[0], 1), weight, dtype=np.float64)
    endmembers_aug = np.concatenate([endmembers_good, weight_col], axis=1)
    pixels_aug = np.concatenate(
        [pixels_good, np.full((pixels_good.shape[0], 1), weight, dtype=np.float64)],
        axis=1,
    )
    return endmembers_aug, pixels_aug


def _linear_unmix_pre_task(
    input_ref: DataRef,
    full_input_region: DataRegion,
    endmembers_ref: DataRef,
    c_inv_ref: DataRef,
    sum_to_unity: bool,
    sum_to_unity_weight: float,
) -> None:
    """Compute ``C_inv = (X^T X)^{-1}`` once for the whole dataset.

    Stripping bad bands and (optionally) appending the sum-to-unity augmentation
    must happen here, not per-tile, so the C_inv used by every tile is consistent.
    """
    client = get_process_storage_client()
    input_meta = client.get_region_meta(input_ref, full_input_region)
    endmember_meta = client.get_meta(endmembers_ref)

    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("LinearUnmixingStage requires a DatasetRegionRef full_input_region")
    num_bands = full_input_region.b1 - full_input_region.b0
    good_band_runs = _combined_good_band_runs(input_meta.bad_bands, endmember_meta.bad_bands, num_bands)

    endmembers_good = _read_endmembers_good(endmembers_ref, good_band_runs)
    if sum_to_unity:
        weight_col = np.full((endmembers_good.shape[0], 1), sum_to_unity_weight, dtype=np.float64)
        endmembers_aug = np.concatenate([endmembers_good, weight_col], axis=1)
    else:
        endmembers_aug = endmembers_good

    c = endmembers_aug @ endmembers_aug.T  # (M, M)
    c_inv = np.linalg.inv(c)

    client.write_data(c_inv_ref, c_inv.astype(np.float64, copy=False))


def _linear_unmix_tile(
    input_ref: DataRef,
    input_region: DataRegion,
    output_write: "WriteSpec",
    endmembers_ref: DataRef,
    c_inv_ref: DataRef,
    sum_to_unity: bool,
    sum_to_unity_weight: float,
    num_endmembers: int,
) -> None:
    """Per-tile worker: solve the normal equations for every valid pixel.

    Nodata pixels (any band == nodata, or any NaN if nodata is NaN) are skipped
    and their output bands are set to NaN.
    """
    client = get_process_storage_client()

    data_tile, region_meta = client.read_region(input_ref, input_region, filter_data=False)
    data_arr = np.asarray(np.ma.getdata(data_tile), dtype=np.float64)
    chunk_h, chunk_w, b_total = data_arr.shape

    endmember_meta = client.get_meta(endmembers_ref)
    good_band_runs = _combined_good_band_runs(region_meta.bad_bands, endmember_meta.bad_bands, b_total)
    good_chunks = split_dataset_tile_by_good_band_runs(data_arr, good_band_runs)
    data_good = np.concatenate(good_chunks, axis=2)  # (h, w, L_good)
    b_good = data_good.shape[2]

    endmembers_good = _read_endmembers_good(endmembers_ref, good_band_runs)
    if endmembers_good.shape != (num_endmembers, b_good):
        raise ValueError(
            f"Endmember matrix shape {endmembers_good.shape} does not match "
            f"expected (M={num_endmembers}, L_good={b_good})"
        )

    c_inv_raw, _ = client.read_data(c_inv_ref)
    c_inv = np.asarray(np.ma.getdata(c_inv_raw), dtype=np.float64)
    if c_inv.ndim == 3:
        c_inv = np.squeeze(c_inv, axis=2)

    flat = data_good.reshape(chunk_h * chunk_w, b_good)

    nodata = region_meta.nodata
    if nodata is None:
        valid_indices = np.arange(chunk_h * chunk_w)
    elif np.isnan(nodata):
        valid_indices = np.where(~np.any(np.isnan(flat), axis=1))[0]
    else:
        valid_indices = np.where(~np.any(flat == nodata, axis=1))[0]

    flat_valid = flat[valid_indices]

    if sum_to_unity:
        endmembers_solve, pixels_solve = _augment_for_sum_to_unity(
            endmembers_good, flat_valid, sum_to_unity_weight
        )
    else:
        endmembers_solve, pixels_solve = endmembers_good, flat_valid

    abundances_valid = pixels_solve @ endmembers_solve.T @ c_inv  # (N_valid, M)

    # RMSE is reported over the physical (unaugmented) bands so it reflects the
    # actual spectral fit, not the sum-to-unity penalty.
    reconstructed = abundances_valid @ endmembers_good  # (N_valid, L_good)
    residuals = flat_valid - reconstructed
    rmse_valid = np.sqrt(np.mean(residuals**2, axis=1))

    output_flat = np.full((chunk_h * chunk_w, num_endmembers + 1), fill_value=np.nan, dtype=np.float32)
    output_flat[valid_indices, :num_endmembers] = abundances_valid.astype(np.float32, copy=False)
    output_flat[valid_indices, num_endmembers] = rmse_valid.astype(np.float32, copy=False)

    client.write_spec(output_write, output_flat.reshape(chunk_h, chunk_w, num_endmembers + 1))


def _linear_unmix_post_task(
    input_ref: DataRef,
    full_input_region: DataRegion,
    output_write: "WriteSpec",
) -> None:
    """Propagate spatial metadata from the input dataset to the abundance output.

    Wavelengths and bad_bands are intentionally dropped: abundance bands are not
    wavelength-indexed, and bad-band flags don't apply to abundance/RMSE values.
    """
    if not isinstance(full_input_region, DatasetRegionRef):
        raise TypeError("LinearUnmixingStage metadata write requires DatasetRegionRef full_input_region")
    client = get_process_storage_client()
    input_region_meta = client.get_region_meta(input_ref, full_input_region)
    output_meta = client.get_meta(output_write.ref)
    nodata = input_region_meta.nodata if input_region_meta.nodata is not None else np.nan
    out_meta = replace(
        output_meta,
        nodata=nodata,
        bad_bands=None,
        wavelengths=None,
        wavelength_units=None,
        crs_wkt=input_region_meta.crs_wkt,
        geotransform=input_region_meta.geotransform,
    )
    client.write_meta(output_write.ref, out_meta)


@dataclass
class LinearUnmixingStage(MapStage):
    """Per-tile unconstrained linear unmixing via the normal equations.

    Outputs ``(H, W, M+1)`` float32: ``M`` abundance bands (indexed to match the
    endmember-list order passed in by the caller) followed by one RMSE band.
    RMSE is the reconstruction residual computed over good bands only.

    Optional sum-to-unity weighting (augmented-matrix trick): when
    ``_sum_to_unity`` is True, ``pre_task_fn`` appends a row of
    ``_sum_to_unity_weight`` to ``X`` before computing ``C = X^T X``, and
    ``task_fn`` appends the same weight to each pixel ``y``. The implementation
    lives in PR 2 — this stage just carries the configuration.
    """

    _output_ref_name: str = "linear_unmix_abundances"
    _endmembers_ref_name: str = _ENDMEMBERS_BROADCAST_KEY
    _c_inv_ref_name: str = "linear_unmix_c_inv"
    _num_endmembers: int = 0
    _sum_to_unity: bool = False
    _sum_to_unity_weight: float = 1.0

    resource_model: ResourceModel = field(
        default_factory=lambda: ResourceModel(
            fixed_overhead_bytes=0,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=1,
        )
    )
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

    def __post_init__(self) -> None:
        if self._endmembers_ref_name not in self.broadcast_input:
            raise ValueError(f"LinearUnmixingStage requires '{self._endmembers_ref_name}' in broadcast_input")
        if self._num_endmembers <= 0:
            raise ValueError("LinearUnmixingStage requires _num_endmembers > 0")
        self.output_bindings = list(self.output_bindings) + [
            DataBinding(self._output_ref_name),
            # C_inv is allocated here so pre_task_fn has somewhere to write it
            # and task_fn can read it back via output_writes[c_inv_name].ref.
            # Treated as an internal intermediate; consumers should ignore it.
            DataBinding(self._c_inv_ref_name),
        ]

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        if not isinstance(input_region, DatasetRegionRef):
            raise TypeError("LinearUnmixingStage expects a DatasetRegionRef input region")
        return DatasetRegionRef(
            y0=input_region.y0,
            y1=input_region.y1,
            x0=input_region.x0,
            x1=input_region.x1,
            b0=0,
            b1=self._num_endmembers + 1,
        )

    def generate_allocation_requests(
        self,
        *,
        input_meta: BasePlanMeta,
        chosen_scheme: Optional[ChunkingScheme],
    ) -> List[AllocationRequest]:
        _ = chosen_scheme
        assert isinstance(input_meta, DatasetPlanMeta), "LinearUnmixingStage requires DatasetPlanMeta"
        n_out_bands = self._num_endmembers + 1
        out_size = input_meta.height * input_meta.width * n_out_bands * np.dtype(np.float32).itemsize
        c_inv_size = self._num_endmembers * self._num_endmembers * np.dtype(np.float64).itemsize
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="dataset",
                residency="ram_cacheable",
                size_est=out_size,
                shape=(input_meta.height, input_meta.width, n_out_bands),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
            AllocationRequest(
                name=self._c_inv_ref_name,
                kind="array",
                residency="ram_cacheable",
                size_est=c_inv_size,
                shape=(self._num_endmembers, self._num_endmembers),
                dtype=np.dtype(np.float64),
                delete_policy=self.get_output_delete_policy(self._c_inv_ref_name),
            ),
        ]

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        endmembers_ref = broadcast_inputs[self._endmembers_ref_name]
        c_inv_ref = output_writes[self._c_inv_ref_name].ref
        return partial(
            _linear_unmix_pre_task,
            input_ref,
            full_input_region,
            endmembers_ref,
            c_inv_ref,
            self._sum_to_unity,
            self._sum_to_unity_weight,
        )

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        output_write = output_writes[self._output_ref_name]
        endmembers_ref = broadcast_inputs[self._endmembers_ref_name]
        c_inv_ref = output_writes[self._c_inv_ref_name].ref
        return partial(
            _linear_unmix_tile,
            input_ref,
            input_region,
            output_write,
            endmembers_ref,
            c_inv_ref,
            self._sum_to_unity,
            self._sum_to_unity_weight,
            self._num_endmembers,
        )

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        _ = broadcast_inputs
        output_write = output_writes[self._output_ref_name]
        return partial(_linear_unmix_post_task, input_ref, full_input_region, output_write)


def get_linear_unmixing_pipeline(
    dataset_ref: DataRef,
    endmembers_ref: DataRef,
    num_endmembers: int,
    output_ref_name: str = "linear_unmix_abundances",
    *,
    sum_to_unity: bool = False,
    sum_to_unity_weight: float = 1.0,
) -> AlgorithmPipeline:
    """Build the linear-unmixing pipeline (single stage).

    Parameters
    ----------
    dataset_ref
        Source hyperspectral cube, shape ``(H, W, L)``.
    endmembers_ref
        Pre-allocated ref holding the endmember matrix of shape
        ``(M, L)`` (float32), on the same wavelength grid as the dataset.
        Ensuring grid alignment is the caller's responsibility.
    num_endmembers
        ``M``. Must equal the first dimension of ``endmembers_ref``.
    output_ref_name
        Name of the abundance output binding. Output cube is
        ``(H, W, M+1)`` float32: ``M`` abundance bands followed by RMSE.
    sum_to_unity
        If True, softly enforce ``sum(abundances) == 1`` via the
        augmented-matrix trick (PR 2).
    sum_to_unity_weight
        Augmented-row weight ``W``. Larger ``W`` produces a stronger penalty
        for abundances that don't sum to one.
    """
    storage_client = get_process_storage_client()
    data_meta = storage_client.get_meta(dataset_ref)
    dataset_plan_meta = DatasetPlanMeta(shape=data_meta.shape, dtype=np.dtype(data_meta.elem_type))

    stage = LinearUnmixingStage(
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        _output_ref_name=output_ref_name,
        _num_endmembers=num_endmembers,
        _sum_to_unity=sum_to_unity,
        _sum_to_unity_weight=sum_to_unity_weight,
        broadcast_input={_ENDMEMBERS_BROADCAST_KEY: endmembers_ref},
    )
    return AlgorithmPipeline([stage])


# ---------------------------------------------------------------------------
# Semantic task
# ---------------------------------------------------------------------------


class LinearUnmixingSemanticTask(QObject, SemanticTask):
    """Semantic task that runs the full linear unmixing pipeline.

    Registers the source dataset, stacks the endmember spectra into a
    ``(M, L)`` float32 array, allocates a storage ref for the endmember
    matrix, and builds the single-stage ``LinearUnmixingStage`` pipeline.

    On completion, the ``(H, W, M+1)`` output cube is read back and emitted
    via ``result_ready``, which is connected to ``_load_result_into_wiser``.
    That slot creates a new dataset whose bands are named after the endmember
    spectra (in order) with the final band named "RMSE", then adds it to the
    WISER application state.

    Args:
        app_state: The running :class:`~wiser.gui.app_state.ApplicationState`.
        app_services: Application services used for dataset registration and
            task allocation.
        source_dataset: The hyperspectral image cube to unmix.
        endmember_spectra: Ordered list of endmember spectra.  Each must span
            the full band count of ``source_dataset`` and be on the same
            wavelength grid (bad-band removal is handled internally by the
            stage).
        output_ref_name: Storage allocation name for the abundance output cube.
            Defaults to ``"linear_unmix_abundances"``.
        name: Optional prefix prepended to the output dataset name.  When
            non-empty, the dataset is named ``"<name>: <source>, <M> endmembers"``;
            when empty, the prefix defaults to ``"Linear Unmix"``.  Must be
            supplied as a keyword argument.

    Signals:
        result_ready: Emitted in ``completion_callback`` with the
            ``(H, W, M+1)`` float32 abundance cube.  Connected to
            ``_load_result_into_wiser``.
    """

    result_ready = Signal(object)

    def __init__(
        self,
        app_state: ApplicationState,
        app_services: AppServices,
        source_dataset: RasterDataSet,
        endmember_spectra: List[Spectrum],
        output_ref_name: str = "linear_unmix_abundances",
        *,
        name: str = "",
    ) -> None:
        QObject.__init__(self)

        if not endmember_spectra:
            raise ValueError("LinearUnmixingSemanticTask requires at least one endmember spectrum.")

        source_ref = app_services.storage_service.register_external(
            ExternalRasterHandle(dataset_obj=source_dataset)
        )

        # Build the endmember names list and the (M, L) float32 matrix.
        # Spectra are assumed to already be on the dataset's wavelength grid
        # (e.g. picked via SpectrumAtPoint or pre-aligned by the caller).
        endmember_names: List[str] = []
        spectrum_arrays: List[np.ndarray] = []
        for i, spec in enumerate(endmember_spectra):
            spec_name = spec.get_name()
            endmember_names.append(spec_name if spec_name else f"Endmember {i + 1}")
            spectrum_arrays.append(np.asarray(spec.get_spectrum(), dtype=np.float32))

        endmember_matrix = np.stack(spectrum_arrays, axis=0)  # (M, L)
        num_endmembers = endmember_matrix.shape[0]

        process_client = get_process_storage_client()
        endmembers_ref = app_services.storage_service.allocate_data(
            AllocationRequest(
                name="lu_endmembers",
                kind="array",
                residency="ram_cacheable",
                size_est=int(endmember_matrix.size * endmember_matrix.dtype.itemsize),
                shape=endmember_matrix.shape,
                dtype=endmember_matrix.dtype,
            )
        )
        process_client.write_data(endmembers_ref, endmember_matrix)

        # Write spectrum metadata to the endmembers ref so the stage's
        # _combined_good_band_runs can AND the endmember bad-band mask with the
        # dataset's bad-band mask.
        #
        # ASSUMPTION: all endmember spectra share the same wavelength grid and
        # bad-band mask.  We read both from the first spectrum only.
        first_spec = endmember_spectra[0]

        raw_bad_bands = first_spec.get_bad_bands()
        em_bad_bands: Optional[np.ndarray] = (
            np.asarray(raw_bad_bands, dtype=np.int8) if raw_bad_bands is not None else None
        )

        em_wavelengths: Optional[np.ndarray] = None
        em_wavelength_units = None
        if first_spec.has_wavelengths():
            wl_list = first_spec.get_wavelengths()
            em_wavelength_units = first_spec.get_wavelength_units()
            em_wavelengths = np.array([w.to(em_wavelength_units).value for w in wl_list], dtype=np.float64)

        em_meta: DataMeta = process_client.get_meta(endmembers_ref)
        process_client.write_meta(
            endmembers_ref,
            replace(
                em_meta,
                bad_bands=em_bad_bands,
                wavelengths=em_wavelengths,
                wavelength_units=em_wavelength_units,
            ),
        )

        pipeline = get_linear_unmixing_pipeline(
            dataset_ref=source_ref,
            endmembers_ref=endmembers_ref,
            num_endmembers=num_endmembers,
            output_ref_name=output_ref_name,
        )

        source_name = source_dataset.get_name() or "Dataset"
        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=source_ref,
            algorithm_pipeline=pipeline,
            task_title="Linear Unmixing",
            task_variables={
                "Source": source_name,
                "Endmembers": num_endmembers,
            },
        )
        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._output_ref_name = output_ref_name
        self._endmember_names: List[str] = endmember_names
        self._prefix_name: str = name
        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        """Read the full abundance cube and emit it for loading on the main thread.

        Runs in the worker process after all tiles finish.  Emits the full
        ``(H, W, M+1)`` float32 cube via ``result_ready`` so the Qt main
        thread can create a dataset from it.

        Args:
            bindings: Mapping of allocation names to resolved
                :class:`~wiser.utils.primitives.DataRef` objects produced by
                the task pipeline.

        Raises:
            KeyError: If the expected output binding is not present.
        """
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing linear unmixing output binding: '{self._output_ref_name}'")

        storage_client = get_process_storage_client()
        meta = storage_client.get_meta(output_ref)
        height, width, n_bands = meta.shape
        region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=n_bands)
        output_data, _ = storage_client.read_region(output_ref, region, filter_data=False)
        self.result_ready.emit(np.asarray(output_data, dtype=np.float32))

    @Slot(object)
    def _load_result_into_wiser(self, abundance_data: object) -> None:
        """Create a WISER dataset from the abundance cube and add it to the app.

        Runs on the Qt main thread (via the ``result_ready`` signal).

        The output dataset has ``M + 1`` bands: one abundance band per
        endmember (in the same order as ``endmember_spectra``) followed by
        one RMSE band.  Band descriptions are set to the endmember names /
        "RMSE".  Spatial metadata (CRS, geotransform) is copied from the
        source dataset.  The nodata value is NaN so that pixels that were
        skipped during unmixing (input nodata pixels) round-trip correctly.

        Args:
            abundance_data: The ``(H, W, M+1)`` float32 array emitted by
                :meth:`completion_callback`.
        """
        abundance_array = np.asarray(abundance_data, dtype=np.float32)  # (H, W, M+1)
        # dataset_from_numpy_array expects (bands, height, width)
        abundance_by_band = abundance_array.transpose(2, 0, 1)  # (M+1, H, W)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        result_dataset = loader.dataset_from_numpy_array(abundance_by_band, cache)

        source_name = self._source_dataset.get_name() or "Dataset"
        num_em = len(self._endmember_names)
        em_str = f"{num_em} endmember{'s' if num_em != 1 else ''}"
        prefix = self._prefix_name if self._prefix_name else "Linear Unmix"
        full_name = f"{prefix}: {source_name}, {em_str}"

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        result_dataset.set_name(self._app_state.unique_dataset_name(full_name))
        result_dataset.set_description(f"Linear unmixing of '{source_name}' with {em_str} ({timestamp})")
        result_dataset.set_data_ignore_value(float("nan"))
        result_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())

        # Name each band: M abundance bands (one per endmember) + RMSE.
        band_descriptions = list(self._endmember_names) + ["RMSE"]
        result_dataset.set_band_descriptions(band_descriptions)

        self._app_state.add_dataset(result_dataset, view_dataset=False)
