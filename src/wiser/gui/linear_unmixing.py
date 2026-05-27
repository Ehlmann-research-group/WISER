import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PySide2.QtCore import Qt
from PySide2.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QTableWidgetItem,
)

from wiser.gui.app_services import AppServices
from wiser.gui.app_state import ApplicationState
from wiser.gui.generated.linear_unmixing_dialog_ui import Ui_LinearUnmixingDialog
from wiser.gui.import_spectra_text import ImportSpectraTextDialog
from wiser.gui.util import StateChange
from wiser.gui.utils import build_trash_button
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.spectrum import Spectrum
from wiser.utils.primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    SpatialTileScheme,
)
from wiser.utils.task_system import (
    AlgorithmPipeline,
    BasePlanMeta,
    DatasetPlanMeta,
    MapStage,
    ResourceModel,
    WriteSpec,
)


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


# ---------------------------------------------------------------------------
# Backend: unconstrained linear-unmixing task stage and pipeline builder.
#
# Solve y = X @ a per pixel via the normal equations a = (X^T X)^{-1} X^T y,
# where X is the (L, M) endmember matrix and y is the L-band pixel spectrum.
# Output is (H, W, M+1): M abundance bands (in endmember-list order) followed
# by one RMSE band over good bands.
# ---------------------------------------------------------------------------

_ENDMEMBERS_BROADCAST_KEY = "linear_unmix_endmembers"


def _linear_unmix_pre_task_stub(*_args: Any, **_kwargs: Any) -> None:
    # PR 2 will: read endmembers (M, L), build augmented X if sum_to_unity is
    # set, compute C = X^T X, invert to C_inv, write C_inv to its allocation.
    raise NotImplementedError("LinearUnmixingStage.pre_task_fn is implemented in PR 2")


def _linear_unmix_task_stub(*_args: Any, **_kwargs: Any) -> None:
    # PR 2 will: read input tile, read endmembers + C_inv, compute
    # A_tile = C_inv @ X^T @ Y_tile, compute RMSE over good bands, write
    # (h, w, M+1) to the output region.
    raise NotImplementedError("LinearUnmixingStage.task_fn is implemented in PR 2")


def _linear_unmix_post_task_stub(*_args: Any, **_kwargs: Any) -> None:
    # PR 2 will: propagate spatial metadata (CRS, geotransform, nodata) from
    # the input dataset onto the abundance output ref.
    raise NotImplementedError("LinearUnmixingStage.post_task_fn is implemented in PR 2")


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
        _ = (input_ref, full_input_region, output_writes, broadcast_inputs)
        return partial(_linear_unmix_pre_task_stub)

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        _ = (input_ref, input_region, output_writes, broadcast_inputs)
        return partial(_linear_unmix_task_stub)

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        _ = (input_ref, full_input_region, output_writes, broadcast_inputs)
        return partial(_linear_unmix_post_task_stub)


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
        ``(M, L)`` (float32), already resampled to the dataset's wavelength
        grid. Resampling is the caller's responsibility — typically the
        ``LinearUnmixingSemanticTask`` (added in a later PR) handles it via
        ``scipy.interpolate.interp1d``.
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
    stage = LinearUnmixingStage(
        _output_ref_name=output_ref_name,
        _num_endmembers=num_endmembers,
        _sum_to_unity=sum_to_unity,
        _sum_to_unity_weight=sum_to_unity_weight,
        broadcast_input={_ENDMEMBERS_BROADCAST_KEY: endmembers_ref},
    )
    _ = dataset_ref  # consumed by the planner via SemanticTask.input_ref, not the stage itself
    return AlgorithmPipeline([stage])
