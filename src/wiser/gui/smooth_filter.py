from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox

from wiser.gui.app_services import AppServices
from wiser.gui.generated.smoothing_filter_dialog_ui import Ui_SmoothingFilterDatasetDialog
from wiser.utils.primitives import (
    DataMeta,
    DataRef,
    DatasetRegionRef,
    ExternalRasterHandle,
    PriorityClass,
)
from wiser.utils.task_stage_utils import (
    NDIMAGE_SMOOTHING_FILTER_REGISTRY,
    SmoothingFilterSpatial,
    SmoothingFilterSpectral,
)
from wiser.utils.task_system import AlgorithmPipeline, DatasetPlanMeta, SemanticTask
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState
    from wiser.raster.dataset import RasterDataSet


class SmoothingDomain(Enum):
    SPATIAL = auto()
    SPECTRAL = auto()


class SmoothingFilterKind(Enum):
    MEAN = "uniform_filter"
    MEDIAN = "median_filter"
    GAUSSIAN = "gaussian_filter"


# Human-readable labels for task titles and task variable display.
_DOMAIN_LABEL: Dict[SmoothingDomain, str] = {
    SmoothingDomain.SPATIAL: "Spatial",
    SmoothingDomain.SPECTRAL: "Spectral",
}

_KIND_LABEL: Dict[SmoothingFilterKind, str] = {
    SmoothingFilterKind.MEAN: "Mean",
    SmoothingFilterKind.MEDIAN: "Median",
    SmoothingFilterKind.GAUSSIAN: "Gaussian",
}


def _build_smoothing_filter_pipeline(
    *,
    input_ref: DataRef,
    domain: SmoothingDomain,
    filter_kind: SmoothingFilterKind,
    mode: str,
    cval: float,
    size: Optional[Union[int, tuple]],
    sigma: Optional[Union[float, tuple]],
    radius: Optional[Union[int, tuple]],
    truncate: Optional[float],
    output_ref_name: str,
) -> AlgorithmPipeline:
    """
    Construct the kwargs dict for the chosen scipy.ndimage function, then build
    and return the appropriate single-stage AlgorithmPipeline.
    """
    from wiser.utils.worker_runtime import get_process_storage_client as _gsc

    storage_client = _gsc()
    dataset_meta = storage_client.get_meta(input_ref)
    if len(dataset_meta.shape) != 3:
        raise ValueError(f"Expected input dataset shape [y][x][b], got {dataset_meta.shape}")

    input_plan_meta = DatasetPlanMeta(
        shape=dataset_meta.shape,
        dtype=np.dtype(dataset_meta.elem_type),
    )

    filter_kwargs: Dict = {"mode": mode, "cval": cval}

    if filter_kind in (SmoothingFilterKind.MEAN, SmoothingFilterKind.MEDIAN):
        if size is None:
            raise ValueError(f"{_KIND_LABEL[filter_kind]} filter requires 'size'.")
        filter_kwargs["size"] = size
    elif filter_kind is SmoothingFilterKind.GAUSSIAN:
        if sigma is None:
            raise ValueError("Gaussian filter requires 'sigma'.")
        if radius is not None and truncate is not None:
            raise ValueError("Specify either 'radius' or 'truncate' for Gaussian, not both.")
        filter_kwargs["sigma"] = sigma
        if radius is not None:
            filter_kwargs["radius"] = radius
        if truncate is not None:
            filter_kwargs["truncate"] = truncate
        # If neither is provided, the stage's _normalize step injects the default truncate (4.0).

    registry_key = filter_kind.value

    if domain is SmoothingDomain.SPATIAL:
        stage = SmoothingFilterSpatial(
            default_executor="process",
            input_plan_meta=input_plan_meta,
            _filter_registry_key=registry_key,
            _filter_kwargs=filter_kwargs,
            _output_ref_name=output_ref_name,
        )
    else:
        stage = SmoothingFilterSpectral(
            default_executor="process",
            input_plan_meta=input_plan_meta,
            _filter_registry_key=registry_key,
            _filter_kwargs=filter_kwargs,
            _output_ref_name=output_ref_name,
        )

    return AlgorithmPipeline([stage])


class SmoothingFilterSemanticTask(QObject, SemanticTask):
    result_ready = Signal(object, object)

    def __init__(
        self,
        app_state: "ApplicationState",
        source_dataset: "RasterDataSet",
        input_ref: DataRef,
        domain: SmoothingDomain,
        filter_kind: SmoothingFilterKind,
        mode: str,
        cval: float = 0.0,
        size: Optional[Union[int, tuple]] = None,
        sigma: Optional[Union[float, tuple]] = None,
        radius: Optional[Union[int, tuple]] = None,
        truncate: Optional[float] = None,
        output_ref_name: str = "smoothing_filtered_dataset",
    ):
        QObject.__init__(self)

        pipeline = _build_smoothing_filter_pipeline(
            input_ref=input_ref,
            domain=domain,
            filter_kind=filter_kind,
            mode=mode,
            cval=cval,
            size=size,
            sigma=sigma,
            radius=radius,
            truncate=truncate,
            output_ref_name=output_ref_name,
        )

        domain_label = _DOMAIN_LABEL[domain]
        kind_label = _KIND_LABEL[filter_kind]
        task_title = f"{domain_label} {kind_label} Smoothing Filter"

        # Build task_variables from only the parameters that were actually provided,
        # so the activity monitor shows relevant information rather than a wall of Nones.
        task_variables: Dict = {"Dataset": source_dataset.get_name()}
        task_variables["Domain"] = domain_label
        task_variables["Mode"] = mode
        task_variables["Cval"] = cval
        if size is not None:
            task_variables["Size"] = size
        if sigma is not None:
            task_variables["Sigma"] = sigma
        if radius is not None:
            task_variables["Radius"] = radius
        if truncate is not None:
            task_variables["Truncate"] = truncate

        SemanticTask.__init__(
            self,
            priority_class=PriorityClass.BACKGROUND,
            input_ref=input_ref,
            algorithm_pipeline=pipeline,
            task_title=task_title,
            task_variables=task_variables,
        )

        self.id = app_state.take_next_id()
        self._app_state = app_state
        self._source_dataset = source_dataset
        self._output_ref_name = output_ref_name
        self._domain = domain
        self._filter_kind = filter_kind

        self.result_ready.connect(self._load_result_into_wiser)

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        output_ref = bindings.get(self._output_ref_name)
        if output_ref is None:
            raise KeyError(f"Missing smoothing filter output binding: {self._output_ref_name!r}")

        storage_client = get_process_storage_client()
        output_meta = storage_client.get_meta(output_ref)
        height, width, bands = output_meta.shape
        output_region = DatasetRegionRef(y0=0, y1=height, x0=0, x1=width, b0=0, b1=bands)
        output_data, _ = storage_client.read_region(output_ref, output_region, filter_data=False)
        self.result_ready.emit(np.asarray(output_data), output_meta)

    @Slot(object, object)
    def _load_result_into_wiser(self, output_data: object, output_meta: object) -> None:
        output_array = np.asarray(output_data)
        # Storage layer uses [y][x][b]; the dataset loader expects [b][y][x].
        output_array_by_band = output_array.transpose(2, 0, 1)

        loader = self._app_state.get_loader()
        cache = self._app_state.get_cache()
        result_dataset = loader.dataset_from_numpy_array(output_array_by_band, cache)

        source_name = self._source_dataset.get_name() or "Dataset"
        domain_label = _DOMAIN_LABEL[self._domain]
        kind_label = _KIND_LABEL[self._filter_kind]
        result_dataset.set_name(
            self._app_state.unique_dataset_name(f"{domain_label} {kind_label} on {source_name}")
        )
        result_dataset.set_description(self._source_dataset.get_description())

        default_display_bands = self._source_dataset.default_display_bands()
        if default_display_bands is not None:
            result_dataset.set_default_display_bands(default_display_bands)

        if self._source_dataset.get_spatial_metadata().get_spatial_ref():
            result_dataset.copy_spatial_metadata(self._source_dataset.get_spatial_metadata())
        if self._source_dataset.has_wavelengths():
            result_dataset.copy_spectral_metadata(self._source_dataset.get_spectral_metadata())

        if isinstance(output_meta, DataMeta):
            result_dataset.set_data_ignore_value(output_meta.nodata)
            if output_meta.bad_bands is not None:
                result_dataset.set_bad_bands(np.asarray(output_meta.bad_bands).astype(int).tolist())
        else:
            # Fallback: carry metadata from source if the output meta is not a DataMeta.
            result_dataset.set_data_ignore_value(self._source_dataset.get_data_ignore_value())
            if self._source_dataset.get_bad_bands() is not None:
                result_dataset.set_bad_bands(self._source_dataset.get_bad_bands())

        self._app_state.add_dataset(result_dataset, view_dataset=False)


# Combo-box options. The user-data carries the canonical value (enum or scipy string).
_AXIS_OPTIONS: Tuple[Tuple[str, SmoothingDomain], ...] = (
    ("Spectral", SmoothingDomain.SPECTRAL),
    ("Spatial", SmoothingDomain.SPATIAL),
)

# scipy.ndimage filter "mode" parameter values.
_MODE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("reflect", "reflect"),
    ("constant", "constant"),
    ("nearest", "nearest"),
    ("mirror", "mirror"),
    ("wrap", "wrap"),
)

_RAD_TRUNC_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Radius", "radius"),
    ("Truncate", "truncate"),
)


class SmoothingFilterDialog(QDialog):
    """
    Dialog for configuring and submitting a SmoothingFilterSemanticTask.

    The kind of filter (mean / median / gaussian) is fixed at construction time. Per-UI parameters
    (axis, mode, cval, size, sigma, radius/truncate) are read directly from the widgets via getters
    rather than mirrored into instance attributes. The only internal state we keep is one snapshot
    per axis (spatial vs. spectral), so a user can switch axes without losing what they typed in
    the other axis's parameter rows.
    """

    def __init__(
        self,
        app_state: "ApplicationState",
        app_services: AppServices,
        filter_kind: SmoothingFilterKind,
        target_dataset_id: Optional[int] = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._app_state = app_state
        self._app_services = app_services
        self._filter_kind = filter_kind
        self._target_dataset_id = target_dataset_id

        self._ui = Ui_SmoothingFilterDatasetDialog()
        self._ui.setupUi(self)

        # The .ui file leaves the truncate ledit with Qt Designer's default name `lineEdit_2`;
        # alias it locally so the rest of this class can read like the rest of the widget names.
        self._ledit_trunc = self._ui.ledit_trunc

        # Per-axis snapshot of the parameter ledits + rad/trunc cbox. We only ever store the most
        # recent state for each axis (at most one dict per axis). State is captured by calling the
        # ledit getters at the moment the user switches axes.
        self._saved_state_by_axis: Dict[SmoothingDomain, Dict[str, Any]] = {}

        self._set_window_title()
        self._setup_validators()
        self._populate_combo_boxes()
        self._set_initial_field_values()
        self._wire_signals()
        self._apply_visibility()

    # region Setup helpers ---------------------------------------------------------------

    def _set_window_title(self) -> None:
        kind_label = _KIND_LABEL[self._filter_kind]
        self.setWindowTitle(f"{kind_label} Smoothing Filter")

    def _setup_validators(self) -> None:
        # Sizes and radii are non-negative integers; sigma, truncate, cval are floats. cval may be
        # any sign; sigma/truncate must be > 0 but the validator only constrains the format -
        # semantic checks happen in `accept()`.
        int_validator = QIntValidator(0, 1_000_000, self)
        pos_float_validator = QDoubleValidator(0.0, 1.0e12, 12, self)
        pos_float_validator.setNotation(QDoubleValidator.StandardNotation)
        signed_float_validator = QDoubleValidator(-1.0e12, 1.0e12, 12, self)
        signed_float_validator.setNotation(QDoubleValidator.StandardNotation)

        self._ui.ledit_size_x.setValidator(int_validator)
        self._ui.ledit_size_y.setValidator(int_validator)
        self._ui.ledit_rad_x.setValidator(int_validator)
        self._ui.ledit_rad_y.setValidator(int_validator)

        self._ui.ledit_sigma.setValidator(pos_float_validator)
        self._ui.ledit_sigma_y.setValidator(pos_float_validator)
        self._ledit_trunc.setValidator(pos_float_validator)

        self._ui.ledit_cval.setValidator(signed_float_validator)

    def _populate_combo_boxes(self) -> None:
        # Block signals so the initial population doesn't trigger axis-change snapshots etc.
        for cbox in (
            self._ui.cbox_dataset,
            self._ui.cbox_axis,
            self._ui.cbox_mode,
            self._ui.cbox_rad_trunc,
        ):
            cbox.blockSignals(True)
            cbox.clear()

        datasets = self._app_state.get_datasets()
        for dataset in datasets:
            ds_id = dataset.get_id()
            if ds_id is None:
                ds_id = self._app_state.take_next_id()
                dataset.set_id(ds_id)
            self._ui.cbox_dataset.addItem(dataset.get_name() or "<unnamed>", ds_id)
        self._ui.cbox_dataset.addItem(self.tr("(no data)"), -1)

        if datasets:
            initial_index = 0
            if self._target_dataset_id is not None:
                found = self._ui.cbox_dataset.findData(self._target_dataset_id)
                if found >= 0:
                    initial_index = found
            self._ui.cbox_dataset.setCurrentIndex(initial_index)
        else:
            # Only "(no data)" exists.
            self._ui.cbox_dataset.setCurrentIndex(0)

        for label, value in _AXIS_OPTIONS:
            self._ui.cbox_axis.addItem(label, value)
        self._ui.cbox_axis.setCurrentIndex(self._ui.cbox_axis.findData(SmoothingDomain.SPECTRAL))

        for label, value in _MODE_OPTIONS:
            self._ui.cbox_mode.addItem(label, value)
        self._ui.cbox_mode.setCurrentIndex(self._ui.cbox_mode.findData("reflect"))

        for label, value in _RAD_TRUNC_OPTIONS:
            self._ui.cbox_rad_trunc.addItem(label, value)
        self._ui.cbox_rad_trunc.setCurrentIndex(self._ui.cbox_rad_trunc.findData("radius"))

        for cbox in (
            self._ui.cbox_dataset,
            self._ui.cbox_axis,
            self._ui.cbox_mode,
            self._ui.cbox_rad_trunc,
        ):
            cbox.blockSignals(False)

        # Disable Ok if no real datasets are loaded.
        ok_button = self._ui.buttonBox.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setEnabled(bool(datasets))

    def _set_initial_field_values(self) -> None:
        self._ui.ledit_cval.setText("0")
        self._ui.ledit_size_x.clear()
        self._ui.ledit_size_y.clear()
        self._ui.ledit_sigma.clear()
        self._ui.ledit_sigma_y.clear()
        self._ui.ledit_rad_x.clear()
        self._ui.ledit_rad_y.clear()
        self._ledit_trunc.clear()

    def _wire_signals(self) -> None:
        self._ui.cbox_axis.currentIndexChanged.connect(self._on_axis_changed)
        self._ui.cbox_rad_trunc.currentIndexChanged.connect(lambda _i: self._apply_visibility())

    # endregion

    # region Visibility ------------------------------------------------------------------

    def _apply_visibility(self) -> None:
        axis = self.get_axis()
        is_spatial = axis is SmoothingDomain.SPATIAL
        is_uniform = self._filter_kind in (SmoothingFilterKind.MEAN, SmoothingFilterKind.MEDIAN)
        is_gaussian = self._filter_kind is SmoothingFilterKind.GAUSSIAN

        # Size rows: only mean/median use "size".
        self._ui.lbl_size_x.setVisible(is_uniform)
        self._ui.ledit_size_x.setVisible(is_uniform)
        self._ui.lbl_size_y.setVisible(is_uniform and is_spatial)
        self._ui.ledit_size_y.setVisible(is_uniform and is_spatial)
        if is_uniform:
            self._ui.lbl_size_x.setText("Size X (width, cols)" if is_spatial else "Size")

        # Sigma rows: only gaussian uses "sigma".
        self._ui.lbl_sigma_x.setVisible(is_gaussian)
        self._ui.ledit_sigma.setVisible(is_gaussian)
        self._ui.lbl_sigma_y.setVisible(is_gaussian and is_spatial)
        self._ui.ledit_sigma_y.setVisible(is_gaussian and is_spatial)
        if is_gaussian:
            self._ui.lbl_sigma_x.setText("Sigma X" if is_spatial else "Sigma")

        # Radius / truncate row: only gaussian; X/Y labels only meaningful in spatial+radius mode.
        self._ui.cbox_rad_trunc.setVisible(is_gaussian)
        rt_kind = self.get_rad_trunc_kind() if is_gaussian else None
        show_rad = is_gaussian and rt_kind == "radius"
        show_trunc = is_gaussian and rt_kind == "truncate"

        self._ui.ledit_rad_x.setVisible(show_rad)
        self._ui.ledit_rad_y.setVisible(show_rad and is_spatial)
        self._ui.lbl_rad_x.setVisible(show_rad and is_spatial)
        self._ui.lbl_rad_y.setVisible(show_rad and is_spatial)
        self._ledit_trunc.setVisible(show_trunc)

    # endregion

    # region Axis-switch state -----------------------------------------------------------

    def _on_axis_changed(self, _index: int) -> None:
        new_axis = self.get_axis()
        # There are exactly two axes, so the "old" axis is the other one.
        old_axis = (
            SmoothingDomain.SPATIAL if new_axis is SmoothingDomain.SPECTRAL else SmoothingDomain.SPECTRAL
        )

        # Snapshot the current ledit/cbox state under the OLD axis by calling the getters now,
        # before we mutate the widgets to reflect the NEW axis.
        self._saved_state_by_axis[old_axis] = self._snapshot_param_state()

        if new_axis in self._saved_state_by_axis:
            self._restore_param_state(self._saved_state_by_axis[new_axis])
        else:
            # No prior state for this axis - reset the per-axis param ledits to empty defaults.
            self._set_initial_field_values()

        self._apply_visibility()

    def _snapshot_param_state(self) -> Dict[str, Any]:
        return {
            "size_x": self._ui.ledit_size_x.text(),
            "size_y": self._ui.ledit_size_y.text(),
            "sigma": self._ui.ledit_sigma.text(),
            "sigma_y": self._ui.ledit_sigma_y.text(),
            "rad_x": self._ui.ledit_rad_x.text(),
            "rad_y": self._ui.ledit_rad_y.text(),
            "trunc": self._ledit_trunc.text(),
            "rad_trunc_kind": self.get_rad_trunc_kind(),
        }

    def _restore_param_state(self, state: Dict[str, Any]) -> None:
        self._ui.ledit_size_x.setText(state.get("size_x", ""))
        self._ui.ledit_size_y.setText(state.get("size_y", ""))
        self._ui.ledit_sigma.setText(state.get("sigma", ""))
        self._ui.ledit_sigma_y.setText(state.get("sigma_y", ""))
        self._ui.ledit_rad_x.setText(state.get("rad_x", ""))
        self._ui.ledit_rad_y.setText(state.get("rad_y", ""))
        self._ledit_trunc.setText(state.get("trunc", ""))

        rt_kind = state.get("rad_trunc_kind", "radius")
        rt_index = self._ui.cbox_rad_trunc.findData(rt_kind)
        if rt_index >= 0:
            self._ui.cbox_rad_trunc.blockSignals(True)
            self._ui.cbox_rad_trunc.setCurrentIndex(rt_index)
            self._ui.cbox_rad_trunc.blockSignals(False)

    # endregion

    # region Getters (UI is the source of truth) -----------------------------------------

    def get_filter_kind(self) -> SmoothingFilterKind:
        return self._filter_kind

    def get_input_dataset(self) -> Optional["RasterDataSet"]:
        ds_id = self._ui.cbox_dataset.currentData()
        if ds_id is None or int(ds_id) < 0:
            return None
        return self._app_state.get_dataset(int(ds_id))

    def get_axis(self) -> SmoothingDomain:
        data = self._ui.cbox_axis.currentData()
        if isinstance(data, SmoothingDomain):
            return data
        return SmoothingDomain.SPECTRAL

    def get_mode(self) -> str:
        data = self._ui.cbox_mode.currentData()
        return str(data) if data is not None else "reflect"

    def get_cval(self) -> float:
        text = self._ui.ledit_cval.text().strip()
        if not text:
            return 0.0
        return float(text)

    def get_rad_trunc_kind(self) -> str:
        data = self._ui.cbox_rad_trunc.currentData()
        return str(data) if data is not None else "radius"

    def get_size(self) -> Optional[Union[int, Tuple[int, int]]]:
        if self._filter_kind not in (SmoothingFilterKind.MEAN, SmoothingFilterKind.MEDIAN):
            return None
        if self.get_axis() is SmoothingDomain.SPECTRAL:
            text = self._ui.ledit_size_x.text().strip()
            if not text:
                return None
            return int(text)
        sx = self._ui.ledit_size_x.text().strip()
        sy = self._ui.ledit_size_y.text().strip()
        if not sx or not sy:
            return None
        # Spatial axes are ordered (axis 0 = y, axis 1 = x) per SmoothingFilterSpatial.
        return (int(sy), int(sx))

    def get_sigma(self) -> Optional[Union[float, Tuple[float, float]]]:
        if self._filter_kind is not SmoothingFilterKind.GAUSSIAN:
            return None
        if self.get_axis() is SmoothingDomain.SPECTRAL:
            text = self._ui.ledit_sigma.text().strip()
            if not text:
                return None
            return float(text)
        sx = self._ui.ledit_sigma.text().strip()
        sy = self._ui.ledit_sigma_y.text().strip()
        if not sx or not sy:
            return None
        return (float(sy), float(sx))

    def get_radius(self) -> Optional[Union[int, Tuple[int, int]]]:
        if self._filter_kind is not SmoothingFilterKind.GAUSSIAN:
            return None
        if self.get_rad_trunc_kind() != "radius":
            return None
        if self.get_axis() is SmoothingDomain.SPECTRAL:
            text = self._ui.ledit_rad_x.text().strip()
            if not text:
                return None
            return int(text)
        rx = self._ui.ledit_rad_x.text().strip()
        ry = self._ui.ledit_rad_y.text().strip()
        if not rx or not ry:
            return None
        return (int(ry), int(rx))

    def get_truncate(self) -> Optional[float]:
        if self._filter_kind is not SmoothingFilterKind.GAUSSIAN:
            return None
        if self.get_rad_trunc_kind() != "truncate":
            return None
        text = self._ledit_trunc.text().strip()
        if not text:
            return None
        return float(text)

    # endregion

    # region Accept ----------------------------------------------------------------------

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, self.windowTitle(), message, QMessageBox.Ok)

    def accept(self) -> None:
        try:
            dataset = self.get_input_dataset()
            if dataset is None:
                self._show_error("Please select an input dataset.")
                return

            axis = self.get_axis()
            mode = self.get_mode()
            cval = self.get_cval()
            size = self.get_size()
            sigma = self.get_sigma()
            radius = self.get_radius()
            truncate = self.get_truncate()
        except ValueError as exc:
            self._show_error(f"Invalid numeric input: {exc}")
            return

        if self._filter_kind in (SmoothingFilterKind.MEAN, SmoothingFilterKind.MEDIAN):
            if size is None:
                self._show_error("Please enter a size value for every visible size field.")
                return
        elif self._filter_kind is SmoothingFilterKind.GAUSSIAN:
            if sigma is None:
                self._show_error("Please enter a sigma value for every visible sigma field.")
                return
            rt_kind = self.get_rad_trunc_kind()
            if rt_kind == "radius" and radius is None:
                self._show_error("Please enter a radius value for every visible radius field.")
                return
            if rt_kind == "truncate" and truncate is None:
                self._show_error("Please enter a truncate value.")
                return

        try:
            dataset_ref = self._app_services.storage_service.register_external(
                ExternalRasterHandle(dataset_obj=dataset)
            )
            task = SmoothingFilterSemanticTask(
                app_state=self._app_state,
                source_dataset=dataset,
                input_ref=dataset_ref,
                domain=axis,
                filter_kind=self._filter_kind,
                mode=mode,
                cval=cval,
                size=size,
                sigma=sigma,
                radius=radius,
                truncate=truncate,
            )
            task_plan = self._app_services.task_planner.plan_semantic_task(task)
            self._app_services.task_manager.register_and_submit_task_plan(
                self._app_services.scheduler, task_plan
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        super().accept()

    # endregion
