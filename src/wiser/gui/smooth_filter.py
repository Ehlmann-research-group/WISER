from enum import Enum, auto
from typing import Dict, Optional, TYPE_CHECKING, Union

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot

from wiser.utils.primitives import DataMeta, DataRef, DatasetRegionRef, PriorityClass
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
        filter_kwargs["sigma"] = sigma
        if radius is not None:
            filter_kwargs["radius"] = radius
        # truncate is injected automatically by the stage's _normalize step (default 4.0).

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
