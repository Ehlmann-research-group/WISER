"""
Async-capable spectrum recomputation built on the semantic task system.

Plan (high level):
- Raster-backed spectra (ROI average, spectrum-at-point) use one ``SequentialStage``
  with ``NoChunkingScheme``: a single work unit reconstructs the ``RasterDataSet``
  from the registered dataset ``DataRef`` in the worker process, runs the same
  math as ``ROIAverageSpectrum`` / ``SpectrumAtPoint`` (``calc_roi_spectrum``,
  ``calc_roi_spectrum`` / ``calc_spectrum_at_point_with_area``), and writes a 1D spectrum output ref.
- ``NumPyArraySpectrum`` does not use the scheduler: a small QObject emits
  ``result_ready`` with the existing array (no recomputation).

Downstream (e.g. ``SpectrumDisplayInfo.generate_plot``) can connect to
``result_ready`` and keep the UI thread responsive; wiring is done in the GUI layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from PySide2.QtCore import QObject, Signal, QTimer

from wiser.raster.dataset import RasterDataSet
from wiser.raster.roi import RegionOfInterest
from wiser.raster.spectrum import (
    NumPyArraySpectrum,
    RasterDataSetSpectrum,
    ROIAverageSpectrum,
    Spectrum,
    SpectrumAtPoint,
    SpectrumAverageMode,
    calc_roi_spectrum,
    calc_spectrum_at_point_with_area,
)
from wiser.utils.primitives import (
    AllocationRequest,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetPlanMeta,
    DeletePolicy,
    NoChunkingScheme,
    SpectrumRef,
)
from wiser.utils.task_system import AlgorithmPipeline, ResourceModel, SemanticTask, SequentialStage, WriteSpec
from wiser.utils.worker_runtime import get_process_storage_client

if TYPE_CHECKING:
    pass

# Broadcast keys for RasterBackedSpectrumComputeStage
COMPUTE_KIND_ROI = "roi_average"
COMPUTE_KIND_POINT = "point"
DEFAULT_SPECTRUM_OUTPUT_NAME = "spectrum_output"


def dataset_plan_meta_from_data_ref(dataset_ref: DataRef) -> DatasetPlanMeta:
    meta = get_process_storage_client().get_meta(dataset_ref)
    h, w, b = meta.shape
    return DatasetPlanMeta(shape=(h, w, b), dtype=np.dtype(meta.elem_type))


def _compute_spectrum_worker(
    input_ref: DataRef,
    input_region: DataRegion,
    output_writes: Dict[str, WriteSpec],
    broadcast_inputs: Dict[str, Any],
) -> None:
    """Top-level worker: reconstruct dataset, compute 1D spectrum, write output."""
    _ = input_region
    client = get_process_storage_client()
    out_name = broadcast_inputs["output_ref_name"]
    output_write = output_writes[out_name]
    dataset = client.reconstruct_external_object(input_ref)
    if not isinstance(dataset, RasterDataSet):
        raise TypeError(f"Expected RasterDataSet from input ref, got {type(dataset)}")

    kind = broadcast_inputs["compute_kind"]
    avg_mode = SpectrumAverageMode[broadcast_inputs["avg_mode_name"]]

    if kind == COMPUTE_KIND_ROI:
        roi = broadcast_inputs["roi"]
        if not isinstance(roi, RegionOfInterest):
            raise TypeError("roi_average requires a RegionOfInterest in broadcast_inputs")
        spectrum = calc_roi_spectrum(dataset, roi, avg_mode)
    elif kind == COMPUTE_KIND_POINT:
        point = broadcast_inputs["point"]
        area = broadcast_inputs["area"]
        spectrum = calc_spectrum_at_point_with_area(
            dataset,
            (int(point[0]), int(point[1])),
            (int(area[0]), int(area[1])),
            avg_mode,
        )
    else:
        raise ValueError(f"Unknown compute_kind: {kind!r}")

    spectrum = np.asarray(spectrum, dtype=np.float32)
    client.write_spec(output_write, spectrum)


@dataclass
class RasterBackedSpectrumComputeStage(SequentialStage):
    """
    One non-chunked stage: full raster input metadata, single spectrum vector output.
    Parameters are passed via ``broadcast_input`` (pickled to workers).
    """

    _output_ref_name: str = DEFAULT_SPECTRUM_OUTPUT_NAME
    _num_bands: int = 0

    def __post_init__(self) -> None:
        if not self.output_bindings:
            self.output_bindings = (DataBinding(self._output_ref_name),)
        bi = dict(self.broadcast_input) if self.broadcast_input else {}
        bi.setdefault("output_ref_name", self._output_ref_name)
        self.broadcast_input = bi

    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        _ = input_region
        return SpectrumRef(self._num_bands)

    def generate_allocation_requests(
        self,
        *,
        input_meta: DatasetPlanMeta,
        chosen_scheme: Any,
    ) -> list[AllocationRequest]:
        _ = chosen_scheme
        return [
            AllocationRequest(
                name=self._output_ref_name,
                kind="spectrum",
                residency="ram_cacheable",
                size_est=self._num_bands * np.dtype(np.float32).itemsize,
                shape=(self._num_bands,),
                dtype=np.dtype(np.float32),
                delete_policy=self.get_output_delete_policy(self._output_ref_name),
            ),
        ]

    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, WriteSpec],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Any:
        return partial(
            _compute_spectrum_worker,
            input_ref,
            input_region,
            output_writes,
            dict(broadcast_inputs),
        )


def get_raster_backed_spectrum_pipeline(
    *,
    dataset_ref: DataRef,
    dataset_plan_meta: DatasetPlanMeta,
    compute_kind: str,
    output_ref_name: str = DEFAULT_SPECTRUM_OUTPUT_NAME,
    roi: Optional[RegionOfInterest] = None,
    point: Optional[Tuple[int, int]] = None,
    area: Tuple[int, int] = (1, 1),
    avg_mode: SpectrumAverageMode = SpectrumAverageMode.MEAN,
) -> AlgorithmPipeline:
    """Build a single-stage pipeline for ROI-average or spectrum-at-point extraction."""
    if compute_kind == COMPUTE_KIND_ROI and roi is None:
        raise ValueError("roi_average requires roi=")
    if compute_kind == COMPUTE_KIND_POINT and point is None:
        raise ValueError("point compute_kind requires point=")

    bands = dataset_plan_meta.bands
    broadcast: Dict[str, Any] = {
        "compute_kind": compute_kind,
        "avg_mode_name": avg_mode.name,
        "output_ref_name": output_ref_name,
    }
    if compute_kind == COMPUTE_KIND_ROI:
        broadcast["roi"] = roi
    else:
        broadcast["point"] = point
        broadcast["area"] = (int(area[0]), int(area[1]))

    stage = RasterBackedSpectrumComputeStage(
        default_executor="process",
        input_plan_meta=dataset_plan_meta,
        resource_model=ResourceModel(
            fixed_overhead_bytes=1024,
            bytes_per_scalar_in=1,
            bytes_per_scalar_out=1,
            scratch_bytes_per_scalar_in=0,
        ),
        chunking_scheme_type=NoChunkingScheme,
        _output_ref_name=output_ref_name,
        _num_bands=bands,
        broadcast_input=broadcast,
        input_binding=DataBinding("__task_input__"),
    )
    return AlgorithmPipeline(stages=[stage])


class RasterBackedSpectrumSemanticTask(QObject, SemanticTask):
    """
    Semantic task that computes a raster-backed spectrum in the work scheduler
    and emits the 1D result on ``result_ready`` when the plan completes.
    """

    result_ready = Signal(object)

    def __init__(
        self,
        *,
        priority_class: Any,
        dataset_ref: DataRef,
        pipeline: AlgorithmPipeline,
        output_ref_name: str = DEFAULT_SPECTRUM_OUTPUT_NAME,
        task_title: str = "Spectrum calculation",
        task_variables: Optional[Dict[str, str]] = None,
    ):
        QObject.__init__(self)
        SemanticTask.__init__(
            self,
            priority_class=priority_class,
            input_ref=dataset_ref,
            algorithm_pipeline=pipeline,
            task_title=task_title,
            task_variables=task_variables or {},
        )
        self._output_ref_name = output_ref_name

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        out_ref = bindings.get(self._output_ref_name)
        if out_ref is None:
            raise KeyError(f"Missing spectrum output binding: {self._output_ref_name}")
        client = get_process_storage_client()
        arr, _ = client.read_data(out_ref, filter_data=False)
        self.result_ready.emit(np.asarray(arr, dtype=np.float32))


class NumPySpectrumImmediateTask(QObject):
    """
    Emits the spectrum array for a ``NumPyArraySpectrum`` without using the scheduler.
    Call ``emit_now()`` (or rely on ``schedule_emit()`` for the next event-loop tick).
    """

    result_ready = Signal(object)

    def __init__(self, spectrum: NumPyArraySpectrum):
        super().__init__()
        self._spectrum = spectrum

    def emit_now(self) -> None:
        self.result_ready.emit(np.asarray(self._spectrum.get_spectrum(), dtype=np.float32))

    def schedule_emit(self) -> None:
        QTimer.singleShot(0, self.emit_now)


SpectrumRecomputeHandle = Union[RasterBackedSpectrumSemanticTask, NumPySpectrumImmediateTask]


def build_spectrum_recompute_task(
    spectrum: Spectrum,
    *,
    dataset_ref: Optional[DataRef] = None,
    priority_class: Any = None,
    task_id: Optional[int] = None,
    output_ref_name: str = DEFAULT_SPECTRUM_OUTPUT_NAME,
) -> SpectrumRecomputeHandle:
    """
    Return a handle with ``result_ready`` for spectrum values.

    - ``RasterDataSetSpectrum`` (ROI / point): requires ``dataset_ref`` registered
      for the same dataset; builds a ``RasterBackedSpectrumSemanticTask``.
    - ``NumPyArraySpectrum``: returns ``NumPySpectrumImmediateTask`` (call ``emit_now()``).
    """
    from wiser.utils.primitives import PriorityClass

    pc = priority_class if priority_class is not None else PriorityClass.BACKGROUND

    if isinstance(spectrum, NumPyArraySpectrum):
        return NumPySpectrumImmediateTask(spectrum)

    if isinstance(spectrum, ROIAverageSpectrum):
        if dataset_ref is None:
            raise ValueError("dataset_ref is required for ROIAverageSpectrum")
        plan_meta = dataset_plan_meta_from_data_ref(dataset_ref)
        pipeline = get_raster_backed_spectrum_pipeline(
            dataset_ref=dataset_ref,
            dataset_plan_meta=plan_meta,
            compute_kind=COMPUTE_KIND_ROI,
            output_ref_name=output_ref_name,
            roi=spectrum.get_roi(),
            avg_mode=spectrum.get_avg_mode(),
        )
        task = RasterBackedSpectrumSemanticTask(
            priority_class=pc,
            dataset_ref=dataset_ref,
            pipeline=pipeline,
            output_ref_name=output_ref_name,
            task_title="ROI average spectrum",
            task_variables={"ROI": spectrum.get_roi().get_name() or ""},
        )
        if task_id is not None:
            task.id = task_id
        return task

    if isinstance(spectrum, SpectrumAtPoint):
        if dataset_ref is None:
            raise ValueError("dataset_ref is required for SpectrumAtPoint")
        plan_meta = dataset_plan_meta_from_data_ref(dataset_ref)
        pipeline = get_raster_backed_spectrum_pipeline(
            dataset_ref=dataset_ref,
            dataset_plan_meta=plan_meta,
            compute_kind=COMPUTE_KIND_POINT,
            output_ref_name=output_ref_name,
            point=spectrum.get_point(),
            area=spectrum.get_area(),
            avg_mode=spectrum.get_avg_mode(),
        )
        p = spectrum.get_point()
        task = RasterBackedSpectrumSemanticTask(
            priority_class=pc,
            dataset_ref=dataset_ref,
            pipeline=pipeline,
            output_ref_name=output_ref_name,
            task_title="Spectrum at point",
            task_variables={"x": str(p[0]), "y": str(p[1])},
        )
        if task_id is not None:
            task.id = task_id
        return task

    if isinstance(spectrum, RasterDataSetSpectrum):
        raise TypeError(
            "Unsupported RasterDataSetSpectrum subclass; use ROIAverageSpectrum or SpectrumAtPoint"
        )

    raise TypeError(f"Unsupported spectrum type: {type(spectrum)!r}")
