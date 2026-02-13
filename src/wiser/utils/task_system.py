from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from .primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    ExecutorType,
    InputKind,
    PriorityClass,
    SingleSpectrumScheme,
    SpatialTileScheme,
    SpectraBatchScheme,
    SpectralBatchDatasetScheme,
)
from .storage_layer import StorageLayer


class SchedulerConfig(Protocol):
    """Scheduler configuration interface used for planning-time typing."""


@dataclass(frozen=True)
class ResourceModel:
    fixed_overhead_bytes: int
    bytes_per_scalar_in: int
    bytes_per_scalar_out: int
    scratch_bytes_per_scalar_in: int


@dataclass
class TaskStage:
    default_executor: ExecutorType
    input_ref: DataRef
    resource_model: ResourceModel

    # Where this stage reads from. It is a key in the task plan's table
    # __task_input__ is the first input to the semantic task
    input_binding: DataBinding = field(default_factory=lambda: DataBinding("__task_input__"))

    output_bindings: Sequence[DataBinding] = field(default_factory=tuple)

    broadcast_input: Dict[str,] = field(default_factory=dict)

    def plan_meta_for(input_ref: DataRef) -> BasePlanMeta:
        """
        Given an input DataRef, output a BasePlanMeta. A BasePlanMeta
        is just a description dimensions in the image cube, spectrum,
        or spectra in the input_ref
        """
        pass


@dataclass
class ReduceStage(TaskStage):
    @abstractmethod
    def reduce_fn():
        pass


@dataclass
class MapStage(TaskStage):
    chunking_scheme_type: type[ChunkingScheme]

    @abstractmethod
    def output_region_for(self, input_region: DataRegion) -> DataRegion:
        """Given an input region described by a DataRegion, return the output region.

        Args:
            input_region (DataRegion): The input region that is given
            to this work unit.

        Returns:
            DataRegion: The output region that the data in the input region
            will map to.
        """
        pass

    def make_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        # we will probably neede a params dict, but I don't know if it
        # will be a UI passed in parameter or something the developer will code
        # params: dict,
        chosen_scheme: ChunkingScheme | None,
    ) -> list[AllocationRequest]:
        """
        Make allocation requests that the Task Planner will
        send to the storage layer.

        :param self: Description
        :param input_meta: Description
        :type input_meta: "BasePlanMeta"
        :param params: Description
        :type params: dict
        :param chosen_scheme: Description
        :type chosen_scheme: ChunkingScheme | None
        :return: Description
        :rtype: list[AllocationRequest]
        """
        pass

    @abstractmethod
    def map_fn(
        self,
        input_region,
        output_ref,
        kwargs,
        broadcast_inputs: list[str] = [],
    ) -> Callable:
        pass


@dataclass(frozen=True)
class BasePlanMeta:
    """Minimal, cheap-to-compute planning metadata."""

    kind: InputKind
    dtype: np.dtype

    @property
    def dtype_bytes(self) -> int:
        return self.dtype.itemsize


@dataclass(frozen=True)
class DatasetPlanMeta(BasePlanMeta):
    """
    Minimal metadata needed to plan chunking and estimate memory for dataset operations.
    """

    kind: InputKind = "dataset"
    shape: Tuple[int, int, int] = (0, 0, 0)  # [y][x][b]

    # Optional performance hints
    gdal_block_shape: Optional[Tuple[int, int]] = None  # (block_h, block_w) if known

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def bands(self) -> int:
        return self.shape[2]

    @property
    def pixels(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class SpectrumPlanMeta(BasePlanMeta):
    """Minimal metadata for a single spectrum (1D array)."""

    kind: InputKind = "spectrum"
    length: int = 0  # number of wavelength samples


@dataclass(frozen=True)
class SpectraListPlanMeta(BasePlanMeta):
    """Minimal metadata for a list of spectra (N spectra, each length L)."""

    kind: InputKind = "spectra_list"
    num_spectra: int = 0
    spectrum_length: int = 0


@dataclass
class AlgorithmPipeline:
    stages: List[TaskStage]


@dataclass
class WriteSpec:
    """One write performed by this unit."""

    name: str  # task stage's output binding (e.g. "pca_image")
    ref: "DataRef"  # where to write
    region: Optional["DataRegion"] = None  # None for text/small outputs


@dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    stage_id: str
    executor_kind: ExecutorType
    input_ref: DataRef
    input_region: DataRegion
    writes: Tuple[WriteSpec, ...]
    fn: Callable[..., Any]
    # I think the params should be in fn (so fn ilike a lambda with params preloaded)
    # but I am still unsure so keeping it for now.
    # params: Dict[str, Any]
    broadcast: Dict[str, "DataRef"]
    # We don't subdivide the ram into i/o, processing, output because the scheduler
    # itself doesn't have divisions
    ram_peak_est_bytes: int
    deps: Tuple[str, ...] = ()  # dependency unit_ids (NOT WorkUnit objects)


@dataclass
class TaskPlan:
    """
    Contains a work unit graph / dependencies.
        - WOrk units for the same stage are bundled together
        - Stages that can be run in parallel are
    """

    plan_id: str
    semantic_task_id: str
    work_units: Dict[str, WorkUnit] = field(
        default_factory=dict
    )  # Each work unit has a parent and/or a child
    stage_work_units: Dict[str, List[str]] = field(default_factory=dict)  # List of work units per stage
    bindings: Dict[str, DataRef] = field(default_factory=dict)


class ChunkingPolicy(Protocol):
    def choose(
        self,
        meta: BasePlanMeta,
        sched: "SchedulerConfig",
        resource_model: ResourceModel,
        scheme_type: type,
        constraints: Dict[str, Any],
    ) -> ChunkingScheme:
        ...


class SimpleChunkingPolicy:
    def choose(
        self,
        meta: BasePlanMeta,
        sched_conf: "SchedulerConfig",
        resource_model: ResourceModel,
        scheme_type: type[ChunkingScheme],
        constraints: Dict[str, Any],
    ) -> ChunkingScheme:
        """
        Choose a chunking scheme based on all of the passed in parameters

        :param self: Description
        :param input_kind: Description
        :param meta: Description
        :type meta: Any
        :param sched_conf: Description
        :type sched_conf: "SchedulerConfig"
        :param resource_model: Description
        :type resource_model: ResourceModel
        :param scheme_type: Description
        :type scheme_type: type
        :param constraints: Description
        :type constraints: Dict[str, Any]
        :return: Description
        :rtype: ChunkingScheme
        """

        # 1) Validate InputKind matches
        scheme_kind = getattr(scheme_type, "kind", None)
        if scheme_kind is None:
            raise TypeError(f"{scheme_type.__name__} must define a class variable `kind` (InputKind).")

        if scheme_kind != meta.kind:
            raise ValueError(
                f"ChunkingScheme InputKind mismatch: scheme_type={scheme_type.__name__} "
                f"has kind={scheme_kind!r}, but meta.kind={meta.kind!r}."
            )

        # 2) Instantiate with simple logic for known schemes
        if scheme_type is SpatialTileScheme:
            # tile_h/tile_w = 1/3 of height/width
            tile_h = max(1, int(meta.height // 3))  # type: ignore[attr-defined]
            tile_w = max(1, int(meta.width // 3))  # type: ignore[attr-defined]
            return SpatialTileScheme(tile_h=tile_h, tile_w=tile_w)

        if scheme_type is SpectralBatchDatasetScheme:
            # band_step = 1/3 of bands
            band_step = max(1, int(meta.bands // 3))  # type: ignore[attr-defined]
            return SpectralBatchDatasetScheme(band_step=band_step)

        if scheme_type is SpectraBatchScheme:
            # batch_size = 1/3 of num_spectra
            batch_size = max(1, int(meta.num_spectra // 3))  # type: ignore[attr-defined]
            return SpectraBatchScheme(batch_size=batch_size)

        if scheme_type is SingleSpectrumScheme:
            return SingleSpectrumScheme()

        # 3) Fallback: instantiate with default constructor
        try:
            return scheme_type()  # type: ignore[call-arg]
        except TypeError as e:
            raise TypeError(
                f"Don't know how to instantiate scheme_type={scheme_type.__name__}. "
                "Either provide a no-arg constructor or add a case in SimpleChunkingPolicy.choose()."
            ) from e


@dataclass
class PlanningContext:
    sched_cfg: "SchedulerConfig"
    storage: StorageLayer
    chunking_policy: ChunkingPolicy


class TaskPlanner:
    """Turns a semantic task (pipeline + params + input) into a TaskPlan (DAG of WorkUnits)."""

    def __init__(self, ctx: PlanningContext):
        self._ctx = ctx
        self._unit_counter = 0

    def _new_unit_id(self, plan_id: str) -> str:
        # TODO: Give unit's a uuid4
        self._unit_counter += 1
        return f"{plan_id}:u{self._unit_counter:06d}"

    def plan_semantic_task(self, semantic_task: SemanticTask) -> TaskPlan:
        """
        Rough flow (map-only draft):
        - bindings["__task_input__"] = semantic_task.input_ref
        - for each stage:
            - resolve stage input ref via bindings
            - choose chunking scheme
            - allocate outputs up front and bind them
            - expand regions -> WorkUnits
            - add dependencies (here: stage barrier; later you can do finer DAG)
        """

        # TODO: give plan id a uuid4
        plan_id = f"plan:{semantic_task.id}"
        plan = TaskPlan(plan_id=plan_id, semantic_task_id=str(semantic_task.id))

        # 1) init bindings
        bindings: Dict[str, DataRef] = {"__task_input__": semantic_task.input_ref}

        plan.bindings.update(bindings)

        # A simple policy: all units in stage i depend on completion of *all* units in stage i-1.
        prev_stage_unit_ids: List[str] = []

        for stage_idx, stage in enumerate(semantic_task.get_algorithm().stages):
            stage_id = f"s{stage_idx:02d}"

            if not isinstance(stage, MapStage):
                raise NotImplementedError("Draft only implements MapStage expansion.")

            # 2) resolve stage input ref
            input_ref = plan.bindings[stage.input_binding.name]

            # 3) compute minimal meta (your real code uses BasePlanMeta from DataRef)
            input_meta = stage.plan_meta_for(input_ref)

            # 4) choose chunking scheme
            scheme = self._ctx.chunking_policy.choose(
                meta=input_meta,
                sched=self._ctx.sched_cfg,
                resource_model=stage.resource_model,
                scheme_type=stage.chunking_scheme_type,
                constraints={},
            )

            # 5) allocate outputs up front
            alloc_reqs = stage.make_allocation_requests(
                input_meta=input_meta,
                # params=semantic_task.params(),
                chosen_scheme=scheme,
            )
            for req in alloc_reqs:
                out_ref = self._ctx.storage.allocate_data(req)
                plan.bindings[req.name] = out_ref

            # 6) expand regions -> WorkUnits
            unit_ids_for_stage: List[str] = []

            for input_region in scheme.iter_chunks(input_meta):
                out_writes: List[WriteSpec] = []
                for ob in stage.output_bindings:
                    out_ref = plan.bindings[ob.name]
                    out_region = stage.output_region_for(input_region)
                    out_writes.append(WriteSpec(name=ob.name, ref=out_ref, region=out_region))

                # 7) estimate RAM (rough)
                ram_est = self._estimate_ram(stage.resource_model, input_region, out_writes, input_meta)

                unit_id = self._new_unit_id(plan_id)
                unit = WorkUnit(
                    unit_id=unit_id,
                    stage_id=stage_id,
                    executor_kind=stage.default_executor,
                    input_ref=input_ref,
                    input_region=input_region,
                    writes=tuple(out_writes),
                    fn=stage.map_fn,
                    # params=dict(stage.params()),
                    broadcast=dict[str, DataRef](stage.broadcast_input),  # name->DataRef
                    ram_peak_est_bytes=ram_est,
                    deps=tuple(prev_stage_unit_ids),
                )

                plan.work_units[unit_id] = unit
                unit_ids_for_stage.append(unit_id)

            plan.stage_work_units[stage_id] = unit_ids_for_stage
            prev_stage_unit_ids = unit_ids_for_stage

        return plan

    def _estimate_ram(
        self,
        rm: ResourceModel,
        input_region: DataRegion,
        writes: Sequence[WriteSpec],
        input_meta: BasePlanMeta,
    ) -> int:
        # Very rough: fixed + per-pixel in/out + scratch. Assumes DataRegion can compute pixel count.
        in_scalar_count = input_region.scalar_count()  # you likely already have this
        out_scalar_count = sum((w.region.scalar_count() if w.region is not None else 0) for w in writes)
        return (
            rm.fixed_overhead_bytes
            + rm.bytes_per_scalar_in * in_scalar_count * input_meta.dtype_bytes
            + rm.bytes_per_scalar_out * out_scalar_count * input_meta.dtype_bytes
            + rm.scratch_bytes_per_scalar_in * in_scalar_count
        )


class SemanticTask(ABC):
    def __init__(
        self,
        priority_class: PriorityClass,
        input_ref: DataRef,
        algorithm_pipeline: AlgorithmPipeline,
        algo_kwargs: Dict,
        output_spec: AllocationRequest,
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self.input_ref = input_ref
        self._priorit_class: PriorityClass = priority_class
        self._output_spec: AllocationRequest = output_spec

        self._algorithm: AlgorithmPipeline = algorithm_pipeline
        self._algo_kwargs: Dict = algo_kwargs

    def get_output_alloc_request(self) -> AllocationRequest:
        return self._output_spec

    def get_algorithm(self) -> AlgorithmPipeline:
        return self._algorithm
