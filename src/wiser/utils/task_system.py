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
from .storage_service import StorageService


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
    input_plan_meta: "BasePlanMeta"
    resource_model: ResourceModel
    fn_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Where this stage reads from. It is a key in the task plan's table
    # __task_input__ is the first input to the semantic task
    input_binding: DataBinding = field(default_factory=lambda: DataBinding("__task_input__"))

    output_bindings: Sequence[DataBinding] = field(default_factory=tuple)

    broadcast_input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReduceStage(TaskStage):
    @abstractmethod
    def reduce_fn():
        raise NotImplementedError("Subclasses must implement reduce_fn")


@dataclass
class MapStage(TaskStage):
    # TODO (Joshua G-K): output_region_for should output a list of DataRegions
    # or at least multiple data regions (make its in a Dict). This is because
    # on task stage should be able to output more than one output.
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme

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
        raise NotImplementedError("Subclasses must implement output_region_for")

    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        # we will probably need a params dict, but I don't know if it
        # will be a UI passed in parameter or something the developer will code
        # params: dict,
        chosen_scheme: ChunkingScheme | None,
    ) -> list[AllocationRequest]:
        """
        Make allocation requests that the Task Planner will
        send to the storage layer. An allocation request should
        not be per each chunked region. It should be for each output.
        For example, if you had a 400x500 dataset that would be chunked
        into tenths, you would only do one allocation of 400x500 still.

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
        raise NotImplementedError("Subclasses must implement generate_allocation_requests")

    @abstractmethod
    def map_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, WriteSpec],  # name -> WriteSpec
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable:
        """
        This function must return a top level callable! It can not return a closure.
        Even though the class has an input_ref, that input_ref may not be made at
        the time this class is made because it may be the output of another stage.
        We will likely remove the input_ref attribute in the future.
        """
        raise NotImplementedError("Subclasses must implement map_fn")


@dataclass(frozen=True)
class BasePlanMeta:
    """Minimal, cheap-to-compute planning metadata."""

    kind: InputKind
    dtype: np.dtype = np.dtype("float32")

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
class WorkUnitMeta:
    """Planning-time I/O metadata keyed by work unit id in TaskPlan."""

    input_ref: DataRef
    input_region: DataRegion
    output_writes: Dict[str, WriteSpec]
    broadcast_inputs: Dict[str, "DataRef"]


@dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    stage_id: str
    priority_class: PriorityClass
    executor_kind: ExecutorType
    fn: Callable[..., Any]
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
    work_units_meta: Dict[str, WorkUnitMeta] = field(default_factory=dict)
    stage_work_units: Dict[str, List[str]] = field(default_factory=dict)  # List of work units per stage
    bindings: Dict[str, DataRef] = field(default_factory=dict)
    fail_fast: bool = True


class ChunkingPolicy(Protocol):
    def choose(
        self,
        meta: BasePlanMeta,
        sched_conf: "SchedulerConfig",
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
            assert isinstance(meta, DatasetPlanMeta)
            # tile_h/tile_w = 1/3 of height/width
            tile_h = max(1, int(meta.height // 3))  # type: ignore[attr-defined]
            tile_w = max(1, int(meta.width // 3))  # type: ignore[attr-defined]
            return SpatialTileScheme(tile_h=tile_h, tile_w=tile_w)

        if scheme_type is SpectralBatchDatasetScheme:
            assert isinstance(meta, DatasetPlanMeta)
            # band_step = 1/3 of bands
            band_step = max(1, int(meta.bands // 3))  # type: ignore[attr-defined]
            return SpectralBatchDatasetScheme(band_step=band_step)

        if scheme_type is SpectraBatchScheme:
            assert isinstance(meta, SpectraListPlanMeta)
            # batch_size = 1/3 of num_spectra
            batch_size = max(1, int(meta.num_spectra // 3))  # type: ignore[attr-defined]
            return SpectraBatchScheme(batch_size=batch_size)

        if scheme_type is SingleSpectrumScheme:
            assert isinstance(meta, SpectrumPlanMeta)
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
    storage: StorageService
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
            # input names should be the same as output names from a previous step
            # unless its __task_input__
            input_ref = plan.bindings[stage.input_binding.name]

            # 3) use stage planning metadata provided by the semantic planner.
            input_meta = stage.input_plan_meta

            # 4) choose chunking scheme
            scheme = self._ctx.chunking_policy.choose(
                meta=input_meta,
                sched_conf=self._ctx.sched_cfg,
                resource_model=stage.resource_model,
                scheme_type=stage.chunking_scheme_type,
                constraints={},
            )

            # 5) allocate outputs up front
            alloc_reqs = stage.generate_allocation_requests(
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
                out_writes: Dict[str, WriteSpec] = {}
                for ob in stage.output_bindings:
                    out_ref = plan.bindings[ob.name]
                    out_region = stage.output_region_for(input_region)
                    out_writes[ob.name] = WriteSpec(name=ob.name, ref=out_ref, region=out_region)

                # 7) estimate RAM (rough)
                ram_est = self._estimate_ram(stage.resource_model, input_region, out_writes, input_meta)

                unit_id = self._new_unit_id(plan_id)
                unit_meta = WorkUnitMeta(
                    input_ref=input_ref,
                    input_region=input_region,
                    output_writes=out_writes,
                    broadcast_inputs=dict[str, DataRef](stage.broadcast_input),
                )
                unit = WorkUnit(
                    unit_id=unit_id,
                    stage_id=stage_id,
                    priority_class=semantic_task.get_priority_class(),
                    executor_kind=stage.default_executor,
                    fn=stage.map_fn(
                        input_ref=unit_meta.input_ref,
                        input_region=unit_meta.input_region,
                        output_writes=unit_meta.output_writes,
                        broadcast_inputs=unit_meta.broadcast_inputs,
                    ),
                    ram_peak_est_bytes=ram_est,
                    deps=tuple(prev_stage_unit_ids),
                )

                plan.work_units[unit_id] = unit
                plan.work_units_meta[unit_id] = unit_meta
                unit_ids_for_stage.append(unit_id)

            plan.stage_work_units[stage_id] = unit_ids_for_stage
            prev_stage_unit_ids = unit_ids_for_stage

        return plan

    def _estimate_ram(
        self,
        rm: ResourceModel,
        input_region: DataRegion,
        writes: Dict[str, WriteSpec],
        input_meta: BasePlanMeta,
    ) -> int:
        # Very rough: fixed + per-pixel in/out + scratch. Assumes DataRegion can compute pixel count.
        in_scalar_count = input_region.scalar_count()  # you likely already have this
        out_scalar_count = sum(
            (w.region.scalar_count() if w.region is not None else 0) for w in writes.values()
        )
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
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self._input_ref = input_ref
        self._priority_class: PriorityClass = priority_class

        self._algorithm: AlgorithmPipeline = algorithm_pipeline

    @property
    def input_ref(self) -> DataRef:
        """Entry-point DataRef for this semantic task."""
        return self._input_ref

    def get_algorithm(self) -> AlgorithmPipeline:
        return self._algorithm

    def get_priority_class(self) -> PriorityClass:
        return self._priority_class
