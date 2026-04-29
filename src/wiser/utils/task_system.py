from __future__ import annotations

from abc import abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
from PySide2.QtCore import QObject, Signal, Slot

from .primitives import (
    AllocationRequest,
    ChunkingScheme,
    DataBinding,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    ExecutorType,
    InputKind,
    NoChunkingScheme,
    PriorityClass,
    SingleSpectrumScheme,
    SpatialTileScheme,
    SpectraBatchScheme,
    SpectraBatchRef,
    SpectralBatchDatasetScheme,
    SpectrumRef,
    WorkUnitDependency,
    BasePlanMeta,
    DatasetPlanMeta,
    DeletePolicy,
    SpectrumPlanMeta,
    SpectraListPlanMeta,
)
from .storage_service import StorageService

if TYPE_CHECKING:
    from wiser.gui.activity_monitor import ActivityMonitorDialog
    from wiser.utils.work_scheduler import SchedulerConfig, WorkScheduler

Number = Union[int, float]


def _noop_post_task() -> None:
    """Default no-op post-task hook for stages that do not need post processing."""
    return None


def _noop_pre_task() -> None:
    """Default no-op pre-task hook for stages that do not need setup work."""
    return None


@dataclass(frozen=True)
class ResourceModel:
    fixed_overhead_bytes: Number
    bytes_per_scalar_in: Number
    bytes_per_scalar_out: Number
    scratch_bytes_per_scalar_in: Number


@dataclass
class TaskStage:
    default_executor: ExecutorType
    input_plan_meta: "BasePlanMeta"  # Describes shape of input data to chunk it
    resource_model: ResourceModel
    chunking_scheme_type: type[ChunkingScheme] = SpatialTileScheme
    work_unit_dependency: WorkUnitDependency = "independent"

    # Where this stage reads from. It is a key in the task plan's table
    # __task_input__ is the first input to the semantic task
    input_binding: DataBinding = field(default_factory=lambda: DataBinding("__task_input__"))

    output_bindings: Sequence[DataBinding] = field(default_factory=list)

    # If the value is a DataBinding it will be substituted for a DataRef at runtime
    broadcast_input: Dict[str, Any] = field(default_factory=dict)
    _output_delete_policies: Dict[str, DeletePolicy] = field(default_factory=dict, init=False, repr=False)
    _default_output_delete_policy: Optional[DeletePolicy] = field(default=None, init=False, repr=False)

    def set_output_delete_policy(self, output_name: str, policy: DeletePolicy) -> None:
        """Override the retention policy for one named output on this stage instance."""

        self._output_delete_policies[output_name] = policy

    def get_output_delete_policy(self, output_name: str) -> Optional[DeletePolicy]:
        """Return the explicit per-output policy, or the planner default for this stage."""

        return self._output_delete_policies.get(output_name, self._default_output_delete_policy)

    def _set_default_output_delete_policy(self, policy: DeletePolicy) -> None:
        """Set the planner-resolved default for outputs without an explicit override."""

        self._default_output_delete_policy = policy

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

    @abstractmethod
    def generate_allocation_requests(
        self,
        *,
        input_meta: "BasePlanMeta",
        chosen_scheme: Optional[ChunkingScheme],
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
    def task_fn(
        self,
        input_ref: DataRef,
        input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        # Type is usually a DataAny but can be any small serializable object
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        """
        This function must return a top level callable! It can not return a closure.
        Even though the class has an input_ref, that input_ref may not be made at
        the time this class is made because it may be the output of another stage.
        """
        raise NotImplementedError("Subclasses must implement task_fn")

    def post_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        """
        Return a small post-processing callable to run once after all work units in
        this stage have completed.

        This hook is intended for lightweight cleanup, metadata updates, or other
        small follow-up work that needs the stage's full input region context. It
        should not be used to load or process large amounts of data, since that
        work belongs in normal chunked stage work units.
        """
        _ = (input_ref, full_input_region, output_writes, broadcast_inputs)
        return _noop_post_task

    def pre_task_fn(
        self,
        input_ref: DataRef,
        full_input_region: DataRegion,
        output_writes: Dict[str, "WriteSpec"],
        broadcast_inputs: Dict[str, Any] = {},
    ) -> Callable[..., None]:
        """
        Return a small setup callable to run once before any work units in this
        stage execute.

        This hook is intended for lightweight stage initialization or staged
        metadata preparation that needs access to the full input region.
        """
        _ = (input_ref, full_input_region, output_writes, broadcast_inputs)
        return _noop_pre_task


@dataclass
class SequentialStage(TaskStage):
    """Stage type where work units are planned as sequential steps."""

    work_unit_dependency: WorkUnitDependency = "sequential"


@dataclass
class MapStage(TaskStage):
    """Stage type where work units in a stage can run independently."""

    work_unit_dependency: WorkUnitDependency = "independent"


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
    # Type is usually a data ref but can be a small serializable object
    broadcast_inputs: Dict[str, Any]


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
        - Work units for the same stage are bundled together
        - Stages that can be run in parallel are
    """

    plan_id: str
    semantic_task_id: str
    work_units: Dict[str, WorkUnit] = field(
        default_factory=dict
    )  # Each work unit has a parent and/or a child
    work_units_meta: Dict[str, WorkUnitMeta] = field(default_factory=dict)
    stage_work_units: Dict[str, List[str]] = field(default_factory=dict)  # List of work units per stage
    # Ordered barrier steps per stage. Each inner list contains units that may run in parallel.
    # Example:
    #   - fully parallel stage: [[u1, u2, u3]]
    #   - fully sequential stage: [[u1], [u2], [u3]]
    #   - mixed: [[u1, u2], [u3], [u4, u5]]
    stage_steps: Dict[str, List[List[str]]] = field(default_factory=dict)
    bindings: Dict[str, DataRef] = field(default_factory=dict)
    produced_ref_ids: set[str] = field(default_factory=set)
    fail_fast: bool = True
    completion_callback: Optional[Callable[[Dict[str, DataRef]], None]] = None
    # The below entries are for displaying to the user. They don't affect the internals
    # of how a task plan is schedule or how data is transferred
    task_title: str = "Generic Task Title"
    task_input_variables: Optional[Dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressUpdate:
    current_iteration: int
    total_iterations: int


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
        scheme_kinds = getattr(scheme_type, "kind", None)
        if not isinstance(scheme_kinds, list) or not scheme_kinds:
            raise TypeError(
                f"{scheme_type.__name__} must define a non-empty class variable `kind` " "(list[RefKind])."
            )

        if meta.kind not in scheme_kinds:
            raise ValueError(
                f"ChunkingScheme InputKind mismatch: scheme_type={scheme_type.__name__} "
                f"has kind={scheme_kinds!r}, but meta.kind={meta.kind}."
            )

        # 2) Instantiate with simple logic for known schemes
        if scheme_type is NoChunkingScheme:
            assert isinstance(meta, (DatasetPlanMeta, SpectrumPlanMeta, SpectraListPlanMeta)), (
                f"The argument meta should be of type DatasetPlanMeta, SpectrumPlanMeta, or "
                f"SpectraListPlanMeta, instead it's of type {type(meta)}"
            )
            return NoChunkingScheme()

        if scheme_type is SpatialTileScheme:
            assert isinstance(
                meta, DatasetPlanMeta
            ), f"The argument meta should be of type DatasetPlanMeta, instead it's of type {type(meta)}"
            tile_h = max(1, int(meta.height // 3))  # type: ignore[attr-defined]
            tile_w = max(1, int(meta.width // 3))  # type: ignore[attr-defined]
            return SpatialTileScheme(tile_h=tile_h, tile_w=tile_w)

        if scheme_type is SpectralBatchDatasetScheme:
            assert isinstance(
                meta, DatasetPlanMeta
            ), f"The argument meta should be of type DatasetPlanMeta, instead it's of type {type(meta)}"
            # band_step = 1/3 of bands
            band_step = max(1, int(meta.bands // 3))  # type: ignore[attr-defined]
            return SpectralBatchDatasetScheme(band_step=band_step)

        if scheme_type is SpectraBatchScheme:
            assert isinstance(
                meta, SpectraListPlanMeta
            ), f"The argument meta should be of type SpectraListPlanMeta, instead it's of type {type(meta)}"
            # batch_size = 1/3 of num_spectra
            batch_size = max(1, int(meta.num_spectra // 3))  # type: ignore[attr-defined]
            return SpectraBatchScheme(batch_size=batch_size)

        if scheme_type is SingleSpectrumScheme:
            assert isinstance(
                meta, SpectrumPlanMeta
            ), f"The argument meta should be of type SpectrumPlanMeta, instead it's of type {type(meta)}"
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

    def _full_input_region_from_meta(self, input_meta: BasePlanMeta) -> DataRegion:
        if isinstance(input_meta, DatasetPlanMeta):
            return DatasetRegionRef(0, input_meta.height, 0, input_meta.width, 0, input_meta.bands)
        if isinstance(input_meta, SpectrumPlanMeta):
            return SpectrumRef(length=input_meta.length)
        if isinstance(input_meta, SpectraListPlanMeta):
            return SpectraBatchRef(i0=0, i1=input_meta.num_spectra, length=input_meta.spectrum_length)
        raise TypeError(f"Unsupported plan meta type for full region construction: {type(input_meta)}")

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
        plan.task_title = semantic_task.task_title
        plan.task_input_variables = semantic_task.task_variables

        # 1) init bindings (extras first so __task_input__ always wins)
        bindings: Dict[str, DataRef] = dict(semantic_task.get_extra_plan_bindings())
        bindings["__task_input__"] = semantic_task.input_ref

        plan.bindings.update(bindings)
        plan.completion_callback = semantic_task.completion_callback

        # A simple policy: all units in stage i depend on completion of *all* units in stage i-1.
        prev_stage_unit_ids: List[str] = []

        for stage_idx, stage in enumerate(semantic_task.get_algorithm().stages):
            stage_id = f"s{stage_idx:02d}"

            if not isinstance(stage, TaskStage):
                raise NotImplementedError("Draft only implements TaskStage expansion.")

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
            stage._set_default_output_delete_policy(DeletePolicy.DELETE_WHEN_RELEASABLE)

            # 5) allocate outputs up front
            alloc_reqs = stage.generate_allocation_requests(
                input_meta=input_meta,
                chosen_scheme=scheme,
            )
            for req in alloc_reqs:
                out_ref = self._ctx.storage.allocate_data(
                    self._resolved_allocation_request(req),
                    owner_plan_id=plan_id,
                    planned_consumer_plan_ids={plan_id},
                )
                plan.bindings[req.name] = out_ref
                plan.produced_ref_ids.add(out_ref.ref_id)

            # 5.5) Substitute out data bindings for data refs
            # Note, data bindings should refer to data refs from
            # previous stages
            stage_broadcast_inputs: Dict[str, Union[Any, DataRef]] = {}
            for input_name, input_value in stage.broadcast_input.items():
                if isinstance(input_value, DataBinding):
                    stage_broadcast_inputs[input_name] = plan.bindings[input_value.name]
                else:
                    stage_broadcast_inputs[input_name] = input_value

            # 6) expand regions -> WorkUnits
            unit_ids_for_stage: List[str] = []
            stage_step_unit_ids: List[List[str]] = []

            full_input_region = self._full_input_region_from_meta(input_meta)
            pre_output_writes: Dict[str, WriteSpec] = {}
            for ob in stage.output_bindings:
                out_ref = plan.bindings[ob.name]
                out_region = stage.output_region_for(full_input_region)
                pre_output_writes[ob.name] = WriteSpec(name=ob.name, ref=out_ref, region=out_region)

            pre_unit_id = self._new_unit_id(plan_id)
            pre_unit_meta = WorkUnitMeta(
                input_ref=input_ref,
                input_region=full_input_region,
                output_writes=pre_output_writes,
                broadcast_inputs=dict[str, Any](stage_broadcast_inputs),
            )
            pre_unit = WorkUnit(
                unit_id=pre_unit_id,
                stage_id=stage_id,
                priority_class=semantic_task.get_priority_class(),
                executor_kind=stage.default_executor,
                fn=stage.pre_task_fn(
                    input_ref=pre_unit_meta.input_ref,
                    full_input_region=pre_unit_meta.input_region,
                    output_writes=pre_unit_meta.output_writes,
                    broadcast_inputs=pre_unit_meta.broadcast_inputs,
                ),
                ram_peak_est_bytes=self._estimate_ram(
                    stage.resource_model,
                    full_input_region,
                    pre_output_writes,
                    input_meta,
                ),
                deps=tuple(prev_stage_unit_ids),
            )

            plan.work_units[pre_unit_id] = pre_unit
            plan.work_units_meta[pre_unit_id] = pre_unit_meta
            unit_ids_for_stage.append(pre_unit_id)
            if stage.work_unit_dependency == "sequential":
                stage_step_unit_ids.append([pre_unit_id])

            chunk_unit_ids: List[str] = []

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
                    broadcast_inputs=dict[str, Any](stage_broadcast_inputs),
                )
                unit = WorkUnit(
                    unit_id=unit_id,
                    stage_id=stage_id,
                    priority_class=semantic_task.get_priority_class(),
                    executor_kind=stage.default_executor,
                    fn=stage.task_fn(
                        input_ref=unit_meta.input_ref,
                        input_region=unit_meta.input_region,
                        output_writes=unit_meta.output_writes,
                        broadcast_inputs=unit_meta.broadcast_inputs,
                    ),
                    ram_peak_est_bytes=ram_est,
                    deps=tuple([*prev_stage_unit_ids, pre_unit_id]),
                )

                plan.work_units[unit_id] = unit
                plan.work_units_meta[unit_id] = unit_meta
                unit_ids_for_stage.append(unit_id)
                chunk_unit_ids.append(unit_id)
                if stage.work_unit_dependency == "sequential":
                    stage_step_unit_ids.append([unit_id])

            post_output_writes: Dict[str, WriteSpec] = {}
            for ob in stage.output_bindings:
                out_ref = plan.bindings[ob.name]
                out_region = stage.output_region_for(full_input_region)
                post_output_writes[ob.name] = WriteSpec(name=ob.name, ref=out_ref, region=out_region)

            post_unit_id = self._new_unit_id(plan_id)
            post_unit_meta = WorkUnitMeta(
                input_ref=input_ref,
                input_region=full_input_region,
                output_writes=post_output_writes,
                broadcast_inputs=dict[str, Any](stage_broadcast_inputs),
            )
            post_unit = WorkUnit(
                unit_id=post_unit_id,
                stage_id=stage_id,
                priority_class=semantic_task.get_priority_class(),
                executor_kind=stage.default_executor,
                fn=stage.post_task_fn(
                    input_ref=post_unit_meta.input_ref,
                    full_input_region=post_unit_meta.input_region,
                    output_writes=post_unit_meta.output_writes,
                    broadcast_inputs=post_unit_meta.broadcast_inputs,
                ),
                ram_peak_est_bytes=self._estimate_ram(
                    stage.resource_model,
                    full_input_region,
                    post_output_writes,
                    input_meta,
                ),
                deps=tuple(chunk_unit_ids if len(chunk_unit_ids) > 0 else [pre_unit_id]),
            )

            plan.work_units[post_unit_id] = post_unit
            plan.work_units_meta[post_unit_id] = post_unit_meta
            unit_ids_for_stage.append(post_unit_id)

            plan.stage_work_units[stage_id] = unit_ids_for_stage
            if stage.work_unit_dependency == "independent":
                stage_steps: List[List[str]] = [[pre_unit_id]]
                if len(chunk_unit_ids) > 0:
                    stage_steps.append(chunk_unit_ids)
                stage_steps.append([post_unit_id])
                plan.stage_steps[stage_id] = stage_steps
            elif stage.work_unit_dependency == "sequential":
                stage_step_unit_ids.append([post_unit_id])
                plan.stage_steps[stage_id] = stage_step_unit_ids
            else:
                raise ValueError(f"Unknown WorkUnitDependency: {stage.work_unit_dependency!r}")
            prev_stage_unit_ids = [post_unit_id]

        return plan

    def _resolved_allocation_request(self, req: AllocationRequest) -> AllocationRequest:
        """
        Ensure every planner-owned output carries a concrete retention policy.

        Stages are expected to provide the effective policy via
        `TaskStage.get_output_delete_policy(...)`. This fallback keeps older stage
        implementations on the same default: delete when releasable unless an
        explicit policy was already attached.
        """

        if req.delete_policy is not None:
            return req
        return AllocationRequest(
            name=req.name,
            kind=req.kind,
            residency=req.residency,
            size_est=req.size_est,
            shape=req.shape,
            dtype=req.dtype,
            chunks=req.chunks,
            tags=req.tags,
            delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
        )

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


class TaskManager(QObject):
    task_finished = Signal(str)
    task_progressed = Signal(object)
    # Emitted when a work unit raises an exception so the UI can append error
    # text for the plan without treating it as a user/system cancellation.
    task_errored = Signal(object)
    # Emitted when a plan is cancelled by scheduler control flow so the UI can
    # move the activity row into the finished section as cancelled.
    task_cancelled = Signal(str)

    def __init__(self, activity_monitor: "ActivityMonitorDialog"):
        super().__init__()
        self._activity_monitor = activity_monitor
        self._activity_ids_by_plan_id: Dict[str, int] = {}
        self._plan_ids_by_activity_id: Dict[int, str] = {}
        self.task_cancelled.connect(self._on_task_cancelled)
        self.task_finished.connect(self._on_task_finished)
        self.task_progressed.connect(self._on_task_progressed)
        self.task_errored.connect(self._on_task_errored)

    def emit_progress_update(self, activity_id: int, numerator: int, denominator: int) -> None:
        if activity_id not in self._plan_ids_by_activity_id:
            raise KeyError(f"Unknown activity monitor activity id: {activity_id}")

        self._activity_monitor.progress_update.emit(
            (
                activity_id,
                ProgressUpdate(
                    current_iteration=max(0, int(numerator)),
                    total_iterations=max(1, int(denominator)),
                ),
            )
        )

    @Slot(object)
    def _on_task_progressed(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 3:
            return

        task_plan_id, numerator, denominator = payload
        if (
            not isinstance(task_plan_id, str)
            or not isinstance(numerator, int)
            or not isinstance(denominator, int)
        ):
            return

        activity_id = self._activity_ids_by_plan_id.get(task_plan_id)
        if activity_id is None:
            return
        self.emit_progress_update(activity_id, numerator, denominator)

    @Slot(str)
    def _on_task_cancelled(self, task_plan_id: str) -> None:
        activity_id = self._activity_ids_by_plan_id.get(task_plan_id)
        if activity_id is None:
            return
        self._activity_monitor.set_task_cancelled(activity_id)

    @Slot(object)
    def _on_task_errored(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return

        task_plan_id, error_message = payload
        if not isinstance(task_plan_id, str) or not isinstance(error_message, str):
            return

        activity_id = self._activity_ids_by_plan_id.get(task_plan_id)
        if activity_id is None:
            return
        self._activity_monitor.append_task_error(activity_id, error_message)

    @Slot(str)
    def _on_task_finished(self, task_plan_id: str) -> None:
        activity_id = self._activity_ids_by_plan_id.get(task_plan_id)
        if activity_id is None:
            return
        self._activity_monitor.set_task_finished(activity_id)

    def register_and_submit_task_plan(self, scheduler: "WorkScheduler", task_plan: TaskPlan) -> Future[None]:
        """
        Registers the task plan to the task gui (the real name will be ActivityMonitorDialog
        (found in [activity_monitor.py](src/wiser/gui/activity_monitor.py))), then submits it?
        """
        task_meta = task_plan.task_input_variables or {
            "plan_id": task_plan.plan_id,
            "semantic_task_id": task_plan.semantic_task_id,
            "stages": str(len(task_plan.stage_work_units)),
            "work_units": str(len(task_plan.work_units)),
        }
        activity_id = self._activity_monitor.register_task(
            title=task_plan.task_title,
            meta=task_meta,
            cancel_callback=lambda: scheduler.cancel_plan(task_plan.plan_id),
        )
        self._activity_ids_by_plan_id[task_plan.plan_id] = activity_id
        self._plan_ids_by_activity_id[activity_id] = task_plan.plan_id
        future = scheduler.run_task_plan(task_plan)
        return future


class SemanticTask:
    def __init__(
        self,
        priority_class: PriorityClass,
        input_ref: DataRef,
        algorithm_pipeline: AlgorithmPipeline,
        task_title: str = "Generic Task Title",
        task_variables: Optional[Dict[str, str]] = None,
        extra_plan_bindings: Optional[Dict[str, DataRef]] = None,
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self._input_ref = input_ref
        self._priority_class: PriorityClass = priority_class

        self._algorithm: AlgorithmPipeline = algorithm_pipeline

        self._task_title = task_title
        self._task_variables = task_variables or dict()
        self._extra_plan_bindings: Dict[str, DataRef] = dict(extra_plan_bindings or ())

    def get_extra_plan_bindings(self) -> Dict[str, DataRef]:
        """Additional :class:`DataRef` entries merged into the task plan before planning.

        The reserved key ``__task_input__`` always maps to :attr:`input_ref` and
        overrides any duplicate key from this mapping.
        """
        return self._extra_plan_bindings

    @property
    def task_title(self) -> str:
        return self._task_title

    @property
    def task_variables(self) -> Dict[str, str]:
        return self._task_variables

    @property
    def input_ref(self) -> DataRef:
        """Entry-point DataRef for this semantic task."""
        return self._input_ref

    def get_algorithm(self) -> AlgorithmPipeline:
        return self._algorithm

    def get_priority_class(self) -> PriorityClass:
        return self._priority_class

    def completion_callback(self, bindings: Dict[str, DataRef]) -> None:
        """
        Hook invoked after the task plan's final work unit completes successfully.

        Implement this in user code to consume the task plan's final `bindings`
        mapping, which contains the `DataRef` objects produced and tracked during
        planning/execution.

        Important: if the task needs the contents of an output `DataRef`, prefer
        reading that data here inside `completion_callback(...)` and then emitting
        or passing along plain Python/NumPy values. Do not defer the `DataRef`
        read to a later Qt slot unless you are certain the ref will still be live;
        output refs may become eligible for reclamation after task completion, so
        delaying the read can create a race with deletion.
        """
        pass
