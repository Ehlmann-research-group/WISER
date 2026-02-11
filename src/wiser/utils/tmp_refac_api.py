from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from .primitives import (
    ExecutorType,
    ChunkingScheme,
    DataBinding,
    DataRegion,
    AllocationRequest,
    InputKind,
    DiskFormat,
    DataRef,
    PriorityClass,
)

from wiser.raster.spectrum import Spectrum

# TODO (Joshua G-K): Change these later to adapt to the system's constraints
CPU_BUDGET = 6
RAM_BUDGET = 4000000000
THREAD_BUDGET = 32


@dataclass(frozen=True)
class ResourceModel:
    fixed_overhead_bytes: int
    bytes_per_pixel_in: int
    bytes_per_pixel_out: int
    scratch_bytes_per_pixel: int


class TaskStage(ABC):
    def __init__(
        self,
        default_executor: ExecutorType,
    ):
        self._executor = default_executor


@dataclass
class ReduceStage(TaskStage):
    resource_model: ResourceModel

    @abstractmethod
    def reduce_fn():
        pass


@dataclass
class MapStage(TaskStage):
    resource_model: ResourceModel
    chunking_scheme_type: type[ChunkingScheme]
    # Where this stage reads from. It is a key in the task plan's table
    # __task_input__ is the first input to the semantic task
    input_binding: DataBinding = field(default_factory=lambda: DataBinding("__task_input__"))

    output_binding: Sequence[DataBinding] = field(default_factory=tuple)

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
        params: dict,
        chosen_scheme: ChunkingScheme | None,
    ) -> list[AllocationRequest]:
        """
        Docstring for make_allocation_requests

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

    # Estimates
    @abstractmethod
    def map_fn(self, input_region, output_ref, kwargs, broadcast_inputs: list[str] = []):
        pass


# list of spectra
# goes through the list of spectra, yeilds chunks of them from i0 to i1 based on spectral step


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


# Union type used by planner + chunk chooser
PlanMeta = Union[DatasetPlanMeta, SpectrumPlanMeta, SpectraListPlanMeta]


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
    params: Dict[str, Any]
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
        input_kind: InputKind,
        meta: PlanMeta,
        sched: "SchedulerConfig",
        resource_model: ResourceModel,
        scheme_type: type,
        constraints: Dict[str, Any],
    ) -> ChunkingScheme:
        ...


class SimpleChunkingPolicy:
    def choose(
        self,
        input_kind: InputKind,
        meta: PlanMeta,
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
        pass


@dataclass(frozen=True)
class StorageConfig:
    """
    This is for storage. So the ram byte limit for the scheduler
    is for computation and the ram byte limit here is for storage
    """

    disk_byte_limit: int
    ram_byte_limit: int


@dataclass
class StorageLayer:
    """
    The main purpose of this is to allocate data and return a reference to it.

    Then to read data and return a reference to it
    :var stage: Description
    """

    data_refs: Dict[str, DataRef] = field(default_factory=dict)  # ref_id and DataRef
    mem_backed_data: Dict[str, Any] = field(default_factory=dict)  # uri, data (for when uri is in memory)

    def allocate_data(
        self,
        desc: AllocationRequest,
        *,
        storage_kind: Optional[DiskFormat] = None,
        ttl_seconds: Optional[int] = None,  # optional: cache eviction hint
    ) -> "DataRef":
        pass

    def write_region(self, data: DataRef, chunk_ref: DataRegion, value: Any) -> bool:
        pass

    def write_data(self, ref: DataRef, value: Any) -> None:  # for text/small arrays
        pass

    def read_data(self, ref_id: str) -> DataRef:
        pass

    def read_region(self, ref_id: str, chunk_ref: DataRegion):
        pass


@dataclass
class PlanningContext:
    sched_cfg: "SchedulerConfig"
    storage: StorageLayer
    chunking_policy: ChunkingPolicy


class TaskPlanner:
    """
    Takes in a SemanticTask. Goes through its AlgorithmPipeline. Goes through each
    Stage in the pipeline and sees the chunking policy reference it wants. Creates a
    chunking scheme object for that stage. Creates work unit using stage and chunking scheme.
    (The region that the chunking scheme gives is just metadata for the future).

    This class will have to be able to fail in submitting a task and if it does fail
    to tell the TaskManager (which is what communicates with the UI)
    """

    def __init__(self, planning_ctx: PlanningContext):
        self._ctx = planning_ctx
        self._queued_tasks: List[SemanticTask]

    def plan_semantic_task(self, semantic_task) -> TaskPlan:
        """
        Docstring for plan_algorithm_pipline

        :param self: Description
        :param algo_pipeline: Description
        :type algo_pipeline: AlgorithmPipeline
        """

        """
        Go through the algorithm pipeline of the semantic task. 

        For each stage:
            Get the resource model, input_sec, output_spec, and Work Scheduler config.

            WIth the input spec, we will decide what Chunking strat to use. With the 
            resource model and work scheduler config we will decide the parameters of
            the chunking strat.
            
            We will then get all of the input regions and use the task stage to map
            them to an output region.

            We will then ask the scheduler for space using the output description.
            We then reconstruct and return a TaskPlan with all of the work units

        """
        pass


class SemanticTask(ABC):
    def __init__(
        self,
        priority_class: PriorityClass,
        algorithm_pipeline: AlgorithmPipeline,
        algo_kwargs: Dict,
        output_spec: AllocationRequest,
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self._priorit_class: PriorityClass = priority_class
        self._output_spec: AllocationRequest = output_spec

        self._algorithm: AlgorithmPipeline = algorithm_pipeline
        self._algo_kwargs: Dict = algo_kwargs

    def get_output_alloc_request(self) -> AllocationRequest:
        return self._output_spec

    def get_algorithm(self) -> Callable:
        return self._algorithm


class SchedulerConfig:
    """
    Specifies the computer resources that we can use
    """

    def __init__(self):
        self._cpu_budget = CPU_BUDGET
        self._ram_budget = RAM_BUDGET
        self._thread_budget = THREAD_BUDGET


class WorkScheduler:
    def __init__(self, config):
        self._config = config
