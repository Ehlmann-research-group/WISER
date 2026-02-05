from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np

from wiser.raster.spectrum import Spectrum

# TODO (Joshua G-K): Change these later to adapt to the system's constraints
CPU_BUDGET = 6
RAM_BUDGET = 4000000000
THREAD_BUDGET = 32


class PriorityClass(Enum):
    INTERACTIVE = "interactive"
    RENDER = "render"
    BACKGROUND = "background"


OutputKind = Literal["dataset", "spectrum", "spectra_list", "array", "json"]
StorageKind = Literal["memmap", "zarr", "json", "in_ram"]


Residency = Literal["spill_required", "ram_cacheable", "pin_when_visible"]


@dataclass(frozen=True)
class OutputDesc:
    kind: OutputKind
    residency: Residency

    # For numeric arrays (dataset/spectrum/spectra_list/array)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None  # "float32", "uint16", etc.
    chunks: Optional[Tuple[int, ...]] = None  # for zarr / chunked storage

    # Optional metadata tags (task_id, stage_id, output_name)
    tags: Optional[Dict[str, str]] = None


# @dataclass(frozen=True)
# class OutputRef:
#     output_id: str
#     kind: OutputKind
#     storage_kind: StorageKind
#     path: str
#     shape: Tuple[int, ...]
#     dtype: str
#     chunks: Optional[Tuple[int, ...]] = None
#     residency: Residency = "spill_required"


# output region types


class AlgorithmPattern(Enum):
    MAP = "map"
    REDUCE = "reduce"
    FILTER = "filter"
    SINGLE_SHOT = "single shot"


ExecutorType = Literal["thread", "process"]


class STAGE_TYPES(Enum):
    MAP = "map"
    REDUCE = "reduce"


InputKind = Literal["dataset", "spectrum", "spectra_list"]


RefKind = Literal["dataset", "spectra", "spectra_list", "json", "arrays"]


@dataclass(frozen=True)
class DataRef:
    kind: RefKind
    ref_id: str  # stable id in storage registry
    storage_kind: StorageKind
    uri: str  # path or locator
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    residency: Residency = "spill_required"


@dataclass(frozen=True)
class ChunkRef(ABC):
    pass


@dataclass(frozen=True)
class DatasetRegionRef(ChunkRef):
    y0: int
    y1: int
    x0: int
    x1: int
    b0: int = 0
    b1: Optional[int] = None  # None = all bands


@dataclass(frozen=True)
class SpectrumRef(ChunkRef):
    # single spectrum, no chunking needed most of the time
    pass


@dataclass(frozen=True)
class SpectraBatchRef(ChunkRef):
    i0: int
    i1: int  # index range into list-of-spectra


# The chunking scheme is what iterates over the data and gives stuff back.
# the chunking ref is what it give sback


@dataclass
class ChunkingScheme(Protocol):
    kind: InputKind = "dataset"

    def iter_chunks(self, meta) -> Iterable["ChunkRef"]:
        ...


@dataclass
class SpatialTileScheme(ChunkingScheme):
    tile_h: int
    tile_w: int

    def iter_chunks(self, meta) -> Iterable[DatasetRegionRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for y0 in range(0, H, self.tile_h):
            y1 = min(H, y0 + self.tile_h)
            for x0 in range(0, W, self.tile_w):
                x1 = min(W, x0 + self.tile_w)
                yield DatasetRegionRef(y0, y1, x0, x1, 0, B)


@dataclass(frozen=True)
class SpectralBatchScheme(ChunkingScheme):
    kind: InputKind = "dataset"
    band_step: int = 32

    def iter_chunks(self, meta) -> Iterable[DatasetRegionRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for b0 in range(0, B, self.band_step):
            b1 = min(B, b0 + self.band_step)
            yield DatasetRegionRef(0, H, 0, W, b0, b1)


@dataclass(frozen=True)
class SingleSpectrumScheme(ChunkingScheme):
    kind: InputKind = "spectrum"

    def iter_chunks(self, meta=None) -> Iterable[SpectrumRef]:
        yield SpectrumRef()


@dataclass(frozen=True)
class SpectraBatchScheme(ChunkingScheme):
    kind: InputKind = "spectra_list"
    batch_size: int = 256

    def iter_chunks(self, meta) -> Iterable[SpectraBatchRef]:
        n = meta.num_spectra
        for i0 in range(0, n, self.batch_size):
            yield SpectraBatchRef(i0=i0, i1=min(n, i0 + self.batch_size))


@dataclass(frozen=True)
class DatasetAccessSpec:
    kind: Literal["dataset"] = "dataset"
    access: Literal["spatial_tiles", "spectral_batches"] = "spatial_tiles"


@dataclass(frozen=True)
class SpectrumAccessSpec:
    kind: Literal["spectrum"] = "spectrum"


@dataclass(frozen=True)
class SpectraListAccessSpec:
    kind: Literal["spectra_list"] = "spectra_list"
    access: Literal["batches"] = "batches"


StageInputSpec = Union[DatasetAccessSpec, SpectrumAccessSpec, SpectraListAccessSpec]


@dataclass(frozen=True)
class ResourceModel:
    fixed_overhead_bytes: int
    bytes_per_pixel_in: int
    bytes_per_pixel_out: int
    scratch_bytes_per_pixel: int


# class InputRef(Protocol):
#     kind: InputKind


# @dataclass(frozen=True)
# class DatasetInputRef:
#     kind: InputKind = "dataset"
#     uri: str = ""
#     subdataset: Optional[str] = None


# @dataclass(frozen=True)
# class SpectrumInputRef:
#     kind: InputKind = "spectrum"
#     uri: Optional[str] = None
#     # If spectra are small, either use an id, or the actual array
#     spectrum_arr: Optional[Spectrum]


# @dataclass(frozen=True)
# class SpectraListInputRef:
#     kind: InputKind = "spectra_list"
#     uri: Optional[str] = None
#     list_id: Optional[str] = None  # app-level identifier


class TaskStage(ABC):
    def __init__(
        self,
        algo_pattern: AlgorithmPattern,
        default_executor: ExecutorType,
    ):
        self._algo_pattern = algo_pattern
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
    input_spec: StageInputSpec
    output_spec: OutputDesc
    input_ref: DataRef = None
    output_ref: DataRef = None

    @abstractmethod
    def output_region_for(self, input_region: ChunkRef) -> ChunkRef:
        """Given an input region described by a ChunkRef, return the output region.

        Args:
            input_region (ChunkRef): The input region that is given
            to this work unit.

        Returns:
            ChunkRef: The output region that the data in the input region
            will map to.
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
    Designed to be lightweight: do NOT put wavelengths/CRS/etc here.
    """

    kind: InputKind = "dataset"
    shape: Tuple[int, int, int] = (0, 0, 0)  # [y][x][b]

    # Optional performance hints (nice-to-have)
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


class WorkUnit:
    unit_id: str
    stage_id: str
    executor_kind: ExecutorType
    input_region: ChunkRef
    fn: Callable[..., Any]
    kwargs: Dict[str, Any]
    # We don't subdivide the ram into i/o, processing, output because the scheduler
    # itself doesn't have divisions
    ram_peak_est_bytes: int
    deps: Tuple["WorkUnit", ...] = ()


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
        input_kind,
        meta: PlanMeta,
        sched_conf: "SchedulerConfig",
        resource_model: ResourceModel,
        scheme_type: type,
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


class StorageLayer:
    """
    The main purpose of this is to allocate data and return a reference to it.

    Then to read data and return a reference to it
    :var stage: Description
    """

    def allocate_data(
        self,
        desc: OutputDesc,
        *,
        preferred_storage: Optional[StorageKind] = None,
        ttl_seconds: Optional[int] = None,  # optional: cache eviction hint
    ) -> "DataRef":
        pass

    def write_region(self, data: DataRef, chunk_ref: ChunkRef) -> bool:
        pass

    def write_data(self, ref: DataRef, value: Any) -> None:  # for text/small arrays
        pass

    def read_data(self, ref_id: str) -> DataRef:
        pass

    def read_region(self, ref_id: str, chunk_ref: ChunkRef):
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
    """

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
        output_spec: OutputDesc,
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self._priorit_class: PriorityClass = priority_class
        self._output_spec: OutputDesc = output_spec

        self._algorithm: AlgorithmPipeline = algorithm_pipeline
        self._algo_kwargs: Dict = algo_kwargs

    def get_output_spec(self) -> OutputDesc:
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
