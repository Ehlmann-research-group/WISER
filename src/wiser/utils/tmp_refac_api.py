from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Literal, Optional, Protocol, Union


class PriorityClass(Enum):
    INTERACTIVE = "interactive"
    RENDER = "render"
    BACKGROUND = "background"


class TaskOutputTypes(Enum):
    DATASET = "dataset"
    TEXT = "text"
    GRAPH = "graph"
    SPECTRA = "spectra"


class ResidencyPreference(Enum):
    SPILL_REQ = "spill required"  # Spills to disk to preserve space
    SPILL_OR_RAM = "spill or ram"
    RAM_REQ = "ram required"


@dataclass(frozen=True)
class OutputSpec:
    """Specifies what output exists and where that output may live"""

    output_type: TaskOutputTypes
    residency_pref: ResidencyPreference


class AlgorithmPattern(Enum):
    MAP = "map"
    REDUCE = "reduce"
    FILTER = "filter"
    SINGLE_SHOT = "single shot"


class ExecutorType(Enum):
    THREAD = "thread"
    PROCESS = "process"


class STAGE_TYPES(Enum):
    MAP = "map"
    REDUCE = "reduce"


InputKind = Literal["dataset", "spectrum", "spectra_list"]


@dataclass(frozen=True)
class ChunkRef:
    pass


@dataclass(frozen=True)
class DatasetChunkRef:
    y0: int
    y1: int
    x0: int
    x1: int
    b0: int = 0
    b1: Optional[int] = None  # None = all bands


@dataclass(frozen=True)
class SpectrumChunkRef:
    # single spectrum, no chunking needed most of the time
    pass


@dataclass(frozen=True)
class SpectraBatchRef:
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

    def iter_chunks(self, meta) -> Iterable[DatasetChunkRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for y0 in range(0, H, self.tile_h):
            y1 = min(H, y0 + self.tile_h)
            for x0 in range(0, W, self.tile_w):
                x1 = min(W, x0 + self.tile_w)
                yield DatasetChunkRef(y0, y1, x0, x1, 0, B)


@dataclass(frozen=True)
class SpectralBatchScheme(ChunkingScheme):
    kind: InputKind = "dataset"
    band_step: int = 32

    def iter_chunks(self, meta) -> Iterable[DatasetChunkRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for b0 in range(0, B, self.band_step):
            b1 = min(B, b0 + self.band_step)
            yield DatasetChunkRef(0, H, 0, W, b0, b1)


@dataclass(frozen=True)
class SingleSpectrumScheme(ChunkingScheme):
    kind: InputKind = "spectrum"

    def iter_chunks(self, meta=None) -> Iterable[SpectrumChunkRef]:
        yield SpectrumChunkRef()


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

    # Estimates
    @abstractmethod
    def map_fn(self, input_ref, output_ref, kwargs, broadcast_inputs: list[str] = []):
        pass


# list of spectra
# goes through the list of spectra, yeilds chunks of them from i0 to i1 based on spectral step


class AlgorithmPipeline:
    pass


class TaskPlan:
    pass


class TaskPlanner:
    """
    Takes in a SemanticTask. Goes through its AlgorithmPipeline. Goes through each
    Stage in the pipeline and sees the chunking policy reference it wants. Creates a
    chunking scheme object for that stage. Creates work unit using stage and chunking scheme.
    (The region that the chunking scheme gives is just metadata for the future).
    """


class SemanticTask(ABC):
    def __init__(
        self,
        priority_class: PriorityClass,
        algorithm_pipeline: AlgorithmPipeline,
        algo_kwargs: Dict,
        output_spec: OutputSpec,
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self._priorit_class: PriorityClass = priority_class
        self._output_spec: OutputSpec = output_spec

        self._algorithm: AlgorithmPipeline = algorithm_pipeline
        self._algo_kwargs: Dict = algo_kwargs

    def get_output_spec(self) -> OutputSpec:
        return self._output_spec

    def get_algorithm(self) -> Callable:
        return self._algorithm
