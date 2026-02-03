from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional


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


class TaskAlgorithm(ABC):
    def __init__(self, algo_pattern: AlgorithmPattern, default_executor: ExecutorType):
        self._algo_pattern = algo_pattern
        self._executor = default_executor


class SemanticTask(ABC):
    def __init__(
        self, priority_class: PriorityClass, algorithm: Callable, algo_kwargs: Dict, output_spec: OutputSpec
    ):
        # The id should be set by whatever uses this task before
        # the task is used
        self.id: Optional[int] = None
        self._priorit_class: PriorityClass = priority_class
        self._output_spec: OutputSpec = output_spec

        self._algorithm = algorithm
        self._algo_kwargs = algo_kwargs

    def get_output_spec(self) -> OutputSpec:
        return self._output_spec

    def get_algorithm(self) -> Callable:
        return self._algorithm
