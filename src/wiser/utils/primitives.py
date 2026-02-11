from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, Iterable, Protocol
import tempfile
from pathlib import Path

import numpy as np


class PriorityClass(Enum):
    INTERACTIVE = "interactive"
    RENDER = "render"
    BACKGROUND = "background"


OutputKind = Literal["dataset", "spectrum", "spectra_list", "array", "json"]
InputKind = Literal["dataset", "spectrum", "spectra_list"]


DiskFormat = Literal["memmap", "zarr", "json"]
RefKind = OutputKind

Residency = Literal["spill_required", "ram_cacheable"]

ExecutorType = Literal["thread", "process"]

MaterializationLocation = Literal["none", "ram", "disk"]


def temp_dir() -> Path:
    return Path(tempfile.gettempdir()) / "wiser"


@dataclass(frozen=True)
class DataRef:
    """
    For actually retrieving the data in disk
    """

    kind: RefKind
    ref_id: str  # stable id in storage registry
    uri: str  # backend-specific identifier (path, uuid, etc.)
    disk_format: Optional[DiskFormat] = None
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    chunks: Optional[Tuple[int, ...]] = None
    residency: Residency = "spill_required"
    materialization_loc: MaterializationLocation = "none"

    def get_byte_estimate(self) -> Optional[int]:
        # Need both to estimate
        if self.shape is None or self.dtype is None:
            return None

        # Make sure shape is valid
        if any(d is None for d in self.shape):
            return None

        # Compute number of elements
        n_elems = 1
        for d in self.shape:
            if d < 0:
                return None
            n_elems *= d

        # Get bytes per element from dtype
        try:
            itemsize = np.dtype(self.dtype).itemsize
        except Exception:
            return None

        return n_elems * itemsize


@dataclass(frozen=True)
class AllocationRequest:
    """
    To reserve the right amount of space on the disk, different
    from DataRef which is the actual handle to access the data.

    The name for AllocationRequest is used to get the underlying data ref.
    """

    name: str  # Unique name to the SemanticTask th
    kind: OutputKind
    residency: Residency
    size_est: int

    # For numeric arrays (dataset/spectrum/spectra_list/array)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[np.dtype] = None
    chunks: Optional[Tuple[int, ...]] = None  # for zarr / chunked storage

    # Optional metadata tags (task_id, stage_id, output_name)
    tags: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class DataBinding:
    """
    Declares that a stage produces a semantic output.

    Does NOT allocate storage.
    Does NOT know shape/dtype.
    """

    name: str
    kind: OutputKind = "dataset"
    residency: Residency = "spill_required"


@dataclass(frozen=True)
class DataRegion(ABC):
    pass


@dataclass(frozen=True)
class DatasetRegionRef(DataRegion):
    y0: int
    y1: int
    x0: int
    x1: int
    b0: int = 0
    b1: Optional[int] = None  # None = all bands


@dataclass(frozen=True)
class SpectrumRef(DataRegion):
    # single spectrum, no chunking needed most of the time
    pass


@dataclass(frozen=True)
class SpectraBatchRef(DataRegion):
    i0: int
    i1: int  # index range into list-of-spectra


# Returns input and output regions (aka ChunkRefs)
@dataclass
class ChunkingScheme(Protocol):
    kind: InputKind = "dataset"

    def iter_chunks(self, meta) -> Iterable["DataRegion"]:
        ...


@dataclass
class SpatialTileScheme:
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
class SpectralBatchScheme:
    kind: InputKind = "dataset"
    band_step: int = 32

    def iter_chunks(self, meta) -> Iterable[DatasetRegionRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for b0 in range(0, B, self.band_step):
            b1 = min(B, b0 + self.band_step)
            yield DatasetRegionRef(0, H, 0, W, b0, b1)


@dataclass(frozen=True)
class SingleSpectrumScheme:
    kind: InputKind = "spectrum"

    def iter_chunks(self, meta=None) -> Iterable[SpectrumRef]:
        yield SpectrumRef()


@dataclass(frozen=True)
class SpectraBatchScheme:
    kind: InputKind = "spectra_list"
    batch_size: int = 256

    def iter_chunks(self, meta) -> Iterable[SpectraBatchRef]:
        n = meta.num_spectra
        for i0 in range(0, n, self.batch_size):
            yield SpectraBatchRef(i0=i0, i1=min(n, i0 + self.batch_size))
