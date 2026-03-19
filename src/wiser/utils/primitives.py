from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from enum import Enum
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
    Iterable,
    Protocol,
    TYPE_CHECKING,
    ClassVar,
)
import tempfile
from pathlib import Path

import numpy as np
from astropy import units as u


class PriorityClass(Enum):
    INTERACTIVE = "interactive"
    RENDER = "render"
    BACKGROUND = "background"


OutputKind = Literal["dataset", "spectrum", "spectra_list", "array", "json"]
InputKind = Literal["dataset", "spectrum", "spectra_list"]
WorkUnitDependency = Literal["independent", "sequential"]

DiskFormat = Literal["memmap", "zarr", "json"]
RefKind = OutputKind

Residency = Literal["spill_required", "ram_cacheable"]

ExecutorType = Literal["thread", "process"]

MaterializationLocation = Literal["none", "ram", "disk"]
RefSource = Literal["internal", "external"]
ExternalParamsFamily = Literal["dataset", "spectra_list", "array"]
ExternalParamsDriver = Literal[
    "netcdf_gdal",
    "pds3_gdal",
    "pds4_gdal",
    "jp2_gdal",
    "envi_gdal",
    "gtiff_gdal",
    "asc_gdal",
    "gdal_generic",
    "envi_sli",
    "memmap",
    "zarr",
]

if TYPE_CHECKING:
    from wiser.utils.task_system import BasePlanMeta

DEFAULT_FLOAT_TYPE = np.float64


def temp_dir() -> Path:
    return Path(tempfile.gettempdir()) / "wiser"


@dataclass(frozen=True)
class BasePlanMeta:
    """Minimal, cheap-to-compute planning metadata needed to chunk data"""

    kind: InputKind
    dtype: np.dtype = np.dtype(DEFAULT_FLOAT_TYPE)

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


@dataclass(frozen=True)
class ExternalParams:
    """
    Reconstruction contract for external disk-backed refs.
    """

    family: ExternalParamsFamily
    driver: ExternalParamsDriver
    kwargs: Dict[str, Any]


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
    dtype: Optional[np.dtype] = None
    chunks: Optional[Tuple[int, ...]] = None
    residency: Residency = "spill_required"
    materialization_loc: MaterializationLocation = "none"
    source: RefSource = "internal"
    readonly: bool = False
    external_params: Optional[ExternalParams] = None

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
            itemsize = self.dtype.itemsize
        except Exception:
            return None

        return n_elems * itemsize


@dataclass(frozen=True)
class DataMeta:
    kind: RefKind
    # [y][x][b] for dataset, [b] for spectrum, [i][b] for spectra_list [i][b]
    shape: Tuple[int, ...]
    elem_type: np.dtype
    wavelengths: Optional[np.ndarray] = None
    wavelength_units: Optional[u.Unit] = None
    nodata: Optional[float | int] = None
    bad_bands: Optional[np.ndarray] = None
    crs_wkt: Optional[str] = None
    geotransform: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class RegionMeta:
    region: DataRegion
    elem_type: np.dtype
    wavelengths: Optional[np.ndarray] = None
    wavelength_units: Optional[u.Unit] = None
    nodata: Optional[float | int] = None
    bad_bands: Optional[np.ndarray] = None
    crs_wkt: Optional[str] = None
    geotransform: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class AllocationRequest:
    """
    To reserve the right amount of space on the disk, different
    from DataRef which is the actual handle to access the data.

    The name for AllocationRequest is used to get the underlying data ref.
    """

    # Unique name to be used as the binding to the DataRef
    # that is allocated by this AllocationRequest
    name: str
    kind: RefKind
    residency: Residency
    size_est: int

    # For numeric arrays (dataset/spectrum/spectra_list/array)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[np.dtype] = None
    chunks: Optional[Tuple[int, ...]] = None  # optional for zarr / chunked storage

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
class DataRegion:
    def scalar_count(self) -> int:
        raise NotImplementedError

    def validate_array_shape(self, arr: np.ndarray) -> None:
        """Validate that an array's shape matches this region."""
        raise NotImplementedError


@dataclass(frozen=True)
class DatasetRegionRef(DataRegion):
    y0: int
    y1: int
    x0: int
    x1: int
    b0: int
    b1: int

    def scalar_count(self) -> int:
        if self.b1 is None:
            raise ValueError("DatasetRegionRef.scalar_count requires b1 to be set.")
        if self.y1 < self.y0 or self.x1 < self.x0 or self.b1 < self.b0:
            raise ValueError("DatasetRegionRef has invalid bounds.")
        return (self.y1 - self.y0) * (self.x1 - self.x0) * (self.b1 - self.b0)

    def validate_array_shape(self, arr: np.ndarray) -> None:
        """
        Validate that `arr` fits this dataset region.

        Expected input array shape is `[y][x][b]` (NumPy shape `(y, x, b)`), where:
        - `y == (y1 - y0)`
        - `x == (x1 - x0)`
        - `b == (b1 - b0)`
        """
        expected_shape = (self.y1 - self.y0, self.x1 - self.x0, self.b1 - self.b0)
        if arr.ndim != 3:
            raise ValueError(
                f"DatasetRegionRef expects a 3D array with shape [y][x][b]; got ndim={arr.ndim}."
            )
        if arr.shape != expected_shape:
            raise ValueError(
                f"DatasetRegionRef expects shape {expected_shape} for bounds "
                f"(y:{self.y0}:{self.y1}, x:{self.x0}:{self.x1}, b:{self.b0}:{self.b1}); got {arr.shape}."
            )


@dataclass(frozen=True)
class SpectrumRef(DataRegion):
    # single spectrum, no chunking needed most of the time
    length: int

    def scalar_count(self) -> int:
        if self.length < 0:
            raise ValueError("SpectrumRef length must be non-negative.")
        return self.length

    def validate_array_shape(self, arr: np.ndarray) -> None:
        """
        Validate that `arr` fits this spectrum region.

        Expected input array shape is `[b]` (NumPy shape `(b,)`), where:
        - `b == length`
        """
        expected_shape = (self.length,)
        if arr.ndim != 1:
            raise ValueError(f"SpectrumRef expects a 1D array with shape [b]; got ndim={arr.ndim}.")
        if arr.shape != expected_shape:
            raise ValueError(f"SpectrumRef expects shape {expected_shape}; got {arr.shape}.")


@dataclass(frozen=True)
class SpectraBatchRef(DataRegion):
    i0: int
    i1: int  # index range into list-of-spectra, exclusive
    length: int

    def scalar_count(self) -> int:
        if self.i1 < self.i0 or self.length < 0:
            raise ValueError("SpectraBatchRef has invalid bounds.")
        return (self.i1 - self.i0) * self.length

    def validate_array_shape(self, arr: np.ndarray) -> None:
        """
        Validate that `arr` fits this spectra batch region.

        Expected input array shape is `[i][b]` (NumPy shape `(i, b)`), where:
        - `i == (i1 - i0)`
        - `b == length`
        """
        expected_shape = (self.i1 - self.i0, self.length)
        if arr.ndim != 2:
            raise ValueError(f"SpectraBatchRef expects a 2D array with shape [i][b]; got ndim={arr.ndim}.")
        if arr.shape != expected_shape:
            raise ValueError(
                f"SpectraBatchRef expects shape {expected_shape} for bounds "
                f"(i:{self.i0}:{self.i1}, b:{self.length}); got {arr.shape}."
            )


# Returns input and output regions (aka ChunkRefs)
@dataclass
class ChunkingScheme(ABC):
    kind: ClassVar[list[RefKind]] = ["dataset"]

    def iter_chunks(self, meta: "BasePlanMeta") -> Iterable["DataRegion"]:
        pass


@dataclass
class NoChunkingScheme(ChunkingScheme):
    kind: ClassVar[list[RefKind]] = ["dataset", "spectrum", "spectra_list"]

    def iter_chunks(self, meta: "BasePlanMeta") -> Iterable["DataRegion"]:
        if isinstance(meta, DatasetPlanMeta):
            yield DatasetRegionRef(0, meta.height, 0, meta.width, 0, meta.bands)
        elif isinstance(meta, SpectrumPlanMeta):
            yield SpectrumRef(meta.length)
        elif isinstance(meta, SpectraListPlanMeta):
            yield SpectraBatchRef(0, meta.num_spectra, meta.spectrum_length)
        else:
            raise ValueError(f"Unsupported meta type: {type(meta)}")


@dataclass
class SpatialTileScheme(ChunkingScheme):
    kind: ClassVar[list[RefKind]] = ["dataset"]
    tile_h: int
    tile_w: int

    def iter_chunks(self, meta: "BasePlanMeta") -> Iterable[DatasetRegionRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for y0 in range(0, H, self.tile_h):
            y1 = min(H, y0 + self.tile_h)
            for x0 in range(0, W, self.tile_w):
                x1 = min(W, x0 + self.tile_w)
                yield DatasetRegionRef(y0, y1, x0, x1, 0, B)


@dataclass
class SpectralBatchDatasetScheme(ChunkingScheme):
    kind: ClassVar[list[RefKind]] = ["dataset", "array"]
    band_step: int = 32

    def iter_chunks(self, meta: "BasePlanMeta") -> Iterable[DatasetRegionRef]:
        H, W, B = meta.height, meta.width, meta.bands
        for b0 in range(0, B, self.band_step):
            b1 = min(B, b0 + self.band_step)
            yield DatasetRegionRef(0, H, 0, W, b0, b1)


@dataclass
class SingleSpectrumScheme(ChunkingScheme):
    kind: ClassVar[list[RefKind]] = ["spectrum"]

    def iter_chunks(self, meta: "BasePlanMeta") -> Iterable[SpectrumRef]:
        yield SpectrumRef(meta.length)


@dataclass
class SpectraBatchScheme(ChunkingScheme):
    kind: ClassVar[list[RefKind]] = ["spectra_list"]
    batch_size: int = 256

    def iter_chunks(self, meta: "BasePlanMeta") -> Iterable[SpectraBatchRef]:
        n = meta.num_spectra
        for i0 in range(0, n, self.batch_size):
            yield SpectraBatchRef(
                i0=i0,
                i1=min(n, i0 + self.batch_size),
                length=meta.spectrum_length,
            )
