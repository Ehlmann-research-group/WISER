from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from abc import ABC
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple, Iterable, Protocol, TYPE_CHECKING, ClassVar
import tempfile
from pathlib import Path

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.spectral_library import SpectralLibrary
    from wiser.raster.spectrum import Spectrum


class PriorityClass(Enum):
    INTERACTIVE = "interactive"
    RENDER = "render"
    BACKGROUND = "background"


class DeletePolicy(Enum):
    """Retention policy for a managed storage object once it becomes reclaimable."""

    KEEP = "keep"
    DELETE_WHEN_RELEASABLE = "delete_when_releasable"


class DeletionState(Enum):
    """Observed runtime position in the deletion lifecycle for a managed storage object."""

    LIVE = "live"
    PENDING_DELETE = "pending_delete"
    DELETED = "deleted"


class ProducerState(Enum):
    """Lifecycle state for the task or plan that is producing a managed output."""

    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


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


def _safe_np_dtype(value: Any) -> np.dtype:
    if value is None:
        return np.dtype("object")
    return np.dtype(value)


def _to_wavelength_array_and_unit(values: Any) -> tuple[Optional[np.ndarray], Any]:
    if values is None:
        return None, None
    if len(values) == 0:
        return np.array([], dtype=np.float64), None
    first = values[0]
    if hasattr(first, "value") and hasattr(first, "unit"):
        arr = np.asarray([v.value for v in values], dtype=np.float64)
        return arr, first.unit
    return np.asarray(values), None


def _derive_region_meta(meta: DataMeta, region: DataRegion) -> RegionMeta:
    wavelengths = meta.wavelengths
    bad_bands = meta.bad_bands
    if isinstance(region, DatasetRegionRef):
        if wavelengths is not None:
            wavelengths = wavelengths[region.b0 : region.b1]
        if bad_bands is not None:
            bad_bands = bad_bands[region.b0 : region.b1]
    return RegionMeta(
        region=region,
        elem_type=meta.elem_type,
        wavelengths=wavelengths,
        wavelength_units=meta.wavelength_units,
        nodata=meta.nodata,
        bad_bands=bad_bands,
        crs_wkt=meta.crs_wkt,
        geotransform=meta.geotransform,
    )


class ExternalHandle(Protocol):
    """Read-only adapter for externally loaded data objects."""

    kind: InputKind

    def read_region(self, region: DataRegion) -> np.ndarray:
        ...

    def get_meta(self) -> DataMeta:
        ...

    def get_region_meta(self, region: DataRegion) -> RegionMeta:
        ...

    def is_same_external_handle(self, other: "ExternalHandle") -> bool:
        ...


@dataclass
class ExternalRasterHandle:
    dataset_obj: "RasterDataSet"
    kind: InputKind = "dataset"

    def read_region(self, region: DataRegion) -> np.ndarray:
        if not isinstance(region, DatasetRegionRef):
            raise TypeError(f"Dataset external read requires DatasetRegionRef, got {type(region)}")
        arr_by_band = self.dataset_obj.get_image_data_subset(
            x=region.x0,
            y=region.y0,
            band=region.b0,
            dx=region.x1 - region.x0,
            dy=region.y1 - region.y0,
            dband=region.b1 - region.b0,
            filter_data_ignore_value=False,
        )
        # RasterDataSet uses [band][y][x]; StorageService dataset regions use [y][x][band].
        return np.asarray(arr_by_band).transpose(1, 2, 0)

    def get_meta(self) -> DataMeta:
        bands, height, width = self.dataset_obj.get_shape()
        wavelengths, wavelength_units = _to_wavelength_array_and_unit(self.dataset_obj.get_wavelengths())
        bad_bands = self.dataset_obj.get_bad_bands()
        return DataMeta(
            kind="dataset",
            shape=(height, width, bands),
            elem_type=_safe_np_dtype(self.dataset_obj.get_elem_type()),
            wavelengths=wavelengths,
            wavelength_units=wavelength_units or self.dataset_obj.get_band_unit(),
            nodata=self.dataset_obj.get_data_ignore_value(),
            bad_bands=np.asarray(bad_bands) if bad_bands is not None else None,
            crs_wkt=self.dataset_obj.get_wkt_spatial_reference(),
            geotransform=tuple(self.dataset_obj.get_geo_transform()),
        )

    def get_region_meta(self, region: DataRegion) -> RegionMeta:
        return _derive_region_meta(self.get_meta(), region)

    def is_same_external_handle(self, other: "ExternalHandle") -> bool:
        if not isinstance(other, ExternalRasterHandle):
            return False

        dataset_id = self.dataset_obj.get_id()
        other_dataset_id = other.dataset_obj.get_id()
        if dataset_id is None or other_dataset_id is None:
            return False

        return dataset_id == other_dataset_id


@dataclass
class ExternalSpectrumHandle:
    spectrum_obj: "Spectrum"
    kind: InputKind = "spectrum"

    def read_region(self, region: DataRegion) -> np.ndarray:
        if not isinstance(region, SpectrumRef):
            raise TypeError(f"Spectrum external read requires SpectrumRef, got {type(region)}")
        spectrum = np.asarray(self.spectrum_obj.get_spectrum())
        return spectrum[: region.length]

    def get_meta(self) -> DataMeta:
        wavelengths, wavelength_units = _to_wavelength_array_and_unit(self.spectrum_obj.get_wavelengths())
        bad_bands = self.spectrum_obj.get_bad_bands()
        return DataMeta(
            kind="spectrum",
            shape=(self.spectrum_obj.num_bands(),),
            elem_type=_safe_np_dtype(self.spectrum_obj.get_elem_type()),
            wavelengths=wavelengths,
            wavelength_units=wavelength_units or self.spectrum_obj.get_wavelength_units(),
            bad_bands=np.asarray(bad_bands) if bad_bands is not None else None,
        )

    def get_region_meta(self, region: DataRegion) -> RegionMeta:
        return _derive_region_meta(self.get_meta(), region)

    def is_same_external_handle(self, other: "ExternalHandle") -> bool:
        if not isinstance(other, ExternalSpectrumHandle):
            return False

        spectrum_id = self.spectrum_obj.get_id()
        other_spectrum_id = other.spectrum_obj.get_id()
        if spectrum_id is None or other_spectrum_id is None:
            return False

        return spectrum_id == other_spectrum_id


@dataclass
class ExternalSpectralLibraryHandle:
    lib_obj: "SpectralLibrary"
    kind: InputKind = "spectra_list"

    def read_region(self, region: DataRegion) -> np.ndarray:
        if not isinstance(region, SpectraBatchRef):
            raise TypeError(f"Spectral library external read requires SpectraBatchRef, got {type(region)}")
        rows: list[np.ndarray] = []
        for i in range(region.i0, region.i1):
            rows.append(np.asarray(self.lib_obj.get_spectrum(i).get_spectrum()))
        if not rows:
            first_dtype = self.get_meta().elem_type
            return np.empty((0, region.length), dtype=first_dtype)
        stacked = np.stack(rows, axis=0)
        if stacked.shape[1] != region.length:
            raise ValueError(
                f"Spectral library chunk length mismatch: expected={region.length}, got={stacked.shape[1]}"
            )
        return stacked

    def get_meta(self) -> DataMeta:
        num_spectra = int(self.lib_obj.num_spectra())
        if num_spectra == 0:
            return DataMeta(
                kind="spectra_list",
                shape=(0, 0),
                elem_type=np.dtype("float32"),
            )
        first = self.lib_obj.get_spectrum(0)
        wavelengths, wavelength_units = _to_wavelength_array_and_unit(first.get_wavelengths())
        bad_bands = first.get_bad_bands()
        return DataMeta(
            kind="spectra_list",
            shape=(num_spectra, first.num_bands()),
            elem_type=_safe_np_dtype(first.get_elem_type()),
            wavelengths=wavelengths,
            wavelength_units=wavelength_units or first.get_wavelength_units(),
            bad_bands=np.asarray(bad_bands) if bad_bands is not None else None,
        )

    def get_region_meta(self, region: DataRegion) -> RegionMeta:
        return _derive_region_meta(self.get_meta(), region)

    def is_same_external_handle(self, other: "ExternalHandle") -> bool:
        if not isinstance(other, ExternalSpectralLibraryHandle):
            return False

        lib_id = self.lib_obj.get_id()
        other_lib_id = other.lib_obj.get_id()
        if lib_id is None or other_lib_id is None:
            return False

        return lib_id == other_lib_id


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
    bad_bands: Optional[np.ndarray] = None  # 0's are bad bands, 1's are good bands
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
    Describes storage that should be allocated for a future output.

    This is a planning-time object. The storage service turns it into a
    `DataRef` plus its service-owned lifetime record.
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
    # When omitted, the stage/planner decides the policy for this output.
    delete_policy: Optional[DeletePolicy] = None


@dataclass
class StorageLeaseRecord:
    """
    Service-owned lifetime state for a managed storage object.

    `delete_policy` is the retention rule we want to enforce eventually.
    `deletion_state` is the current runtime status while the object moves
    through that lifecycle.
    """

    ref_id: str
    backend_kind: str
    owner_plan_id: Optional[str] = None
    planned_consumer_plan_ids: set[str] = dataclass_field(default_factory=set)
    borrowers: Dict[str, int] = dataclass_field(default_factory=dict)
    pins: Dict[str, int] = dataclass_field(default_factory=dict)
    producer_state: ProducerState = ProducerState.WRITING
    # Policy answers "should we reclaim this when it becomes safe?"
    delete_policy: DeletePolicy = DeletePolicy.KEEP
    # State answers "what has happened so far in the deletion lifecycle?"
    deletion_state: DeletionState = DeletionState.LIVE
    external_owned: bool = False


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
