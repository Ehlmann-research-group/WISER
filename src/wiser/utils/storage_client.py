from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import uuid

import numpy as np
import zarr
from wiser.raster.dataset import RasterDataSet
from wiser.raster.serializable import SerializedForm
from wiser.raster.spectrum import Spectrum
from wiser.raster.dataset_impl import (
    ASC_GDALRasterDataImpl,
    ENVI_GDALRasterDataImpl,
    GDALRasterDataImpl,
    GTiff_GDALRasterDataImpl,
    JP2_GDALRasterDataImpl,
    NetCDF_GDALRasterDataImpl,
    PDS3_GDALRasterDataImpl,
    PDS4_GDALRasterDataImpl,
)
from wiser.raster.envi_spectral_library import ENVISpectralLibrary

from .primitives import (
    DataMeta,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    RegionMeta,
    SpectraBatchRef,
    SpectrumRef,
)
from .storage_service import (
    AccessDescriptor,
    ExternalDiskAccessDescriptor,
    ExternalJsonRamAccessDescriptor,
    ExternalRamAccessDescriptor,
    JsonDiskAccessDescriptor,
    JsonRamAccessDescriptor,
    MemmapAccessDescriptor,
    RamAccessDescriptor,
    SharedMemArrayDescriptor,
    StorageService,
    ZarrAccessDescriptor,
)

if TYPE_CHECKING:
    from .task_system import WriteSpec


# ---------------------------------------------------------------------------
# ROI-backed spectra list (roi_proxy driver)
# ---------------------------------------------------------------------------

# Per-process cache of reconstructed RoiBackedSpectraList helpers, keyed by the
# ROI ref's ref_id.  A given worker process therefore reopens the source
# dataset (GDAL handle) at most once per ROI ref, regardless of how many
# SpectraBatchRef tiles it ends up reading.
# No lock needed: stages use ProcessPoolExecutor so each worker is a separate
# process with its own private copy of this dict.
_ROI_HELPER_CACHE: Dict[str, "RoiBackedSpectraList"] = {}


class RoiBackedSpectraList:
    """
    Helper that exposes the pixels of a Region-of-Interest as a virtual
    ``(N, b)`` spectra list backed by a source :class:`RasterDataSet`.

    Constructed lazily in worker processes from the rectangle decomposition
    sent by the storage service via the ``roi_proxy`` external-params driver.
    See :class:`wiser.utils.primitives.ExternalRoiHandle` (built in the main
    process at registration time) for the matching producer side.

    Index space:
        Spectra are numbered in rectangle-traversal order. Within each
        rectangle, pixels are enumerated in row-major (y-then-x) order, the
        same convention used by :meth:`RasterDataSet.get_all_bands_at_rect`
        after ``transpose(1, 2, 0).reshape(-1, b)``.

    Thread-safety:
        ``read_batch`` is safe to call from multiple threads concurrently
        provided the underlying :class:`RasterDataSet` is itself thread-safe.
    """

    def __init__(
        self,
        rects: List[List[int]],
        prefix_sums: List[int],
        source_dataset: RasterDataSet,
    ) -> None:
        # _rects: shape (R, 4), dtype intp.  Each row is one axis-aligned
        # rectangle in ABSOLUTE image coordinates:
        #   col 0 — x_start  (left,   inclusive)
        #   col 1 — x_end    (right,  inclusive)
        #   col 2 — y_start  (top,    inclusive)
        #   col 3 — y_end    (bottom, inclusive)
        self._rects: np.ndarray = (
            np.asarray(rects, dtype=np.intp).reshape(-1, 4) if rects else np.empty((0, 4), dtype=np.intp)
        )
        # _prefix_sums: shape (R+1,), dtype intp.
        #   _prefix_sums[r]   = first spectra index belonging to rect r
        #   _prefix_sums[r+1] = one past the last spectra index of rect r
        #   _prefix_sums[-1]  = N (total ROI pixel count)
        self._prefix_sums: np.ndarray = np.asarray(prefix_sums, dtype=np.intp)
        if self._prefix_sums.size == 0:
            self._prefix_sums = np.array([0], dtype=np.intp)
        self._source: RasterDataSet = source_dataset

    @property
    def total_pixels(self) -> int:
        return int(self._prefix_sums[-1])

    def read_batch(self, i0: int, i1: int) -> np.ndarray:
        """
        Read spectra ``[i0, i1)`` from the source dataset using the
        precomputed rectangle decomposition.

        Returns an ``(i1 - i0, b)`` ``ndarray`` in source dtype.  Raw values
        only — masking of nodata/bad-bands is the caller's responsibility
        (handled by ``_mask_data_ignore_and_bad_bands`` on the
        :class:`StorageClient` read path).
        """
        i0 = int(i0)
        i1 = int(i1)
        n_out = i1 - i0
        if n_out <= 0:
            return np.empty((0, self._source.get_num_bands()), dtype=np.float64)

        total = self.total_pixels
        if i0 < 0 or i1 > total:
            raise IndexError(
                f"RoiBackedSpectraList batch [{i0}, {i1}) out of range for ROI with {total} pixels"
            )

        # Binary-search the prefix sums to find the first and last covering rects.
        first_rect = int(np.searchsorted(self._prefix_sums, i0, side="right")) - 1
        last_rect = int(np.searchsorted(self._prefix_sums, i1 - 1, side="right")) - 1

        result_chunks: list = []
        for r in range(first_rect, last_rect + 1):
            rect = self._rects[r]
            abs_x_start = int(rect[0])
            abs_x_end = int(rect[1])
            abs_y_start = int(rect[2])
            abs_y_end = int(rect[3])
            dx = abs_x_end - abs_x_start + 1
            dy = abs_y_end - abs_y_start + 1

            # GDAL/RasterDataSet read: returns (b, dy, dx).  filter_bad_values
            # is False so callers downstream see the raw nodata sentinels and
            # can apply consistent masking via the standard region path.
            arr_byx = self._source.get_all_bands_at_rect(
                abs_x_start, abs_y_start, dx, dy, filter_bad_values=False
            )
            arr_byx = np.asarray(arr_byx)
            if arr_byx.ndim == 2:
                arr_byx = arr_byx[np.newaxis, :, :]

            # (b, dy, dx) -> (dy, dx, b) -> (dy*dx, b) in row-major order.
            arr_flat = arr_byx.transpose(1, 2, 0).reshape(-1, arr_byx.shape[0])

            rect_start = int(self._prefix_sums[r])
            rect_end = int(self._prefix_sums[r + 1])
            local_start = max(i0, rect_start) - rect_start
            local_end = min(i1, rect_end) - rect_start

            result_chunks.append(arr_flat[local_start:local_end, :])

        if not result_chunks:
            return np.empty((0, self._source.get_num_bands()), dtype=np.float64)
        return np.vstack(result_chunks)


@dataclass
class StorageClient:
    """
    Handles accessing and modifying storage from a worker process.
    """

    service: StorageService
    service_address: Tuple[str, int]
    service_authkey: bytes
    _conn: Optional[Connection] = field(default=None, init=False, repr=False)
    _shared_mem_handles: Dict[str, SharedMemory] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._connect_to_service()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None

        for shm in self._shared_mem_handles.values():
            try:
                shm.close()
            except Exception:
                pass
        self._shared_mem_handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _connect_to_service(self) -> None:
        if self.service_address is None or self.service_authkey is None:
            raise RuntimeError(
                "StorageClient requires service_address and service_authkey at construction time"
            )
        self._conn = Client(self.service_address, authkey=self.service_authkey)

    def _rpc_call(self, method: str, **params: Any) -> Any:
        if self._conn is None:
            raise RuntimeError("StorageClient is not connected to StorageService")

        request_id = uuid.uuid4().hex
        request = {"request_id": request_id, "method": method, "params": params}
        self._conn.send(request)
        response = self._conn.recv()

        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid RPC response type: {type(response).__name__}")
        response_request_id = response.get("request_id")
        if response_request_id != request_id:
            raise RuntimeError(
                "Mismatched RPC response request_id: " f"expected={request_id}, got={response_request_id}"
            )
        if response.get("ok") is True:
            return response.get("result")

        error = response.get("error")
        if not isinstance(error, dict):
            raise RuntimeError(f"Invalid RPC error response for method={method}: {response!r}")
        code = error.get("code", "UNKNOWN_ERROR")
        message = error.get("message", "RPC call failed")
        details = error.get("details")
        raise RuntimeError(f"Storage RPC {method} failed [{code}]: {message} details={details}")

    def read_data_ref(self, ref: DataRef) -> DataRef:
        return self._rpc_call("read_data_ref", ref=ref)

    def get_access(
        self,
        ref: DataRef,
        region: Optional[DataRegion],
        mode: Literal["r", "rw"] = "r",
    ) -> AccessDescriptor:
        return self._rpc_call("get_access", ref=ref, region=region, mode=mode)

    def read_data(
        self,
        ref: DataRef,
        *,
        filter_data: bool = True,
    ) -> tuple[np.ndarray | np.ma.MaskedArray, RegionMeta]:
        """
        Read the whole object for `ref`.

        For RAM-backed refs, this returns a copied NumPy array. The client
        attaches to shared memory only for the duration of the read so worker
        functions do not leak attached shared-memory handles.

        Dimension conventions:
        - dataset refs return arrays shaped as [y][x][b]
        - spectrum refs return arrays shaped as [b]
        - spectra_list refs return arrays shaped as [i][b]
        """
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=None, mode="r")
        if isinstance(
            desc, (JsonDiskAccessDescriptor, JsonRamAccessDescriptor, ExternalJsonRamAccessDescriptor)
        ):
            raise TypeError("read_data not supported for JSON; use read_json_value")

        whole_region = self._whole_region_from_meta(desc.meta)
        region_meta: RegionMeta = self._rpc_call("get_region_meta", ref=desc.ref, region=whole_region)

        if isinstance(desc, RamAccessDescriptor):
            data = self._read_ram_array_copy(desc.ref.uri, whole_region)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data, region_meta, filter_data_ignore_and_bad_bands=filter_data
                ),
                region_meta,
            )

        if isinstance(desc, ExternalRamAccessDescriptor):
            data = self._read_shared_mem_descriptor_copy(desc.shared_mem, whole_region)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data, region_meta, filter_data_ignore_and_bad_bands=filter_data
                ),
                region_meta,
            )

        if isinstance(desc, ExternalDiskAccessDescriptor):
            arr = self._read_external_region(desc.ref, whole_region)
            # TODO: Don't copy for GDAL-backed datasets.
            data = np.array(arr, copy=True)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data, region_meta, filter_data_ignore_and_bad_bands=filter_data
                ),
                region_meta,
            )

        if isinstance(desc, MemmapAccessDescriptor):
            mm = np.load(str(desc.path), mmap_mode="r")
            arr = self._read_region_from_array(mm, whole_region)
            data = np.array(arr, copy=True)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data, region_meta, filter_data_ignore_and_bad_bands=filter_data
                ),
                region_meta,
            )

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r")
            arr = self._read_region_from_array(grp[desc.array_name], whole_region)
            data = np.array(arr, copy=True)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data, region_meta, filter_data_ignore_and_bad_bands=filter_data
                ),
                region_meta,
            )

        raise ValueError(f"Unknown access descriptor: {type(desc)}")

    def read_region(
        self,
        ref: DataRef,
        region: DataRegion,
        *,
        filter_data: bool = True,
    ) -> tuple[np.ndarray | np.ma.MaskedArray, RegionMeta]:
        """
        Read the specified region for `ref`.

        Dimension conventions:
        - dataset regions return arrays shaped as [y][x][b]
        - spectrum regions return arrays shaped as [b]
        - spectra_list regions return arrays shaped as [i][b]
        """
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=region, mode="r")
        if desc.region_meta is None:
            raise ValueError("Region metadata is required for region reads")
        if isinstance(
            desc, (JsonDiskAccessDescriptor, JsonRamAccessDescriptor, ExternalJsonRamAccessDescriptor)
        ):
            raise TypeError("read_region not supported for JSON; use read_json_value")

        if isinstance(desc, ExternalRamAccessDescriptor):
            data = self._read_shared_mem_descriptor_copy(desc.shared_mem, region)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data,
                    desc.region_meta,
                    filter_data_ignore_and_bad_bands=filter_data,
                ),
                desc.region_meta,
            )

        if isinstance(desc, ExternalDiskAccessDescriptor):
            arr = self._read_external_region(desc.ref, region)
            # TODO: Don't copy for GDAL-backed datasets.
            data = np.array(arr, copy=True)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data,
                    desc.region_meta,
                    filter_data_ignore_and_bad_bands=filter_data,
                ),
                desc.region_meta,
            )

        if isinstance(desc, RamAccessDescriptor):
            data = self._read_ram_array_copy(desc.ref.uri, region)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data,
                    desc.region_meta,
                    filter_data_ignore_and_bad_bands=filter_data,
                ),
                desc.region_meta,
            )

        if isinstance(desc, MemmapAccessDescriptor):
            mm = np.load(str(desc.path), mmap_mode="r")
            arr = self._read_region_from_array(mm, region)
            data = np.array(arr, copy=True)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data,
                    desc.region_meta,
                    filter_data_ignore_and_bad_bands=filter_data,
                ),
                desc.region_meta,
            )

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r")
            arr = self._read_region_from_array(grp[desc.array_name], region)
            data = np.array(arr, copy=True)
            return (
                self._mask_data_ignore_and_bad_bands(
                    data,
                    desc.region_meta,
                    filter_data_ignore_and_bad_bands=filter_data,
                ),
                desc.region_meta,
            )

        raise ValueError(f"Unknown access descriptor: {type(desc)}")

    def write_region(self, ref: DataRef, region: DataRegion, value: Any) -> None:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=region, mode="rw")
        self._write_access_value(desc=desc, value=value, region=region, op_name="write_region")

    def write_data(self, ref: DataRef, value: Any) -> None:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=None, mode="rw")
        self._write_access_value(desc=desc, value=value, region=None, op_name="write_data")

    def write_spec(self, write_spec: "WriteSpec", value: Any) -> None:
        region = write_spec.region
        desc: AccessDescriptor = self._rpc_call(
            "get_access",
            ref=write_spec.ref,
            region=region,
            mode="rw",
        )
        self._write_access_value(desc=desc, value=value, region=region, op_name="write_spec")

    def read_json_value(self, ref: DataRef) -> Any:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=None, mode="r")
        if isinstance(desc, ExternalJsonRamAccessDescriptor):
            return desc.value
        if isinstance(desc, JsonRamAccessDescriptor):
            return desc.value
        if not isinstance(desc, JsonDiskAccessDescriptor):
            raise TypeError("read_json_value requires a JSON ref")
        return self._rpc_call("read_json_value", ref=desc.ref)

    def write_json_value(self, ref: DataRef, value: Any) -> None:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=None, mode="rw")
        if isinstance(desc, ExternalJsonRamAccessDescriptor):
            raise PermissionError(
                f"write_json_value is not allowed for external/read-only refs: {desc.ref.ref_id}"
            )
        if isinstance(desc, JsonRamAccessDescriptor):
            self._rpc_call("write_json_ram_value", ref=desc.ref, value=value)
            return
        if not isinstance(desc, JsonDiskAccessDescriptor):
            raise TypeError("write_json_value requires a JSON ref")
        self._rpc_call("write_json_value", ref=desc.ref, value=value)

    def _read_ram_array_copy(self, uri: str, region: DataRegion) -> np.ndarray:
        descriptor = self._get_ram_descriptor(uri)
        return self._read_shared_mem_descriptor_copy(descriptor, region)

    def _read_shared_mem_descriptor_copy(
        self,
        descriptor: SharedMemArrayDescriptor,
        region: DataRegion,
    ) -> np.ndarray:
        return self._with_shared_mem_array(
            descriptor,
            lambda arr: np.array(self._read_region_from_array(arr, region), copy=True),
        )

    def _get_ram_descriptor(self, uri: str) -> SharedMemArrayDescriptor:
        return self._rpc_call("get_ram_descriptor", uri=uri)

    def get_external_object_serialized_form(self, ref: DataRef) -> SerializedForm:
        """
        Fetch the SerializedForm for a registered external object.

        This lower-level helper is used when a worker needs the serialized
        description of an external raster or spectrum object held by the
        StorageService.
        """
        return self._rpc_call("get_external_object_serialized_form", ref=ref)

    def reconstruct_external_object(self, ref: DataRef) -> Union[RasterDataSet, Spectrum]:
        """
        Reconstruct an external raster or spectrum object inside the caller's process.

        The object is recreated from the SerializedForm returned by the
        StorageService for an external handle associated with `ref`.
        """
        serialized_form = self.get_external_object_serialized_form(ref)
        serializable_class = serialized_form.get_serializable_class()
        reconstructed = serializable_class.deserialize_into_class(serialized_form)

        if isinstance(reconstructed, RasterDataSet):
            return reconstructed
        if isinstance(reconstructed, Spectrum):
            return reconstructed

        raise TypeError(
            "External object reconstruction returned unsupported type: " f"{type(reconstructed).__name__}"
        )

    def _with_shared_mem_array(
        self,
        descriptor: SharedMemArrayDescriptor,
        fn: Callable[[np.ndarray], Any],
    ) -> Any:
        shm = SharedMemory(name=descriptor.name, create=False)
        try:
            arr = np.ndarray(
                shape=descriptor.shape,
                dtype=np.dtype(descriptor.dtype_str),
                buffer=shm.buf,
                strides=descriptor.strides,
            )
            return fn(arr)
        finally:
            shm.close()

    def _get_or_attach_shm(self, descriptor: SharedMemArrayDescriptor) -> SharedMemory:
        shm = self._shared_mem_handles.get(descriptor.name)
        if shm is not None:
            return shm
        shm = SharedMemory(name=descriptor.name, create=False)
        self._shared_mem_handles[descriptor.name] = shm
        return shm

    def _read_region_from_array(self, arr: Any, region: DataRegion) -> Any:
        if isinstance(region, DatasetRegionRef):
            return arr[
                region.y0 : region.y1,
                region.x0 : region.x1,
                region.b0 : (region.b1 if region.b1 is not None else None),
            ]
        if isinstance(region, SpectrumRef):
            return arr[...]
        if isinstance(region, SpectraBatchRef):
            return arr[region.i0 : region.i1]
        raise TypeError(f"Unknown DataRegion type: {type(region)}")

    def _read_external_region(self, ref: DataRef, region: DataRegion) -> Any:
        params = ref.external_params
        if params is None:
            raise ValueError(
                f"External disk ref_id={ref.ref_id} is missing external_params needed for reconstruction"
            )

        if params.family == "dataset":
            dataset = self._reconstruct_external_dataset(ref)
            if not isinstance(region, DatasetRegionRef):
                raise TypeError(
                    f"External dataset read requires DatasetRegionRef, "
                    f"got {type(region)} for ref_id={ref.ref_id}"
                )
            # We set data ignore value to false because we want accessing
            # from a dataset and accessing from a shared memory array to
            # return an np.ndarray
            arr_by_band = dataset.get_image_data_subset(
                x=region.x0,
                y=region.y0,
                band=region.b0,
                dx=region.x1 - region.x0,
                dy=region.y1 - region.y0,
                dband=(region.b1 - region.b0) if region.b1 is not None else (dataset.num_bands() - region.b0),
                filter_data_ignore_value=False,
            )
            return np.asarray(arr_by_band).transpose(1, 2, 0)

        if params.family == "spectra_list":
            if params.driver != "envi_sli":
                raise ValueError(
                    f"Unsupported spectra_list external driver={params.driver} for ref_id={ref.ref_id}"
                )
            path = params.kwargs.get("path")
            if path is None:
                raise ValueError(f"Missing 'path' in external_params.kwargs for ref_id={ref.ref_id}")
            lib = ENVISpectralLibrary(str(path))
            if not isinstance(region, SpectraBatchRef):
                raise TypeError(
                    f"External spectra_list read requires SpectraBatchRef, "
                    f"got {type(region)} for ref_id={ref.ref_id}"
                )
            rows: list[np.ndarray] = []
            for i in range(region.i0, region.i1):
                rows.append(np.asarray(lib.get_spectrum(i).get_spectrum()))
            if not rows:
                return np.empty((0, region.length), dtype=lib.get_elem_type())
            stacked = np.stack(rows, axis=0)
            if stacked.shape[1] != region.length:
                raise ValueError(
                    f"Spectra list chunk length mismatch: expected={region.length}, got={stacked.shape[1]}"
                )
            return stacked

        if params.family == "array":
            if params.driver == "memmap":
                path = params.kwargs.get("path")
                if path is None:
                    raise ValueError(f"Missing 'path' in external_params.kwargs for ref_id={ref.ref_id}")
                arr = np.load(str(path), mmap_mode="r")
                return self._read_region_from_array(arr, region)
            if params.driver == "zarr":
                store_path = params.kwargs.get("store_path")
                array_name = params.kwargs.get("array_name", "data")
                if store_path is None:
                    raise ValueError(
                        f"Missing 'store_path' in external_params.kwargs for ref_id={ref.ref_id}"
                    )
                store = zarr.DirectoryStore(str(store_path))
                grp = zarr.open_group(store=store, mode="r")
                return self._read_region_from_array(grp[array_name], region)
            raise ValueError(f"Unsupported array external driver={params.driver} for ref_id={ref.ref_id}")

        raise ValueError(f"Unsupported external_params.family={params.family} for ref_id={ref.ref_id}")

    def _reconstruct_external_dataset(self, ref: DataRef) -> RasterDataSet:
        params = ref.external_params
        if params is None:
            raise ValueError(f"External dataset ref_id={ref.ref_id} is missing external_params")
        if params.family != "dataset":
            raise ValueError(
                f"Expected external_params.family='dataset' for ref_id={ref.ref_id}, got {params.family}"
            )
        path = params.kwargs.get("path")
        if path is None:
            raise ValueError(f"Missing 'path' in external_params.kwargs for ref_id={ref.ref_id}")

        if params.driver == "netcdf_gdal":
            impls = NetCDF_GDALRasterDataImpl.try_load_file(
                str(path),
                subdataset_name=params.kwargs.get("subdataset_name"),
                interactive=False,
            )
        elif params.driver == "envi_gdal":
            impls = ENVI_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        elif params.driver == "gtiff_gdal":
            impls = GTiff_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        elif params.driver == "asc_gdal":
            impls = ASC_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        elif params.driver == "pds3_gdal":
            impls = PDS3_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        elif params.driver == "pds4_gdal":
            impls = PDS4_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        elif params.driver == "jp2_gdal":
            impls = JP2_GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        elif params.driver == "gdal_generic":
            impls = GDALRasterDataImpl.try_load_file(str(path), interactive=False)
        else:
            raise ValueError(f"Unsupported dataset external driver={params.driver} for ref_id={ref.ref_id}")

        if not impls:
            raise ValueError(
                f"Driver {params.driver} could not load external dataset path={path} for ref_id={ref.ref_id}"
            )
        return RasterDataSet(impls[0], data_cache=None)

    def _write_region_into_array(self, arr: Any, region: DataRegion, value: Any) -> None:
        if isinstance(region, DatasetRegionRef):
            arr[
                region.y0 : region.y1,
                region.x0 : region.x1,
                region.b0 : (region.b1 if region.b1 is not None else None),
            ] = value
            return
        if isinstance(region, SpectrumRef):
            arr[...] = value
            return
        if isinstance(region, SpectraBatchRef):
            arr[region.i0 : region.i1] = value
            return
        raise TypeError(f"Unknown DataRegion type: {type(region)}")

    def _write_access_value(
        self,
        *,
        desc: AccessDescriptor,
        value: Any,
        region: Optional[DataRegion],
        op_name: str,
    ) -> None:
        if isinstance(desc, RamAccessDescriptor):
            descriptor = self._get_ram_descriptor(desc.ref.uri)
            self._with_shared_mem_array(
                descriptor,
                # Lambdas cannot contain `arr[...] = value`, so use the underlying
                # item-assignment method when writing the full shared-memory array.
                lambda arr: arr.__setitem__(Ellipsis, value)
                if region is None
                else self._write_region_into_array(arr, region, value),
            )
            return

        if isinstance(desc, MemmapAccessDescriptor):
            arr = np.load(str(desc.path), mmap_mode="r+")
            if region is None:
                arr[...] = value
            else:
                self._write_region_into_array(arr, region, value)
            if hasattr(arr, "flush"):
                arr.flush()
            return

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r+")
            if region is None:
                grp[desc.array_name][...] = value
            else:
                self._write_region_into_array(grp[desc.array_name], region, value)
            return

        raise TypeError(
            f"StorageClient.{op_name} currently supports RAM, memmap, and zarr access descriptors."
            f"\nIt does not support {type(desc)} descriptors."
        )

    def _mask_data_ignore_and_bad_bands(
        self,
        data: np.ndarray | np.ma.MaskedArray,
        region_meta: RegionMeta,
        filter_data_ignore_and_bad_bands: bool,
    ) -> np.ndarray | np.ma.MaskedArray:
        """
        Mask `nodata` values and bad bands for region reads.

        Bad-band metadata is treated as describing the spectral axis, which is
        expected to be the last axis for all supported region shapes:
        - dataset: [y][x][b]
        - spectrum: [b]
        - spectra_list: [i][b]
        """
        if not filter_data_ignore_and_bad_bands:
            return data

        arr = np.ma.array(data, copy=False)
        raw = np.ma.getdata(arr)
        combined_mask = np.ma.getmaskarray(arr)

        if region_meta.nodata is not None:
            if np.isnan(region_meta.nodata):
                nodata_mask = np.isnan(raw)
            else:
                nodata_mask = raw == region_meta.nodata
            combined_mask = np.ma.mask_or(combined_mask, nodata_mask)

        if region_meta.bad_bands is not None:
            if raw.ndim == 0:
                raise ValueError("bad_bands metadata requires array data with at least one dimension")

            band_count = raw.shape[-1]
            bad_band_mask = np.asarray(region_meta.bad_bands) == 0
            if bad_band_mask.shape != (band_count,):
                raise ValueError(
                    f"Expected bad_bands shape {(band_count,)} "
                    f" for region {type(region_meta.region).__name__}, "
                    f"got {bad_band_mask.shape}"
                )

            # Build a mask shape that spans only the spectral axis and uses
            # singleton dimensions for every leading axis, so NumPy can
            # broadcast the 1D bad-band mask across the whole region.
            broadcast_shape = (1,) * (raw.ndim - 1) + (band_count,)
            bad_band_mask = bad_band_mask.reshape(broadcast_shape)
            combined_mask = np.ma.mask_or(combined_mask, np.broadcast_to(bad_band_mask, raw.shape))

        return np.ma.array(raw, mask=combined_mask, copy=False)

    def get_meta(self, ref: DataRef) -> DataMeta:
        return self._rpc_call("get_meta", ref=ref)

    def write_meta(self, ref: DataRef, meta: DataMeta) -> None:
        self._rpc_call("write_meta", ref=ref, meta=meta)

    def get_region_meta(self, ref: DataRef, region: DataRegion) -> RegionMeta:
        return self._rpc_call("get_region_meta", ref=ref, region=region)

    def _whole_region_from_meta(self, meta: DataMeta) -> DataRegion:
        if meta.kind == "dataset":
            if len(meta.shape) != 3:
                raise ValueError(f"Dataset meta.shape must be 3D, got {meta.shape}")
            return DatasetRegionRef(0, meta.shape[0], 0, meta.shape[1], 0, meta.shape[2])
        if meta.kind == "spectrum":
            if len(meta.shape) != 1:
                raise ValueError(f"Spectrum meta.shape must be 1D, got {meta.shape}")
            return SpectrumRef(length=meta.shape[0])
        if meta.kind == "spectra_list":
            if len(meta.shape) != 2:
                raise ValueError(f"Spectra list meta.shape must be 2D, got {meta.shape}")
            return SpectraBatchRef(i0=0, i1=meta.shape[0], length=meta.shape[1])

        if len(meta.shape) == 1:
            return SpectrumRef(length=meta.shape[0])
        if len(meta.shape) == 2:
            return SpectraBatchRef(i0=0, i1=meta.shape[0], length=meta.shape[1])
        if len(meta.shape) == 3:
            return DatasetRegionRef(0, meta.shape[0], 0, meta.shape[1], 0, meta.shape[2])
        raise TypeError(
            f"Cannot derive whole-data region for kind={meta.kind} shape={meta.shape}; "
            "use read_region with an explicit DataRegion"
        )
