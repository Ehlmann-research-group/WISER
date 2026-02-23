from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Literal, Optional, Tuple
import uuid

import numpy as np
import zarr
from wiser.raster.dataset import RasterDataSet
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

    def read_data(self, ref: DataRef) -> tuple[np.ndarray, RegionMeta]:
        """
        Read the whole object for `ref`.

        For `RamAccessDescriptor`, the returned array is a shared-memory view
        (not a copy), backed by an attached `SharedMemory` segment.
        """
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=None, mode="r")
        if isinstance(
            desc, (JsonDiskAccessDescriptor, JsonRamAccessDescriptor, ExternalJsonRamAccessDescriptor)
        ):
            raise TypeError("read_data not supported for JSON; use read_json_value")

        whole_region = self._whole_region_from_meta(desc.meta)
        region_meta: RegionMeta = self._rpc_call("get_region_meta", ref=desc.ref, region=whole_region)

        if isinstance(desc, RamAccessDescriptor):
            arr = self._read_ram_array_view(desc.ref.uri)
            arr_region = self._read_region_from_array(arr, whole_region)
            return np.asarray(arr_region), region_meta

        if isinstance(desc, ExternalRamAccessDescriptor):
            arr = self._read_shared_mem_descriptor_view(desc.shared_mem)
            arr_region = self._read_region_from_array(arr, whole_region)
            return np.asarray(arr_region), region_meta

        if isinstance(desc, ExternalDiskAccessDescriptor):
            arr = self._read_external_region(desc.ref, whole_region)
            # TODO: Don't copy for GDAL-backed datasets.
            return np.array(arr, copy=True), region_meta

        if isinstance(desc, MemmapAccessDescriptor):
            mm = np.load(str(desc.path), mmap_mode="r")
            arr = self._read_region_from_array(mm, whole_region)
            return np.array(arr, copy=True), region_meta

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r")
            arr = self._read_region_from_array(grp[desc.array_name], whole_region)
            return np.array(arr, copy=True), region_meta

        raise ValueError(f"Unknown access descriptor: {type(desc)}")

    def read_region(self, ref: DataRef, region: DataRegion) -> tuple[np.ndarray, RegionMeta]:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=region, mode="r")
        if desc.region_meta is None:
            raise ValueError("Region metadata is required for region reads")
        if isinstance(
            desc, (JsonDiskAccessDescriptor, JsonRamAccessDescriptor, ExternalJsonRamAccessDescriptor)
        ):
            raise TypeError("read_region not supported for JSON; use read_json_value")

        if isinstance(desc, ExternalRamAccessDescriptor):
            arr = self._read_shared_mem_descriptor_view(desc.shared_mem)
            arr = self._read_region_from_array(arr, region)
            return np.asarray(arr), desc.region_meta

        if isinstance(desc, ExternalDiskAccessDescriptor):
            arr = self._read_external_region(desc.ref, region)
            # TODO: Don't copy for GDAL-backed datasets.
            return np.array(arr, copy=True), desc.region_meta

        if isinstance(desc, RamAccessDescriptor):
            arr = self._read_ram_array_view(desc.ref.uri)
            arr = self._read_region_from_array(arr, region)
            return np.asarray(arr), desc.region_meta

        if isinstance(desc, MemmapAccessDescriptor):
            mm = np.load(str(desc.path), mmap_mode="r")
            arr = self._read_region_from_array(mm, region)
            return np.array(arr, copy=True), desc.region_meta

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r")
            arr = self._read_region_from_array(grp[desc.array_name], region)
            return np.array(arr, copy=True), desc.region_meta

        raise ValueError(f"Unknown access descriptor: {type(desc)}")

    def write_region(self, ref: DataRef, region: DataRegion, value: Any) -> None:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=region, mode="rw")
        if isinstance(desc, MemmapAccessDescriptor):
            arr = np.load(str(desc.path), mmap_mode="r+")
            self._write_region_into_array(arr, region, value)
            if hasattr(arr, "flush"):
                arr.flush()
            return
        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r+")
            self._write_region_into_array(grp[desc.array_name], region, value)
            return
        raise TypeError(
            "StorageClient.write_region currently supports only memmap and zarr access descriptors"
        )

    def write_data(self, ref: DataRef, value: Any) -> None:
        desc: AccessDescriptor = self._rpc_call("get_access", ref=ref, region=None, mode="rw")
        if isinstance(desc, RamAccessDescriptor):
            arr = self._read_ram_array_view(desc.ref.uri)
            arr[...] = value
            return
        if isinstance(desc, MemmapAccessDescriptor):
            arr = np.load(str(desc.path), mmap_mode="r+")
            arr[...] = value
            if hasattr(arr, "flush"):
                arr.flush()
            return
        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r+")
            grp[desc.array_name][...] = value
            return
        raise TypeError(
            "StorageClient.write_data currently supports RAM, memmap, and zarr access descriptors."
            f"\nIt does not support {type(desc)} descriptors."
        )

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

    def _read_ram_array_view(self, uri: str) -> np.ndarray:
        descriptor = self._get_ram_descriptor(uri)
        return self._read_shared_mem_descriptor_view(descriptor)

    def _read_shared_mem_descriptor_view(self, descriptor: SharedMemArrayDescriptor) -> np.ndarray:
        shm = self._get_or_attach_shm(descriptor)
        return np.ndarray(
            shape=descriptor.shape,
            dtype=np.dtype(descriptor.dtype_str),
            buffer=shm.buf,
            strides=descriptor.strides,
        )

    def _get_ram_descriptor(self, uri: str) -> SharedMemArrayDescriptor:
        return self._rpc_call("get_ram_descriptor", uri=uri)

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

    def get_meta(self, ref: DataRef) -> DataMeta:
        return self._rpc_call("get_meta", ref=ref)

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
