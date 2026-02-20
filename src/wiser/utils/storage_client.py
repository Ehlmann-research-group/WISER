from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Literal, Optional

import numpy as np
import zarr

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
    _shared_mem_handles: Dict[str, SharedMemory] = field(default_factory=dict, init=False, repr=False)

    def close(self) -> None:
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

    def read_data_ref(self, ref: DataRef) -> DataRef:
        desc = self.service.get_access(ref, region=None, mode="r")
        return desc.ref

    def read_data(self, ref: DataRef) -> tuple[np.ndarray, RegionMeta]:
        """
        Read the whole object for `ref`.

        For `RamAccessDescriptor`, the returned array is a shared-memory view
        (not a copy), backed by an attached `SharedMemory` segment.
        """
        desc = self.service.get_access(ref, region=None, mode="r")
        if isinstance(
            desc, (JsonDiskAccessDescriptor, JsonRamAccessDescriptor, ExternalJsonRamAccessDescriptor)
        ):
            raise TypeError("read_data not supported for JSON; use read_json_value")

        whole_region = self._whole_region_from_meta(desc.meta)
        region_meta = self.service.get_region_meta(desc.ref, whole_region)

        if isinstance(desc, RamAccessDescriptor):
            arr = self._read_ram_array_view(desc.ref.uri)
            arr_region = self._read_region_from_array(arr, whole_region)
            return np.asarray(arr_region), region_meta

        if isinstance(desc, ExternalRamAccessDescriptor):
            arr = self._read_shared_mem_descriptor_view(desc.shared_mem)
            arr_region = self._read_region_from_array(arr, whole_region)
            return np.asarray(arr_region), region_meta

        if isinstance(desc, ExternalDiskAccessDescriptor):
            arr = self._read_external_region(desc.ref.ref_id, whole_region)
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
        if not uri.startswith("mem://"):
            raise ValueError(f"Expected mem:// uri, got: {uri}")
        try:
            descriptor = self.service.ram_objects[uri]
        except KeyError as e:
            raise KeyError(f"No RAM object for uri={uri}") from e
        if not isinstance(descriptor, SharedMemArrayDescriptor):
            raise TypeError(f"RAM object for uri={uri} is not a SharedMemArrayDescriptor")
        return descriptor

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

    def _read_external_region(self, ref_id: str, region: DataRegion) -> Any:
        try:
            handle = self.service.external_handles[ref_id]
        except KeyError as e:
            raise KeyError(f"No external handle for ref_id={ref_id}") from e
        return handle.read_region(region)

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
        return self.service.get_meta(ref)

    def get_region_meta(self, ref: DataRef, region: DataRegion) -> RegionMeta:
        return self.service.get_region_meta(ref, region)

    def get_access(
        self,
        ref: DataRef,
        region: Optional[DataRegion],
        mode: Literal["r", "rw"] = "r",
    ) -> AccessDescriptor:
        return self.service.get_access(ref, region, mode=mode)

    def read_region(self, ref: DataRef, region: DataRegion) -> tuple[np.ndarray, RegionMeta]:
        desc = self.service.get_access(ref, region, mode="r")
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
            arr = self._read_external_region(desc.ref.ref_id, region)
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
        desc = self.service.get_access(ref, region, mode="rw")
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
        desc = self.service.get_access(ref, region=None, mode="rw")
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
        raise TypeError("StorageClient.write_data currently supports only memmap and zarr access descriptors")

    def read_json_value(self, ref: DataRef) -> Any:
        desc = self.service.get_access(ref, region=None, mode="r")
        if isinstance(desc, ExternalJsonRamAccessDescriptor):
            return desc.value
        if isinstance(desc, JsonRamAccessDescriptor):
            return desc.value
        if not isinstance(desc, JsonDiskAccessDescriptor):
            raise TypeError("read_json_value requires a JSON ref")
        return self.service.read_json_value(desc.ref)

    def write_json_value(self, ref: DataRef, value: Any) -> None:
        desc = self.service.get_access(ref, region=None, mode="rw")
        if isinstance(desc, ExternalJsonRamAccessDescriptor):
            raise PermissionError(
                f"write_json_value is not allowed for external/read-only refs: {desc.ref.ref_id}"
            )
        if isinstance(desc, JsonRamAccessDescriptor):
            self.service.write_json_ram_value(desc.ref, value)
            return
        if not isinstance(desc, JsonDiskAccessDescriptor):
            raise TypeError("write_json_value requires a JSON ref")
        self.service.write_json_value(desc.ref, value)

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
