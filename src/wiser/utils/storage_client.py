from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

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
    ExternalRamAccessDescriptor,
    JsonAccessDescriptor,
    MemmapAccessDescriptor,
    RamAccessDescriptor,
    StorageService,
    ZarrAccessDescriptor,
)


@dataclass
class StorageClient:
    service: StorageService

    def read_data_ref(self, ref: DataRef) -> DataRef:
        desc = self.service.get_access(ref, region=None, mode="r")
        return desc.ref

    def read_data(self, ref: DataRef) -> tuple[np.ndarray, RegionMeta]:
        desc = self.service.get_access(ref, region=None, mode="r")
        if isinstance(desc, JsonAccessDescriptor):
            raise TypeError("read_data not supported for JSON; use read_json_value")

        whole_region = self._whole_region_from_meta(desc.meta)
        region_meta = self.service.get_region_meta(desc.ref, whole_region)

        # TODO: Make this function open up these descriptors.
        # ExternalDisk should be reconstructed, external ram and ram should be shared memory
        if isinstance(desc, (ExternalRamAccessDescriptor, ExternalDiskAccessDescriptor, RamAccessDescriptor)):
            arr = self.service.read_region(desc.ref, whole_region)
            return np.asarray(arr), region_meta

        if isinstance(desc, MemmapAccessDescriptor):
            mm = np.load(str(desc.path), mmap_mode="r")
            arr = self.service._read_region_from_array(mm, whole_region)
            return np.asarray(arr), region_meta

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r")
            arr = self.service._read_region_from_array(grp[desc.array_name], whole_region)
            return np.asarray(arr), region_meta

        raise ValueError(f"Unknown access descriptor: {type(desc)}")

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
        if isinstance(desc, JsonAccessDescriptor):
            raise TypeError("read_region not supported for JSON; use read_json_value")

        if desc.ref.source == "external":
            arr = self.service.read_region(desc.ref, region)
            return np.asarray(arr), desc.region_meta

        if isinstance(desc, RamAccessDescriptor):
            arr = self.service.read_region(desc.ref, region)
            return np.asarray(arr), desc.region_meta

        if isinstance(desc, MemmapAccessDescriptor):
            mm = np.load(str(desc.path), mmap_mode="r")
            arr = self.service._read_region_from_array(mm, region)
            return np.asarray(arr), desc.region_meta

        if isinstance(desc, ZarrAccessDescriptor):
            store = zarr.DirectoryStore(str(desc.store_path))
            grp = zarr.open_group(store=store, mode="r")
            arr = self.service._read_region_from_array(grp[desc.array_name], region)
            return np.asarray(arr), desc.region_meta

        raise ValueError(f"Unknown access descriptor: {type(desc)}")

    def write_region(self, ref: DataRef, region: DataRegion, value: Any) -> None:
        desc = self.service.get_access(ref, region, mode="rw")
        if isinstance(desc, JsonAccessDescriptor):
            raise TypeError("write_region not supported for JSON; use write_json_value")
        self.service.write_region(desc.ref, region, value)

    def write_data(self, ref: DataRef, value: Any) -> None:
        desc = self.service.get_access(ref, region=None, mode="rw")
        if isinstance(desc, JsonAccessDescriptor):
            self.service.write_json_value(desc.ref, value)
            return
        self.service.write_data(desc.ref, value)

    def read_json_value(self, ref: DataRef) -> Any:
        desc = self.service.get_access(ref, region=None, mode="r")
        if not isinstance(desc, JsonAccessDescriptor):
            raise TypeError("read_json_value requires a JSON ref")
        return self.service.read_json_value(desc.ref)

    def write_json_value(self, ref: DataRef, value: Any) -> None:
        desc = self.service.get_access(ref, region=None, mode="rw")
        if not isinstance(desc, JsonAccessDescriptor):
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
