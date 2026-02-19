from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import zarr

from .primitives import DataMeta, DataRef, DataRegion, RegionMeta
from .storage_service import (
    AccessDescriptor,
    JsonAccessDescriptor,
    MemmapAccessDescriptor,
    RamAccessDescriptor,
    StorageService,
    ZarrAccessDescriptor,
)


@dataclass
class StorageClient:
    service: StorageService

    def read_data(self, ref: DataRef) -> DataRef:
        return self.service.read_data(ref)

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
