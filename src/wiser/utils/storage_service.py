from __future__ import annotations

from dataclasses import dataclass, field, replace
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple
import json
import logging
import uuid
from urllib.parse import quote, unquote, urlparse

import numpy as np
import zarr

from .primitives import (
    AllocationRequest,
    DataMeta,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    DiskFormat,
    RefKind,
    RegionMeta,
    SpectraBatchRef,
    SpectrumRef,
    temp_dir,
)
from .storage_layer import ExternalHandle

logger = logging.getLogger(__name__)


def _safe_np_dtype(value: Any) -> np.dtype:
    if value is None:
        return np.dtype("object")
    return np.dtype(value)


@dataclass(frozen=True)
class AccessDescriptor:
    ref: DataRef
    mode: Literal["r", "rw"]
    region: Optional[DataRegion]
    meta: DataMeta
    region_meta: Optional[RegionMeta]


@dataclass(frozen=True)
class RamAccessDescriptor(AccessDescriptor):
    pass


@dataclass(frozen=True)
class ExternalRamAccessDescriptor(AccessDescriptor):
    shared_mem: SharedMemArrayDescriptor


@dataclass(frozen=True)
class ExternalDiskAccessDescriptor(AccessDescriptor):
    pass


@dataclass(frozen=True)
class MemmapAccessDescriptor(AccessDescriptor):
    path: Path


@dataclass(frozen=True)
class ZarrAccessDescriptor(AccessDescriptor):
    store_path: Path
    array_name: str = "data"


@dataclass(frozen=True)
class JsonDiskAccessDescriptor(AccessDescriptor):
    path: Path


@dataclass(frozen=True)
class JsonRamAccessDescriptor(AccessDescriptor):
    value: Any


@dataclass(frozen=True)
class ExternalJsonRamAccessDescriptor(AccessDescriptor):
    value: Any


@dataclass(frozen=True)
class SharedMemArrayDescriptor:
    """
    Minimal descriptor for a NumPy array stored in shared memory.

    - name: SharedMemory name (string handle)
    - shape: array shape
    - dtype_str: dtype in string form (e.g. "float32", "int64")
    - strides: optional; needed for non-contiguous views
    - nbytes: total bytes allocated (useful for sanity checks)
    """

    name: str
    shape: Tuple[int, ...]
    dtype_str: str
    strides: Optional[Tuple[int, ...]]
    nbytes: int


@dataclass
class StorageService:
    root_dir: Path = temp_dir()
    ram_byte_limit: Optional[int] = None
    disk_byte_limit: Optional[int] = None

    data_refs: Dict[str, DataRef] = field(default_factory=dict)
    ram_objects: Dict[str, Any] = field(default_factory=dict)  # uri -> shared mem descriptor or JSON object
    ram_est_bytes: Dict[str, int] = field(default_factory=dict)
    meta_by_ref: Dict[str, DataMeta] = field(default_factory=dict)
    external_handles: Dict[str, ExternalHandle] = field(default_factory=dict)
    _shared_mem_handles: Dict[str, SharedMemory] = field(default_factory=dict, init=False, repr=False)
    _shared_memory_manager: SharedMemoryManager = field(init=False, repr=False)

    _ram_used_bytes: int = 0

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._shared_memory_manager = SharedMemoryManager()
        self._shared_memory_manager.start()

    def close(self) -> None:
        for shm in self._shared_mem_handles.values():
            try:
                shm.close()
            except Exception:
                logger.debug("SharedMemory close failed during shutdown", exc_info=True)
        self._shared_mem_handles.clear()
        self.ram_objects.clear()
        self.ram_est_bytes.clear()
        try:
            self._shared_memory_manager.shutdown()
        except Exception:
            logger.debug("SharedMemoryManager shutdown failed", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # External registration
    # -------------------------------------------------------------------------
    def register_external(self, handle: ExternalHandle) -> DataRef:
        ref_id = self._new_ref_id()
        meta = handle.get_meta()
        ref = DataRef(
            kind=meta.kind,
            ref_id=ref_id,
            uri=f"external://{ref_id}",
            disk_format=None,
            shape=tuple(meta.shape),
            dtype=_safe_np_dtype(meta.elem_type),
            chunks=None,
            residency="spill_required",
            materialization_loc="none",
            source="external",
            readonly=True,
        )
        self.data_refs[ref_id] = ref
        self.external_handles[ref_id] = handle
        self.meta_by_ref[ref_id] = meta
        logger.info("register_external ref_id=%s kind=%s", ref_id, ref.kind)
        return ref

    # -------------------------------------------------------------------------
    # Allocation
    # -------------------------------------------------------------------------
    def allocate_data(
        self,
        desc: AllocationRequest,
        *,
        preferred_storage: Optional[DiskFormat] = None,
        ttl_seconds: Optional[int] = None,
    ) -> DataRef:
        _ = ttl_seconds
        ref_id = self._new_ref_id()
        kind: RefKind = desc.kind

        can_allocate_shared = desc.kind != "json" and desc.shape is not None and desc.dtype is not None
        want_ram = desc.residency == "ram_cacheable"
        if want_ram and can_allocate_shared and self._can_fit_in_ram(desc):
            uri = f"mem://{ref_id}"
            shm_desc = self._allocate_in_ram_object(uri, desc)
            self.ram_est_bytes[uri] = desc.size_est
            self._ram_used_bytes += self._estimate_bytes(shm_desc, fallback_est=desc.size_est)

            ref = DataRef(
                kind=kind,
                ref_id=ref_id,
                uri=uri,
                disk_format=None,
                shape=tuple(desc.shape) if desc.shape is not None else None,
                dtype=desc.dtype,
                chunks=tuple(desc.chunks) if desc.chunks is not None else None,
                residency=desc.residency,
                materialization_loc="ram",
                source="internal",
                readonly=False,
            )
            self.data_refs[ref_id] = ref
            self.meta_by_ref[ref_id] = self._meta_from_ref(ref)
            logger.info(
                "Created DataRef ref_id=%s uri=%s materialization_loc=%s disk_format=%s",
                ref.ref_id,
                ref.uri,
                ref.materialization_loc,
                ref.disk_format,
            )
            return ref

        disk_kind = preferred_storage or self._choose_disk_format(desc)
        if disk_kind == "json":
            path = self.root_dir / f"{ref_id}.json"
            path.write_text("null", encoding="utf-8")
            ref = DataRef(
                kind="json",
                ref_id=ref_id,
                uri=self._path_to_file_uri(path),
                disk_format="json",
                residency=desc.residency,
                materialization_loc="disk",
                source="internal",
                readonly=False,
            )
            self.data_refs[ref_id] = ref
            self.meta_by_ref[ref_id] = self._meta_from_ref(ref)
            logger.info(
                "Created DataRef ref_id=%s uri=%s materialization_loc=%s disk_format=%s",
                ref.ref_id,
                ref.uri,
                ref.materialization_loc,
                ref.disk_format,
            )
            return ref

        if disk_kind == "memmap":
            if desc.shape is None or desc.dtype is None:
                raise ValueError("memmap allocation requires desc.shape and desc.dtype")
            path = self.root_dir / f"{ref_id}.npy"
            mm = np.lib.format.open_memmap(
                filename=str(path),
                mode="w+",
                dtype=desc.dtype,
                shape=tuple(desc.shape),
            )
            mm[...] = 0
            mm.flush()
            del mm
            ref = DataRef(
                kind=kind,
                ref_id=ref_id,
                uri=self._path_to_file_uri(path),
                disk_format="memmap",
                shape=tuple(desc.shape),
                dtype=desc.dtype,
                chunks=None,
                residency=desc.residency,
                materialization_loc="disk",
                source="internal",
                readonly=False,
            )
            self.data_refs[ref_id] = ref
            self.meta_by_ref[ref_id] = self._meta_from_ref(ref)
            return ref

        if disk_kind == "zarr":
            if desc.shape is None or desc.dtype is None:
                raise ValueError("zarr allocation requires desc.shape and desc.dtype")
            store_path = self.root_dir / f"{ref_id}.zarr"
            store = zarr.DirectoryStore(str(store_path))
            root = zarr.group(store=store, overwrite=True)
            chunks = tuple(desc.chunks) if desc.chunks is not None else None
            arr = root.zeros(
                name="data",
                shape=tuple(desc.shape),
                chunks=chunks,
                dtype=desc.dtype,
            )
            arr.attrs["_wiser_ref_id"] = ref_id
            ref = DataRef(
                kind=kind,
                ref_id=ref_id,
                uri=self._path_to_zarr_uri(store_path),
                disk_format="zarr",
                shape=tuple(desc.shape),
                dtype=desc.dtype,
                chunks=chunks,
                residency=desc.residency,
                materialization_loc="disk",
                source="internal",
                readonly=False,
            )
            self.data_refs[ref_id] = ref
            self.meta_by_ref[ref_id] = self._meta_from_ref(ref)
            logger.info(
                "Created DataRef ref_id=%s uri=%s materialization_loc=%s disk_format=%s",
                ref.ref_id,
                ref.uri,
                ref.materialization_loc,
                ref.disk_format,
            )
            return ref

        raise ValueError(f"Unknown DiskFormat: {disk_kind}")

    # -------------------------------------------------------------------------
    # Access endpoint
    # -------------------------------------------------------------------------
    def get_access(
        self,
        ref: DataRef,
        region: Optional[DataRegion],
        mode: Literal["r", "rw"] = "r",
    ) -> AccessDescriptor:
        canonical = self.read_data_ref(ref)
        if mode == "rw":
            self._ensure_writable(canonical, op_name="get_access(rw)")

        meta = self.get_meta(canonical)
        region_meta = self.get_region_meta(canonical, region) if region is not None else None

        if canonical.source == "external":
            if canonical.kind == "json":
                return ExternalJsonRamAccessDescriptor(
                    ref=canonical,
                    mode=mode,
                    region=region,
                    meta=meta,
                    region_meta=region_meta,
                    value=self._get_external_json_value(canonical),
                )
            if canonical.materialization_loc in ("none", "ram"):
                shared_mem = self._ensure_external_ram_shared(canonical, meta)
                return ExternalRamAccessDescriptor(
                    ref=canonical,
                    mode=mode,
                    region=region,
                    meta=meta,
                    region_meta=region_meta,
                    shared_mem=shared_mem,
                )
            if canonical.materialization_loc == "disk":
                return ExternalDiskAccessDescriptor(
                    ref=canonical,
                    mode=mode,
                    region=region,
                    meta=meta,
                    region_meta=region_meta,
                )
            raise ValueError(
                f"Unsupported external materialization_loc={canonical.materialization_loc} "
                f"for ref_id={canonical.ref_id}"
            )

        if canonical.materialization_loc == "ram":
            return RamAccessDescriptor(
                ref=canonical,
                mode=mode,
                region=region,
                meta=meta,
                region_meta=region_meta,
            )

        if canonical.materialization_loc == "disk":
            if canonical.disk_format == "memmap":
                return MemmapAccessDescriptor(
                    ref=canonical,
                    mode=mode,
                    region=region,
                    meta=meta,
                    region_meta=region_meta,
                    path=self._file_uri_to_path(canonical.uri),
                )
            if canonical.disk_format == "zarr":
                return ZarrAccessDescriptor(
                    ref=canonical,
                    mode=mode,
                    region=region,
                    meta=meta,
                    region_meta=region_meta,
                    store_path=self._zarr_uri_to_path(canonical.uri),
                )
            if canonical.disk_format == "json":
                return JsonDiskAccessDescriptor(
                    ref=canonical,
                    mode=mode,
                    region=region,
                    meta=meta,
                    region_meta=region_meta,
                    path=self._file_uri_to_path(canonical.uri),
                )
        if canonical.materialization_loc == "ram" and canonical.kind == "json":
            return JsonRamAccessDescriptor(
                ref=canonical,
                mode=mode,
                region=region,
                meta=meta,
                region_meta=region_meta,
                value=self._get_json_ram_value(canonical),
            )

        raise ValueError(
            f"Unsupported materialization/storage: {canonical.materialization_loc}/{canonical.disk_format}"
        )

    # -------------------------------------------------------------------------
    # Read/write API
    # -------------------------------------------------------------------------
    def read_data_ref(self, ref: DataRef) -> DataRef:
        try:
            return self.data_refs[ref.ref_id]
        except KeyError as e:
            raise KeyError(f"Unknown ref_id: {ref.ref_id}") from e

    def read_data(self, ref: DataRef) -> tuple[np.ndarray, RegionMeta]:
        desc = self.get_access(ref, region=None, mode="r")
        if isinstance(
            desc,
            (JsonDiskAccessDescriptor, JsonRamAccessDescriptor, ExternalJsonRamAccessDescriptor),
        ):
            raise TypeError("read_data not supported for JSON; use read_json_value")

        whole_region = self._whole_region_from_meta(desc.meta)
        region_meta = self.get_region_meta(desc.ref, whole_region)
        arr = self.read_region(desc.ref, whole_region)
        return np.asarray(arr), region_meta

    def read_region(self, ref: DataRef, region: DataRegion) -> Any:
        canonical = self.read_data_ref(ref)
        if canonical.source == "external":
            return self.external_handles[canonical.ref_id].read_region(region)

        if canonical.materialization_loc == "ram":
            return self._read_region_from_ram(canonical.uri, region)

        if canonical.materialization_loc != "disk":
            raise ValueError(f"Ref is not materialized: {canonical.ref_id}")

        if canonical.disk_format == "json":
            raise TypeError("read_region not supported for JSON; use read_json_value")
        if canonical.disk_format == "memmap":
            arr = np.load(str(self._file_uri_to_path(canonical.uri)), mmap_mode="r")
            return self._read_region_from_array(arr, region)
        if canonical.disk_format == "zarr":
            z = self._open_zarr_array(canonical.uri, mode="r")
            return self._read_region_from_array(z, region)
        raise ValueError(f"Unsupported disk format: {canonical.disk_format}")

    def write_region(self, ref: DataRef, region: DataRegion, value: Any) -> None:
        canonical = self.read_data_ref(ref)
        self._ensure_writable(canonical)

        if canonical.materialization_loc == "ram":
            self._write_region_into_ram(canonical.uri, region, value)
            return

        if canonical.materialization_loc != "disk":
            raise ValueError(f"Ref is not materialized: {canonical.ref_id}")

        if canonical.disk_format == "json":
            raise TypeError("write_region not supported for JSON; use write_json_value")
        if canonical.disk_format == "memmap":
            arr = np.load(str(self._file_uri_to_path(canonical.uri)), mmap_mode="r+")
            self._write_region_into_array(arr, region, value)
            if hasattr(arr, "flush"):
                arr.flush()
            return
        if canonical.disk_format == "zarr":
            z = self._open_zarr_array(canonical.uri, mode="r+")
            self._write_region_into_array(z, region, value)
            return
        raise ValueError(f"Unsupported disk format: {canonical.disk_format}")

    def write_data(self, ref: DataRef, value: Any) -> None:
        canonical = self.read_data_ref(ref)
        self._ensure_writable(canonical)

        if canonical.materialization_loc == "ram":
            existing = self.ram_objects.get(canonical.uri)
            if existing is None:
                raise KeyError(f"No RAM object for uri={canonical.uri}")
            self._write_array_into_shared_mem(existing, value)
            return

        if canonical.materialization_loc != "disk":
            raise ValueError(f"Ref is not materialized: {canonical}")

        if canonical.disk_format == "json":
            self.write_json_value(canonical, value)
            return
        if canonical.disk_format == "memmap":
            arr = np.load(str(self._file_uri_to_path(canonical.uri)), mmap_mode="r+")
            arr[...] = value
            if hasattr(arr, "flush"):
                arr.flush()
            return
        if canonical.disk_format == "zarr":
            z = self._open_zarr_array(canonical.uri, mode="r+")
            z[...] = value
            return
        raise ValueError(f"Unsupported disk format: {canonical.disk_format}")

    def read_json_value(self, ref: DataRef) -> Any:
        canonical = self.read_data_ref(ref)
        if canonical.disk_format != "json":
            raise TypeError("read_json_value requires a JSON ref")
        path = self._file_uri_to_path(canonical.uri)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_json_value(self, ref: DataRef, value: Any) -> None:
        canonical = self.read_data_ref(ref)
        self._ensure_writable(canonical, op_name="write_json_value")
        if canonical.disk_format != "json":
            raise TypeError("write_json_value requires a JSON ref")
        path = self._file_uri_to_path(canonical.uri)
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f)

    def write_json_ram_value(self, ref: DataRef, value: Any) -> None:
        canonical = self.read_data_ref(ref)
        self._ensure_writable(canonical, op_name="write_json_ram_value")
        if canonical.kind != "json":
            raise TypeError("write_json_ram_value requires ref.kind == 'json'")
        if canonical.materialization_loc != "ram":
            raise TypeError("write_json_ram_value requires a RAM-backed JSON ref")
        self.ram_objects[canonical.uri] = value

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    def get_meta(self, ref: DataRef) -> DataMeta:
        canonical = self.read_data_ref(ref)
        if canonical.ref_id in self.meta_by_ref:
            return self.meta_by_ref[canonical.ref_id]

        if canonical.source == "external":
            meta = self.external_handles[canonical.ref_id].get_meta()
        else:
            meta = self._meta_from_ref(canonical)
        self.meta_by_ref[canonical.ref_id] = meta
        return meta

    def get_region_meta(self, ref: DataRef, region: DataRegion) -> RegionMeta:
        canonical = self.read_data_ref(ref)
        if canonical.source == "external":
            return self.external_handles[canonical.ref_id].get_region_meta(region)
        return self._derive_region_meta(self.get_meta(canonical), region)

    def set_meta(self, ref: DataRef, meta: DataMeta) -> None:
        canonical = self.read_data_ref(ref)
        self._ensure_writable(canonical, op_name="set_meta")
        self.meta_by_ref[canonical.ref_id] = meta

    def update_meta(self, ref: DataRef, **fields: Any) -> DataMeta:
        canonical = self.read_data_ref(ref)
        self._ensure_writable(canonical, op_name="update_meta")
        updated = replace(self.get_meta(canonical), **fields)
        self.meta_by_ref[canonical.ref_id] = updated
        return updated

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _choose_disk_format(self, desc: AllocationRequest) -> DiskFormat:
        if desc.kind == "json":
            return "json"
        return "memmap"

    def _get_ram_array(self, uri: str) -> np.ndarray:
        if not uri.startswith("mem://"):
            raise ValueError(f"Expected mem:// uri, got: {uri}")
        descriptor = self.ram_objects.get(uri)
        if descriptor is None:
            raise KeyError(f"No RAM object for uri={uri}")
        if not isinstance(descriptor, SharedMemArrayDescriptor):
            raise TypeError(f"RAM object for uri={uri} is not a SharedMemArrayDescriptor")
        return self._with_shared_mem_array(descriptor, lambda arr: np.asarray(arr).copy())

    def _derive_region_meta(self, meta: DataMeta, region: DataRegion) -> RegionMeta:
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

        # Fallback for "array" and other generic numeric refs.
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

    def _open_zarr_array(self, uri: str, mode: str) -> zarr.Array:
        if not uri.startswith("zarr://"):
            raise ValueError(f"Expected zarr:// uri, got: {uri}")
        store_path = self._zarr_uri_to_path(uri)
        store = zarr.DirectoryStore(str(store_path))
        grp = zarr.open_group(store=store, mode=mode)
        return grp["data"]

    def _file_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"Not a file URI: {uri}")
        raw = parsed.path if parsed.path else parsed.netloc
        return Path(unquote(raw)).resolve()

    def _zarr_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "zarr":
            raise ValueError(f"Not a zarr URI: {uri}")
        raw = parsed.path if parsed.path else parsed.netloc
        return Path(unquote(raw)).resolve()

    def _path_to_file_uri(self, path: Path) -> str:
        return f"file://{quote(str(path.resolve()))}"

    def _path_to_zarr_uri(self, path: Path) -> str:
        return f"zarr://{quote(str(path.resolve()))}"

    def _can_fit_in_ram(self, desc: AllocationRequest) -> bool:
        if self.ram_byte_limit is None:
            return True
        needed = self._estimate_request_bytes(desc)
        return (self._ram_used_bytes + needed) <= self.ram_byte_limit

    def _estimate_request_bytes(self, desc: AllocationRequest) -> int:
        if desc.kind == "json":
            return 1024
        if desc.shape is None or desc.dtype is None:
            return 10**18
        n = 1
        for dim in desc.shape:
            n *= int(dim)
        return n * np.dtype(desc.dtype).itemsize

    def _estimate_bytes(self, obj: Any, fallback_est: int) -> int:
        if isinstance(obj, SharedMemArrayDescriptor):
            return int(obj.nbytes)
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        return fallback_est

    def _ensure_writable(self, ref: DataRef, op_name: str = "write") -> None:
        if ref.readonly or ref.source == "external":
            raise PermissionError(f"{op_name} is not allowed for external/read-only refs: {ref.ref_id}")

    def _meta_from_ref(self, ref: DataRef) -> DataMeta:
        return DataMeta(
            kind=ref.kind,
            shape=tuple(ref.shape) if ref.shape is not None else (),
            elem_type=_safe_np_dtype(ref.dtype),
        )

    def _allocate_in_ram_object(self, uri: str, desc: AllocationRequest) -> SharedMemArrayDescriptor:
        if desc.shape is None or desc.dtype is None:
            raise ValueError("RAM shared-memory allocation requires desc.shape and desc.dtype")
        shape = tuple(int(dim) for dim in desc.shape)
        dtype = np.dtype(desc.dtype)
        nbytes = int(np.prod(shape, dtype=np.int64) * dtype.itemsize)
        shm: SharedMemory = self._shared_memory_manager.SharedMemory(size=nbytes)
        shm_desc = SharedMemArrayDescriptor(
            name=shm.name,
            shape=shape,
            dtype_str=dtype.str,
            strides=None,
            nbytes=nbytes,
        )
        self._shared_mem_handles[uri] = shm
        self.ram_objects[uri] = shm_desc
        self._with_shared_mem_array(shm_desc, lambda arr: arr.fill(0))
        return shm_desc

    def _attach_shared_mem(self, descriptor: SharedMemArrayDescriptor) -> SharedMemory:
        return SharedMemory(name=descriptor.name, create=False)

    def _array_from_descriptor(self, shm: SharedMemory, descriptor: SharedMemArrayDescriptor) -> np.ndarray:
        return np.ndarray(
            shape=descriptor.shape,
            dtype=np.dtype(descriptor.dtype_str),
            buffer=shm.buf,
            strides=descriptor.strides,
        )

    def _with_shared_mem_array(
        self,
        descriptor: SharedMemArrayDescriptor,
        fn: Callable[[np.ndarray], Any],
    ) -> Any:
        shm = self._attach_shared_mem(descriptor)
        try:
            arr = self._array_from_descriptor(shm, descriptor)
            return fn(arr)
        finally:
            shm.close()

    def _read_region_from_ram(self, uri: str, region: DataRegion) -> Any:
        """
        Read a region from a RAM-backed shared memory object and return a
        fresh independent NumPy array copy.

        Raises:
            KeyError: If no RAM-backed object exists for the given URI.
        """
        descriptor = self.ram_objects.get(uri)
        if descriptor is None:
            raise KeyError(f"No RAM object for uri={uri}")
        if not isinstance(descriptor, SharedMemArrayDescriptor):
            raise TypeError(f"RAM object for uri={uri} is not a SharedMemArrayDescriptor")
        return self._with_shared_mem_array(
            descriptor,
            lambda arr: np.asarray(self._read_region_from_array(arr, region)).copy(),
        )

    def _write_region_into_ram(self, uri: str, region: DataRegion, value: Any) -> None:
        descriptor = self.ram_objects.get(uri)
        if descriptor is None:
            raise KeyError(f"No RAM object for uri={uri}")
        if not isinstance(descriptor, SharedMemArrayDescriptor):
            raise TypeError(f"RAM object for uri={uri} is not a SharedMemArrayDescriptor")
        self._with_shared_mem_array(
            descriptor,
            lambda arr: self._write_region_into_array(arr, region, value),
        )

    def _write_array_into_shared_mem(self, descriptor: SharedMemArrayDescriptor, value: Any) -> None:
        def _writer(arr: np.ndarray) -> None:
            arr[...] = value

        self._with_shared_mem_array(descriptor, _writer)

    def _ensure_external_ram_shared(self, ref: DataRef, meta: DataMeta) -> SharedMemArrayDescriptor:
        existing = self.ram_objects.get(ref.uri)
        if existing is not None:
            return existing

        if not (ref.source == "external" and ref.readonly):
            raise ValueError("_ensure_external_ram_shared requires an external read-only ref")
        if meta.kind == "json":
            raise TypeError("External RAM shared-memory materialization is not supported for JSON refs")
        if not meta.shape:
            raise ValueError(
                f"Cannot materialize external shared memory for ref_id={ref.ref_id} without shape"
            )

        whole_region = self._whole_region_from_meta(meta)
        value = self.external_handles[ref.ref_id].read_region(whole_region)
        arr = np.asarray(value)
        nbytes = int(arr.nbytes)
        shm: SharedMemory = self._shared_memory_manager.SharedMemory(size=nbytes)
        shm_desc = SharedMemArrayDescriptor(
            name=shm.name,
            shape=tuple(arr.shape),
            dtype_str=np.dtype(arr.dtype).str,
            strides=tuple(arr.strides),
            nbytes=nbytes,
        )
        self._shared_mem_handles[ref.uri] = shm
        self.ram_objects[ref.uri] = shm_desc
        self._with_shared_mem_array(shm_desc, lambda target: np.copyto(target, arr))
        return shm_desc

    def _get_external_json_value(self, ref: DataRef) -> Any:
        cached = self.ram_objects.get(ref.uri)
        if cached is not None and not isinstance(cached, SharedMemArrayDescriptor):
            return cached
        if ref.ref_id not in self.external_handles:
            raise KeyError(f"No external handle for ref_id={ref.ref_id}")
        raise TypeError(
            f"External JSON ref {ref.ref_id} does not have a cached RAM value; "
            "populate service.ram_objects[ref.uri] with a JSON object before requesting access"
        )

    def _get_json_ram_value(self, ref: DataRef) -> Any:
        cached = self.ram_objects.get(ref.uri)
        if cached is None:
            raise KeyError(f"No RAM JSON object for uri={ref.uri}")
        if isinstance(cached, SharedMemArrayDescriptor):
            raise TypeError(f"RAM object for uri={ref.uri} is shared-memory array, not JSON")
        return cached

    def _new_ref_id(self) -> str:
        return uuid.uuid4().hex
