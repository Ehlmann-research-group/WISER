# ==================== imports (keep with StorageLayer) ====================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json
import uuid
import tempfile
from urllib.parse import urlparse, quote, unquote

import numpy as np
import zarr

from .primitives import (
    DataRef,
    AllocationRequest,
    DiskFormat,
    DataRegion,
    RefKind,
    SpectraBatchRef,
    SpectrumRef,
    DatasetRegionRef,
    temp_dir,
)


# ==================== StorageLayer ====================
@dataclass
class StorageLayer:
    """
    Allocates, writes, and reads data backed by either:
      - RAM:     mem://<ref_id>                 (stored in mem_backed_data)
      - Disk:    file:///abs/path/<id>.npy      (memmap-style .npy via numpy)
      - Disk:    zarr:///abs/path/<id>.zarr     (zarr DirectoryStore)
      - Disk:    file:///abs/path/<id>.json     (json file)

    This StorageLayer implements a simple policy:
      - If desc.residency == "ram_cacheable" and there is enough RAM budget,
        allocate in RAM (materialization_loc="ram", uri="mem://...").
      - Otherwise (spill_required OR insufficient RAM), allocate on disk
        (materialization_loc="disk", uri="file://..." or "zarr://...").

    Notes:
      - `data_refs` is a registry: ref_id -> DataRef
      - `mem_backed_data` stores the actual RAM objects: mem://... -> object
      - For now, we do NOT keep both a RAM and disk copy at once.
    """

    # Where disk-backed allocations are created
    root_dir: Path = temp_dir()

    # Simple RAM budget model (bytes). If None, treat as "infinite" for now.
    ram_byte_limit: Optional[int] = None
    disk_byte_limit: Optional[int] = None

    # Registries
    data_refs: Dict[str, DataRef] = field(default_factory=dict)  # ref_id -> DataRef
    mem_backed_data: Dict[str, Any] = field(default_factory=dict)  # mem://... -> object
    mem_backed_est: Dict[str, int] = field(default_factory=dict)

    # Track rough RAM usage for budget decisions (best-effort)
    _ram_used_bytes: int = 0

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def allocate_data(
        self,
        desc: AllocationRequest,
        *,
        preferred_storage: Optional[DiskFormat] = None,
        ttl_seconds: Optional[int] = None,  # ignored for now
    ) -> DataRef:
        """
        Allocate storage according to residency and (optional) preferred_storage.

        Rules:
          - If desc.residency == "ram_cacheable" and we have room => allocate RAM.
          - Else allocate disk:
              - If preferred_storage is provided, use it (memmap/zarr/json).
              - Else choose:
                  * json -> json
                  * numeric -> memmap (default; zarr only if explicitly preferred)
        """
        _ = ttl_seconds  # ignored per request

        ref_id = self._new_ref_id()
        kind: RefKind = desc.kind

        # Decide RAM vs disk
        want_ram = desc.residency == "ram_cacheable"
        if want_ram and self._can_fit_in_ram(desc):
            uri = f"mem://{ref_id}"
            obj = self._allocate_in_ram_object(desc)
            self.mem_backed_data[uri] = obj
            self.mem_backed_est[uri] = desc.size_est
            self._ram_used_bytes += self._estimate_bytes(obj, fallback_est=desc.size_est)

            ref = DataRef(
                kind=kind,
                ref_id=ref_id,
                uri=uri,
                disk_format=None,  # not disk-backed
                shape=tuple(desc.shape) if desc.shape is not None else None,
                dtype=str(desc.dtype) if desc.dtype is not None else None,
                chunks=tuple(desc.chunks) if desc.chunks is not None else None,
                residency=desc.residency,
                materialization_loc="ram",
            )
            self.data_refs[ref_id] = ref
            return ref

        # Otherwise: allocate to disk
        disk_kind = preferred_storage or self._choose_disk_format(desc)

        if disk_kind == "json":
            path = self.root_dir / f"{ref_id}.json"
            uri = self._path_to_file_uri(path)
            # Create placeholder file
            path.write_text("null", encoding="utf-8")

            ref = DataRef(
                kind="json",
                ref_id=ref_id,
                uri=uri,
                disk_format="json",
                residency=desc.residency,
                materialization_loc="disk",
            )
            self.data_refs[ref_id] = ref
            return ref

        if disk_kind == "memmap":
            if desc.shape is None or desc.dtype is None:
                raise ValueError("memmap allocation requires desc.shape and desc.dtype")

            path = self.root_dir / f"{ref_id}.npy"
            uri = self._path_to_file_uri(path)

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
                uri=uri,
                disk_format="memmap",
                shape=tuple(desc.shape),
                dtype=str(desc.dtype),
                chunks=None,
                residency=desc.residency,
                materialization_loc="disk",
            )
            self.data_refs[ref_id] = ref
            return ref

        if disk_kind == "zarr":
            if desc.shape is None or desc.dtype is None:
                raise ValueError("zarr allocation requires desc.shape and desc.dtype")

            store_path = self.root_dir / f"{ref_id}.zarr"
            uri = self._path_to_zarr_uri(store_path)

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
                uri=uri,
                disk_format="zarr",
                shape=tuple(desc.shape),
                dtype=str(desc.dtype),
                chunks=chunks,
                residency=desc.residency,
                materialization_loc="disk",
            )
            self.data_refs[ref_id] = ref
            return ref

        raise ValueError(f"Unknown DiskFormat: {disk_kind}")

    def write_region(self, data: DataRef, chunk_ref: DataRegion, value: Any) -> bool:
        """
        Write `value` into `data` at region `chunk_ref`.

        Returns True on success, False on error.
        """
        try:
            if data.materialization_loc == "ram":
                obj = self._get_ram_object(data.uri)
                self._write_region_into_array(obj, chunk_ref, value)
                return True

            if data.materialization_loc == "disk":
                if data.disk_format == "json":
                    raise TypeError("write_region is not supported for JSON")
                if data.disk_format == "memmap":
                    path = self._file_uri_to_path(data.uri)
                    arr = np.load(str(path), mmap_mode="r+")
                    self._write_region_into_array(arr, chunk_ref, value)
                    if hasattr(arr, "flush"):
                        arr.flush()
                    return True
                if data.disk_format == "zarr":
                    z = self._open_zarr_array(data.uri, mode="r+")
                    self._write_region_into_array(z, chunk_ref, value)
                    return True

            raise ValueError(
                f"Unsupported materialization/storage: {data.materialization_loc}/{data.disk_format}"
            )
        except Exception:
            return False

    def write_data(self, ref: DataRef, value: Any) -> None:
        """
        Write entire object for `ref`.

        - RAM: store/overwrite in mem_backed_data (or assign into ndarray)
        - JSON: json dump to file
        - memmap: open mmap and assign
        - zarr: open and assign
        """
        if ref.materialization_loc == "ram":
            existing = self.mem_backed_data.get(ref.uri, None)
            if isinstance(existing, np.ndarray) and isinstance(value, np.ndarray):
                existing[...] = value
            else:
                # replace; update rough accounting
                est_bytes = self.mem_backed_est[ref.uri]
                if existing is not None:
                    self._ram_used_bytes -= self._estimate_bytes(existing, fallback=est_bytes)
                self.mem_backed_data[ref.uri] = value
                self._ram_used_bytes += self._estimate_bytes(value, fallback_est=est_bytes)
            return

        if ref.materialization_loc != "disk":
            raise ValueError(f"Ref is not materialized: {ref}")

        if ref.disk_format == "json":
            path = self._file_uri_to_path(ref.uri)
            with path.open("w", encoding="utf-8") as f:
                json.dump(value, f)
            return

        if ref.disk_format == "memmap":
            path = self._file_uri_to_path(ref.uri)
            arr = np.load(str(path), mmap_mode="r+")
            arr[...] = value
            if hasattr(arr, "flush"):
                arr.flush()
            return

        if ref.disk_format == "zarr":
            z = self._open_zarr_array(ref.uri, mode="r+")
            z[...] = value
            return

        raise ValueError(f"Unsupported disk format: {ref.disk_format}")

    def read_data(self, ref_id: str) -> DataRef:
        """
        Return the DataRef handle (metadata + locator).
        """
        try:
            return self.data_refs[ref_id]
        except KeyError as e:
            raise KeyError(f"Unknown ref_id: {ref_id}") from e

    def read_region(self, ref_id: str, chunk_ref: DataRegion) -> Any:
        """
        Read a region specified by `chunk_ref`.
        For JSON refs, ignores chunk_ref and returns the whole object.
        """
        ref = self.read_data(ref_id)

        if ref.materialization_loc == "ram":
            obj = self._get_ram_object(ref.uri)
            return self._read_region_from_array(obj, chunk_ref)

        if ref.materialization_loc == "disk":
            if ref.disk_format == "json":
                path = self._file_uri_to_path(ref.uri)
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)

            if ref.disk_format == "memmap":
                path = self._file_uri_to_path(ref.uri)
                arr = np.load(str(path), mmap_mode="r")
                return self._read_region_from_array(arr, chunk_ref)

            if ref.disk_format == "zarr":
                z = self._open_zarr_array(ref.uri, mode="r")
                return self._read_region_from_array(z, chunk_ref)

            raise ValueError(f"Unsupported disk format: {ref.disk_format}")

        raise ValueError(f"Ref is not materialized: {ref_id}")

    # -------------------------------------------------------------------------
    # Policy helpers
    # -------------------------------------------------------------------------

    def _choose_disk_format(self, desc: AllocationRequest) -> DiskFormat:
        # json requests always go to json
        if desc.kind == "json":
            return "json"
        # numeric defaults to memmap; caller can override with preferred_storage="zarr"
        return "memmap"

    def _can_fit_in_ram(self, desc: AllocationRequest) -> bool:
        if self.ram_byte_limit is None:
            return True
        needed = self._estimate_request_bytes(desc)
        return (self._ram_used_bytes + needed) <= self.ram_byte_limit

    def _estimate_request_bytes(self, desc: AllocationRequest) -> int:
        if desc.kind == "json":
            # unknown until written; assume small
            return 1024
        if desc.shape is None or desc.dtype is None:
            # unknown; assume can't fit safely
            return 10**18
        n = 1
        for d in desc.shape:
            n *= int(d)
        return n * np.dtype(desc.dtype).itemsize

    # -------------------------------------------------------------------------
    # Allocation helpers
    # -------------------------------------------------------------------------

    def _allocate_in_ram_object(self, desc: AllocationRequest) -> Any:
        if desc.kind == "json":
            return {}  # default empty object for JSON-in-RAM use
        if desc.shape is None or desc.dtype is None:
            # allow late write_data to fill it
            return None
        return np.zeros(tuple(desc.shape), dtype=desc.dtype)

    # -------------------------------------------------------------------------
    # Region slicing helpers
    # -------------------------------------------------------------------------

    def _read_region_from_array(self, arr: Any, chunk_ref: DataRegion) -> Any:
        if isinstance(chunk_ref, DatasetRegionRef):
            return arr[
                chunk_ref.y0 : chunk_ref.y1,
                chunk_ref.x0 : chunk_ref.x1,
                chunk_ref.b0 : (chunk_ref.b1 if chunk_ref.b1 is not None else None),
            ]
        if isinstance(chunk_ref, SpectrumRef):
            return arr[...]
        if isinstance(chunk_ref, SpectraBatchRef):
            return arr[chunk_ref.i0 : chunk_ref.i1]
        raise TypeError(f"Unknown DataRegion type: {type(chunk_ref)}")

    def _write_region_into_array(self, arr: Any, chunk_ref: DataRegion, value: Any) -> None:
        if isinstance(chunk_ref, DatasetRegionRef):
            arr[
                chunk_ref.y0 : chunk_ref.y1,
                chunk_ref.x0 : chunk_ref.x1,
                chunk_ref.b0 : (chunk_ref.b1 if chunk_ref.b1 is not None else None),
            ] = value
            return
        if isinstance(chunk_ref, SpectrumRef):
            arr[...] = value
            return
        if isinstance(chunk_ref, SpectraBatchRef):
            arr[chunk_ref.i0 : chunk_ref.i1] = value
            return
        raise TypeError(f"Unknown DataRegion type: {type(chunk_ref)}")

    # -------------------------------------------------------------------------
    # RAM helpers
    # -------------------------------------------------------------------------

    def _get_ram_object(self, uri: str) -> Any:
        if not uri.startswith("mem://"):
            raise ValueError(f"Expected mem:// uri, got: {uri}")
        try:
            return self.mem_backed_data[uri]
        except KeyError as e:
            raise KeyError(f"No RAM object for uri={uri}") from e

    def _estimate_bytes(self, obj: Any, fallback_est: int) -> int:
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        return fallback_est

    # -------------------------------------------------------------------------
    # Disk helpers
    # -------------------------------------------------------------------------

    def _open_zarr_array(self, uri: str, mode: str) -> zarr.Array:
        if not uri.startswith("zarr://"):
            raise ValueError(f"Expected zarr:// uri, got: {uri}")
        store_path = self._zarr_uri_to_path(uri)
        store = zarr.DirectoryStore(str(store_path))
        grp = zarr.open_group(store=store, mode=mode)
        return grp["data"]

    # -------------------------------------------------------------------------
    # URI helpers
    # -------------------------------------------------------------------------

    def _new_ref_id(self) -> str:
        return uuid.uuid4().hex

    def _path_to_file_uri(self, path: Path) -> str:
        p = path.resolve()
        return f"file://{quote(str(p))}"

    def _file_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"Not a file URI: {uri}")
        return Path(unquote(parsed.path)).resolve()

    def _path_to_zarr_uri(self, path: Path) -> str:
        p = path.resolve()
        return f"zarr://{quote(str(p))}"

    def _zarr_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "zarr":
            raise ValueError(f"Not a zarr URI: {uri}")
        return Path(unquote(parsed.path)).resolve()
