from __future__ import annotations

from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection, Listener
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import secrets
import threading
import traceback
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union, TYPE_CHECKING
import json
import logging
import uuid
from urllib.parse import quote, unquote, urlparse

import numpy as np
import zarr

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
from wiser.raster.dataset import RasterDataSet
from wiser.raster.envi_spectral_library import ENVISpectralLibrary

from .primitives import (
    _safe_np_dtype,
    AllocationRequest,
    DataMeta,
    DeletePolicy,
    DeletionState,
    ExternalHandle,
    ExternalParams,
    DataRef,
    DataRegion,
    DatasetRegionRef,
    DiskFormat,
    InputKind,
    ProducerState,
    RefKind,
    RegionMeta,
    StorageLeaseRecord,
    SpectraBatchRef,
    SpectrumRef,
    temp_dir,
)

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet

logger = logging.getLogger(__name__)


def shared_mem_exists(shared_mem_name: str) -> bool:
    try:
        shm = SharedMemory(name=shared_mem_name, create=False)
    except FileNotFoundError:
        return False

    try:
        return True
    finally:
        shm.close()


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
    """
    Central authority for storage-backed data and its service-side lifetime state.

    `DataRef` identifies how data is accessed. `StorageLeaseRecord` tracks whether
    an internally managed backing object is still live, pending deletion, or has
    already been reclaimed.
    """

    root_dir: Path = temp_dir()
    listener_host: str = "127.0.0.1"
    ram_byte_limit: Optional[int] = None
    disk_byte_limit: Optional[int] = None

    data_refs: Dict[str, DataRef] = field(default_factory=dict)
    ram_objects: Dict[str, Any] = field(default_factory=dict)  # uri -> shared mem descriptor or JSON object
    ram_est_bytes: Dict[str, int] = field(default_factory=dict)
    meta_by_ref: Dict[str, DataMeta] = field(default_factory=dict)
    external_handles: Dict[str, ExternalHandle] = field(default_factory=dict)
    lease_records: Dict[str, StorageLeaseRecord] = field(default_factory=dict)
    # Keep one service-owned handle open per live shared-memory allocation.
    # This is intentional: on Windows a named mapping disappears once the last
    # handle closes, so workers/clients can attach transiently while the service
    # retains the owning handle until explicit deletion.
    _shared_mem_handles: Dict[str, SharedMemory] = field(default_factory=dict, init=False, repr=False)
    # Historical mapping for debugging: once a RAM ref has been backed by a named
    # shared-memory segment, keep the name here even after the live handle is gone.
    _shared_mem_handles_names: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _shared_memory_manager: SharedMemoryManager = field(init=False, repr=False)
    _listener: Optional[Listener] = field(default=None, init=False, repr=False)
    _listener_address: Optional[Tuple[str, int]] = field(default=None, init=False)
    _listener_authkey: bytes = field(default=b"", init=False, repr=False)
    _listener_stop_event: threading.Event = field(init=False, repr=False)
    _accept_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _active_connections: Dict[int, Connection] = field(default_factory=dict, init=False, repr=False)
    _connection_threads: Dict[int, threading.Thread] = field(default_factory=dict, init=False, repr=False)
    _rpc_allowlist: Dict[str, Callable[..., Any]] = field(default_factory=dict, init=False, repr=False)

    _ram_used_bytes: int = 0

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._listener_stop_event = threading.Event()
        self._listener_authkey = secrets.token_bytes(32)
        self._start_listener()
        self._register_rpc_methods()
        self._shared_memory_manager = SharedMemoryManager()
        self._shared_memory_manager.start()

    def close(self) -> None:
        self._listener_stop_event.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                logger.debug("Listener close failed during shutdown", exc_info=True)
            finally:
                self._listener = None

        for conn in list(self._active_connections.values()):
            try:
                conn.close()
            except Exception:
                logger.debug("Connection close failed during shutdown", exc_info=True)
        self._active_connections.clear()

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None

        for thread in list(self._connection_threads.values()):
            thread.join(timeout=1.0)
        self._connection_threads.clear()

        for shm in self._shared_mem_handles.values():
            try:
                shm.close()
            except Exception:
                logger.debug("SharedMemory close failed during shutdown", exc_info=True)
        self._shared_mem_handles.clear()
        self.ram_objects.clear()
        self.ram_est_bytes.clear()
        self.lease_records.clear()
        try:
            self._shared_memory_manager.shutdown()
        except Exception:
            logger.debug("SharedMemoryManager shutdown failed", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def listener_address(self) -> Tuple[str, int]:
        if self._listener_address is None:
            raise RuntimeError("StorageService listener has not been initialized")
        return self._listener_address

    @property
    def listener_authkey(self) -> bytes:
        return self._listener_authkey

    def get_connection_bootstrap(self) -> tuple[Tuple[str, int], bytes]:
        return self.listener_address, self.listener_authkey

    def get_shared_mem_name_for_ref(self, ref_id: str) -> Optional[str]:
        """
        Return the named shared-memory segment historically associated with `ref_id`.

        This is intentionally debug-oriented: the returned name may outlive the
        live service-owned handle so callers can probe whether the OS-level segment
        still exists after normal service bookkeeping has released it.
        """

        ref = self.data_refs.get(ref_id)
        if ref is None:
            return None
        return self._shared_mem_handles_names.get(ref.uri)

    def _start_listener(self) -> None:
        self._listener = Listener((self.listener_host, 0), authkey=self._listener_authkey)
        address = self._listener.address
        if not (isinstance(address, tuple) and len(address) == 2):
            raise RuntimeError(f"Unexpected listener address format: {address!r}")
        self._listener_address = (str(address[0]), int(address[1]))
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._listener_stop_event.is_set():
            listener = self._listener
            if listener is None:
                break
            try:
                conn = listener.accept()
            except (OSError, EOFError):
                if self._listener_stop_event.is_set():
                    break
                logger.debug("Listener accept failed", exc_info=True)
                continue

            conn_id = id(conn)
            self._active_connections[conn_id] = conn
            thread = threading.Thread(target=self._handle_connection, args=(conn,), daemon=True)
            self._connection_threads[conn_id] = thread
            thread.start()

    def _handle_connection(self, conn: Connection) -> None:
        conn_id = id(conn)
        try:
            while not self._listener_stop_event.is_set():
                request = conn.recv()
                response = self._dispatch_rpc_request(request)
                conn.send(response)
        except EOFError:
            logger.debug("StorageService client disconnected")
        except (OSError, ConnectionError):
            if not self._listener_stop_event.is_set():
                logger.debug("StorageService connection handler error", exc_info=True)
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("StorageService connection close failed", exc_info=True)
            self._active_connections.pop(conn_id, None)
            self._connection_threads.pop(conn_id, None)

    def _register_rpc_methods(self) -> None:
        self._rpc_allowlist = {
            "read_data_ref": self.read_data_ref,
            "get_access": self.get_access,
            "get_meta": self.get_meta,
            "write_meta": self.set_meta,
            "get_region_meta": self.get_region_meta,
            "read_json_value": self.read_json_value,
            "write_json_value": self.write_json_value,
            "write_json_ram_value": self.write_json_ram_value,
            "get_ram_descriptor": self.get_ram_descriptor,
        }

    def _dispatch_rpc_request(self, request: Any) -> dict[str, Any]:
        request_id = None
        try:
            if not isinstance(request, dict):
                return self._rpc_error_response(
                    request_id=None,
                    code="BAD_REQUEST",
                    message="RPC request must be a dictionary",
                    details={"received_type": type(request).__name__},
                )

            request_id = request.get("request_id")
            method_name = request.get("method")
            params = request.get("params", {})

            if not isinstance(request_id, str) or not request_id:
                return self._rpc_error_response(
                    request_id=request_id,
                    code="BAD_REQUEST",
                    message="request_id must be a non-empty string",
                    details={},
                )
            if not isinstance(method_name, str) or not method_name:
                return self._rpc_error_response(
                    request_id=request_id,
                    code="BAD_REQUEST",
                    message="method must be a non-empty string",
                    details={},
                )
            if not isinstance(params, dict):
                return self._rpc_error_response(
                    request_id=request_id,
                    code="BAD_REQUEST",
                    message="params must be a dictionary",
                    details={"received_type": type(params).__name__},
                )

            handler = self._rpc_allowlist.get(method_name)
            if handler is None:
                return self._rpc_error_response(
                    request_id=request_id,
                    code="METHOD_NOT_ALLOWED",
                    message=f"Unknown method: {method_name}",
                    details={"method": method_name},
                )

            result = handler(**params)
            return {"request_id": request_id, "ok": True, "result": result}
        except Exception as exc:
            return self._rpc_error_response(
                request_id=request_id,
                code="INTERNAL_ERROR",
                message=str(exc),
                details={
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )

    def _rpc_error_response(
        self,
        request_id: Optional[str],
        *,
        code: str,
        message: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
        }

    # -------------------------------------------------------------------------
    # External registration
    # -------------------------------------------------------------------------
    def register_external(self, handle: ExternalHandle) -> DataRef:
        ref_id = self._new_ref_id()
        meta = handle.get_meta()
        external_params = self._build_external_params(handle)
        # TODO: After the below check, check the type of ExternalHandle
        # (ExternalRasterHandle, vs ExternalSpectraListHandle, vs ExternalSpectrumHandle)
        # then check their "_obj"s to see if they are RAM loaded. Or think of a better way
        # to do this.
        materialization_loc: Literal["none", "ram", "disk"] = (
            "disk" if external_params is not None else "none"
        )

        ref = DataRef(
            kind=meta.kind,
            ref_id=ref_id,
            uri=f"external://{ref_id}",
            disk_format=None,
            shape=tuple(meta.shape),
            dtype=_safe_np_dtype(meta.elem_type),
            chunks=None,
            residency="spill_required",
            materialization_loc=materialization_loc,
            source="external",
            readonly=True,
            external_params=external_params,
        )
        self.data_refs[ref_id] = ref
        self.external_handles[ref_id] = handle
        self.meta_by_ref[ref_id] = meta
        logger.info("register_external ref_id=%s kind=%s", ref_id, ref.kind)
        return ref

    def _build_external_params(self, handle: ExternalHandle) -> Optional[ExternalParams]:
        if handle.kind == "dataset" and hasattr(handle, "dataset_obj"):
            dataset_obj: RasterDataSet = getattr(handle, "dataset_obj")
            serialized = dataset_obj.get_serialized_form()
            serialize_value = serialized.get_serialize_value()
            metadata = serialized.get_metadata()
            if not isinstance(serialize_value, str):
                return None

            impl_type = metadata.get("impl_type")
            impl = dataset_obj.get_impl()
            if isinstance(impl, NetCDF_GDALRasterDataImpl) or impl_type == "NetCDF_GDALRasterDataImpl":
                return ExternalParams(
                    family="dataset",
                    driver="netcdf_gdal",
                    kwargs={
                        "path": serialize_value,
                        "subdataset_name": metadata.get("subdataset_name"),
                    },
                )
            if isinstance(impl, ENVI_GDALRasterDataImpl) or impl_type == "ENVI_GDALRasterDataImpl":
                return ExternalParams(
                    family="dataset",
                    driver="envi_gdal",
                    kwargs={"path": serialize_value},
                )
            if isinstance(impl, GTiff_GDALRasterDataImpl):
                return ExternalParams(
                    family="dataset",
                    driver="gtiff_gdal",
                    kwargs={"path": serialize_value},
                )
            if isinstance(impl, ASC_GDALRasterDataImpl):
                return ExternalParams(
                    family="dataset",
                    driver="asc_gdal",
                    kwargs={"path": serialize_value},
                )
            if isinstance(impl, PDS3_GDALRasterDataImpl) or impl_type == "PDS3_GDALRasterDataImpl":
                return ExternalParams(
                    family="dataset",
                    driver="pds3_gdal",
                    kwargs={"path": serialize_value},
                )
            if isinstance(impl, PDS4_GDALRasterDataImpl):
                return ExternalParams(
                    family="dataset",
                    driver="pds4_gdal",
                    kwargs={"path": serialize_value},
                )
            if isinstance(impl, JP2_GDALRasterDataImpl) or impl_type == "JP2_GDAL_PDR_RasterDataImpl":
                return ExternalParams(
                    family="dataset",
                    driver="jp2_gdal",
                    kwargs={"path": serialize_value},
                )
            if isinstance(impl, GDALRasterDataImpl) or impl_type == "GDALRasterDataImpl":
                return ExternalParams(
                    family="dataset",
                    driver="gdal_generic",
                    kwargs={"path": serialize_value},
                )
            return None

        if handle.kind == "spectra_list" and hasattr(handle, "lib_obj"):
            lib_obj = getattr(handle, "lib_obj")
            if isinstance(lib_obj, ENVISpectralLibrary):
                filepaths = lib_obj.get_filepaths()
                if filepaths:
                    return ExternalParams(
                        family="spectra_list",
                        driver="envi_sli",
                        kwargs={"path": filepaths[0]},
                    )
            return None

        # External spectrum refs are currently RAM/no-disk reconstruction only.
        return None

    # -------------------------------------------------------------------------
    # Allocation
    # -------------------------------------------------------------------------
    def allocate_data(
        self,
        desc: AllocationRequest,
        *,
        preferred_storage: Optional[DiskFormat] = None,
        ttl_seconds: Optional[int] = None,
        owner_plan_id: Optional[str] = None,
        planned_consumer_plan_ids: Optional[set[str]] = None,
    ) -> DataRef:
        _ = ttl_seconds
        ref_id = self._new_ref_id()
        kind: RefKind = desc.kind
        effective_delete_policy = desc.delete_policy if desc.delete_policy is not None else DeletePolicy.KEEP

        can_allocate_shared = desc.kind != "json" and desc.shape is not None and desc.dtype is not None
        want_ram = desc.residency == "ram_cacheable"
        if want_ram and desc.kind == "json":
            uri = f"mem://{ref_id}"
            self.ram_objects[uri] = {}
            self.ram_est_bytes[uri] = desc.size_est
            self._ram_used_bytes += desc.size_est
            ref = DataRef(
                kind="json",
                ref_id=ref_id,
                uri=uri,
                disk_format=None,
                residency=desc.residency,
                materialization_loc="ram",
                source="internal",
                readonly=False,
            )
            self.data_refs[ref_id] = ref
            self.meta_by_ref[ref_id] = self._meta_from_ref(ref)
            self._create_lease_record(
                ref,
                backend_kind="json",
                delete_policy=effective_delete_policy,
                owner_plan_id=owner_plan_id,
                planned_consumer_plan_ids=planned_consumer_plan_ids,
            )
            logger.info(
                "Created DataRef ref_id=%s uri=%s materialization_loc=%s disk_format=%s",
                ref.ref_id,
                ref.uri,
                ref.materialization_loc,
                ref.disk_format,
            )
            return ref

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
            self._create_lease_record(
                ref,
                backend_kind="ram_shm",
                delete_policy=effective_delete_policy,
                owner_plan_id=owner_plan_id,
                planned_consumer_plan_ids=planned_consumer_plan_ids,
            )
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
            self._create_lease_record(
                ref,
                backend_kind="json",
                delete_policy=effective_delete_policy,
                owner_plan_id=owner_plan_id,
                planned_consumer_plan_ids=planned_consumer_plan_ids,
            )
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
            self._create_lease_record(
                ref,
                backend_kind="memmap",
                delete_policy=effective_delete_policy,
                owner_plan_id=owner_plan_id,
                planned_consumer_plan_ids=planned_consumer_plan_ids,
            )
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
            self._create_lease_record(
                ref,
                backend_kind="zarr",
                delete_policy=effective_delete_policy,
                owner_plan_id=owner_plan_id,
                planned_consumer_plan_ids=planned_consumer_plan_ids,
            )
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

        if canonical.materialization_loc == "ram":
            return RamAccessDescriptor(
                ref=canonical,
                mode=mode,
                region=region,
                meta=meta,
                region_meta=region_meta,
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

    def get_ram_descriptor(self, uri: str) -> SharedMemArrayDescriptor:
        if not uri.startswith("mem://"):
            raise ValueError(f"Expected mem:// uri, got: {uri}")
        try:
            descriptor = self.ram_objects[uri]
        except KeyError as e:
            raise KeyError(f"No RAM object for uri={uri}") from e
        if not isinstance(descriptor, SharedMemArrayDescriptor):
            raise TypeError(f"RAM object for uri={uri} is not a SharedMemArrayDescriptor")
        return descriptor

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

    def get_lease_record(self, ref: Union[DataRef, str]) -> StorageLeaseRecord:
        """Return the service-owned lifetime record for an internally managed ref."""

        ref_id = ref if isinstance(ref, str) else self.read_data_ref(ref).ref_id
        if ref_id not in self.lease_records:
            raise KeyError(f"No storage lease record for ref_id={ref_id}")
        return self.lease_records[ref_id]

    def set_delete_policy(self, ref: Union[DataRef, str], policy: DeletePolicy) -> StorageLeaseRecord:
        """Update retention policy and immediately re-evaluate reclaimability."""

        ref_id = ref if isinstance(ref, str) else self.read_data_ref(ref).ref_id
        record = self.get_lease_record(ref_id)
        record.delete_policy = policy
        return self.try_reclaim(ref_id)

    def register_plan_consumer(self, ref: Union[DataRef, str], plan_id: str) -> StorageLeaseRecord:
        """Record that a task plan still depends on this managed ref."""

        record = self.get_lease_record(ref)
        record.planned_consumer_plan_ids.add(plan_id)
        return self.try_reclaim(record.ref_id)

    def release_plan_consumer(self, ref: Union[DataRef, str], plan_id: str) -> StorageLeaseRecord:
        """Release one plan-level dependency and re-run the reclaim decision."""

        record = self.get_lease_record(ref)
        record.planned_consumer_plan_ids.discard(plan_id)
        return self.try_reclaim(record.ref_id)

    def mark_producer_completed(self, ref: Union[DataRef, str]) -> StorageLeaseRecord:
        """Mark an output as successfully produced and re-run the reclaim decision."""

        return self._mark_producer_state(ref, ProducerState.COMPLETED)

    def mark_producer_failed(self, ref: Union[DataRef, str]) -> StorageLeaseRecord:
        """Mark an output as failed so cleanup can proceed through the same reclaim path."""

        return self._mark_producer_state(ref, ProducerState.FAILED)

    def mark_producer_aborted(self, ref: Union[DataRef, str]) -> StorageLeaseRecord:
        """Mark an output as aborted so cleanup can proceed through the same reclaim path."""

        return self._mark_producer_state(ref, ProducerState.ABORTED)

    def try_reclaim(self, ref: Union[DataRef, str]) -> StorageLeaseRecord:
        """
        Re-evaluate whether a managed ref should remain live, become pending delete,
        or be reclaimed immediately.

        The policy/state split matters here:
        - `delete_policy` says whether reclamation is desired.
        - `deletion_state` records what is currently true at runtime.
        """
        record = self.get_lease_record(ref)
        if record.deletion_state == DeletionState.DELETED:
            return record
        if self._should_reclaim(record):
            self._delete_managed_ref(record.ref_id)
            record.deletion_state = DeletionState.DELETED
            return record
        if self._is_pending_delete(record):
            record.deletion_state = DeletionState.PENDING_DELETE
        else:
            record.deletion_state = DeletionState.LIVE
        return record

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _choose_disk_format(self, desc: AllocationRequest) -> DiskFormat:
        if desc.kind == "json":
            return "json"
        return "memmap"

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

    def _create_lease_record(
        self,
        ref: DataRef,
        *,
        backend_kind: str,
        delete_policy: DeletePolicy,
        owner_plan_id: Optional[str] = None,
        planned_consumer_plan_ids: Optional[set[str]] = None,
        external_owned: bool = False,
    ) -> StorageLeaseRecord:
        """Create the initial lifetime record for a newly allocated managed ref."""

        record = StorageLeaseRecord(
            ref_id=ref.ref_id,
            backend_kind=backend_kind,
            owner_plan_id=owner_plan_id,
            planned_consumer_plan_ids=set(planned_consumer_plan_ids or ()),
            delete_policy=delete_policy,
            producer_state=ProducerState.WRITING,
            deletion_state=DeletionState.LIVE,
            external_owned=external_owned,
        )
        self.lease_records[ref.ref_id] = record
        return record

    def _mark_producer_state(self, ref: Union[DataRef, str], state: ProducerState) -> StorageLeaseRecord:
        """Transition producer lifecycle state and immediately re-check reclamation."""

        record = self.get_lease_record(ref)
        record.producer_state = state
        return self.try_reclaim(record.ref_id)

    def _is_terminal_producer_state(self, state: ProducerState) -> bool:
        return state in {ProducerState.COMPLETED, ProducerState.FAILED, ProducerState.ABORTED}

    def _is_pending_delete(self, record: StorageLeaseRecord) -> bool:
        """
        True when policy says the object should be reclaimed, but some live dependency
        still blocks deletion.
        """

        return (
            record.delete_policy == DeletePolicy.DELETE_WHEN_RELEASABLE
            and self._is_terminal_producer_state(record.producer_state)
            and not record.external_owned
            and (bool(record.planned_consumer_plan_ids) or bool(record.borrowers) or bool(record.pins))
        )

    def _should_reclaim(self, record: StorageLeaseRecord) -> bool:
        """Return True only when the record is fully reclaimable under service rules."""

        return (
            record.delete_policy == DeletePolicy.DELETE_WHEN_RELEASABLE
            and self._is_terminal_producer_state(record.producer_state)
            and not record.planned_consumer_plan_ids
            and not record.borrowers
            and not record.pins
            and not record.external_owned
        )

    def _delete_managed_ref(self, ref_id: str) -> None:
        """
        Remove service-owned backing storage for an internal ref.

        External refs are intentionally excluded from this path; the service may
        track them for access, but it does not own their underlying source object.
        """

        ref = self.data_refs.get(ref_id)
        if ref is None:
            return

        self.meta_by_ref.pop(ref_id, None)
        self.data_refs.pop(ref_id, None)

        if ref.materialization_loc == "ram":
            descriptor = self.ram_objects.pop(ref.uri, None)
            est = self.ram_est_bytes.pop(ref.uri, 0)
            self._ram_used_bytes = max(0, self._ram_used_bytes - est)
            owner_handle = self._shared_mem_handles.pop(ref.uri, None)
            if descriptor is not None and isinstance(descriptor, SharedMemArrayDescriptor):
                try:
                    # Unlink while the service-owned handle is still open so the
                    # shared-memory name remains resolvable on Windows. After the
                    # name is removed, closing the final owner handle releases the
                    # backing storage.
                    backing = SharedMemory(name=descriptor.name, create=False)
                    try:
                        backing.unlink()
                    finally:
                        backing.close()
                except FileNotFoundError:
                    pass
                except Exception:
                    logger.debug("SharedMemory unlink failed during ref deletion", exc_info=True)
            if owner_handle is not None:
                try:
                    owner_handle.close()
                except Exception:
                    logger.debug("SharedMemory owner-handle close failed during ref deletion", exc_info=True)
            return

        if ref.materialization_loc != "disk":
            return

        try:
            if ref.disk_format in {"memmap", "json"}:
                self._file_uri_to_path(ref.uri).unlink(missing_ok=True)
            elif ref.disk_format == "zarr":
                store_path = self._zarr_uri_to_path(ref.uri)
                if store_path.exists():
                    import shutil

                    shutil.rmtree(store_path, ignore_errors=True)
        except Exception:
            logger.debug("Managed ref deletion failed for ref_id=%s", ref_id, exc_info=True)

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
        self._shared_mem_handles_names[uri] = shm.name
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
        self._shared_mem_handles_names[ref.uri] = shm.name
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
