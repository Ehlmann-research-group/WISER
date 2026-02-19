from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol
import json
import logging
import uuid
from urllib.parse import urlparse, quote, unquote

import numpy as np
import zarr

from .primitives import (
    AllocationRequest,
    DataMeta,
    DataRef,
    DatasetRegionRef,
    RefKind,
    RegionMeta,
    DataRegion,
    DiskFormat,
    InputKind,
    SpectraBatchRef,
    SpectrumRef,
    temp_dir,
)

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.spectral_library import SpectralLibrary
    from wiser.raster.spectrum import Spectrum


logger = logging.getLogger(__name__)


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

    def read_region(self, region: DataRegion) -> Any:
        ...

    def get_meta(self) -> DataMeta:
        ...

    def get_region_meta(self, region: DataRegion) -> RegionMeta:
        ...


@dataclass
class ExternalRasterHandle:
    dataset_obj: "RasterDataSet"
    kind: InputKind = "dataset"

    def read_region(self, region: DataRegion) -> Any:
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
        # RasterDataSet uses [band][y][x]; StorageLayer dataset regions use [y][x][band].
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


@dataclass
class ExternalSpectrumHandle:
    spectrum_obj: "Spectrum"
    kind: InputKind = "spectrum"

    def read_region(self, region: DataRegion) -> Any:
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


@dataclass
class ExternalSpectralLibraryHandle:
    lib_obj: "SpectralLibrary"
    kind: InputKind = "spectra_list"

    def read_region(self, region: DataRegion) -> Any:
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
    external_handles: Dict[str, ExternalHandle] = field(default_factory=dict)
    meta_by_ref: Dict[str, DataMeta] = field(default_factory=dict)

    # Track rough RAM usage for budget decisions (best-effort)
    _ram_used_bytes: int = 0

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def register_external_dataset(self, obj: "RasterDataSet") -> DataRef:
        return self._register_external(ExternalRasterHandle(dataset_obj=obj))

    def register_external_spectrum(self, obj: "Spectrum") -> DataRef:
        return self._register_external(ExternalSpectrumHandle(spectrum_obj=obj))

    def register_external_spectral_library(self, obj: "SpectralLibrary") -> DataRef:
        return self._register_external(ExternalSpectralLibraryHandle(lib_obj=obj))

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
            logger.info(
                "Allocating RAM object for ref_id=%s uri=%s kind=%s shape=%s dtype=%s",
                ref_id,
                uri,
                kind,
                desc.shape,
                desc.dtype,
            )
            self.mem_backed_data[uri] = obj
            self.mem_backed_est[uri] = desc.size_est
            self._ram_used_bytes += self._estimate_bytes(obj, fallback_est=desc.size_est)

            ref = DataRef(
                kind=kind,
                ref_id=ref_id,
                uri=uri,
                disk_format=None,  # not disk-backed
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
                dtype=desc.dtype,
                chunks=None,
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

    def write_region(self, data: DataRef, chunk_ref: DataRegion, value: Any) -> None:
        """
        Write `value` into `data` at region `chunk_ref`. Should not be used with json.

        Returns True on success, False on error.
        """
        self._ensure_writable(data)
        try:
            if data.materialization_loc == "ram":
                obj = self._get_ram_object(data.uri)
                self._write_region_into_array(obj, chunk_ref, value)
                logger.info(
                    "write_region success materialization_loc=%s uri=%s disk_format=%s region=%s",
                    data.materialization_loc,
                    data.uri,
                    data.disk_format,
                    chunk_ref,
                )
                return

            if data.materialization_loc == "disk":
                if data.disk_format == "json":
                    raise TypeError("write_region is not supported for JSON")
                if data.disk_format == "memmap":
                    path = self._file_uri_to_path(data.uri)
                    arr = np.load(str(path), mmap_mode="r+")
                    self._write_region_into_array(arr, chunk_ref, value)
                    if hasattr(arr, "flush"):
                        arr.flush()
                    logger.info(
                        "write_region success materialization_loc=%s uri=%s disk_format=%s region=%s",
                        data.materialization_loc,
                        data.uri,
                        data.disk_format,
                        chunk_ref,
                    )
                    return
                if data.disk_format == "zarr":
                    z = self._open_zarr_array(data.uri, mode="r+")
                    self._write_region_into_array(z, chunk_ref, value)
                    logger.info(
                        "write_region success materialization_loc=%s uri=%s disk_format=%s region=%s",
                        data.materialization_loc,
                        data.uri,
                        data.disk_format,
                        chunk_ref,
                    )
                    return

            raise ValueError(
                f"Unsupported materialization/storage: {data.materialization_loc}/{data.disk_format}"
            )
        except Exception as e:
            logger.exception(
                "write_region failed materialization_loc=%s uri=%s disk_format=%s region=%s",
                data.materialization_loc,
                data.uri,
                data.disk_format,
                chunk_ref,
            )
            raise e

    def write_data(self, ref: DataRef, value: Any) -> None:
        """
        Write entire object for `ref`.

        - RAM: store/overwrite in mem_backed_data (or assign into ndarray)
        - JSON: json dump to file
        - memmap: open mmap and assign
        - zarr: open and assign
        """
        self._ensure_writable(ref)
        logger.info(
            "write_data uri=%s materialization_loc=%s disk_format=%s",
            ref.uri,
            ref.materialization_loc,
            ref.disk_format,
        )
        if ref.materialization_loc == "ram":
            existing = self.mem_backed_data.get(ref.uri, None)
            if isinstance(existing, np.ndarray) and isinstance(value, np.ndarray):
                existing[...] = value
            else:
                # replace; update rough accounting
                est_bytes = self.mem_backed_est[ref.uri]
                if existing is not None:
                    self._ram_used_bytes -= self._estimate_bytes(existing, fallback_est=est_bytes)
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
            ref = self.data_refs[ref_id]
            logger.info("read_data ref_id=%s uri=%s", ref_id, ref.uri)
            return ref
        except KeyError as e:
            raise KeyError(f"Unknown ref_id: {ref_id}") from e

    def read_region(self, ref_id: str, chunk_ref: DataRegion) -> Any:
        """
        Read a region specified by `chunk_ref`.
        For JSON refs, ignores chunk_ref and returns the whole object.
        """
        ref = self.read_data(ref_id)
        if ref.source == "external":
            logger.info("read_region external ref_id=%s region=%s", ref.ref_id, chunk_ref)
            return self.external_handles[ref.ref_id].read_region(chunk_ref)

        logger.info(
            "read_region materialization_loc=%s uri=%s disk_format=%s region=%s",
            ref.materialization_loc,
            ref.uri,
            ref.disk_format,
            chunk_ref,
        )

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

    def get_meta(self, ref_id: str) -> DataMeta:
        if ref_id in self.meta_by_ref:
            return self.meta_by_ref[ref_id]
        ref = self.read_data(ref_id)
        if ref.source == "external":
            meta = self.external_handles[ref.ref_id].get_meta()
        else:
            meta = self._meta_from_ref(ref)
        self.meta_by_ref[ref_id] = meta
        return meta

    def get_region_meta(self, ref_id: str, region: DataRegion) -> RegionMeta:
        ref = self.read_data(ref_id)
        if ref.source == "external":
            return self.external_handles[ref.ref_id].get_region_meta(region)
        return _derive_region_meta(self.get_meta(ref_id), region)

    def set_meta(self, ref_id: str, meta: DataMeta) -> None:
        ref = self.read_data(ref_id)
        self._ensure_writable(ref, op_name="set_meta")
        self.meta_by_ref[ref_id] = meta

    def update_meta(self, ref_id: str, **fields: Any) -> DataMeta:
        ref = self.read_data(ref_id)
        self._ensure_writable(ref, op_name="update_meta")
        updated = replace(self.get_meta(ref_id), **fields)
        self.meta_by_ref[ref_id] = updated
        return updated

    # -------------------------------------------------------------------------
    # Policy helpers
    # -------------------------------------------------------------------------

    def _choose_disk_format(self, desc: AllocationRequest) -> DiskFormat:
        # json requests always go to json
        if desc.kind == "json":
            return "json"
        # numeric defaults to memmap; caller can override with preferred_storage="zarr"
        return "memmap"

    def _register_external(self, handle: ExternalHandle) -> DataRef:
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

    def _ensure_writable(self, ref: DataRef, op_name: str = "write") -> None:
        if ref.readonly or ref.source == "external":
            raise PermissionError(f"{op_name} is not allowed for external/read-only refs: {ref.ref_id}")

    def _meta_from_ref(self, ref: DataRef) -> DataMeta:
        return DataMeta(
            kind=ref.kind,
            shape=tuple(ref.shape) if ref.shape is not None else (),
            elem_type=_safe_np_dtype(ref.dtype),
        )

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
        raw = parsed.path if parsed.path else parsed.netloc

        p = Path(unquote(raw))
        return p.resolve()

    def _path_to_zarr_uri(self, path: Path) -> str:
        p = path.resolve()
        return f"zarr://{quote(str(p))}"

    def _zarr_uri_to_path(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "zarr":
            raise ValueError(f"Not a zarr URI: {uri}")
        raw = parsed.path if parsed.path else parsed.netloc

        p = Path(unquote(raw))
        return p.resolve()
