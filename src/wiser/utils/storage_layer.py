# ==================== imports (keep with StorageLayer) ====================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import zarr

from .primitives import (
    DataRef,
    AllocationRequest,
    DiskFormat,
    DataRegion,
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
        pass

    def write_region(self, data: DataRef, chunk_ref: DataRegion, value: Any) -> bool:
        """
        Write `value` into `data` at region `chunk_ref`.

        Returns True on success, False on error.
        """
        pass

    def write_data(self, ref: DataRef, value: Any) -> None:
        """
        Write entire object for `ref`.

        - RAM: store/overwrite in mem_backed_data (or assign into ndarray)
        - JSON: json dump to file
        - memmap: open mmap and assign
        - zarr: open and assign
        """
        pass

    def read_data(self, ref_id: str) -> DataRef:
        """
        Return the DataRef handle (metadata + locator).
        """
        pass

    def read_region(self, ref_id: str, chunk_ref: DataRegion) -> Any:
        """
        Read a region specified by `chunk_ref`.
        For JSON refs, ignores chunk_ref and returns the whole object.
        """
        pass

    # -------------------------------------------------------------------------
    # Policy helpers
    # -------------------------------------------------------------------------

    def _choose_disk_format(self, desc: AllocationRequest) -> DiskFormat:
        pass

    def _can_fit_in_ram(self, desc: AllocationRequest) -> bool:
        pass

    def _estimate_request_bytes(self, desc: AllocationRequest) -> int:
        pass

    # -------------------------------------------------------------------------
    # Allocation helpers
    # -------------------------------------------------------------------------

    def _allocate_in_ram_object(self, desc: AllocationRequest) -> Any:
        pass

    # -------------------------------------------------------------------------
    # Region slicing helpers
    # -------------------------------------------------------------------------

    def _read_region_from_array(self, arr: Any, chunk_ref: DataRegion) -> Any:
        pass

    def _write_region_into_array(self, arr: Any, chunk_ref: DataRegion, value: Any) -> None:
        pass

    # -------------------------------------------------------------------------
    # RAM helpers
    # -------------------------------------------------------------------------

    def _get_ram_object(self, uri: str) -> Any:
        pass

    def _estimate_bytes(self, obj: Any, fallback_est: int) -> int:
        pass

    # -------------------------------------------------------------------------
    # Disk helpers
    # -------------------------------------------------------------------------

    def _open_zarr_array(self, uri: str, mode: str) -> zarr.Array:
        pass

    # -------------------------------------------------------------------------
    # URI helpers
    # -------------------------------------------------------------------------

    def _new_ref_id(self) -> str:
        pass

    def _path_to_file_uri(self, path: Path) -> str:
        pass

    def _file_uri_to_path(self, uri: str) -> Path:
        pass

    def _path_to_zarr_uri(self, path: Path) -> str:
        pass

    def _zarr_uri_to_path(self, uri: str) -> Path:
        pass
