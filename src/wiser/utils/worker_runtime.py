from __future__ import annotations

from typing import Optional

from .storage_client import StorageClient

_PROCESS_STORAGE_CLIENT: Optional[StorageClient] = None


def initialize_thread_worker() -> None:
    # Reserved for future thread-local setup.
    return


def initialize_process_storage_client(service_address: tuple[str, int], service_authkey: bytes) -> None:
    global _PROCESS_STORAGE_CLIENT
    _PROCESS_STORAGE_CLIENT = StorageClient(
        service=None,  # type: ignore[arg-type]
        service_address=service_address,
        service_authkey=service_authkey,
    )


def get_process_storage_client() -> StorageClient:
    if _PROCESS_STORAGE_CLIENT is None:
        raise RuntimeError(
            "Process storage client is not initialized in this worker. "
            "Make sure ProcessPoolExecutor uses initialize_process_storage_client as initializer."
        )
    return _PROCESS_STORAGE_CLIENT
