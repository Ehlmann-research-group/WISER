from concurrent.futures import ProcessPoolExecutor, wait
import tempfile
from typing import Optional
import unittest

import numpy as np
import tests.context

from wiser.utils.primitives import AllocationRequest, DatasetRegionRef
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_service import StorageService

_WORKER_CLIENT: Optional[StorageClient] = None


def _pool_init_storage_client(service_address: tuple[str, int], service_authkey: bytes) -> None:
    # We make this global (module-level) because this function is called per process, so each
    # process will have it's own _WORKER_CLIENT. This function is called in the process's
    # intializer so there is no return value.
    global _WORKER_CLIENT
    _WORKER_CLIENT = StorageClient(
        service=None,  # type: ignore[arg-type]
        service_address=service_address,
        service_authkey=service_authkey,
    )


def _pool_write_block(ref, y0: int, y1: int, fill_value: float) -> tuple[int, int, float]:
    if _WORKER_CLIENT is None:
        raise RuntimeError("Worker StorageClient was not initialized")
    block = np.full((y1 - y0, 6, 7), fill_value, dtype=np.float32)
    _WORKER_CLIENT.write_region(
        ref,
        DatasetRegionRef(y0=y0, y1=y1, x0=0, x1=6, b0=0, b1=7),
        block,
    )
    return (y0, y1, fill_value)


class TestStorageConnectionInfra(unittest.TestCase):
    def test_service_bootstrap_exposes_address_and_authkey(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            try:
                address, authkey = service.get_connection_bootstrap()
                self.assertEqual(address[0], "127.0.0.1")
                self.assertIsInstance(address[1], int)
                self.assertGreater(address[1], 0)
                self.assertIsInstance(authkey, bytes)
                self.assertGreater(len(authkey), 0)
            finally:
                service.close()

    def test_client_connects_on_init_and_close_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)
                self.assertIsNotNone(client._conn)
                self.assertTrue(client._conn.readable)
                self.assertTrue(client._conn.writable)
            finally:
                if client is not None:
                    client.close()
                    client.close()
                service.close()
                service.close()

    def test_client_rpc_unknown_method_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)
                with self.assertRaises(RuntimeError) as exc:
                    client._rpc_call("definitely_not_allowed")
                self.assertIn("METHOD_NOT_ALLOWED", str(exc.exception))
            finally:
                if client is not None:
                    client.close()
                service.close()

    def _run_parallel_write_test(
        self,
        *,
        preferred_storage,
        residency: str,
        ram_byte_limit=None,
        chunks=None,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir, ram_byte_limit=ram_byte_limit)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)
                ref = service.allocate_data(
                    AllocationRequest(
                        name="parallel_write_array",
                        kind="dataset",
                        residency=residency,
                        size_est=5 * 6 * 7 * np.dtype(np.float32).itemsize,
                        shape=(5, 6, 7),
                        dtype=np.dtype(np.float32),
                        chunks=chunks,
                    ),
                    preferred_storage=preferred_storage,
                )

                with ProcessPoolExecutor(
                    max_workers=3,
                    initializer=_pool_init_storage_client,
                    initargs=(address, authkey),
                ) as pool:
                    futures = [
                        pool.submit(_pool_write_block, ref, 0, 2, 1.0),
                        pool.submit(_pool_write_block, ref, 2, 3, 2.0),
                        pool.submit(_pool_write_block, ref, 3, 5, 3.0),
                    ]
                    wait(futures)
                    for future in futures:
                        future.result()

                got, _ = client.read_data(ref)
                expected = np.empty((5, 6, 7), dtype=np.float32)
                expected[0:2, :, :] = 1.0
                expected[2:3, :, :] = 2.0
                expected[3:5, :, :] = 3.0
                np.testing.assert_allclose(got, expected)
            finally:
                if client is not None:
                    client.close()
                service.close()

    def test_process_pool_parallel_writes_ram_accessible(self):
        self._run_parallel_write_test(
            preferred_storage=None,
            residency="ram_cacheable",
            ram_byte_limit=10_000_000,
        )

    def test_process_pool_parallel_writes_memmap(self):
        self._run_parallel_write_test(
            preferred_storage="memmap",
            residency="spill_required",
        )

    def test_process_pool_parallel_writes_zarr(self):
        self._run_parallel_write_test(
            preferred_storage="zarr",
            residency="spill_required",
            chunks=(1, 6, 7),
        )
