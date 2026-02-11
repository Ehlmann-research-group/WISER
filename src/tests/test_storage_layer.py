"""Unit, functional, and integration tests for the storage layer
in the task management and execution system.
"""
import tempfile
import unittest

import numpy as np

import tests.context
from wiser.utils.primitives import AllocationRequest
from wiser.utils.storage_layer import StorageLayer

import pdb


class TestStorageLayer(unittest.TestCase):
    def test_allocate_data_dataset_spill_required_allocates_disk_memmap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)
            request = AllocationRequest(
                name="dataset_spill",
                kind="dataset",
                residency="spill_required",
                size_est=288,
                shape=(3, 4, 6),
                dtype=np.float32,
                chunks=None,
            )

            ref = storage.allocate_data(request)

            self.assertEqual(ref.kind, "dataset")
            self.assertEqual(ref.residency, "spill_required")
            self.assertEqual(ref.materialization_loc, "disk")
            self.assertEqual(ref.disk_format, "memmap")
            self.assertEqual(ref.shape, (3, 4, 6))
            self.assertEqual(ref.dtype, np.float32)
            self.assertIsNone(ref.chunks)
            self.assertTrue(ref.uri.startswith("file://"))
            self.assertTrue(storage._file_uri_to_path(ref.uri).exists())

    def test_allocate_data_dataset_ram_cacheable_allocates_ram(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)

            request_size_est = 288
            request = AllocationRequest(
                name="dataset_ram",
                kind="dataset",
                residency="ram_cacheable",
                size_est=request_size_est,
                shape=(3, 4, 6),
                dtype=np.float32,
                chunks=None,
            )

            ref = storage.allocate_data(request)

            self.assertEqual(ref.kind, "dataset")
            self.assertEqual(ref.residency, "ram_cacheable")
            self.assertEqual(ref.materialization_loc, "ram")
            self.assertIsNone(ref.disk_format)
            self.assertEqual(ref.shape, (3, 4, 6))
            self.assertEqual(ref.dtype, np.float32)
            self.assertIsNone(ref.chunks)
            self.assertTrue(ref.uri.startswith("mem://"))
            self.assertIn(ref.uri, storage.mem_backed_data)
            arr = storage.mem_backed_data[ref.uri]
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.shape, (3, 4, 6))
            self.assertEqual(arr.dtype, np.dtype(np.float32))
            self.assertEqual(storage.mem_backed_est[ref.uri], request_size_est)

    def test_allocate_data_dataset_spill_required_with_preferred_zarr(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)
            request = AllocationRequest(
                name="dataset_spill_zarr",
                kind="dataset",
                residency="spill_required",
                size_est=288,
                shape=(3, 4, 6),
                dtype=np.float32,
                chunks=None,
            )

            ref = storage.allocate_data(request, preferred_storage="zarr")

            self.assertEqual(ref.kind, "dataset")
            self.assertEqual(ref.residency, "spill_required")
            self.assertEqual(ref.materialization_loc, "disk")
            self.assertEqual(ref.disk_format, "zarr")
            self.assertEqual(ref.shape, (3, 4, 6))
            self.assertEqual(ref.dtype, np.float32)
            self.assertIsNone(ref.chunks)
            self.assertTrue(ref.uri.startswith("zarr://"))
            path = storage._zarr_uri_to_path(ref.uri)
            self.assertTrue(path.exists() and path.suffix == ".zarr")
