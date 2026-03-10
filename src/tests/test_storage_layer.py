"""Unit, functional, and integration tests for the storage layer
in the task management and execution system.
"""
import tempfile
import unittest

import numpy as np
from astropy import units as u
import tests.context
from wiser.utils.primitives import (
    AllocationRequest,
    DataMeta,
    DatasetRegionRef,
    SpectraBatchRef,
    SpectraBatchScheme,
    SpectrumRef,
)
from wiser.utils.task_system import SpectraListPlanMeta
from wiser.utils.storage_layer import StorageLayer


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
                dtype=np.dtype(np.float32),
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
            path = storage._file_uri_to_path(ref.uri)
            self.assertTrue(path.exists() and path.suffix == ".npy")

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
                dtype=np.dtype(np.float32),
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
                dtype=np.dtype(np.float32),
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

    def test_allocate_data_dataset_ram_cacheable_fits_in_ram_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 3*4*6 float32 values = 288 bytes, so this should fit in RAM.
            storage = StorageLayer(root_dir=tmp_dir, ram_byte_limit=288)
            request = AllocationRequest(
                name="dataset_ram_fit_limit",
                kind="dataset",
                residency="ram_cacheable",
                size_est=288,
                shape=(3, 4, 6),
                dtype=np.dtype(np.float32),
                chunks=None,
            )

            ref = storage.allocate_data(request)

            self.assertEqual(ref.kind, "dataset")
            self.assertEqual(ref.residency, "ram_cacheable")
            self.assertEqual(ref.materialization_loc, "ram")
            self.assertIsNone(ref.disk_format)
            self.assertTrue(ref.uri.startswith("mem://"))
            self.assertIn(ref.uri, storage.mem_backed_data)
            self.assertEqual(storage.mem_backed_est[ref.uri], 288)

    def test_allocate_data_dataset_ram_cacheable_spills_when_over_ram_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 3*4*6 float32 values = 288 bytes, so this should spill to disk.
            storage = StorageLayer(root_dir=tmp_dir, ram_byte_limit=287)
            request = AllocationRequest(
                name="dataset_ram_spill_limit",
                kind="dataset",
                residency="ram_cacheable",
                size_est=288,
                shape=(3, 4, 6),
                dtype=np.dtype(np.float32),
                chunks=None,
            )

            ref = storage.allocate_data(request)

            self.assertEqual(ref.kind, "dataset")
            self.assertEqual(ref.residency, "ram_cacheable")
            self.assertEqual(ref.materialization_loc, "disk")
            self.assertEqual(ref.disk_format, "memmap")
            self.assertTrue(ref.uri.startswith("file://"))
            path = storage._file_uri_to_path(ref.uri)
            self.assertTrue(path.exists() and path.suffix == ".npy")

    def test_write_and_read_full_dataset_all_threes_ram(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir, ram_byte_limit=560)
            request = AllocationRequest(
                name="dataset_all_threes_ram",
                kind="dataset",
                residency="ram_cacheable",
                size_est=560,  # 4*5*7 float32 values
                shape=(4, 5, 7),
                dtype=np.dtype(np.float32),
                chunks=None,
            )
            ref = storage.allocate_data(request)
            self.assertEqual(ref.materialization_loc, "ram")

            value = np.full((4, 5, 7), 3, dtype=np.float32)
            storage.write_data(ref, value)

            region = DatasetRegionRef(y0=0, y1=4, x0=0, x1=5, b0=0, b1=7)
            result = storage.read_region(ref.ref_id, region)
            np.testing.assert_array_equal(result, value)

    def test_write_and_read_full_dataset_all_threes_zarr_disk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)
            request = AllocationRequest(
                name="dataset_all_threes_zarr",
                kind="dataset",
                residency="spill_required",
                size_est=560,  # 4*5*7 float32 values
                shape=(4, 5, 7),
                dtype=np.dtype(np.float32),
                chunks=None,
            )
            ref = storage.allocate_data(request, preferred_storage="zarr")
            self.assertEqual(ref.materialization_loc, "disk")
            self.assertEqual(ref.disk_format, "zarr")

            value = np.full((4, 5, 7), 3, dtype=np.float32)
            storage.write_data(ref, value)

            region = DatasetRegionRef(y0=0, y1=4, x0=0, x1=5, b0=0, b1=7)
            result = storage.read_region(ref.ref_id, region)
            np.testing.assert_array_equal(result, value)

    def test_write_regions_then_read_back_dataset_ram(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir, ram_byte_limit=560)
            request = AllocationRequest(
                name="dataset_regions_ram",
                kind="dataset",
                residency="ram_cacheable",
                size_est=560,  # 4*5*7 float32 values
                shape=(4, 5, 7),
                dtype=np.dtype(np.float32),
                chunks=None,
            )
            ref = storage.allocate_data(request)
            self.assertEqual(ref.materialization_loc, "ram")

            region_first = DatasetRegionRef(y0=0, y1=1, x0=0, x1=5, b0=0, b1=7)
            region_rest = DatasetRegionRef(y0=1, y1=4, x0=0, x1=5, b0=0, b1=7)
            value_first = np.full((1, 5, 7), 2, dtype=np.float32)
            value_rest = np.full((3, 5, 7), 3, dtype=np.float32)

            storage.write_region(ref, region_first, value_first)
            storage.write_region(ref, region_rest, value_rest)

            got_first = storage.read_region(ref.ref_id, region_first)
            got_rest = storage.read_region(ref.ref_id, region_rest)
            np.testing.assert_array_equal(got_first, value_first)
            np.testing.assert_array_equal(got_rest, value_rest)

            whole_region = DatasetRegionRef(y0=0, y1=4, x0=0, x1=5, b0=0, b1=7)
            got_whole = storage.read_region(ref.ref_id, whole_region)
            expected = np.full((4, 5, 7), 3, dtype=np.float32)
            expected[0:1, :, :] = 2
            np.testing.assert_array_equal(got_whole, expected)

    def test_write_regions_then_read_back_dataset_zarr_disk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)
            request = AllocationRequest(
                name="dataset_regions_zarr",
                kind="dataset",
                residency="spill_required",
                size_est=560,  # 4*5*7 float32 values
                shape=(4, 5, 7),
                dtype=np.dtype(np.float32),
                chunks=None,
            )
            ref = storage.allocate_data(request, preferred_storage="zarr")
            self.assertEqual(ref.materialization_loc, "disk")
            self.assertEqual(ref.disk_format, "zarr")

            region_first = DatasetRegionRef(y0=0, y1=1, x0=0, x1=5, b0=0, b1=7)
            region_rest = DatasetRegionRef(y0=1, y1=4, x0=0, x1=5, b0=0, b1=7)
            value_first = np.full((1, 5, 7), 2, dtype=np.float32)
            value_rest = np.full((3, 5, 7), 3, dtype=np.float32)

            storage.write_region(ref, region_first, value_first)
            storage.write_region(ref, region_rest, value_rest)

            got_first = storage.read_region(ref.ref_id, region_first)
            got_rest = storage.read_region(ref.ref_id, region_rest)
            np.testing.assert_array_equal(got_first, value_first)
            np.testing.assert_array_equal(got_rest, value_rest)

            whole_region = DatasetRegionRef(y0=0, y1=4, x0=0, x1=5, b0=0, b1=7)
            got_whole = storage.read_region(ref.ref_id, whole_region)
            expected = np.full((4, 5, 7), 3, dtype=np.float32)
            expected[0:1, :, :] = 2
            np.testing.assert_array_equal(got_whole, expected)

    def test_register_external_dataset_read_and_meta(self):
        class _FakeDataset:
            def __init__(self):
                self._arr = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)

            def get_shape(self):
                return self._arr.shape

            def get_elem_type(self):
                return self._arr.dtype

            def get_wavelengths(self):
                return [500.0, 700.0]

            def get_band_unit(self) -> u.Unit:
                return u.nm

            def get_data_ignore_value(self):
                return -9999.0

            def get_bad_bands(self):
                return [1, 0]

            def get_wkt_spatial_reference(self):
                return "EPSG:4326"

            def get_geo_transform(self):
                return (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

            def get_image_data_subset(self, x, y, band, dx, dy, dband, filter_data_ignore_value=False):
                _ = filter_data_ignore_value
                return self._arr[band : band + dband, y : y + dy, x : x + dx]

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)
            ref = storage.register_external_dataset(_FakeDataset())

            self.assertEqual(ref.source, "external")
            self.assertTrue(ref.readonly)
            self.assertEqual(ref.materialization_loc, "none")

            region = DatasetRegionRef(y0=1, y1=3, x0=1, x1=4, b0=0, b1=2)
            got = storage.read_region(ref.ref_id, region)
            self.assertEqual(got.shape, (2, 3, 2))

            meta = storage.get_meta(ref.ref_id)
            self.assertEqual(meta.kind, "dataset")
            self.assertEqual(meta.shape, (4, 5, 2))
            self.assertEqual(meta.elem_type, np.float32)
            self.assertEqual(meta.wavelength_units, u.nm)
            self.assertTrue(np.array_equal(meta.bad_bands, np.array([1, 0])))
            self.assertEqual(meta.crs_wkt, "EPSG:4326")
            self.assertEqual(meta.geotransform, (0.0, 1.0, 0.0, 0.0, 0.0, -1.0))

            region_meta = storage.get_region_meta(ref.ref_id, DatasetRegionRef(0, 2, 0, 2, 0, 1))
            self.assertTrue(np.array_equal(region_meta.wavelengths, np.array([500.0])))

    def test_external_refs_refuse_writes_and_meta_updates(self):
        class _FakeSpectrum:
            def __init__(self):
                self._arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            def get_spectrum(self):
                return self._arr

            def num_bands(self):
                return self._arr.shape[0]

            def get_elem_type(self):
                return self._arr.dtype

            def get_wavelengths(self):
                return [0.5, 1.0, 2.0]

            def get_wavelength_units(self):
                return None

            def get_bad_bands(self):
                return np.array([1, 1, 1], dtype=np.bool_)

        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageLayer(root_dir=tmp_dir)
            ref = storage.register_external_spectrum(_FakeSpectrum())

            with self.assertRaises(PermissionError):
                storage.write_data(ref, np.array([9.0, 9.0, 9.0], dtype=np.float32))
            with self.assertRaises(PermissionError):
                storage.write_region(ref, SpectrumRef(length=3), np.array([9.0, 9.0, 9.0], dtype=np.float32))
            with self.assertRaises(PermissionError):
                storage.set_meta(
                    ref.ref_id, DataMeta(kind="spectrum", shape=(3,), elem_type=np.dtype(np.float32))
                )
            with self.assertRaises(PermissionError):
                storage.update_meta(ref.ref_id, nodata=-1)

    def test_spectra_batch_scheme_sets_length_and_exclusive_bounds(self):
        meta = SpectraListPlanMeta(
            kind="spectra_list",
            dtype=np.dtype(np.float32),
            num_spectra=10,
            spectrum_length=4,
        )
        scheme = SpectraBatchScheme(batch_size=6)
        chunks = list(scheme.iter_chunks(meta))

        self.assertEqual(
            chunks, [SpectraBatchRef(i0=0, i1=6, length=4), SpectraBatchRef(i0=6, i1=10, length=4)]
        )
        self.assertEqual(chunks[0].scalar_count(), 24)
