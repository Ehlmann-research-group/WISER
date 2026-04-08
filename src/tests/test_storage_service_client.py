import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import tests.context

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset import band_info_list_equal
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl, NumPyRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import (
    AllocationRequest,
    DatasetRegionRef,
    DeletePolicy,
    DeletionState,
    ProducerState,
)
from wiser.utils.storage_client import StorageClient
from wiser.utils.primitives import ExternalRasterHandle
from wiser.utils.multiprocessing_context import CTX
from wiser.utils.storage_service import StorageService

import pytest

pytestmark = [
    pytest.mark.storage,
]

_WORKER_CLIENT: Optional[StorageClient] = None


def _pool_init_storage_client(service_address: tuple[str, int], service_authkey: bytes) -> None:
    global _WORKER_CLIENT
    _WORKER_CLIENT = StorageClient(
        service=None,  # type: ignore[arg-type]
        service_address=service_address,
        service_authkey=service_authkey,
    )


def _pool_read_external_ram_array(ref) -> tuple[tuple[int, ...], float]:
    if _WORKER_CLIENT is None:
        raise RuntimeError("Worker StorageClient was not initialized")
    arr, _ = _WORKER_CLIENT.read_data(ref)
    arr = np.asarray(arr, dtype=np.float32)
    return arr.shape, float(arr.sum())


def _pool_shift_y_diff_external_to_internal(input_ref, output_ref) -> tuple[tuple[int, ...], float]:
    if _WORKER_CLIENT is None:
        raise RuntimeError("Worker StorageClient was not initialized")
    data, _ = _WORKER_CLIENT.read_data(input_ref)
    data = np.asarray(data, dtype=np.float32)
    noise = data[:-1, :, :] - data[1:, :, :]
    _WORKER_CLIENT.write_data(output_ref, noise)
    return noise.shape, float(noise.sum())


class TestStorageServiceClient(unittest.TestCase):
    def _assert_meta_equal(self, left, right) -> None:
        self.assertEqual(left.kind, right.kind)
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(np.dtype(left.elem_type), np.dtype(right.elem_type))
        self.assertEqual(left.wavelength_units, right.wavelength_units)
        self.assertEqual(left.nodata, right.nodata)
        self.assertEqual(left.crs_wkt, right.crs_wkt)
        self.assertEqual(left.geotransform, right.geotransform)

        if left.wavelengths is None or right.wavelengths is None:
            self.assertEqual(left.wavelengths, right.wavelengths)
        else:
            np.testing.assert_array_equal(left.wavelengths, right.wavelengths)

        if left.bad_bands is None or right.bad_bands is None:
            self.assertEqual(left.bad_bands, right.bad_bands)
        else:
            np.testing.assert_array_equal(left.bad_bands, right.bad_bands)

    def _assert_array_and_mask_equal(self, actual, expected) -> None:
        actual_ma = np.ma.array(actual, copy=False)
        expected_ma = np.ma.array(expected, copy=False)

        actual_mask = np.ma.getmaskarray(actual_ma)
        expected_mask = np.ma.getmaskarray(expected_ma)
        np.testing.assert_array_equal(actual_mask, expected_mask)

        valid = ~expected_mask
        np.testing.assert_allclose(
            np.asarray(actual_ma.data)[valid],
            np.asarray(expected_ma.data)[valid],
            equal_nan=True,
        )

    def test_internal_allocation_creates_storage_lease_record(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            try:
                ref = service.allocate_data(
                    AllocationRequest(
                        name="lease_record_dataset",
                        kind="dataset",
                        residency="ram_cacheable",
                        size_est=2 * 3 * 4 * np.dtype(np.float32).itemsize,
                        shape=(2, 3, 4),
                        dtype=np.dtype(np.float32),
                    )
                )

                record = service.get_lease_record(ref)
                self.assertEqual(record.ref_id, ref.ref_id)
                self.assertEqual(record.backend_kind, "ram_shm")
                self.assertEqual(record.delete_policy, DeletePolicy.KEEP)
                self.assertEqual(record.producer_state, ProducerState.WRITING)
                self.assertEqual(record.deletion_state, DeletionState.LIVE)
                self.assertEqual(record.borrowers, {})
                self.assertEqual(record.pins, {})
                self.assertEqual(record.planned_consumer_plan_ids, set())
                self.assertFalse(record.external_owned)
            finally:
                service.close()

    def test_delete_when_releasable_allocation_reclaims_when_producer_completes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            try:
                ref = service.allocate_data(
                    AllocationRequest(
                        name="delete_when_done_dataset",
                        kind="dataset",
                        residency="spill_required",
                        size_est=2 * 3 * 4 * np.dtype(np.float32).itemsize,
                        shape=(2, 3, 4),
                        dtype=np.dtype(np.float32),
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    )
                )

                record = service.mark_producer_completed(ref)
                self.assertEqual(record.deletion_state, DeletionState.DELETED)
                self.assertEqual(record.producer_state, ProducerState.COMPLETED)
                self.assertNotIn(ref.ref_id, service.data_refs)
                with self.assertRaises(KeyError):
                    service.read_data_ref(ref)
            finally:
                service.close()

    def test_delete_when_releasable_allocation_becomes_pending_delete_when_blocked(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            try:
                ref = service.allocate_data(
                    AllocationRequest(
                        name="pending_delete_dataset",
                        kind="dataset",
                        residency="spill_required",
                        size_est=2 * 3 * 4 * np.dtype(np.float32).itemsize,
                        shape=(2, 3, 4),
                        dtype=np.dtype(np.float32),
                        delete_policy=DeletePolicy.DELETE_WHEN_RELEASABLE,
                    )
                )

                record = service.get_lease_record(ref)
                record.planned_consumer_plan_ids.add("plan:child")

                record = service.mark_producer_completed(ref)
                self.assertEqual(record.deletion_state, DeletionState.PENDING_DELETE)
                self.assertIn(ref.ref_id, service.data_refs)
            finally:
                service.close()

    def test_external_disk_backed_dataset_read_data_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)

            # External disk-backed dataset from test fixtures.
            loader = RasterDataLoader()
            fixture_path = (
                Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "caltech_425_7_7_nm"
            )
            disk_dataset = loader.load_from_file(str(fixture_path), interactive=False)[0]
            disk_ref = service.register_external(ExternalRasterHandle(dataset_obj=disk_dataset))
            disk_ref = replace(disk_ref, materialization_loc="disk")
            service.data_refs[disk_ref.ref_id] = disk_ref

            expected_disk = np.asarray(disk_dataset.get_image_data()).transpose(1, 2, 0)
            got_disk, disk_region_meta = client.read_data(disk_ref)
            np.testing.assert_allclose(got_disk, expected_disk, equal_nan=True)

            disk_meta_service = service.get_meta(disk_ref)
            disk_meta_client = client.get_meta(disk_ref)
            self._assert_meta_equal(disk_meta_service, disk_meta_client)
            self.assertEqual(np.dtype(disk_region_meta.elem_type), np.dtype(disk_meta_client.elem_type))
            self.assertEqual(
                disk_region_meta.region,
                DatasetRegionRef(
                    0, expected_disk.shape[0], 0, expected_disk.shape[1], 0, expected_disk.shape[2]
                ),
            )

    def test_external_disk_backed_netcdf_reflectance_read_data_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)

            fixture_path = (
                Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "netcdf.nc"
            )
            netcdf_impl = NetCDF_GDALRasterDataImpl.try_load_file(
                str(fixture_path),
                subdataset_name="reflectance",
                interactive=False,
            )[0]
            disk_dataset = RasterDataSet(netcdf_impl)

            disk_ref = service.register_external(ExternalRasterHandle(dataset_obj=disk_dataset))
            disk_ref = replace(disk_ref, materialization_loc="disk")
            service.data_refs[disk_ref.ref_id] = disk_ref

            expected_disk = np.asarray(disk_dataset.get_image_data()).transpose(1, 2, 0)
            got_disk, disk_region_meta = client.read_data(disk_ref)
            np.testing.assert_allclose(got_disk, expected_disk, equal_nan=True)

            disk_meta_service = service.get_meta(disk_ref)
            disk_meta_client = client.get_meta(disk_ref)
            self._assert_meta_equal(disk_meta_service, disk_meta_client)
            self.assertEqual(np.dtype(disk_region_meta.elem_type), np.dtype(disk_meta_client.elem_type))
            self.assertEqual(
                disk_region_meta.region,
                DatasetRegionRef(
                    0, expected_disk.shape[0], 0, expected_disk.shape[1], 0, expected_disk.shape[2]
                ),
            )

    def test_external_ram_backed_dataset_read_data_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)

            # External RAM-backed dataset using NumPyRasterDataImpl.
            arr_band_first = (np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) * 1.5) - 4.25
            ram_dataset = RasterDataSet(NumPyRasterDataImpl(arr_band_first))
            ram_ref = service.register_external(ExternalRasterHandle(dataset_obj=ram_dataset))
            ram_ref = replace(ram_ref, materialization_loc="ram")
            service.data_refs[ram_ref.ref_id] = ram_ref

            expected_ram = arr_band_first.transpose(1, 2, 0)
            got_ram, ram_region_meta = client.read_data(ram_ref)
            np.testing.assert_allclose(got_ram, expected_ram, equal_nan=True)

            ram_meta_service = service.get_meta(ram_ref)
            ram_meta_client = client.get_meta(ram_ref)
            self._assert_meta_equal(ram_meta_service, ram_meta_client)
            self.assertEqual(np.dtype(ram_region_meta.elem_type), np.dtype(ram_meta_client.elem_type))
            self.assertEqual(
                ram_region_meta.region,
                DatasetRegionRef(
                    0, expected_ram.shape[0], 0, expected_ram.shape[1], 0, expected_ram.shape[2]
                ),
            )

    def test_external_ram_backed_dataset_read_data_from_separate_process(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)

                arr_band_first = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
                ram_dataset = RasterDataSet(NumPyRasterDataImpl(arr_band_first))
                ram_ref = service.register_external(ExternalRasterHandle(dataset_obj=ram_dataset))
                ram_ref = replace(ram_ref, materialization_loc="ram")
                service.data_refs[ram_ref.ref_id] = ram_ref

                expected = arr_band_first.transpose(1, 2, 0)

                # This regression test isolates the cross-process attach path for external
                # RAM-backed refs. It is narrower than the MNF failure: if this passes,
                # then external shared-memory reads from a worker are functional and the
                # remaining bug is likely elsewhere in the task/output write path.
                with ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=CTX,
                    initializer=_pool_init_storage_client,
                    initargs=(address, authkey),
                ) as pool:
                    future = pool.submit(_pool_read_external_ram_array, ram_ref)
                    got_shape, got_sum = future.result(timeout=30)

                self.assertEqual(got_shape, expected.shape)
                self.assertAlmostEqual(got_sum, float(expected.sum()), places=5)
            finally:
                if client is not None:
                    client.close()
                service.close()

    def test_separate_process_can_read_external_ram_and_write_internal_ram_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir, ram_byte_limit=10_000_000)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)

                arr_band_first = np.array(
                    [
                        [[1.0, 3.0], [2.0, 4.0]],
                        [[10.0, 30.0], [20.0, 40.0]],
                        [[100.0, 300.0], [200.0, 400.0]],
                    ],
                    dtype=np.float32,
                )
                ram_dataset = RasterDataSet(NumPyRasterDataImpl(arr_band_first))
                input_ref = service.register_external(ExternalRasterHandle(dataset_obj=ram_dataset))
                input_ref = replace(input_ref, materialization_loc="ram")
                service.data_refs[input_ref.ref_id] = input_ref

                expected_input = arr_band_first.transpose(1, 2, 0)
                output_shape = (expected_input.shape[0] - 1, expected_input.shape[1], expected_input.shape[2])
                output_ref = service.allocate_data(
                    AllocationRequest(
                        name="shift_y_diff_output",
                        kind="dataset",
                        residency="ram_cacheable",
                        size_est=int(np.prod(output_shape) * np.dtype(np.float32).itemsize),
                        shape=output_shape,
                        dtype=np.dtype(np.float32),
                    )
                )

                # This reproduces the MNF stage access pattern more closely than a pure read:
                # a worker process reads an external RAM-backed input and writes an internally
                # allocated RAM-backed output ref.
                with ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=CTX,
                    initializer=_pool_init_storage_client,
                    initargs=(address, authkey),
                ) as pool:
                    future = pool.submit(_pool_shift_y_diff_external_to_internal, input_ref, output_ref)
                    got_shape, got_sum = future.result(timeout=30)

                expected_noise = expected_input[:-1, :, :] - expected_input[1:, :, :]
                self.assertEqual(got_shape, expected_noise.shape)
                self.assertAlmostEqual(got_sum, float(expected_noise.sum()), places=5)

                output_data, _ = client.read_data(output_ref)
                np.testing.assert_allclose(output_data, expected_noise, atol=1e-6)
            finally:
                if client is not None:
                    client.close()
                service.close()

    def test_internal_disk_backed_dataset_write_then_client_read_data_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)

            ref = service.allocate_data(
                AllocationRequest(
                    name="internal_disk_dataset",
                    kind="dataset",
                    residency="spill_required",
                    size_est=4 * 3 * 6 * np.dtype(np.float32).itemsize,
                    shape=(4, 3, 6),
                    dtype=np.dtype(np.float32),
                ),
                preferred_storage="memmap",
            )
            self.assertEqual(ref.materialization_loc, "disk")
            self.assertEqual(ref.disk_format, "memmap")

            expected = ((np.arange(4 * 3 * 6, dtype=np.float32).reshape(4, 3, 6) % 7.0) * 2.25) - 3.0
            client.write_data(ref, expected)

            got, region_meta = client.read_data(ref)
            np.testing.assert_allclose(got, expected, equal_nan=True)
            # Must delete got to close the memmapped array.
            del got

            meta_service = service.get_meta(ref)
            meta_client = client.get_meta(ref)
            self._assert_meta_equal(meta_service, meta_client)
            self.assertEqual(np.dtype(region_meta.elem_type), np.dtype(expected.dtype))
            self.assertEqual(region_meta.region, DatasetRegionRef(0, 4, 0, 3, 0, 6))

    def test_internal_ram_backed_dataset_write_then_client_read_data_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir, ram_byte_limit=10_000_000)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)

            ref = service.allocate_data(
                AllocationRequest(
                    name="internal_ram_dataset",
                    kind="dataset",
                    residency="ram_cacheable",
                    size_est=3 * 5 * 4 * np.dtype(np.float32).itemsize,
                    shape=(3, 5, 4),
                    dtype=np.dtype(np.float32),
                )
            )
            self.assertEqual(ref.materialization_loc, "ram")

            expected = (np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4) / 3.0) + 0.125
            client.write_data(ref, expected)

            got, region_meta = client.read_data(ref)
            np.testing.assert_allclose(got, expected, equal_nan=True)

            meta_service = service.get_meta(ref)
            meta_client = client.get_meta(ref)
            self._assert_meta_equal(meta_service, meta_client)
            self.assertEqual(np.dtype(region_meta.elem_type), np.dtype(expected.dtype))
            self.assertEqual(region_meta.region, DatasetRegionRef(0, 3, 0, 5, 0, 4))

    def test_external_dataset_filtered_reads_match_raster_dataset_values_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)
            try:
                loader = RasterDataLoader()
                fixture_path = (
                    Path(__file__).resolve().parent
                    / ".."
                    / "test_utils"
                    / "test_datasets"
                    / "caltech_425_7_7_nm"
                )
                dataset = loader.load_from_file(str(fixture_path), interactive=False)[0]
                ref = service.register_external(ExternalRasterHandle(dataset_obj=dataset))
                ref = replace(ref, materialization_loc="disk")
                service.data_refs[ref.ref_id] = ref

                # Compare full-cube read.
                expected_image = np.ma.array(
                    dataset.get_image_data(filter_data_ignore_value=True),
                    copy=False,
                ).transpose(1, 2, 0)
                bad_bands = dataset.get_bad_bands()
                if bad_bands is not None:
                    bad_band_mask = np.asarray(bad_bands, dtype=np.int8) == 0
                    expected_mask = np.ma.getmaskarray(expected_image).copy()
                    expected_mask[:, :, bad_band_mask] = True
                    expected_image = np.ma.array(
                        np.ma.getdata(expected_image),
                        mask=expected_mask,
                        copy=False,
                    )
                got_image, _ = client.read_data(ref, filter_data=True)
                self._assert_array_and_mask_equal(got_image, expected_image)

                # Compare subset read.
                x, y, b = 1, 2, 3
                dx, dy, db = 3, 3, 5
                expected_subset = np.ma.array(
                    dataset.get_image_data_subset(
                        x=x,
                        y=y,
                        band=b,
                        dx=dx,
                        dy=dy,
                        dband=db,
                        filter_data_ignore_value=True,
                    ),
                    copy=False,
                ).transpose(1, 2, 0)
                subset_region = DatasetRegionRef(y, y + dy, x, x + dx, b, b + db)
                got_subset, _ = client.read_region(
                    ref,
                    subset_region,
                    filter_data=True,
                )
                self._assert_array_and_mask_equal(got_subset, expected_subset)

                # Compare single-band read.
                band_index = 10
                expected_band = dataset.get_band_data(
                    band_index=band_index,
                    filter_data_ignore_value=True,
                )
                h, w, _ = ref.shape
                band_region = DatasetRegionRef(0, h, 0, w, band_index, band_index + 1)
                got_band_3d, _ = client.read_region(
                    ref,
                    band_region,
                    filter_data=True,
                )
                got_band = np.ma.array(got_band_3d, copy=False)[:, :, 0]
                self._assert_array_and_mask_equal(got_band, expected_band)
            finally:
                client.close()
                service.close()

    def test_client_can_reconstruct_external_dataset_with_matching_metadata_and_masked_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            address, authkey = service.get_connection_bootstrap()
            client = StorageClient(service=service, service_address=address, service_authkey=authkey)
            try:
                loader = RasterDataLoader()
                fixture_path = (
                    Path(__file__).resolve().parent
                    / ".."
                    / "test_utils"
                    / "test_datasets"
                    / "caltech_425_6_6_data_ignore.hdr"
                )
                dataset = loader.load_from_file(str(fixture_path), interactive=False)[0]
                ref = service.register_external(ExternalRasterHandle(dataset_obj=dataset))

                reconstructed = client.reconstruct_external_object(ref)
                self.assertIsInstance(reconstructed, RasterDataSet)

                self.assertEqual(reconstructed.get_bad_bands(), dataset.get_bad_bands())
                self.assertEqual(reconstructed.get_data_ignore_value(), dataset.get_data_ignore_value())
                self.assertEqual(
                    reconstructed.get_wkt_spatial_reference(),
                    dataset.get_wkt_spatial_reference(),
                )
                self.assertEqual(reconstructed.get_geo_transform(), dataset.get_geo_transform())
                self.assertEqual(reconstructed.get_width(), dataset.get_width())
                self.assertEqual(reconstructed.get_height(), dataset.get_height())
                self.assertEqual(reconstructed.num_bands(), dataset.num_bands())
                self.assertEqual(reconstructed.get_band_unit(), dataset.get_band_unit())
                self.assertTrue(band_info_list_equal(reconstructed._band_info, dataset._band_info))

                expected_raw = np.ma.array(
                    dataset.get_image_data(filter_data_ignore_value=False),
                    copy=False,
                ).transpose(1, 2, 0)
                actual_raw = np.ma.array(
                    reconstructed.get_image_data(filter_data_ignore_value=False),
                    copy=False,
                ).transpose(1, 2, 0)
                self._assert_array_and_mask_equal(actual_raw, expected_raw)

                expected_masked = np.ma.array(
                    dataset.get_image_data(filter_data_ignore_value=True),
                    copy=False,
                ).transpose(1, 2, 0)
                actual_masked = np.ma.array(
                    reconstructed.get_image_data(filter_data_ignore_value=True),
                    copy=False,
                ).transpose(1, 2, 0)
                self._assert_array_and_mask_equal(actual_masked, expected_masked)
            finally:
                client.close()
                service.close()
