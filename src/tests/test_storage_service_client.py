import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import tests.context

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl, NumPyRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.utils.primitives import AllocationRequest, DatasetRegionRef
from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_layer import ExternalRasterHandle
from wiser.utils.storage_service import StorageService


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
                    dtype=np.float32,
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
                    dtype=np.float32,
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
                got_image, _ = client.read_data(ref, filter_data_ignore_value=True)
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
                    filter_data_ignore_value=True,
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
                    filter_data_ignore_value=True,
                )
                got_band = np.ma.array(got_band_3d, copy=False)[:, :, 0]
                self._assert_array_and_mask_equal(got_band, expected_band)
            finally:
                client.close()
                service.close()
