import os
import unittest

import tests.context  # This is needed for when we run the file with pytest <file-name>.py
# import context  # This is needed for when we run the file with python <file-name>.py

import numpy as np
import astropy.units as u
from osgeo import osr

from test_utils.test_model import WiserTestModel
from test_utils.test_arrays import (
    sam_sff_arr_reg,
    spec_bbl_caltech_425_7_7,
    spec_wvl_caltech_425_7_7,
)

from wiser.raster.dataset import (
    RasterDataSet,
    RasterDataBand,
    RasterDataDynamicBand,
)
from wiser.raster.spectrum import NumPyArraySpectrum, SpectrumAtPoint

import pytest

pytestmark = [
    pytest.mark.smoke,
]


class TestDataSerialization(unittest.TestCase):
    def setUp(self):
        self.test_model = WiserTestModel()

    def tearDown(self):
        self.test_model.close_app()
        del self.test_model

    def test_raster_data_set(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))

        target_path = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "test_utils",
                "test_datasets",
                "caltech_4_100_150_nm.hdr",
            )
        )

        ds = self.test_model.load_dataset(target_path)

        serializedForm = ds.get_serialized_form()

        reconstructed_dataset: RasterDataSet = serializedForm.get_serializable_class().deserialize_into_class(
            serializedForm=serializedForm,
        )

        assert reconstructed_dataset.is_metadata_same(
            ds
        ), "The reconstructed dataset has different metadata from the original dataset"

    def test_numpy_dataset_serialization(self):
        arr = sam_sff_arr_reg
        ds = self.test_model.load_dataset(arr)
        serialized_ds = ds.get_serialized_form()

        reconstructed_dataset: RasterDataSet = serialized_ds.get_serializable_class().deserialize_into_class(
            serializedForm=serialized_ds,
        )

        assert reconstructed_dataset.is_metadata_same(
            ds
        ), "The reconstructed dataset has different metadata from the original dataset"

    def test_numpy_dataset_serialization_preserves_full_metadata(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "test_utils",
                "test_datasets",
                "caltech_425_7_7_nm.hdr",
            )
        )

        source_ds = self.test_model.load_dataset(target_path)
        ds = self.test_model.load_dataset(source_ds.get_image_data())
        ds.set_bad_bands(spec_bbl_caltech_425_7_7)
        ds.set_data_ignore_value(-9999)
        ds.update_band_info(spec_wvl_caltech_425_7_7)

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(4326)
        expected_wkt = spatial_ref.ExportToWkt()
        expected_geo_transform = (100.0, 1.0, 0.0, 200.0, 0.0, -1.0)

        ds._set_spatial_reference(spatial_ref)
        ds._set_wkt(expected_wkt)
        ds._set_geo_transform(expected_geo_transform)

        serialized_ds_form = ds.get_serialized_form()
        reconstructed_dataset: RasterDataSet = (
            serialized_ds_form.get_serializable_class().deserialize_into_class(
                serializedForm=serialized_ds_form,
            )
        )

        assert reconstructed_dataset.is_metadata_same(
            ds
        ), "The reconstructed dataset has different metadata from the original dataset"
        assert reconstructed_dataset.get_bad_bands() == spec_bbl_caltech_425_7_7
        assert reconstructed_dataset.get_data_ignore_value() == -9999
        assert reconstructed_dataset.get_wavelengths() == spec_wvl_caltech_425_7_7
        assert reconstructed_dataset.get_band_unit() == u.nm
        assert reconstructed_dataset.get_geo_transform() == expected_geo_transform
        assert reconstructed_dataset.get_wkt_spatial_reference() == expected_wkt
        assert reconstructed_dataset.get_spatial_ref().IsSame(spatial_ref) == 1

    def test_raster_data_band(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "test_utils",
                "test_datasets",
                "caltech_4_100_150_nm.hdr",
            )
        )
        ds = self.test_model.load_dataset(target_path)
        band = RasterDataBand(ds, 2)
        serializedForm = band.get_serialized_form()
        reconstructed_band: RasterDataBand = serializedForm.get_serializable_class().deserialize_into_class(
            serializedForm=serializedForm,
        )

        assert reconstructed_band.is_metadata_same(
            band
        ), "The reconstructed band has different metadata from the original band"

    def test_raster_data_dynamic_band_index(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "test_utils",
                "test_datasets",
                "caltech_4_100_150_nm.hdr",
            )
        )
        ds = self.test_model.load_dataset(target_path)
        band = RasterDataDynamicBand(ds, band_index=0)
        serializedForm = band.get_serialized_form()
        reconstructed_band: RasterDataDynamicBand = (
            serializedForm.get_serializable_class().deserialize_into_class(
                serializedForm=serializedForm,
            )
        )

        assert reconstructed_band.is_metadata_same(
            band
        ), "The reconstructed band has different metadata from the original band"

    def test_raster_data_dynamic_band_wavelength(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "test_utils",
                "test_datasets",
                "caltech_4_100_150_nm.hdr",
            )
        )
        ds = self.test_model.load_dataset(target_path)
        band = RasterDataDynamicBand(ds, wavelength_value=572, wavelength_units="nm", epsilon=0.1)
        serializedForm = band.get_serialized_form()
        reconstructed_band: RasterDataDynamicBand = (
            serializedForm.get_serializable_class().deserialize_into_class(
                serializedForm=serializedForm,
            )
        )

        assert reconstructed_band.is_metadata_same(
            band
        ), "The reconstructed band has different metadata from the original band"

    def test_numpy_array_spectrum(self):
        arr = np.array([1, 2, 3, 4, 5])
        wvl = [100 * u.nm, 200 * u.nm, 300 * u.nm, 400 * u.nm, 500 * u.nm]
        spectrum = NumPyArraySpectrum(arr, name="test", wavelengths=wvl)
        serializedForm = spectrum.get_serialized_form()
        reconstructed_spectrum: NumPyArraySpectrum = (
            serializedForm.get_serializable_class().deserialize_into_class(
                serializedForm=serializedForm,
            )
        )

        assert np.allclose(
            reconstructed_spectrum.get_spectrum(), spectrum.get_spectrum()
        ), "The reconstructed spectrum has different metadata from the original spectrum"

    def test_spectrum_at_point(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.normpath(
            os.path.join(
                current_dir,
                "..",
                "test_utils",
                "test_datasets",
                "caltech_4_100_150_nm.hdr",
            )
        )
        ds = self.test_model.load_dataset(target_path)
        spectrum = SpectrumAtPoint(ds, (1, 1))
        serializedForm = spectrum.get_serialized_form()
        reconstructed_spectrum: SpectrumAtPoint = (
            serializedForm.get_serializable_class().deserialize_into_class(
                serializedForm=serializedForm,
            )
        )

        assert np.allclose(
            reconstructed_spectrum.get_spectrum(), spectrum.get_spectrum()
        ), "The reconstructed spectrum has different metadata from the original spectrum"

    # def test_netcdf_serialization(self):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # target_path = os.path.normpath(os.path.join(current_dir, "..", "test_utils", "test_datasets", "netcdf.nc"))  # noqa: E501

    # ds = self.test_model.load_dataset(target_path)

    # serializedForm = ds.get_serialized_form()

    #   reconstructed_dataset: RasterDataSet = (
    #     serializedForm.get_serializable_class().deserialize_into_class(
    #        serializedForm=serializedForm,
    #     )
    # )

    # assert np.allclose(reconstructed_dataset.get_image_data(), ds.get_image_data), \
    #     "The reconstructed dataset has different metadata from the original dataset"


if __name__ == "__main__":
    test_data_serialization = TestDataSerialization()
    test_data_serialization.setUp()
    try:
        # test_data_serialization.test_raster_data_dynamic_band_index()
        test_data_serialization.test_raster_data_dynamic_band_wavelength()
    finally:
        test_data_serialization.tearDown()
