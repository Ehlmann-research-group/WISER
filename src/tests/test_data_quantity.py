"""
The data quantity a dataset declares for its values, and the spectrum-plot
y-axis label derived from the displayed spectra's quantities.
"""

import os
import shutil

import netCDF4 as nc
import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401 -- adds src/ to sys.path

from test_utils.test_model import WiserTestModel

from wiser.gui.spectrum_plot import y_axis_label
from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import NetCDF_GDALRasterDataImpl
from wiser.raster.loader import RasterDataLoader
from wiser.raster.spectrum import SpectrumAtPoint

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_utils", "test_datasets")
EMIT_SUBSET = os.path.join(DATASETS, "EMIT_L2A_RFL_001_20260821T180123_2623312_016_subset_aid0000.nc")
EMIT_FLAT_CROP = os.path.join(DATASETS, "EMIT_L2A_RFL_001_20241006T165148_2428011_003_crop.nc")
PLAIN_NETCDF = os.path.join(DATASETS, "netcdf.nc")
ENVI_NAME = "caltech_4_100_150_nm"
GTIFF_WAVELENGTH_DESCRIPTIONS = os.path.join(DATASETS, "caltech_4_100_150_nm_epsg4087.tif")


def _open_netcdf(path):
    impl = NetCDF_GDALRasterDataImpl.try_load_file(path, subdataset_name="reflectance", interactive=False)[0]
    return RasterDataSet(impl)


def _envi_copy(tmp_path, extra_header_lines):
    """Copy the caltech ENVI fixture into tmp_path, appending header lines."""
    data = tmp_path / ENVI_NAME
    shutil.copyfile(os.path.join(DATASETS, ENVI_NAME), data)
    with open(os.path.join(DATASETS, ENVI_NAME + ".hdr")) as src:
        header = src.read()
    with open(str(data) + ".hdr", "w") as dst:
        dst.write(header.rstrip("\n") + "\n" + "".join(line + "\n" for line in extra_header_lines))
    return str(data)


def _load(path):
    return RasterDataLoader().load_from_file(path, interactive=False)[0]


def _write_gtiff(path, unit_types):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), 3, 2, len(unit_types), gdal.GDT_Float32)
    for i, unit in enumerate(unit_types, start=1):
        band = ds.GetRasterBand(i)
        band.WriteArray(np.zeros((2, 3), dtype=np.float32))
        band.SetDescription(f"Band {i}")
        band.SetUnitType(unit)
    ds.FlushCache()
    ds = None
    return str(path)


# ======================================================
# RasterDataSet.get_data_quantity()


def test_emit_netcdf_quantity_from_long_name_and_units():
    assert _open_netcdf(EMIT_SUBSET).get_data_quantity() == "Surface Reflectance (unitless)"


def test_single_variable_netcdf_quantity_from_long_name_and_units():
    assert _load(EMIT_FLAT_CROP).get_data_quantity() == "Surface Reflectance (unitless)"


def test_netcdf_without_attributes_has_no_quantity():
    assert _open_netcdf(PLAIN_NETCDF).get_data_quantity() is None


def test_netcdf_non_string_long_name_raises(tmp_path):
    path = str(tmp_path / "numeric_long_name.nc")
    shutil.copyfile(PLAIN_NETCDF, path)
    with nc.Dataset(path, "a") as ds:
        ds["reflectance"].long_name = 5
    with pytest.raises(ValueError, match="long_name"):
        _open_netcdf(path)


def test_envi_z_plot_titles_give_quantity(tmp_path):
    path = _envi_copy(tmp_path, ["z plot titles = {Wavelength, Reflectance (x10000)}"])
    assert _load(path).get_data_quantity() == "Reflectance (x10000)"


def test_envi_z_plot_titles_and_data_units(tmp_path):
    path = _envi_copy(tmp_path, ["z plot titles = {Wavelength, Radiance}", "data units = uW/cm^2/sr/nm"])
    assert _load(path).get_data_quantity() == "Radiance (uW/cm^2/sr/nm)"


def test_envi_data_units_only(tmp_path):
    path = _envi_copy(tmp_path, ["data units = percent"])
    assert _load(path).get_data_quantity() == "percent"


def test_envi_without_quantity_fields():
    assert _load(os.path.join(DATASETS, ENVI_NAME)).get_data_quantity() is None


def test_gtiff_shared_unit_type_gives_quantity(tmp_path):
    path = _write_gtiff(tmp_path / "units.tif", ["W/m^2/sr/um", "W/m^2/sr/um"])
    assert _load(path).get_data_quantity() == "W/m^2/sr/um"


def test_gtiff_differing_unit_types_have_no_quantity(tmp_path):
    path = _write_gtiff(tmp_path / "mixed.tif", ["W/m^2/sr/um", "K"])
    assert _load(path).get_data_quantity() is None


def test_gtiff_band_names_are_not_a_quantity():
    assert _load(GTIFF_WAVELENGTH_DESCRIPTIONS).get_data_quantity() is None


def test_numpy_dataset_has_no_quantity():
    loader = RasterDataLoader()
    ds = loader.dataset_from_numpy_array(np.zeros((2, 3, 4), dtype=np.float32), None)
    assert ds.get_data_quantity() is None


# ======================================================
# gui.spectrum_plot.y_axis_label()


def test_y_axis_label_agreeing_quantities():
    assert y_axis_label(["Reflectance", "Reflectance"]) == "Reflectance"


def test_y_axis_label_disagreeing_quantities():
    assert y_axis_label(["Reflectance", "Radiance"]) == "Data value (mixed units)"


def test_y_axis_label_known_and_unknown_quantity_are_mixed():
    assert y_axis_label(["Reflectance", None]) == "Data value (mixed units)"


def test_y_axis_label_no_quantities():
    assert y_axis_label([None, None]) == "Value"
    assert y_axis_label([]) == "Value"


# ======================================================
# Spectrum plot y-axis label


@pytest.mark.smoke
class TestSpectrumPlotYLabel:
    @pytest.fixture(autouse=True)
    def model(self):
        self.test_model = WiserTestModel()
        yield
        self.test_model.close_app()
        del self.test_model

    def _add(self, dataset):
        self.test_model.app_state.add_dataset(dataset)
        return dataset

    def _ylabel(self):
        return self.test_model.spectrum_plot._axes.get_ylabel()

    def test_emit_spectrum_labels_y_axis(self):
        ds = self._add(_open_netcdf(EMIT_SUBSET))
        self.test_model.set_active_spectrum(SpectrumAtPoint(ds, (0, 0)))
        assert self._ylabel() == "Surface Reflectance (unitless)"

    def test_envi_z_plot_titles_label_y_axis(self, tmp_path):
        ds = self._add(_load(_envi_copy(tmp_path, ["z plot titles = {Wavelength, Reflectance}"])))
        self.test_model.set_active_spectrum(SpectrumAtPoint(ds, (0, 0)))
        assert self._ylabel() == "Reflectance"

    def test_mixed_quantities_label_y_axis(self, tmp_path):
        envi = self._add(_load(_envi_copy(tmp_path, ["z plot titles = {Wavelength, Reflectance}"])))
        emit = self._add(_open_netcdf(EMIT_SUBSET))
        self.test_model.collect_spectrum(SpectrumAtPoint(envi, (0, 0)))
        self.test_model.set_active_spectrum(SpectrumAtPoint(emit, (0, 0)))
        assert self._ylabel() == "Data value (mixed units)"

    def test_no_quantity_labels_y_axis_value(self):
        ds = self._add(_load(os.path.join(DATASETS, ENVI_NAME)))
        self.test_model.set_active_spectrum(SpectrumAtPoint(ds, (0, 0)))
        assert self._ylabel() == "Value"

    def test_user_label_wins_until_cleared(self, tmp_path):
        ds = self._add(_load(_envi_copy(tmp_path, ["z plot titles = {Wavelength, Reflectance}"])))
        plot = self.test_model.spectrum_plot
        plot.set_y_label("My axis")
        self.test_model.set_active_spectrum(SpectrumAtPoint(ds, (0, 0)))
        assert self._ylabel() == "My axis"

        plot.set_font_size("axes", 8.0)
        self.test_model.set_active_spectrum(SpectrumAtPoint(ds, (1, 1)))
        assert self._ylabel() == "My axis"

        plot.set_y_label("")
        assert self._ylabel() == "Reflectance"
