"""EMIT netCDF reflectance is presented on the orthorectified grid its GLT defines.

EMIT L2A files store ``reflectance(downtrack, crosstrack, bands)`` in raw swath
geometry, while the global ``geotransform`` / ``spatial_ref`` describe the
orthorectified grid that ``location/glt_x`` / ``location/glt_y`` map into.
Applying that transform to swath pixels misplaces them by kilometres, so WISER
re-indexes the swath through the GLT and serves the ortho grid.

The geolocation check is independent of WISER's read path:  it reads the GLT
and the per-swath-pixel ``location/lat`` / ``location/lon`` with netCDF4 and
compares them with the declared geotransform.
"""

import os
import shutil

import netCDF4 as nc
import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401 -- adds src/ to sys.path

from PySide6.QtWidgets import QApplication

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import GDALRasterDataImpl, NetCDF_GDALRasterDataImpl
from wiser.raster.loader import RasterDataLoader

from wiser.gui.subdataset_file_opener_dialog import SubdatasetFileOpenerDialog  # noqa: E402 -- after wiser.raster, which it imports cyclically

DATASETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_utils", "test_datasets")
EMIT_SUBSET = os.path.join(DATASETS, "EMIT_L2A_RFL_001_20260821T180123_2623312_016_subset_aid0000.nc")
EMIT_FLAT_CROP = os.path.join(DATASETS, "EMIT_L2A_RFL_001_20241006T165148_2428011_003_crop.nc")
PLAIN_NETCDF = os.path.join(DATASETS, "netcdf.nc")

with nc.Dataset(EMIT_SUBSET) as _ds:
    DECLARED_GEOTRANSFORM = tuple(float(v) for v in _ds.geotransform)
FILL = -9999.0


def _read_raw(path):
    """GLT, lat/lon and reflectance read directly with netCDF4, with no masking."""
    with nc.Dataset(path) as ds:
        out = {}
        for name in ("location/glt_x", "location/glt_y", "location/lat", "location/lon", "reflectance"):
            var = ds[name]
            var.set_auto_mask(False)
            out[name.split("/")[-1]] = np.asarray(var[:])
        out["good_wavelengths"] = np.asarray(ds["sensor_band_parameters/good_wavelengths"][:])
    return out


def _open(path, variable="reflectance"):
    return NetCDF_GDALRasterDataImpl.try_load_file(path, subdataset_name=variable, interactive=False)[0]


@pytest.fixture(scope="module")
def raw():
    return _read_raw(EMIT_SUBSET)


@pytest.fixture(scope="module")
def impl():
    return _open(EMIT_SUBSET)


def test_dimensions_and_geotransform_are_the_ortho_grid(impl, raw):
    assert raw["glt_x"].shape == (185, 157)
    assert (impl.get_height(), impl.get_width()) == raw["glt_x"].shape
    assert impl.num_bands() == 285
    np.testing.assert_allclose(impl.read_geo_transform(), DECLARED_GEOTRANSFORM, rtol=0, atol=1e-12)
    assert impl.read_spatial_ref().GetAuthorityCode(None) == "4326"


def test_ortho_pixels_are_where_their_swath_pixel_is(impl, raw):
    gt = impl.read_geo_transform()
    ortho_y, ortho_x = np.nonzero(raw["glt_x"] != 0)
    swath_y = raw["glt_y"][ortho_y, ortho_x] - 1
    swath_x = raw["glt_x"][ortho_y, ortho_x] - 1

    centre_lon = gt[0] + (ortho_x + 0.5) * gt[1]
    centre_lat = gt[3] + (ortho_y + 0.5) * gt[5]
    error_px = np.hypot(
        (centre_lon - raw["lon"][swath_y, swath_x]) / gt[1],
        (centre_lat - raw["lat"][swath_y, swath_x]) / gt[5],
    )

    assert len(error_px) == 449
    assert error_px.max() < 1.5, f"max geolocation error {error_px.max():.3f} ortho pixels"

    # The swath lat/lon as WISER serves them land on the same ortho pixels.
    wiser_lat = _open(EMIT_SUBSET, "/location/lat").get_band_data(0)
    wiser_lon = _open(EMIT_SUBSET, "/location/lon").get_band_data(0)
    np.testing.assert_array_equal(wiser_lat[ortho_y, ortho_x], raw["lat"][swath_y, swath_x])
    np.testing.assert_array_equal(wiser_lon[ortho_y, ortho_x], raw["lon"][swath_y, swath_x])


def test_spectra_come_from_the_glt_source_pixel(impl, raw):
    ortho_y, ortho_x = np.nonzero(raw["glt_x"] != 0)
    cube = impl.get_image_data()
    assert cube.shape == (285, 185, 157)

    for i in np.linspace(0, len(ortho_y) - 1, 12).astype(int):
        y, x = ortho_y[i], ortho_x[i]
        expected = raw["reflectance"][raw["glt_y"][y, x] - 1, raw["glt_x"][y, x] - 1, :]
        np.testing.assert_array_equal(impl.get_all_bands_at(x, y), expected)
        np.testing.assert_array_equal(cube[:, y, x], expected)

    y, x = ortho_y[0], ortho_x[0]
    np.testing.assert_array_equal(impl.get_band_data(40), cube[40])
    np.testing.assert_array_equal(impl.get_multiple_band_data([3, 7]), cube[[3, 7]])
    np.testing.assert_array_equal(impl.get_all_bands_at_rect(x, y, 4, 3), cube[:, y : y + 3, x : x + 4])
    np.testing.assert_array_equal(
        impl.get_image_data_subset(x, y, 10, 4, 3, 5), cube[10:15, y : y + 3, x : x + 4]
    )
    np.testing.assert_array_equal(
        impl.get_image_data_subset(x, y, 10, 4, 3, 1), cube[10, y : y + 3, x : x + 4]
    )
    np.testing.assert_array_equal(impl.sample_band_data(40, 2), cube[40, ::2, ::2][:92, :78])


def test_ortho_pixels_without_source_are_ignored(impl, raw):
    no_source = raw["glt_x"] == 0
    assert no_source.sum() == 28596
    assert impl.read_data_ignore_value() == FILL
    assert np.all(impl.get_band_data(0)[no_source] == FILL)

    dataset = RasterDataSet(impl, None)
    band = dataset.get_band_data(0)
    assert isinstance(band, np.ma.MaskedArray)
    np.testing.assert_array_equal(np.ma.getmaskarray(band), no_source)


def test_band_metadata_is_unchanged(impl, raw):
    info = impl.read_band_info()
    assert len(info) == 285
    with nc.Dataset(EMIT_SUBSET) as ds:
        wavelengths = np.asarray(ds["sensor_band_parameters/wavelengths"][:])
    np.testing.assert_allclose([b["wavelength"].value for b in info], wavelengths)
    bad_bands = impl.read_bad_bands()
    assert list(bad_bands) == raw["good_wavelengths"].tolist()
    assert bad_bands.count(0) == 30


def _copy(tmp_path):
    path = str(tmp_path / "emit.nc")
    shutil.copy(EMIT_SUBSET, path)
    return path


def _edit_glt(path, edit):
    with nc.Dataset(path, "a") as ds:
        glt_x, glt_y = ds["location/glt_x"], ds["location/glt_y"]
        glt_x.set_auto_mask(False)
        glt_y.set_auto_mask(False)
        x, y = glt_x[:], glt_y[:]
        edit(x, y)
        glt_x[:] = x
        glt_y[:] = y


def _first_source(x):
    return tuple(np.argwhere(x != 0)[0])


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda x, y: x.__setitem__(_first_source(x), 37), "outside the 1-based 18 x 36 swath"),
        (lambda x, y: y.__setitem__(_first_source(x), 19), "outside the 1-based 18 x 36 swath"),
        (lambda x, y: y.__setitem__(_first_source(x), -3), "outside the 1-based 18 x 36 swath"),
        (lambda x, y: y.__setitem__(_first_source(x), 0), "disagree on which pixels are 0"),
    ],
    ids=["column-past-swath", "row-past-swath", "negative-row", "half-empty-entry"],
)
def test_out_of_range_glt_raises(tmp_path, edit, message):
    path = _copy(tmp_path)
    _edit_glt(path, edit)

    with pytest.raises(ValueError, match=message) as excinfo:
        _open(path)
    assert path in str(excinfo.value)

    with pytest.raises(ValueError, match=message):
        RasterDataLoader().load_from_file(path, interactive=False)


def _write_swath_file(
    path,
    glt_x_dims=("ortho_y", "ortho_x"),
    refl_dims=("downtrack", "crosstrack", "bands"),
    fill=True,
    lat_name="lat",
):
    with nc.Dataset(path, "w") as ds:
        for name, size in (
            ("downtrack", 3),
            ("crosstrack", 4),
            ("bands", 2),
            ("ortho_y", 5),
            ("ortho_x", 6),
            ("other", 7),
        ):
            ds.createDimension(name, size)
        refl = ds.createVariable(
            "reflectance", "f4", refl_dims, fill_value=np.float32(FILL) if fill else False
        )
        refl[:] = np.arange(np.prod(refl.shape), dtype="f4").reshape(refl.shape)
        location = ds.createGroup("location")
        for name in (lat_name, "lon"):
            location.createVariable(name, "f8", ("downtrack", "crosstrack"))[:] = 0.0
        glt_x = location.createVariable("glt_x", "i4", glt_x_dims)
        glt_y = location.createVariable("glt_y", "i4", ("ortho_y", "ortho_x"))
        glt_x[:] = np.ones(glt_x.shape, dtype="i4")
        glt_y[:] = np.ones(glt_y.shape, dtype="i4")
        glt_x[0, 0] = 0
        glt_y[0, 0] = 0


def test_synthetic_swath_file_opens_on_the_ortho_grid(tmp_path):
    path = str(tmp_path / "swath.nc")
    _write_swath_file(path)
    impl = _open(path)
    assert (impl.get_height(), impl.get_width()) == (5, 6)
    band = impl.get_band_data(1)
    assert band[0, 0] == FILL
    assert np.all(band.flat[1:] == 1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"glt_x_dims": ("ortho_y", "other")}, "must be integer tables of one 2D shape"),
        ({"refl_dims": ("crosstrack", "downtrack", "bands")}, "which is not the location/lat swath"),
        ({"fill": False}, "has no fill value"),
        ({"lat_name": "latitude"}, "has a GLT but WISER cannot match 'reflectance'"),
    ],
    ids=["glt-shape-mismatch", "swath-transposed", "no-fill-value", "no-location-lat"],
)
def test_inconsistent_glt_raises(tmp_path, kwargs, message):
    path = str(tmp_path / "swath.nc")
    _write_swath_file(path, **kwargs)
    with pytest.raises(ValueError, match=message):
        _open(path)


def test_flat_emit_crop_still_opens_as_before():
    impl = NetCDF_GDALRasterDataImpl.try_load_file(EMIT_FLAT_CROP, interactive=False)[0]
    assert type(impl) is GDALRasterDataImpl
    reference = gdal.Open(EMIT_FLAT_CROP)
    assert (impl.get_width(), impl.get_height(), impl.num_bands()) == (32, 32, 3)
    np.testing.assert_array_equal(impl.get_image_data(), reference.ReadAsArray())


def test_netcdf_without_location_reads_as_before():
    impl = _open(PLAIN_NETCDF)
    assert not impl.reads_are_transformed()
    reference = gdal.Open(f'NETCDF:"{PLAIN_NETCDF}":reflectance')
    assert (impl.get_width(), impl.get_height()) == (reference.RasterXSize, reference.RasterYSize)
    np.testing.assert_array_equal(impl.get_image_data(), reference.ReadAsArray())


def test_serialized_reopen_is_the_ortho_dataset(impl):
    dataset = RasterDataSet(impl, None)
    reopened = RasterDataSet.deserialize_into_class(dataset.get_serialized_form())
    assert (reopened.get_width(), reopened.get_height()) == (157, 185)
    assert reopened.get_geo_transform() == dataset.get_geo_transform()
    np.testing.assert_array_equal(reopened.get_impl().get_image_data(), impl.get_image_data())


def test_subdataset_dialog_opens_the_ortho_dataset(impl):
    QApplication.instance() or QApplication([])
    gdal_dataset = gdal.Open(EMIT_SUBSET)
    with nc.Dataset(EMIT_SUBSET) as netcdf_dataset:
        dialog = SubdatasetFileOpenerDialog(gdal_dataset, netcdf_dataset)
        cbox = dialog._ui.cbox_subdataset_choice
        cbox.setCurrentIndex(next(i for i in range(cbox.count()) if cbox.itemData(i)[0] == "reflectance"))
        dialog.accept()
        chosen = dialog.netcdf_impl
        assert (chosen.get_width(), chosen.get_height()) == (157, 185)
        np.testing.assert_allclose(chosen.read_geo_transform(), DECLARED_GEOTRANSFORM, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(chosen.get_band_data(40), impl.get_band_data(40))


def test_save_as_writes_the_ortho_grid(impl, tmp_path):
    dataset = RasterDataSet(impl, None)
    path = str(tmp_path / "ortho_envi")
    RasterDataLoader().save_dataset_as(dataset, path, "ENVI", {})

    saved = gdal.Open(path)
    assert (saved.RasterXSize, saved.RasterYSize, saved.RasterCount) == (157, 185, 285)
    # ENVI "map info" stores the origin to about 1e-8 degree (millimetres).
    np.testing.assert_allclose(saved.GetGeoTransform(), DECLARED_GEOTRANSFORM, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(saved.GetRasterBand(41).ReadAsArray(), impl.get_band_data(40))


@pytest.mark.parametrize(
    "read",
    [
        lambda impl: impl.get_all_bands_at(-1, 0),
        lambda impl: impl.get_all_bands_at(157, 0),
        lambda impl: impl.get_all_bands_at(0, 185),
        lambda impl: impl.get_all_bands_at_rect(-1, 50, 3, 3),
        lambda impl: impl.get_all_bands_at_rect(155, 50, 3, 3),
        lambda impl: impl.get_all_bands_at_rect(-5, -5, 2, 2),
        lambda impl: impl.get_image_data_subset(-1, 0, 0, 2, 2, 3),
        lambda impl: impl.get_image_data_subset(156, 184, 0, 2, 2, 3),
        lambda impl: impl.get_image_data_subset(157, 0, 0, 1, 1, 1),
    ],
    ids=[
        "pixel-left",
        "pixel-right",
        "pixel-below",
        "rect-left",
        "rect-right",
        "rect-wholly-negative",
        "subset-left",
        "subset-corner",
        "subset-1x1-past-width",
    ],
)
def test_reads_outside_the_ortho_grid_raise(impl, read):
    with pytest.raises(ValueError, match="is not inside the 157 x 185 ortho grid"):
        read(impl)


def test_off_grid_area_average_gets_the_caller_fallback(impl):
    from PySide6.QtCore import QRect

    from wiser.raster.spectrum import calc_rect_spectrum

    plain = RasterDataSet(_open(PLAIN_NETCDF), None)
    assert np.all(np.isnan(calc_rect_spectrum(plain, QRect(-5, -5, 2, 2))))

    ortho = RasterDataSet(impl, None)
    for rect in (QRect(-5, -5, 2, 2), QRect(-1, 50, 3, 3)):
        spectrum = calc_rect_spectrum(ortho, rect)
        assert spectrum.shape == (285,)
        assert np.all(np.isnan(spectrum))


def test_chunked_multi_band_read_matches_netcdf(raw, monkeypatch):
    chunked = _open(EMIT_SUBSET)
    # Two bands of the 18 x 36 swath window per read, so the bands span several reads.
    monkeypatch.setattr(chunked, "_ORTHO_READ_CHUNK_BYTES", 2 * 18 * 36 * 4)
    bands = list(range(20, 27))
    has_source = raw["glt_x"] != 0
    expected = np.full((len(bands), 185, 157), FILL, dtype=np.float32)
    expected[:, has_source] = raw["reflectance"][
        raw["glt_y"][has_source] - 1, raw["glt_x"][has_source] - 1, :
    ][:, bands].T

    np.testing.assert_array_equal(chunked.get_multiple_band_data(bands), expected)
    np.testing.assert_array_equal(chunked.get_image_data_subset(0, 0, 20, 157, 185, 7), expected)
