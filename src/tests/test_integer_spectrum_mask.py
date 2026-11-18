"""Tests that spectra read from integer datasets mark no-data and bad bands as NaN.

An integer array cannot hold NaN, so a filtered spectrum read promotes integer
data to a floating-point type before masking.  Unfiltered reads keep the
source dtype.
"""

import warnings

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication

import tests.context  # noqa: F401

from wiser.gui.pixel_value_widget import PixelValueWidget
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectra_export import export_roi_pixel_spectra
from wiser.raster.spectrum import calc_rect_spectrum, calc_roi_spectrum

IGNORE = -9999
BAD_BANDS = [1, 0, 1]


@pytest.fixture(params=[np.int16, np.uint16, np.int32])
def int_cube(request):
    """
    A (3, 2, 2) integer cube.  Pixel (x=1, y=1) holds the ignore value on every
    band, pixel (x=0, y=1) holds it on band 2 only, and pixel (x=1, y=0) holds a
    real zero on band 0.  Band 1 is flagged bad.
    """
    dtype = np.dtype(request.param)
    ignore = IGNORE if np.issubdtype(dtype, np.signedinteger) else np.iinfo(dtype).max
    data = (np.arange(3 * 2 * 2).reshape(3, 2, 2) + 100).astype(dtype)
    data[:, 1, 1] = ignore
    data[2, 1, 0] = ignore
    data[0, 0, 1] = 0
    ds = RasterDataLoader().dataset_from_numpy_array(data)
    ds.set_data_ignore_value(ignore)
    ds.set_bad_bands(BAD_BANDS)
    return ds, data


def test_pixel_spectrum_masks_ignore_value_and_bad_band_as_nan(int_cube):
    ds, data = int_cube
    assert np.isnan(ds.get_all_bands_at(1, 1)).all()

    spectrum = ds.get_all_bands_at(0, 1)
    np.testing.assert_array_equal(np.isnan(spectrum), [False, True, True])
    assert spectrum[0] == data[0, 1, 0]


def test_pixel_spectrum_keeps_real_zero(int_cube):
    ds, data = int_cube
    spectrum = ds.get_all_bands_at(1, 0)
    np.testing.assert_array_equal(spectrum, [0, np.nan, data[2, 0, 1]])


def test_pixel_spectrum_holds_integer_values_exactly(int_cube):
    ds, data = int_cube
    spectrum = ds.get_all_bands_at(0, 0)
    assert np.issubdtype(spectrum.dtype, np.floating)
    np.testing.assert_array_equal(spectrum[[0, 2]], data[[0, 2], 0, 0])


def test_unfiltered_pixel_spectrum_keeps_source_dtype(int_cube):
    ds, data = int_cube
    spectrum = ds.get_all_bands_at(1, 1, filter_bad_values=False, filter_data_ignore_value=False)
    assert spectrum.dtype == data.dtype
    np.testing.assert_array_equal(spectrum, data[:, 1, 1])


def test_rect_read_masks_ignore_value_and_bad_band_as_nan(int_cube):
    ds, data = int_cube
    block = ds.get_all_bands_at_rect(0, 0, 2, 2)
    expected = np.zeros(data.shape, dtype=bool)
    expected[:, 1, 1] = True
    expected[2, 1, 0] = True
    expected[1] = True
    np.testing.assert_array_equal(np.isnan(block), expected)


def test_unfiltered_rect_read_keeps_source_dtype(int_cube):
    ds, data = int_cube
    block = ds.get_all_bands_at_rect(0, 0, 2, 2, filter_bad_values=False)
    assert block.dtype == data.dtype
    np.testing.assert_array_equal(block, data)


def _expected_mean(data):
    valid = data.astype(np.float64)
    valid[:, 1, 1] = np.nan
    valid[2, 1, 0] = np.nan
    valid[1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(valid.reshape(3, -1), axis=1)


def test_rect_and_roi_mean_spectrum_exclude_ignore_value_and_bad_band(int_cube):
    ds, data = int_cube
    expected = _expected_mean(data)

    roi = RegionOfInterest()
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(2, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rect_spectrum = calc_rect_spectrum(ds, QRect(0, 0, 2, 2))
        roi_spectrum = calc_roi_spectrum(ds, roi)
    np.testing.assert_allclose(rect_spectrum, expected)
    np.testing.assert_allclose(roi_spectrum, expected)


def test_roi_pixel_export_writes_nan_for_ignore_value_and_bad_band(int_cube, tmp_path):
    ds, data = int_cube
    roi = RegionOfInterest()
    roi.add_selection(RectangleSelection(QPoint(1, 1), QPoint(2, 2)))
    path = tmp_path / "pixels.txt"
    export_roi_pixel_spectra(str(path), ds, roi)
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:]]
    assert [row[1] for row in rows] == ["NaN", "NaN", "NaN"]


def test_pixel_value_widget_formats_integer_values_without_exponent():
    values = np.array([65535.0, np.nan, 0.0])
    assert PixelValueWidget._format_values(values, (0,), integer=True) == "Val: 65535"
    assert PixelValueWidget._format_values(values, (0, 1, 2), integer=True) == "R: 65535  G: nan  B: 0"
    assert PixelValueWidget._format_values(values, (0,), integer=False) == "Val: 6.554e+04"


def test_pixel_value_widget_shows_integer_dataset_values_as_whole_numbers():
    QApplication.instance() or QApplication([])
    data = np.array([[[65000]], [[IGNORE]], [[0]]], dtype=np.int32)
    ds = RasterDataLoader().dataset_from_numpy_array(data)
    ds.set_data_ignore_value(IGNORE)
    widget = PixelValueWidget()
    widget.update_pixel_values(ds, (0, 0), (0, 1, 2))
    assert widget._ui.lbl_display_values.text() == "R: 65000  G: nan  B: 0"
