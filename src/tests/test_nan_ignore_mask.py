"""Tests that every RasterDataSet read masks exactly the declared data ignore value.

A NaN ignore value masks the NaN pixels.  A finite ignore value masks the pixels
equal to it and leaves undeclared NaN pixels as they are.
"""

import math

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect

import tests.context  # noqa: F401

from wiser.gui.similarity_transform_dialog import create_rotated_scaled_dataset
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.raster.spectrum import calc_rect_spectrum, calc_roi_spectrum

# Pixel (x=1, y=2) is NaN on every band; pixel (x=3, y=0) is NaN on band 1 only.
_NAN_PIXELS = [(slice(None), 2, 1), (1, 0, 3)]


def _nan_cube():
    data = (np.arange(3 * 4 * 5).reshape(3, 4, 5) + 1).astype(np.float32)
    for index in _NAN_PIXELS:
        data[index] = np.nan
    return data


def _dataset(data, ignore_value):
    ds = RasterDataLoader().dataset_from_numpy_array(data)
    ds.set_data_ignore_value(ignore_value)
    return ds


@pytest.fixture
def nan_ignore_dataset():
    data = _nan_cube()
    return _dataset(data, np.nan), data


def _assert_masks_nan(arr, expected):
    np.testing.assert_array_equal(np.ma.getmaskarray(arr), np.isnan(expected))


def test_image_data_masks_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    _assert_masks_nan(ds.get_image_data(), data)


def test_image_data_subset_masks_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    _assert_masks_nan(ds.get_image_data_subset(0, 0, 0, 5, 4, 3), data)


@pytest.mark.parametrize("band", [0, 1, 2])
def test_band_data_masks_nan(nan_ignore_dataset, band):
    ds, data = nan_ignore_dataset
    _assert_masks_nan(ds.get_band_data(band), data[band])


def test_band_data_normalized_masks_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    _assert_masks_nan(ds.get_band_data_normalized(1), data[1])


def test_sample_band_data_masks_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    _assert_masks_nan(ds.sample_band_data(1, 1), data[1])


def test_multiple_band_data_masks_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    _assert_masks_nan(ds.get_multiple_band_data([0, 1]), data[[0, 1]])


def test_unfiltered_read_leaves_nan_unmasked(nan_ignore_dataset):
    ds, _ = nan_ignore_dataset
    assert not np.ma.getmaskarray(ds.get_band_data(1, filter_data_ignore_value=False)).any()


def test_pixel_spectrum_reads_nan_as_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    np.testing.assert_array_equal(np.isnan(ds.get_all_bands_at(3, 0)), [False, True, False])
    np.testing.assert_array_equal(ds.get_all_bands_at(0, 0), data[:, 0, 0])


def test_band_stats_exclude_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    stats = ds.get_band_stats(1)
    assert stats.get_min() == np.nanmin(data[1])
    assert stats.get_max() == np.nanmax(data[1])


def test_rect_and_roi_mean_spectrum_exclude_nan(nan_ignore_dataset):
    ds, data = nan_ignore_dataset
    expected = np.nanmean(data.reshape(3, -1), axis=1)
    assert np.isfinite(expected).all()

    rect_spectrum = calc_rect_spectrum(ds, QRect(0, 0, 5, 4))
    np.testing.assert_allclose(rect_spectrum, expected)

    roi = RegionOfInterest()
    roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(5, 4)))
    np.testing.assert_allclose(calc_roi_spectrum(ds, roi), expected)


def test_finite_ignore_value_leaves_nan_unmasked():
    data = _nan_cube()
    data[0, 0, 0] = -9999.0
    ds = _dataset(data, -9999.0)
    band = ds.get_band_data(0)
    expected = np.zeros(data[0].shape, dtype=bool)
    expected[0, 0] = True
    np.testing.assert_array_equal(np.ma.getmaskarray(band), expected)


@pytest.mark.parametrize("filter_bad_values", [True, False])
def test_pixel_spectrum_masks_finite_ignore_value_on_every_band(filter_bad_values):
    data = (np.arange(3 * 4 * 5).reshape(3, 4, 5) + 1).astype(np.float32)
    data[:, 2, 1] = -9999.0
    ds = _dataset(data, -9999.0)
    ds.set_bad_bands([1, 0, 1])

    spectrum = ds.get_all_bands_at(1, 2, filter_bad_values=filter_bad_values)
    assert np.isnan(spectrum).all()

    rect = ds.get_all_bands_at_rect(1, 2, 1, 1)
    assert np.isnan(rect).all()


def test_mask_data_ignore_value_keeps_existing_mask():
    from wiser.raster.utils import mask_data_ignore_value

    arr = np.ma.masked_array([1.0, np.nan, 3.0], mask=[True, False, False])
    np.testing.assert_array_equal(
        np.ma.getmaskarray(mask_data_ignore_value(arr, np.nan)), [True, True, False]
    )
    np.testing.assert_array_equal(np.ma.getmaskarray(mask_data_ignore_value(arr, 3.0)), [True, False, True])


def test_rotated_float_output_reopens_with_nan_fill_masked(tmp_path):
    out_path = str(tmp_path / "rotated.tif")
    ds = _dataset((np.arange(2 * 6 * 8).reshape(2, 6, 8) + 1).astype(np.float32), None)
    rotation = 30.0
    create_rotated_scaled_dataset(ds, rotation, 1.0, cv2.INTER_NEAREST, out_path)

    reopened = RasterDataLoader().load_from_file(out_path, interactive=False)[0]
    assert math.isnan(reopened.get_data_ignore_value())
    band = reopened.get_band_data(0)
    np.testing.assert_array_equal(np.ma.getmaskarray(band), np.isnan(np.ma.getdata(band)))
    assert band.mask[0, 0]
    assert not band.mask[band.shape[0] // 2, band.shape[1] // 2]
