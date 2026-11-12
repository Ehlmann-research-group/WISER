"""Tests that ``write_raster_to_dataset`` raises when GDAL cannot write the array."""

import numpy as np
import pytest
from osgeo import gdal

import tests.context  # noqa: F401

from wiser.bandmath.utils import write_raster_to_dataset


@pytest.fixture
def dataset():
    ds = gdal.GetDriverByName("MEM").Create("", 8, 6, 2, gdal.GDT_Float32)
    yield ds
    ds = None


def test_full_extent_array_is_written(dataset):
    data = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
    write_raster_to_dataset(dataset, [0, 1], data, gdal.GDT_Float32)
    np.testing.assert_array_equal(dataset.ReadAsArray(), data)


@pytest.mark.parametrize("use_exceptions", [True, False])
@pytest.mark.parametrize("shape", [(2, 5, 8), (2, 6, 9), (1, 6, 8)])
def test_array_not_matching_dataset_raises(dataset, use_exceptions, shape):
    with gdal.ExceptionMgr(useExceptions=use_exceptions):
        with pytest.raises(ValueError):
            write_raster_to_dataset(dataset, [0, 1], np.ones(shape, np.float32), gdal.GDT_Float32)


def test_failed_write_raises_without_gdal_exceptions(dataset):
    data = np.ones((2, 6, 8), np.float32)
    with gdal.ExceptionMgr(useExceptions=False):
        with pytest.raises(RuntimeError):
            write_raster_to_dataset(dataset, [0, 5], data, gdal.GDT_Float32)
