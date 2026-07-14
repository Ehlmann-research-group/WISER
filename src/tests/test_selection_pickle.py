"""Tests that Selection and RegionOfInterest round-trip cleanly through pickle.

RasterDataSet is not directly pickleable (it wraps a GDAL handle).  The
__getstate__ / __setstate__ hooks on Selection replace the dataset with a
SerializedForm on pickling and rebuild it on unpickling.  These tests verify
the contract end-to-end using small in-memory NumPy datasets so they run fast
and without any GUI dependencies.
"""
import pickle

import numpy as np
import pytest
from PySide6.QtCore import QPoint

from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import (
    MultiPixelSelection,
    PolygonSelection,
    RectangleSelection,
    SinglePixelSelection,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.smoke,
]


def _make_dataset(bands: int = 3, height: int = 5, width: int = 5):
    """Return a small in-memory RasterDataSet backed by a NumPy array."""
    arr = np.arange(1, bands + 1, dtype=np.float32).reshape((bands, 1, 1)) * np.ones(
        (bands, height, width), dtype=np.float32
    )
    return RasterDataLoader().dataset_from_numpy_array(arr)


def _roundtrip(obj):
    return pickle.loads(pickle.dumps(obj))


class TestSelectionPickleNoDataset:
    """Selections without an associated dataset should pickle without errors."""

    def test_single_pixel_no_dataset(self):
        sel = SinglePixelSelection(pixel=QPoint(2, 3))
        sel2 = _roundtrip(sel)
        assert sel2.get_pixel().toTuple() == (2, 3)
        assert sel2.get_dataset() is None

    def test_multi_pixel_no_dataset(self):
        pixels = [QPoint(0, 0), QPoint(1, 2), QPoint(3, 4)]
        sel = MultiPixelSelection(pixels=pixels)
        sel2 = _roundtrip(sel)
        assert sel2.get_all_pixels() == sel.get_all_pixels()
        assert sel2.get_dataset() is None

    def test_rectangle_no_dataset(self):
        sel = RectangleSelection(QPoint(1, 1), QPoint(4, 3))
        sel2 = _roundtrip(sel)
        assert sel2.get_all_pixels() == sel.get_all_pixels()

    def test_polygon_no_dataset(self):
        sel = PolygonSelection([QPoint(0, 0), QPoint(4, 0), QPoint(2, 4)])
        sel2 = _roundtrip(sel)
        assert sel2.get_all_pixels() == sel.get_all_pixels()


class TestSelectionPickleWithDataset:
    """Selections that hold a RasterDataSet must rebuild the dataset on unpickle."""

    def test_single_pixel_with_dataset(self):
        dataset = _make_dataset()
        sel = SinglePixelSelection(pixel=QPoint(1, 2), dataset=dataset)
        sel2 = _roundtrip(sel)

        assert sel2.get_pixel().toTuple() == (1, 2)
        assert sel2.get_dataset() is not None
        assert sel2.get_dataset().get_width() == dataset.get_width()
        assert sel2.get_dataset().get_height() == dataset.get_height()
        assert sel2.get_dataset().num_bands() == dataset.num_bands()

    def test_multi_pixel_with_dataset(self):
        dataset = _make_dataset()
        pixels = [QPoint(0, 0), QPoint(2, 1), QPoint(4, 4)]
        sel = MultiPixelSelection(pixels=pixels, dataset=dataset)
        sel2 = _roundtrip(sel)

        assert sel2.get_all_pixels() == sel.get_all_pixels()
        assert sel2.get_dataset() is not None
        assert sel2.get_dataset().num_bands() == dataset.num_bands()

    def test_dataset_data_preserved_after_roundtrip(self):
        """Verify that pixel reads from the rebuilt dataset match the original."""
        dataset = _make_dataset(bands=4, height=6, width=6)
        sel = MultiPixelSelection(pixels=[QPoint(0, 0)], dataset=dataset)
        sel2 = _roundtrip(sel)

        original_band = dataset.get_band_data(0)
        restored_band = sel2.get_dataset().get_band_data(0)
        np.testing.assert_array_equal(original_band, restored_band)


class TestROIPickle:
    """RegionOfInterest with a list of selections must round-trip end-to-end."""

    def test_roi_multi_selection_roundtrip(self):
        """ROI with selections both with and without a dataset round-trips cleanly."""
        dataset = _make_dataset()

        roi = RegionOfInterest(name="test_roi", color="blue")
        roi.set_description("pickle smoke test")

        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(4, 3)))
        roi.add_selection(PolygonSelection([QPoint(0, 0), QPoint(3, 0), QPoint(1, 3)]))
        roi.add_selection(
            MultiPixelSelection(
                pixels=[QPoint(0, 0), QPoint(2, 2), QPoint(4, 4)],
                dataset=dataset,
            )
        )

        roi2 = _roundtrip(roi)

        assert roi2.get_name() == roi.get_name()
        assert roi2.get_description() == roi.get_description()
        assert len(roi2.get_selections()) == 3
        assert roi2.get_all_pixels() == roi.get_all_pixels()

        multi_sel = roi2.get_selections()[2]
        assert isinstance(multi_sel, MultiPixelSelection)
        assert multi_sel.get_dataset() is not None
        assert multi_sel.get_dataset().num_bands() == dataset.num_bands()

    def test_roi_no_dataset_selections(self):
        """ROI whose selections have no datasets also round-trips correctly."""
        roi = RegionOfInterest(name="no_dataset_roi")
        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(5, 5)))
        roi.add_selection(MultiPixelSelection(pixels=[QPoint(1, 1), QPoint(2, 3)]))

        roi2 = _roundtrip(roi)

        assert roi2.get_name() == roi.get_name()
        assert roi2.get_all_pixels() == roi.get_all_pixels()
        for sel in roi2.get_selections():
            assert sel.get_dataset() is None
