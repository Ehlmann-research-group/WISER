from pathlib import Path
import types
import unittest
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pytest

import tests.context  # noqa: F401

from PySide2.QtCore import QPoint

from test_utils.memory_cleanup import release_kept_refs
from test_utils.test_model import WiserTestModel

from wiser.gui.app_services import AppServices
from wiser.gui.mnf import get_noise_covariance_subpipeline
from wiser.raster.loader import RasterDataLoader
from wiser.raster.roi import RegionOfInterest
from wiser.raster.selection import RectangleSelection
from wiser.utils.primitives import (
    DeletePolicy,
    ExternalRasterHandle,
    ExternalRoiHandle,
    NoiseMethodType,
    PriorityClass,
)
from wiser.utils.storage_client import RoiBackedSpectraList, StorageClient
from wiser.utils.task_stage_utils import _flatten_valid_rows
from wiser.utils.task_system import AlgorithmPipeline, SemanticTask


if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.utils.task_system import TaskPlan


pytestmark = [
    pytest.mark.integration,
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


_JPL_425_DATASET_PATH = (
    Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets" / "jpl_425_7_7.hdr"
).resolve()


def _load_jpl_425_dataset() -> "RasterDataSet":
    """Load the 425-band, 7x7 JPL ENVI test dataset."""
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.dataset_impl import ENVI_GDALRasterDataImpl

    impls = ENVI_GDALRasterDataImpl.try_load_file(str(_JPL_425_DATASET_PATH), interactive=False)
    assert impls, f"Could not load {_JPL_425_DATASET_PATH}"
    return RasterDataSet(impls[0], data_cache=None)


def _synthetic_dataset(arr_band_first: np.ndarray, *, dataset_id: int) -> "RasterDataSet":
    """Wrap a ``[b][y][x]`` NumPy array as a :class:`RasterDataSet`."""
    loader = RasterDataLoader()
    dataset = loader.dataset_from_numpy_array(arr_band_first)
    dataset.set_id(dataset_id)
    return dataset


class TestExternalRoiHandle(unittest.TestCase):
    """``ExternalRoiHandle.get_meta()`` shape and ``bad_bands`` forwarding."""

    def test_meta_shape_and_bad_bands_forwarding(self):
        # 8-band, 4x4 synthetic dataset.  arr shape: [b][y][x].
        arr = np.arange(8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)
        dataset = _synthetic_dataset(arr, dataset_id=70001)

        # 0 = bad, 1 = good.  Mark band 3 as bad.
        bad_bands = [1, 1, 1, 0, 1, 1, 1, 1]
        dataset.set_bad_bands(bad_bands)

        # RectangleSelection uses exclusive endpoints — (0,0)-(3,3) covers
        # x in [0,1,2] and y in [0,1,2] -> 9 pixels.
        roi = RegionOfInterest(name="rect_3x3")
        roi.set_id(80001)
        roi.add_selection(RectangleSelection(QPoint(0, 0), QPoint(3, 3)))
        self.assertEqual(len(roi.get_all_pixels()), 9)

        handle = ExternalRoiHandle(roi=roi, source_dataset=dataset)
        meta = handle.get_meta()

        self.assertEqual(meta.kind, "spectra_list")
        self.assertEqual(meta.shape, (9, 8))
        self.assertIsNotNone(meta.bad_bands)
        np.testing.assert_array_equal(np.asarray(meta.bad_bands), np.asarray(bad_bands))
        self.assertEqual(np.dtype(meta.elem_type), np.dtype(np.float32))

    def test_meta_empty_roi_has_zero_pixels(self):
        arr = np.zeros((3, 2, 2), dtype=np.float32)
        dataset = _synthetic_dataset(arr, dataset_id=70002)

        roi = RegionOfInterest(name="empty")
        roi.set_id(80002)
        self.assertEqual(len(roi.get_all_pixels()), 0)

        handle = ExternalRoiHandle(roi=roi, source_dataset=dataset)
        meta = handle.get_meta()
        self.assertEqual(meta.shape, (0, 3))


class TestRoiBackedSpectraListReadBatch(unittest.TestCase):
    """``RoiBackedSpectraList.read_batch`` matches direct numpy indexing."""

    @staticmethod
    def _expected_pixels_for_rects(arr_band_first, rects):
        """Reconstruct the (N, b) expected order for a list of rects.

        ``arr_band_first`` is ``[b][y][x]``.  Pixels inside each rectangle are
        enumerated in row-major (y-then-x) order, matching the read path in
        :meth:`RoiBackedSpectraList.read_batch`.
        """
        chunks = []
        for x0, x1, y0, y1 in rects:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    chunks.append(arr_band_first[:, y, x])
        return np.stack(chunks, axis=0)

    def test_read_batch_full_single_rect(self):
        # 5-band, 4x4 dataset.
        arr = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
        dataset = _synthetic_dataset(arr, dataset_id=71001)

        # Single 2x2 rect: x in [1,2], y in [0,1] -> 4 pixels.
        rects = [[1, 2, 0, 1]]
        prefix_sums = [0, 4]
        helper = RoiBackedSpectraList(rects, prefix_sums, dataset)

        self.assertEqual(helper.total_pixels, 4)
        batch = helper.read_batch(0, helper.total_pixels)

        expected = self._expected_pixels_for_rects(arr, rects)
        self.assertEqual(batch.shape, (4, 5))
        np.testing.assert_allclose(batch, expected)

    def test_read_batch_partial_window_spans_two_rects(self):
        # 3-band, 4x4 dataset.
        arr = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        dataset = _synthetic_dataset(arr, dataset_id=71002)

        # Two disjoint single-row rects:
        #   rect 0: x in [0,1], y=0 -> 2 pixels at indices 0..1
        #   rect 1: x in [2,3], y=2 -> 2 pixels at indices 2..3
        rects = [[0, 1, 0, 0], [2, 3, 2, 2]]
        prefix_sums = [0, 2, 4]
        helper = RoiBackedSpectraList(rects, prefix_sums, dataset)

        # Window [1, 3) crosses the rect boundary: last pixel of rect 0 +
        # first pixel of rect 1.
        batch = helper.read_batch(1, 3)
        expected = np.stack([arr[:, 0, 1], arr[:, 2, 2]], axis=0)
        self.assertEqual(batch.shape, (2, 3))
        np.testing.assert_allclose(batch, expected)

        # Full read should equal the concatenation of both rects.
        full = helper.read_batch(0, helper.total_pixels)
        full_expected = self._expected_pixels_for_rects(arr, rects)
        np.testing.assert_allclose(full, full_expected)

    def test_read_batch_empty_window(self):
        arr = np.zeros((2, 3, 3), dtype=np.float32)
        dataset = _synthetic_dataset(arr, dataset_id=71003)

        rects = [[0, 2, 0, 2]]
        prefix_sums = [0, 9]
        helper = RoiBackedSpectraList(rects, prefix_sums, dataset)

        empty = helper.read_batch(4, 4)
        self.assertEqual(empty.shape, (0, 2))

    def test_read_batch_out_of_range_raises(self):
        arr = np.zeros((2, 3, 3), dtype=np.float32)
        dataset = _synthetic_dataset(arr, dataset_id=71004)

        helper = RoiBackedSpectraList([[0, 2, 0, 2]], [0, 9], dataset)
        with self.assertRaises(IndexError):
            helper.read_batch(0, 100)


class TestFlattenValidRows2D(unittest.TestCase):
    """Equivalent 3-D and 2-D inputs flatten to the same ``(K, b_good)``."""

    @staticmethod
    def _meta(bad_bands):
        """Minimal stand-in for the ``region_meta`` argument.

        ``_good_band_mask_for_region_meta`` reads only ``.bad_bands`` from the
        region meta, so a :class:`types.SimpleNamespace` is sufficient.
        """
        if bad_bands is None:
            return types.SimpleNamespace(bad_bands=None)
        return types.SimpleNamespace(bad_bands=np.asarray(bad_bands, dtype=np.int32))

    def test_2d_matches_3d_with_bad_bands_and_invalid_rows(self):
        rng = np.random.default_rng(seed=123)
        b = 6
        tile_3d = rng.standard_normal((2, 3, b)).astype(np.float64)

        # Inject some invalid rows so the filter actually drops pixels.
        tile_3d[0, 1, 2] = np.nan
        tile_3d[1, 2, 4] = np.inf

        # Band 1 is "bad" (0 = bad, 1 = good in the convention used by
        # _good_band_mask_for_region_meta).
        bad_bands = [1, 0, 1, 1, 1, 1]
        meta = self._meta(bad_bands)

        flat_3d = _flatten_valid_rows(tile_3d, meta)

        # Same data viewed as a 2-D spectra-list tile.
        tile_2d = tile_3d.reshape(-1, b)
        flat_2d = _flatten_valid_rows(tile_2d, meta)

        np.testing.assert_array_equal(flat_3d, flat_2d)
        # The good-band mask drops band 1 -> result has b - 1 columns.
        self.assertEqual(flat_3d.shape[1], b - 1)
        # All-bad-row count: pixel (0,1) and pixel (1,2) -> 2 dropped of 6.
        self.assertEqual(flat_3d.shape[0], 2 * 3 - 2)

    def test_2d_no_bad_bands_no_invalid_passes_through(self):
        b = 4
        tile_2d = np.arange(5 * b, dtype=np.float64).reshape(5, b)

        flat = _flatten_valid_rows(tile_2d, self._meta(None))
        self.assertEqual(flat.shape, (5, b))
        np.testing.assert_array_equal(flat, tile_2d)

    def test_2d_drops_rows_with_masked_values(self):
        b = 3
        raw = np.arange(4 * b, dtype=np.float64).reshape(4, b)
        mask = np.zeros_like(raw, dtype=bool)
        # Mask row 2 entirely.
        mask[2, :] = True
        tile_2d = np.ma.MaskedArray(raw, mask=mask)

        flat = _flatten_valid_rows(tile_2d, self._meta(None))
        # Row 2 is dropped: 4 - 1 = 3 surviving rows.
        self.assertEqual(flat.shape, (3, b))
        np.testing.assert_array_equal(flat, np.delete(raw, 2, axis=0))


if __name__ == "__main__":
    unittest.main()
