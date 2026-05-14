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


if __name__ == "__main__":
    unittest.main()
