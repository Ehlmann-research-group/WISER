import numpy as np

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from PySide2 import QtCore, QtWidgets, QtTest, QtGui


# Shared fixture for the PCA/MNF "unmasked NaN/Inf" regression tests in
# test_pca_task.py and test_mnf.py. The cube carries one bad band, a nodata
# sentinel at fully-masked pixels, and unmasked NaN/+Inf/-Inf in good bands —
# the case that made PCA feed NaN into sklearn before the cleaning fix.
NAN_INF_BANDS, NAN_INF_HEIGHT, NAN_INF_WIDTH = 5, 12, 12
NAN_INF_DATA_IGNORE_VALUE = -9999.0
# Band 2 is bad -> 4 good bands remain.
NAN_INF_BAD_BANDS = [1, 1, 0, 1, 1]
# (y, x) pixels cleaning must drop; the reduction output must carry the nodata
# fill at exactly these locations.
NAN_INF_INVALID_YX = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def build_unmasked_nan_inf_cube() -> np.ndarray:
    """Return a ``[b][y][x]`` float32 cube for the NaN/Inf regression tests.

    Pixels (0,0) and (1,1) are fully nodata (every band is the sentinel, so they
    are masked at read time). Good-band pixels (2,2)/(3,3)/(4,4) carry an unmasked
    NaN/+Inf/-Inf respectively, so neither bad-band nor nodata masking removes
    them. Pair with :data:`NAN_INF_BAD_BANDS` and :data:`NAN_INF_DATA_IGNORE_VALUE`
    via ``dataset.set_bad_bands`` / ``dataset.set_data_ignore_value``.
    """
    rng = np.random.default_rng(0)
    cube = (
        rng.standard_normal((NAN_INF_BANDS, NAN_INF_HEIGHT, NAN_INF_WIDTH)).astype(np.float32) * 50.0 + 1000.0
    )
    cube[:, 0, 0] = NAN_INF_DATA_IGNORE_VALUE
    cube[:, 1, 1] = NAN_INF_DATA_IGNORE_VALUE
    cube[0, 2, 2] = np.nan
    cube[1, 3, 3] = np.inf
    cube[3, 4, 4] = -np.inf
    return cube


def assert_reduction_drops_invalid_pixels(test_case, output, num_components: int) -> None:
    """Assert a PCA/MNF reduction wrote the nodata fill at every invalid pixel and
    finite components everywhere else.

    ``output`` is the ``[y][x][component]`` cube read back from the pipeline.
    """
    out = np.asarray(np.ma.getdata(output))
    test_case.assertEqual(out.shape, (NAN_INF_HEIGHT, NAN_INF_WIDTH, num_components))

    valid_mask = np.ones((NAN_INF_HEIGHT, NAN_INF_WIDTH), dtype=bool)
    for y, x in NAN_INF_INVALID_YX:
        valid_mask[y, x] = False
        test_case.assertTrue(
            np.allclose(out[y, x, :], NAN_INF_DATA_IGNORE_VALUE),
            f"Invalid pixel {(y, x)} should carry the nodata fill, got {out[y, x, :]}",
        )
    # Every surviving pixel produced finite components (no NaN/Inf leaked through).
    test_case.assertTrue(np.all(np.isfinite(out[valid_mask])))


def are_pixels_close(pixel1, pixel2) -> bool:
    """
    Helper functions to determine if two pixels are close. Used for when scrolling
    in zoom pane and center's don't exactly align.
    """
    if isinstance(pixel1, (QPoint, QPointF)):
        pixel1 = (pixel1.x(), pixel1.y())

    if isinstance(pixel2, (QPoint, QPointF)):
        pixel2 = (pixel2.x(), pixel2.y())

    pixel1_diff = abs(pixel1[0] - pixel1[1])
    pixel2_diff = abs(pixel2[0] - pixel2[1])

    diff_similar = abs(pixel1_diff - pixel2_diff) <= 2

    epsilon = 2
    return abs(pixel1[0] - pixel2[0]) <= epsilon and diff_similar


def are_qrects_close(qrect1: QRect, qrect2: QRect, epsilon=3) -> bool:
    """
    Helper functions to determine if two qrects are close. Used for when scrolling
    in zoom pane and center's don't exactly align. Not exact alignment seems to
    occur when we enter the event loop.
    """
    top_left_1 = qrect1.topLeft()
    top_left_1 = (top_left_1.x(), top_left_1.y())

    width_1 = qrect1.width()
    height_1 = qrect1.height()

    top_left_2 = qrect2.topLeft()
    top_left_2 = (top_left_2.x(), top_left_2.y())

    width_2 = qrect2.width()
    height_2 = qrect2.height()

    width_diff = abs(width_1 - width_2)
    height_diff = abs(height_1 - height_2)

    diff_similar = abs(width_diff - height_diff) <= epsilon

    return top_left_1 == top_left_2 and diff_similar and width_diff <= epsilon and height_diff <= epsilon
