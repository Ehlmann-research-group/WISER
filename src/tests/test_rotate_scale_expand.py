"""Tests for the fill and mask handling of ``cv2_rotate_scale_expand``."""

import unittest

import cv2
import numpy as np
import pytest

import tests.context  # noqa: F401

from wiser.gui.util import cv2_rotate_scale_expand

pytestmark = [
    pytest.mark.unit,
]

CONSTANT = 7.5

BLENDING_INTERPOLATIONS = {
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

TRANSFORMS = [(35.0, 1.0), (35.0, 0.5), (35.0, 2.0), (90.0, 1.0), (0.0, 1.0), (12.0, 1.3)]


def _constant(shape, mask=None):
    data = np.full(shape, CONSTANT, dtype=np.float32)
    if mask is None:
        mask = np.zeros(shape, dtype=bool)
    return np.ma.MaskedArray(data, mask=mask)


class TestRotateScaleExpandFill(unittest.TestCase):
    def _assert_unmasked_equal_constant(self, out):
        self.assertIsInstance(out, np.ma.MaskedArray)
        valid = ~np.ma.getmaskarray(out)
        self.assertTrue(valid.any())
        np.testing.assert_allclose(out.data[valid], CONSTANT, rtol=1e-5)

    def test_unmasked_output_never_blends_with_the_border_fill(self):
        for name, interp in BLENDING_INTERPOLATIONS.items():
            for angle, scale in TRANSFORMS:
                with self.subTest(interp=name, angle=angle, scale=scale):
                    out = cv2_rotate_scale_expand(_constant((40, 60)), angle, scale, interp=interp)
                    self._assert_unmasked_equal_constant(out)

    def test_unmasked_output_never_blends_with_masked_source_pixels(self):
        mask = np.zeros((40, 60, 3), dtype=bool)
        mask[18:22, 25:30, 0] = True
        mask[5, :, 1] = True
        mask[:, 50, 2] = True
        for name, interp in BLENDING_INTERPOLATIONS.items():
            for angle, scale in TRANSFORMS:
                with self.subTest(interp=name, angle=angle, scale=scale):
                    out = cv2_rotate_scale_expand(_constant((40, 60, 3), mask), angle, scale, interp=interp)
                    self._assert_unmasked_equal_constant(out)

    def test_identity_transform_keeps_every_valid_pixel(self):
        mask = np.zeros((40, 60), dtype=bool)
        mask[18:22, 25:30] = True
        for name, interp in BLENDING_INTERPOLATIONS.items():
            with self.subTest(interp=name):
                out = cv2_rotate_scale_expand(_constant((40, 60), mask), 0.0, 1.0, interp=interp)
                np.testing.assert_array_equal(np.ma.getmaskarray(out), mask)

    def test_nearest_output_is_the_nearest_warp_of_data_and_mask(self):
        rng = np.random.default_rng(0)
        data = rng.random((40, 60, 3)).astype(np.float32)
        mask = rng.random((40, 60, 3)) < 0.1
        angle, scale = 35.0, 0.5
        out = cv2_rotate_scale_expand(
            np.ma.MaskedArray(data, mask=mask), angle, scale, interp=cv2.INTER_NEAREST
        )

        M = cv2.getRotationMatrix2D((30.0, 20.0), angle, scale)
        new_w, new_h = out.shape[1], out.shape[0]
        M[0, 2] += new_w / 2 - 30.0
        M[1, 2] += new_h / 2 - 20.0
        expected_data = cv2.warpAffine(
            np.where(mask, 0, data).astype(np.float32),
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        expected_valid = cv2.warpAffine(
            (~mask).astype(np.uint8) * 255,
            M,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        np.testing.assert_array_equal(out.data, expected_data)
        np.testing.assert_array_equal(np.ma.getmaskarray(out), expected_valid == 0)


if __name__ == "__main__":
    unittest.main()
