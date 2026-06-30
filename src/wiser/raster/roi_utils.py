"""
Utilities for decomposing a RegionOfInterest into a minimal set of axis-aligned
rectangles (greedy run-length encoding), used both by spectral averaging
(spectrum.py) and the ROI-backed storage handle (primitives.py).

Rectangle format throughout: ``[x_start, x_end, y_start, y_end]`` in
whichever coordinate system the caller uses (local raster or absolute image).
All returned arrays have dtype ``np.intp`` and shape ``(R, 4)`` — even when
empty (shape ``(0, 4)``).
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, List

import numpy as np
from PySide6.QtCore import QRect

if TYPE_CHECKING:
    from wiser.raster.roi import RegionOfInterest


def find_rectangles_in_row(row: np.ndarray, y: int) -> List[np.ndarray]:
    """Return a list of ``[x_start, x_end, y, y]`` arrays for every run of 1s in *row*."""
    rectangles = []
    start = None

    for x in range(len(row)):
        if row[x] == 1 and start is None:
            start = x
        elif row[x] == 0 and start is not None:
            rectangles.append(np.array([start, x - 1, y, y]))
            start = None

    if start is not None:
        rectangles.append(np.array([start, len(row) - 1, y, y]))

    return rectangles


def raster_to_combined_rectangles_x_axis(raster: np.ndarray) -> np.ndarray:
    """
    Greedy rectangle packing along the x-axis (rows scanned top-to-bottom).

    Adjacent rows that share the exact same run of 1s are merged into a single
    rectangle spanning all those rows.

    Returns shape ``(R, 4)`` with columns ``[x_start, x_end, y_start, y_end]``
    in raster-local coordinates.  Returns shape ``(0, 4)`` when the raster
    contains no active pixels.
    """
    rectangles = []
    previous_row_rectangles: deque = deque()

    for y in range(raster.shape[0]):
        row = raster[y]
        current_row_rectangles = find_rectangles_in_row(row, y)

        for _ in range(len(previous_row_rectangles)):
            prev_rect = previous_row_rectangles.pop()
            prev_x_start, prev_x_end, prev_y_start, prev_y_end = prev_rect

            merged = False
            for curr_rect in current_row_rectangles:
                x_start, x_end, y_start, y_end = curr_rect
                if prev_x_start == x_start and prev_x_end == x_end and prev_y_end == y - 1:
                    curr_rect[-2] = prev_y_start
                    merged = True
                    break

            if not merged:
                rectangles.append(np.array(prev_rect))

        previous_row_rectangles = current_row_rectangles

    rectangles += list(previous_row_rectangles)

    if not rectangles:
        return np.empty((0, 4), dtype=np.intp)
    return np.array(rectangles, dtype=np.intp)


def raster_to_combined_rectangles_y_axis(raster_y: np.ndarray) -> np.ndarray:
    """
    Greedy rectangle packing along the y-axis (columns scanned left-to-right).

    Transposes the raster, runs the x-axis algorithm, then swaps coordinate
    columns back so the result uses the same ``[x_start, x_end, y_start, y_end]``
    format as :func:`raster_to_combined_rectangles_x_axis`.

    Returns shape ``(R, 4)``, or ``(0, 4)`` when empty.
    """
    raster_x = raster_y.T
    rectangles_x = raster_to_combined_rectangles_x_axis(raster_x)
    if len(rectangles_x) == 0:
        return np.empty((0, 4), dtype=np.intp)
    # In the transposed raster, column 0/1 are y-coords and column 2/3 are x-coords
    # of the original; swap them back to [x_start, x_end, y_start, y_end].
    return rectangles_x[:, [2, 3, 0, 1]].copy()


def array_to_qrects(array: np.ndarray) -> List[QRect]:
    """Convert an ``(R, 4)`` rect array into a list of :class:`QRect` objects."""
    qrects = []
    for row in array:
        x1, x2, y1, y2 = row
        qrects.append(QRect(int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)))
    return qrects


def create_raster_from_roi(roi: "RegionOfInterest") -> np.ndarray:
    """
    Rasterize *roi* into a binary ``uint8`` array bounded by the ROI's bounding
    box.  Overlapping selections are deduplicated automatically because
    :meth:`RegionOfInterest.get_all_pixels` returns a set.

    The returned array has shape ``(bbox.height(), bbox.width())`` with 1 where
    a pixel is part of the ROI and 0 elsewhere.  The vectorised implementation
    is substantially faster than the previous pixel-by-pixel loop for large ROIs.
    """
    bbox = roi.get_bounding_box()
    pixels = roi.get_all_pixels()  # Set[Tuple[x, y]]

    xmin = bbox.left()
    ymin = bbox.top()
    height = bbox.height()
    width = bbox.width()

    raster = np.zeros((height, width), dtype=np.uint8)

    if pixels:
        pixels_arr = np.array(list(pixels), dtype=np.intp)  # (N, 2): col0=x, col1=y
        col_idxs = pixels_arr[:, 0] - xmin  # x - xmin → column index
        row_idxs = pixels_arr[:, 1] - ymin  # y - ymin → row index
        raster[row_idxs, col_idxs] = 1

    return raster
