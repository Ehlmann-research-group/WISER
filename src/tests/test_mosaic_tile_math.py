"""Unit tests for the tile-grid math backing the mosaic pixel cache (EPIC #629, #674).

Pure arithmetic — no widget, no ``QApplication``, no GDAL. These cover the three
building blocks the tiled pixel cache is keyed on: snapping a continuous camera scale to
a discrete power-of-two zoom bucket, the world rectangle a cell covers, and the set of
cells (with an optional prefetch ring) covering a viewport.
"""
import math
import unittest

import tests.context  # noqa: F401  (adds src/ to sys.path)
from wiser.gui.mosaic_view import (
    _PREFETCH_RING,
    _TILE_PX,
    _bucket_wupp,
    _tile_world_extent,
    _tiles_covering_extent,
    _zoom_bucket,
)

import pytest

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
]


class TestZoomBucket(unittest.TestCase):
    def test_exact_powers_of_two_map_to_their_exponent(self) -> None:
        for exp in (-12, -3, 0, 5, 14):
            self.assertEqual(_zoom_bucket(2.0**exp), exp)
            self.assertEqual(_bucket_wupp(_zoom_bucket(2.0**exp)), 2.0**exp)

    def test_snaps_to_nearest_bucket(self) -> None:
        # 27.4 is between 2^4=16 and 2^5=32; log2(27.4)~=4.78 -> rounds to 5.
        self.assertEqual(_zoom_bucket(27.4), 5)
        # 20.0 -> log2~=4.32 -> rounds to 4.
        self.assertEqual(_zoom_bucket(20.0), 4)

    def test_draw_scale_stays_within_sqrt2(self) -> None:
        # bucket_wupp / wupp must land in [1/sqrt2, sqrt2] for any continuous wupp,
        # which is the whole point of rounding rather than flooring.
        lo, hi = 1.0 / math.sqrt(2.0), math.sqrt(2.0)
        for wupp in (0.0003, 1.0, 13.7, 27.4, 999.0, 1e6):
            scale = _bucket_wupp(_zoom_bucket(wupp)) / wupp
            self.assertTrue(lo - 1e-9 <= scale <= hi + 1e-9, f"{wupp} -> {scale}")

    def test_degenerate_scales_fall_back_to_zero(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            self.assertEqual(_zoom_bucket(bad), 0)


class TestTileWorldExtent(unittest.TestCase):
    def test_cell_spans_tile_px_times_bucket_resolution(self) -> None:
        bucket = 5  # bucket_wupp = 32
        tws = _TILE_PX * 32.0
        self.assertEqual(_tile_world_extent(bucket, 0, 0), (0.0, 0.0, tws, tws))
        self.assertEqual(
            _tile_world_extent(bucket, 3, -2),
            (3 * tws, -2 * tws, 4 * tws, -1 * tws),
        )

    def test_adjacent_cells_share_an_edge_without_gap_or_overlap(self) -> None:
        bucket = 2
        left = _tile_world_extent(bucket, 0, 0)
        right = _tile_world_extent(bucket, 1, 0)
        self.assertEqual(left[2], right[0])  # left max_x == right min_x


class TestTilesCoveringExtent(unittest.TestCase):
    def test_single_cell_when_extent_is_inside_one_tile(self) -> None:
        bucket = 0  # bucket_wupp = 1 -> tws = _TILE_PX
        # A small rect fully inside cell (0, 0).
        cells = _tiles_covering_extent(bucket, (1.0, 1.0, 10.0, 10.0))
        self.assertEqual(cells, [(0, 0)])

    def test_covers_all_straddled_cells(self) -> None:
        bucket = 0
        tws = _TILE_PX
        # Straddle a 2x2 block of cells.
        extent = (tws - 5.0, tws - 5.0, tws + 5.0, tws + 5.0)
        cells = set(_tiles_covering_extent(bucket, extent))
        self.assertEqual(cells, {(0, 0), (1, 0), (0, 1), (1, 1)})

    def test_exact_boundary_does_not_pull_in_the_next_cell(self) -> None:
        bucket = 0
        tws = _TILE_PX
        # max exactly on the cell-1 boundary should stay within cell 0 (half-open).
        cells = _tiles_covering_extent(bucket, (0.0, 0.0, float(tws), float(tws)))
        self.assertEqual(set(cells), {(0, 0)})

    def test_prefetch_ring_adds_one_cell_on_every_side(self) -> None:
        bucket = 0
        no_ring = set(_tiles_covering_extent(bucket, (1.0, 1.0, 10.0, 10.0), ring=0))
        with_ring = set(_tiles_covering_extent(bucket, (1.0, 1.0, 10.0, 10.0), ring=1))
        self.assertEqual(no_ring, {(0, 0)})
        # A 1-ring around a single cell is the surrounding 3x3 block.
        self.assertEqual(
            with_ring,
            {(c, r) for c in (-1, 0, 1) for r in (-1, 0, 1)},
        )
        self.assertEqual(_PREFETCH_RING, 1)  # the wired-up default matches this test

    def test_degenerate_extent_yields_no_cells(self) -> None:
        self.assertEqual(_tiles_covering_extent(0, (5.0, 5.0, 5.0, 5.0)), [])
        self.assertEqual(_tiles_covering_extent(0, (10.0, 0.0, 0.0, 10.0)), [])


if __name__ == "__main__":
    unittest.main()
