"""Unit tests for the three-tier in-memory cache in ``wiser.raster.data_cache`` (#772).

Small NumPy arrays only -- no widget, no ``QApplication``, no GDAL. These cover the
contract every caller of the render and computation caches relies on: a miss followed
by ``add_cache_item`` stores the entry, the next equivalent request hits, and the
running byte count stays in step with what is actually held so that eviction keeps the
cache inside its capacity.
"""
import unittest

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

from wiser.raster.data_cache import (
    Cache,
    ComputationCache,
    DataCache,
    HistogramCache,
    RenderCache,
)

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.unit,
    pytest.mark.regression,
]

# The caches only need the "dataset" to be hashable, so a sentinel string stands in for
# a RasterDataSet throughout and keeps these tests free of GDAL.
_DATASET = "dataset-a"
_OTHER_DATASET = "dataset-b"


def _array(n_bytes: int) -> np.ndarray:
    """
    Returns a float64 array occupying exactly n_bytes bytes.

    Args:
        n_bytes (int): The size of the array in bytes; must be a multiple of 8.
    """
    assert n_bytes % 8 == 0, "float64 arrays are sized in multiples of 8 bytes"
    return np.arange(n_bytes // 8, dtype=np.float64)


class TestAddCacheItemStoresNewKeys(unittest.TestCase):
    """A miss followed by an add must be served from the cache next time."""

    def test_computation_cache_serves_a_hit_after_a_miss(self) -> None:
        cache = ComputationCache()
        key = cache.get_cache_key(_DATASET, band_index=3, normalized=True)
        arr = _array(800)

        # The sequence every read path in dataset.py follows: look up, read on a
        # miss, then store what was read.
        self.assertFalse(cache.in_cache(key))
        self.assertIsNone(cache.get_cache_item(key))
        self.assertTrue(cache.add_cache_item(key, arr))

        self.assertTrue(cache.in_cache(key))
        np.testing.assert_array_equal(cache.get_cache_item(key), arr)

    def test_render_cache_serves_a_hit_after_a_miss(self) -> None:
        cache = RenderCache()
        key = cache.get_cache_key(_DATASET, (0, 1, 2), ("linear",), None)
        img = _array(1600)

        self.assertTrue(cache.add_cache_item(key, img))
        np.testing.assert_array_equal(cache.get_cache_item(key), img)

    def test_distinct_keys_are_kept_side_by_side(self) -> None:
        cache = ComputationCache()
        first_key = cache.get_cache_key(_DATASET, band_index=0, normalized=False)
        second_key = cache.get_cache_key(_DATASET, band_index=1, normalized=False)
        normalized_key = cache.get_cache_key(_DATASET, band_index=0, normalized=True)
        first, second, normalized = _array(80), _array(160), _array(240)

        cache.add_cache_item(first_key, first)
        cache.add_cache_item(second_key, second)
        cache.add_cache_item(normalized_key, normalized)

        np.testing.assert_array_equal(cache.get_cache_item(first_key), first)
        np.testing.assert_array_equal(cache.get_cache_item(second_key), second)
        np.testing.assert_array_equal(cache.get_cache_item(normalized_key), normalized)

    def test_masked_arrays_round_trip(self) -> None:
        # get_image_data() masks the data-ignore value before storing, so the
        # computation cache holds masked arrays as well as plain ones.
        cache = ComputationCache()
        key = cache.get_cache_key(_DATASET)
        masked = np.ma.masked_values(np.array([1.0, -9999.0, 3.0]), -9999.0)

        self.assertTrue(cache.add_cache_item(key, masked))

        cached = cache.get_cache_item(key)
        self.assertIsInstance(cached, np.ma.MaskedArray)
        np.testing.assert_array_equal(cached.mask, masked.mask)


class TestSizeAccounting(unittest.TestCase):
    """_size must equal the bytes actually held, including across replacements."""

    def test_size_follows_the_stored_arrays(self) -> None:
        cache = ComputationCache()
        first, second = _array(800), _array(1600)

        cache.add_cache_item(cache.get_cache_key(_DATASET, band_index=0), first)
        self.assertEqual(cache._size, first.nbytes)
        cache.add_cache_item(cache.get_cache_key(_DATASET, band_index=1), second)
        self.assertEqual(cache._size, first.nbytes + second.nbytes)

    def test_replacing_a_key_does_not_double_count(self) -> None:
        cache = ComputationCache()
        key = cache.get_cache_key(_DATASET, band_index=0)
        original, replacement = _array(800), _array(2400)

        cache.add_cache_item(key, original)
        self.assertTrue(cache.add_cache_item(key, replacement))

        self.assertEqual(len(cache._cache), 1)
        self.assertEqual(cache._size, replacement.nbytes)
        np.testing.assert_array_equal(cache.get_cache_item(key), replacement)

    def test_remove_cache_item_gives_back_its_bytes(self) -> None:
        cache = ComputationCache()
        key = cache.get_cache_key(_DATASET, band_index=0)

        cache.add_cache_item(key, _array(800))
        cache.remove_cache_item(key)

        self.assertEqual(cache._size, 0)
        self.assertIsNone(cache.get_cache_item(key))

    def test_clear_keys_from_partial_drops_only_that_dataset(self) -> None:
        cache = ComputationCache()
        keep = cache.get_cache_key(_OTHER_DATASET, band_index=0)
        drop_first = cache.get_cache_key(_DATASET, band_index=0)
        drop_second = cache.get_cache_key(_DATASET, band_index=1)
        kept_array = _array(800)

        cache.add_cache_item(keep, kept_array)
        cache.add_cache_item(drop_first, _array(800))
        cache.add_cache_item(drop_second, _array(1600))

        cache.clear_keys_from_partial(cache.get_partial_key(_DATASET))

        self.assertIsNone(cache.get_cache_item(drop_first))
        self.assertIsNone(cache.get_cache_item(drop_second))
        np.testing.assert_array_equal(cache.get_cache_item(keep), kept_array)
        self.assertEqual(cache._size, kept_array.nbytes)

    def test_clear_cache_empties_a_populated_cache(self) -> None:
        cache = ComputationCache()
        for band in range(3):
            cache.add_cache_item(cache.get_cache_key(_DATASET, band_index=band), _array(800))
        self.assertEqual(len(cache._cache), 3)

        cache.clear_cache()

        self.assertEqual(len(cache._cache), 0)
        self.assertEqual(cache._size, 0)


class TestCapacityAndEviction(unittest.TestCase):
    """Now that entries are stored, capacity is what keeps the caches off the heap."""

    def test_an_entry_larger_than_capacity_is_refused(self) -> None:
        cache = Cache(capacity=800)

        self.assertFalse(cache.add_cache_item(1, _array(1600)))
        self.assertEqual(len(cache._cache), 0)
        self.assertEqual(cache._size, 0)

    def test_eviction_keeps_the_cache_within_capacity(self) -> None:
        cache = ComputationCache(capacity=2400)

        for band in range(4):
            cache.add_cache_item(cache.get_cache_key(_DATASET, band_index=band), _array(800))
            self.assertLessEqual(cache._size, cache._capacity)

        self.assertEqual(cache._size, sum(value.nbytes for value in cache._cache.values()))

    def test_eviction_takes_the_oldest_entry_first(self) -> None:
        cache = ComputationCache(capacity=2400)
        keys = [cache.get_cache_key(_DATASET, band_index=band) for band in range(4)]

        for key in keys:
            cache.add_cache_item(key, _array(800))

        oldest, rest = keys[0], keys[1:]
        self.assertIsNone(cache.get_cache_item(oldest))
        for key in rest:
            self.assertIsNotNone(cache.get_cache_item(key))

    def test_a_replacement_does_not_evict_an_unrelated_entry(self) -> None:
        # Re-storing a key with a same-sized array must not push the running size
        # over capacity and cost the cache an entry it should have kept.
        cache = ComputationCache(capacity=1600)
        first = cache.get_cache_key(_DATASET, band_index=0)
        second = cache.get_cache_key(_DATASET, band_index=1)
        cache.add_cache_item(first, _array(800))
        cache.add_cache_item(second, _array(800))

        cache.add_cache_item(second, _array(800))

        self.assertIsNotNone(cache.get_cache_item(first))
        self.assertIsNotNone(cache.get_cache_item(second))
        self.assertEqual(cache._size, 1600)


class TestHistogramCacheUnaffected(unittest.TestCase):
    """HistogramCache already stored on a miss; it must keep doing so."""

    def test_bins_and_edges_round_trip(self) -> None:
        cache = HistogramCache()
        key = cache.get_cache_key(_DATASET, 0, "linear", "none", 0.0, 1.0)
        bins, edges = _array(800), _array(808)

        cache.add_cache_item(key, (bins, edges))

        cached_bins, cached_edges = cache.get_cache_item(key)
        np.testing.assert_array_equal(cached_bins, bins)
        np.testing.assert_array_equal(cached_edges, edges)
        self.assertEqual(cache._size, bins.nbytes + edges.nbytes)


class TestDataCacheTiers(unittest.TestCase):
    def test_each_tier_is_its_own_cache(self) -> None:
        data_cache = DataCache()

        self.assertIsInstance(data_cache.get_render_cache(), RenderCache)
        self.assertIsInstance(data_cache.get_computation_cache(), ComputationCache)
        self.assertIsInstance(data_cache.get_histogram_cache(), HistogramCache)
        self.assertIsNot(data_cache.get_render_cache(), data_cache.get_computation_cache())


if __name__ == "__main__":
    unittest.main()
