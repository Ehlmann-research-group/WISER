"""Band min/max statistics are computed once per (band, data ignore value).

The display normalization of a band needs its min and max.  Scanning the full band
for them on every render is a large share of render time on big images, so the
dataset caches them; the cache must be keyed so a lookup finds what was stored, and
so a different data ignore value (which changes which pixels count) recomputes.
"""

from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

import wiser.raster.dataset as dsmod
from wiser.raster.loader import RasterDataLoader

pytestmark = [pytest.mark.functional]

_DATASETS = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets"
_DATA_IGNORE_HDR = (_DATASETS / "caltech_425_6_6_data_ignore.hdr").resolve()
_BAND = 64


class _CountingNumpy:
    """Stands in for ``numpy`` in the dataset module, counting band min scans."""

    def __init__(self):
        self.nanmin_calls = 0

    def nanmin(self, *args, **kwargs):
        self.nanmin_calls += 1
        return np.nanmin(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.fixture
def counting_np(monkeypatch):
    counter = _CountingNumpy()
    monkeypatch.setattr(dsmod, "np", counter)
    return counter


def _load():
    # No data cache, so every call reaches the band-stats cache.
    ds = RasterDataLoader().load_from_file(str(_DATA_IGNORE_HDR))[0]
    assert ds.get_cache() is None
    assert ds.get_data_ignore_value() == -9999.0
    return ds


def test_normalized_band_computes_min_max_once(counting_np):
    ds = _load()
    first = ds.get_band_data_normalized(_BAND)
    second = ds.get_band_data_normalized(_BAND)

    assert counting_np.nanmin_calls == 1
    np.testing.assert_array_equal(first, second)


def test_band_stats_and_normalized_band_share_the_cache(counting_np):
    ds = _load()
    stats = ds.get_band_stats(_BAND)
    calls = counting_np.nanmin_calls

    ds.get_band_data_normalized(_BAND)
    assert counting_np.nanmin_calls == calls
    again = ds.get_band_stats(_BAND)
    assert (again.get_min(), again.get_max()) == (stats.get_min(), stats.get_max())


def test_changing_ignore_value_recomputes_stats(counting_np):
    ds = _load()
    ds.get_band_data_normalized(_BAND)
    stats_before = ds.get_band_stats(_BAND)
    assert counting_np.nanmin_calls == 1

    raw = np.asarray(ds._impl.get_band_data(_BAND), dtype=np.float64)
    valid = raw[raw != -9999.0]
    new_ignore = float(valid.min())
    ds.set_data_ignore_value(new_ignore)

    normalized = ds.get_band_data_normalized(_BAND)
    stats_after = ds.get_band_stats(_BAND)
    assert counting_np.nanmin_calls == 2

    remaining = raw[raw != new_ignore]
    assert stats_after.get_min() != stats_before.get_min()
    assert np.isclose(stats_after.get_min(), remaining.min())
    assert np.isclose(stats_after.get_max(), remaining.max())
    assert np.isclose(normalized.compressed().min(), 0.0)
