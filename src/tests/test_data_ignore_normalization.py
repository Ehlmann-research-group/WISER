"""Regression tests: the data ignore value must never enter band statistics.

A dataset's "data ignore value" is a sentinel (e.g. -9999), not real data.  Band
statistics (min/max) are what the whole app uses to normalize a band for display.
If the sentinel leaks into those stats, ``band_min`` collapses to the sentinel and
every real pixel normalizes to ~1.0 -- the image renders as a uniform white field
with the ignore pixels black, in every view (main, context, zoom, georeferencer),
because they all share the dataset's cached stats.

The leak is triggered whenever the stats are first computed from an *unfiltered*
array -- e.g. a feature that reads ``get_band_data(filter_data_ignore_value=False)``
before the band is displayed.  These tests pin the invariant at the dataset level so
it holds regardless of which caller warms the stats cache first.
"""
from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

from wiser.raster.loader import RasterDataLoader

pytestmark = [pytest.mark.functional]

_DATASETS = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets"
_DATA_IGNORE_HDR = (_DATASETS / "caltech_425_6_6_data_ignore.hdr").resolve()

# A band with real (non-sentinel) data in this dataset.
_BAND = 64


def _load():
    ds = RasterDataLoader().load_from_file(str(_DATA_IGNORE_HDR))[0]
    assert ds.get_data_ignore_value() == -9999.0
    return ds


def _real_min_max(ds, band):
    """The true min/max of the band with the sentinel excluded, straight from impl."""
    raw = np.asarray(ds._impl.get_band_data(band), dtype=np.float64)
    valid = raw[raw != ds.get_data_ignore_value()]
    return valid.min(), valid.max()


def test_band_stats_exclude_ignore_value_after_raw_read():
    """A prior unfiltered read must not poison the cached band statistics."""
    ds = _load()
    real_min, real_max = _real_min_max(ds, _BAND)

    # A feature reads the raw band (sentinel still present) -- this used to seed
    # the shared band-stats cache with band_min == the sentinel.
    ds.get_band_data(_BAND, filter_data_ignore_value=False)

    stats = ds.get_band_stats(_BAND)
    assert stats.get_min() != ds.get_data_ignore_value()
    assert np.isclose(stats.get_min(), real_min)
    assert np.isclose(stats.get_max(), real_max)


def test_normalized_display_not_washed_out_after_raw_read():
    """The display-normalized band must span a real range, not collapse to uniform."""
    ds = _load()

    # Poison attempt: unfiltered read before the display path runs.
    ds.get_band_data(_BAND, filter_data_ignore_value=False)

    arr = ds.get_band_data_normalized(_BAND)
    assert isinstance(arr, np.ma.masked_array)
    unmasked = arr.compressed()

    # Real pixels must not all collapse to a single value (the wash-out symptom).
    assert not np.isclose(
        unmasked.min(), unmasked.max()
    ), "Normalized band is uniform -- the data ignore value poisoned the stretch."
    # Proper min/max normalization puts valid data across the full [0, 1] range.
    assert np.isclose(unmasked.min(), 0.0)
    assert np.isclose(unmasked.max(), 1.0)


def test_normalized_display_matches_regardless_of_prior_reads():
    """Display normalization is identical whether or not an unfiltered read happened first."""
    clean = _load().get_band_data_normalized(_BAND).compressed()

    poisoned_ds = _load()
    poisoned_ds.get_band_data(_BAND, filter_data_ignore_value=False)
    after_raw = poisoned_ds.get_band_data_normalized(_BAND).compressed()

    np.testing.assert_allclose(np.sort(after_raw), np.sort(clean), rtol=1e-6, atol=1e-6)
