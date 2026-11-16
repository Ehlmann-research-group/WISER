"""Histogram-equalize stretch output matches np.interp through the CDF exactly.

The equalize stretch maps each normalized value through the histogram's CDF by
linear interpolation between left bin edges.  For uniform edges it locates the bin
by index arithmetic instead of a search; the result must be bit-identical to the
np.interp mapping, including values on an edge, outside [0, 1], NaN, and the wild
values that masked (data-ignore) pixels carry after normalization.
"""

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

from wiser.gui.rasterview import make_channel_image_numba, make_channel_image_python
from wiser.gui.stretch_builder import create_histogram_numba, create_histogram_python
from wiser.raster.stretch import StretchHistEqualize, StretchHistEqualizeUsingNumba
from wiser.utils.numba_wrapper import numba_njit_wrapper

pytestmark = [pytest.mark.smoke]


def _interp_python(a, edges, cdf):
    np.copyto(a, np.interp(a, edges[:-1], cdf))


@numba_njit_wrapper(non_njit_func=_interp_python)
def _interp_numba(a, edges, cdf):
    out = np.interp(a, edges[:-1], cdf)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            a[i, j] = out[i, j]


def _band(edges):
    rng = np.random.default_rng(11)
    a = rng.random((200, 600)).astype(np.float32)
    a[0, :8] = [0.0, 1.0, np.nan, -0.25, 1.5, np.inf, -np.inf, 0.5]
    # Every edge value exactly, and the float32 values next to each one.
    e = np.asarray(edges, dtype=np.float32)
    a[1, : e.size] = e
    a[2, : e.size] = np.nextafter(e, np.float32(-1.0))
    a[3, : e.size] = np.nextafter(e, np.float32(2.0))
    # A data-ignore value of -9999 normalized against a [10, 500] band.
    a[4:20, 10:60] = (-9999.0 - 10.0) / 490.0
    return a


def _histogram(make_histogram, conditioner):
    data = np.random.default_rng(5).random(50_000)
    if conditioner == "uniform_jittered":
        # Uniform edges within the tolerance, each moved off the exact grid so
        # the index arithmetic lands one bin off for values between the grid
        # point and the edge, and its one-bin correction is exercised.
        edges = np.linspace(0.05, 0.95, 301)
        step = edges[1] - edges[0]
        edges[1:-2] += np.random.default_rng(7).choice([-5e-4, 5e-4], size=298) * step
        return np.histogram(data, bins=edges)
    bins, edges = make_histogram(data, 0.0, 1.0)
    if conditioner == "sqrt":
        edges = np.sqrt(edges)
    elif conditioner == "log":
        edges = np.log2(1 + edges)
    return bins, edges


@pytest.mark.parametrize("conditioner", ["none", "uniform_jittered", "sqrt", "log"])
@pytest.mark.parametrize(
    "stretch_cls, make_histogram, reference, make_channel",
    [
        (StretchHistEqualize, create_histogram_python, _interp_python, make_channel_image_python),
        (StretchHistEqualizeUsingNumba, create_histogram_numba, _interp_numba, make_channel_image_numba),
    ],
)
def test_equalize_matches_interp(stretch_cls, make_histogram, reference, make_channel, conditioner):
    bins, edges = _histogram(make_histogram, conditioner)
    stretch = stretch_cls(bins, edges)
    # Uniform edges take the index-arithmetic path; conditioned edges are not uniform.
    assert (stretch._step > 0.0) == (conditioner in ("none", "uniform_jittered"))

    band = _band(edges)
    expected = band.copy()
    reference(expected, np.asarray(stretch._histo_edges), np.asarray(stretch._cdf))
    actual = band.copy()
    stretch.apply(actual)
    assert np.array_equal(actual, expected, equal_nan=True)

    assert np.array_equal(make_channel(band, stretch, None), make_channel(expected, None, None))
