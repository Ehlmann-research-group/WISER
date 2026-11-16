"""The numba stretch classes the stretch builder picks for large bands must render.

For bands larger than ``ARRAY_NUMBA_THRESHOLD`` the stretch builder builds the numba
jitclass stretches.  ``RasterView`` asks every stretch part ``requires_all_bands()``
to decide between the per-channel and the joint render path, so a jitclass that
cannot answer leaves every pane unrendered.  The threshold is forced to zero here so
the numba classes are exercised on a small cube.
"""

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

import wiser.gui.rasterview as rvmod
import wiser.gui.stretch_builder as sbmod
import wiser.raster.utils as rutils
from wiser.raster.stretch import (
    StretchBase,
    StretchBaseUsingNumba,
    StretchHistEqualize,
    StretchHistEqualizeUsingNumba,
    StretchLinear,
    StretchLinearUsingNumba,
    StretchLog2,
    StretchLog2UsingNumba,
    StretchSquareRoot,
    StretchSquareRootUsingNumba,
)
from test_utils.test_model import WiserTestModel

pytestmark = [pytest.mark.smoke]


@pytest.fixture
def model(monkeypatch):
    for mod in (sbmod, rvmod, rutils):
        monkeypatch.setattr(mod, "ARRAY_NUMBA_THRESHOLD", 0)
    m = WiserTestModel()
    yield m
    m.close_app()


def _cube():
    rng = np.random.default_rng(3)
    return rng.random((3, 30, 40), dtype=np.float32) * 100.0


def _pane_rasterviews(m):
    return {
        "main": m.get_main_view_rv((0, 0)),
        "context": m.get_context_pane_rasterview(),
        "zoom": m.get_zoom_pane_rasterview(),
    }


def _expected_image(rv):
    ds = rv.get_raster_data()
    channels = []
    for band, stretch in zip(rv.get_display_bands(), rv.get_stretches()):
        arr = ds.get_band_data_normalized(band)
        data = arr.data if isinstance(arr, np.ma.masked_array) else arr
        s1, s2 = stretch.get_stretches()
        channels.append(rvmod.make_channel_image(data, s1, s2))
    return rvmod.make_rgb_image(*channels)


@pytest.mark.parametrize(
    "click, numba_type",
    [
        ("click_stretch_linear_2_5", StretchLinearUsingNumba),
        ("click_stretch_hist_equalize", StretchHistEqualizeUsingNumba),
    ],
)
def test_numba_stretch_redraws_all_panes(model, click, numba_type):
    model.load_dataset(_cube())
    before = {name: rv.get_image_data().copy() for name, rv in _pane_rasterviews(model).items()}

    getattr(model, click)()

    for name, rv in _pane_rasterviews(model).items():
        stretch = rv.get_stretches()[0]
        assert isinstance(stretch.get_stretches()[0], numba_type), name
        image = rv.get_image_data()
        assert not np.array_equal(image, before[name]), f"{name} pane did not redraw"
        assert np.array_equal(image, _expected_image(rv)), name


@pytest.mark.parametrize(
    "numba_stretch, plain_stretch",
    [
        (StretchBaseUsingNumba(), StretchBase()),
        (StretchLinearUsingNumba(0.2, 0.8), StretchLinear(0.2, 0.8)),
        (
            StretchHistEqualizeUsingNumba(np.ones(4, dtype=np.int64), np.linspace(0.0, 1.0, 5)),
            StretchHistEqualize(np.ones(4, dtype=np.int64), np.linspace(0.0, 1.0, 5)),
        ),
        (StretchSquareRootUsingNumba(), StretchSquareRoot()),
        (StretchLog2UsingNumba(), StretchLog2()),
    ],
)
def test_numba_stretch_requires_all_bands_matches_plain(numba_stretch, plain_stretch):
    assert numba_stretch.requires_all_bands() is plain_stretch.requires_all_bands()
