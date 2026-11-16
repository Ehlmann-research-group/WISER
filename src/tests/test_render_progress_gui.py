"""Strip-wise rendering under one cancellable progress dialog.

A stretch change re-renders the main view, context pane and zoom pane on the GUI
thread.  ``RasterView`` stretches and packs each band in row strips, reporting
each strip to one ``RenderProgress`` and checking for a cancel between strips.
These tests pin that the strips give the whole-band result exactly, that progress
runs monotonically to completion, that a cancel leaves every pane and the stretch
state as they were, and that small images never show the dialog.
"""

from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

import wiser.gui.rasterview as rvmod
import wiser.gui.stretch_builder as sbmod
import wiser.raster.utils as rutils
from wiser.gui.render_progress import RenderProgress
from wiser.raster.stretch import (
    StretchComposite,
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

_DATASETS = Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets"
_DATA_IGNORE_HDR = str((_DATASETS / "caltech_425_6_6_data_ignore.hdr").resolve())


def _band(shape=(37, 50)):
    a = np.random.default_rng(1).random(shape).astype(np.float32)
    a[0, :4] = [0.0, 1.0, -0.5, 1.5]
    return a


def _stretch_pair(kind, numba):
    data = _band().ravel()
    bins, edges = np.histogram(data, bins=512, range=(0.0, 1.0))
    linear = StretchLinearUsingNumba(0.2, 0.7) if numba else StretchLinear(0.2, 0.7)
    equalize = StretchHistEqualizeUsingNumba(bins, edges) if numba else StretchHistEqualize(bins, edges)
    sqrt = StretchSquareRootUsingNumba() if numba else StretchSquareRoot()
    log2 = StretchLog2UsingNumba() if numba else StretchLog2()
    return {
        "linear": (linear, None),
        "equalize": (equalize, None),
        "sqrt": StretchComposite(sqrt, linear).get_stretches(),
        "log2": StretchComposite(log2, equalize).get_stretches(),
        "none": (None, None),
    }[kind]


@pytest.mark.parametrize("numba", [False, True])
@pytest.mark.parametrize("kind", ["linear", "equalize", "sqrt", "log2", "none"])
def test_channel_strips_match_whole_band(monkeypatch, kind, numba):
    if numba:
        monkeypatch.setattr(rvmod, "ARRAY_NUMBA_THRESHOLD", 0)
    band = _band()
    s1, s2 = _stretch_pair(kind, numba)
    whole = rvmod.make_channel_image(band, s1, s2)
    strips = rvmod.make_channel_image_strips(band, s1, s2, 5, lambda done, total: None)
    assert strips.dtype == np.uint8
    assert np.array_equal(strips, whole)


@pytest.mark.parametrize("numba", [False, True])
def test_strip_progress_is_monotonic_to_completion(monkeypatch, numba):
    if numba:
        monkeypatch.setattr(rvmod, "ARRAY_NUMBA_THRESHOLD", 0)
    band = _band()
    calls = []
    channel = rvmod.make_channel_image_strips(band, None, None, 5, lambda d, t: calls.append((d, t)))
    assert [d for d, _ in calls] == [5, 10, 15, 20, 25, 30, 35, 37]
    assert {t for _, t in calls} == {37}

    calls.clear()
    rgb = rvmod.make_rgb_image_strips(channel, channel, channel, 8, lambda d, t: calls.append((d, t)))
    assert [d for d, _ in calls] == [8, 16, 24, 32, 37]
    assert np.array_equal(rgb, rvmod.make_rgb_image(channel, channel, channel))


def test_strip_rows_is_sized_by_bytes():
    assert rvmod.strip_rows(4000) == rvmod.RENDER_STRIP_BYTES // (4 * 4000)
    assert rvmod.strip_rows(rvmod.RENDER_STRIP_BYTES) == 1


# --- application-level ------------------------------------------------------


@pytest.fixture
def model(monkeypatch):
    # Strips of 4 rows on the 40-pixel-wide cube, so a render has many steps.
    monkeypatch.setattr(rvmod, "RENDER_STRIP_BYTES", 4 * 40 * 4)
    m = WiserTestModel()
    yield m
    m.close_app()


def _cube():
    return np.random.default_rng(3).random((3, 30, 40), dtype=np.float32) * 100.0


def _panes(m):
    return {
        "main": m.get_main_view_rv((0, 0)),
        "context": m.get_context_pane_rasterview(),
        "zoom": m.get_zoom_pane_rasterview(),
    }


def _whole_band_image(rv):
    """The image as the whole-band renderer built it: stretch, zero masked pixels, pack."""
    ds = rv.get_raster_data()
    channels = []
    for band, stretch in zip(rv.get_display_bands(), rv.get_stretches()):
        arr = ds.get_band_data_normalized(band)
        data = arr.data if isinstance(arr, np.ma.masked_array) else arr
        s1, s2 = stretch.get_stretches() if stretch else (None, None)
        channel = rvmod.make_channel_image(data, s1, s2)
        if isinstance(arr, np.ma.masked_array):
            channel[arr.mask] = 0
        channels.append(channel)
    if len(channels) == 1:
        return rvmod.make_grayscale_image(channels[0], rv.get_colormap())
    return rvmod.make_rgb_image(*channels)


def _record_steps(monkeypatch):
    """Record (done, total) after every step of every render progress."""
    records = []
    original = RenderProgress.step

    def step(self):
        records.append((self._done + 1, self._total))
        original(self)

    monkeypatch.setattr(RenderProgress, "step", step)
    return records


@pytest.mark.parametrize(
    "dataset, click",
    [
        ("cube", "click_stretch_linear_2_5"),
        ("cube", "click_stretch_hist_equalize"),
        ("cube", "click_log_conditioner"),
        ("ignore", "click_stretch_linear_2_5"),
        ("ignore", "click_stretch_hist_equalize"),
        ("ignore", "click_sqrt_conditioner"),
    ],
)
def test_strip_render_matches_whole_band_render(model, dataset, click):
    model.load_dataset(_cube() if dataset == "cube" else _DATA_IGNORE_HDR)
    getattr(model, click)()
    for name, rv in _panes(model).items():
        assert np.array_equal(rv.get_image_data(), _whole_band_image(rv)), name


@pytest.mark.parametrize("click", ["click_stretch_linear_2_5", "click_stretch_decorrelation", "load"])
def test_progress_runs_monotonically_to_completion(model, monkeypatch, click):
    if click != "load":
        model.load_dataset(_cube())
        model.get_stretch_builder()
    records = _record_steps(monkeypatch)
    if click == "load":
        model.load_dataset(_cube())
    elif click == "click_stretch_decorrelation":
        model.get_stretch_config()._ui.rb_stretch_decorrelation.click()
    else:
        getattr(model, click)()

    done = [d for d, _ in records]
    assert len(done) > 8
    assert done == list(range(1, len(done) + 1))
    assert records[-1][0] == records[-1][1]


def _pane_state(m):
    return {
        name: (rv.get_unscaled_pixmap(), rv.get_image_data(), rv.get_image_data().copy(), rv.get_stretches())
        for name, rv in _panes(m).items()
    }


@pytest.mark.parametrize("when", ["during_first_pane", "after_first_pane"])
@pytest.mark.parametrize("click", ["click_stretch_hist_equalize", "click_stretch_decorrelation"])
def test_cancel_keeps_every_pane_and_the_stretch_state(model, monkeypatch, when, click):
    ds = model.load_dataset(_cube())
    model.click_stretch_linear_2_5()
    bands = model.get_main_view_rv((0, 0)).get_display_bands()
    stretches_before = model.app_state.get_stretches(ds.get_id(), bands)
    before = _pane_state(model)

    original = RenderProgress.step
    seen = {"steps": 0, "renders": 0, "progress": None}

    def step(self):
        seen["steps"] += 1
        seen["progress"] = self
        if when == "during_first_pane" and seen["steps"] == 3:
            self.cancel()
        if when == "after_first_pane" and seen["renders"] == 1:
            self.cancel()
        original(self)

    original_render = rvmod.RasterView._render_display_image

    def render(self, *args):
        result = original_render(self, *args)
        seen["renders"] += 1
        return result

    monkeypatch.setattr(RenderProgress, "step", step)
    monkeypatch.setattr(rvmod.RasterView, "_render_display_image", render)

    if click == "click_stretch_decorrelation":
        model.get_stretch_config()._ui.rb_stretch_decorrelation.click()
    else:
        getattr(model, click)()

    assert seen["progress"].was_cancelled()
    # During the first pane no render completes; after it, exactly one has and
    # its pane must have been rolled back.
    assert seen["renders"] == (0 if when == "during_first_pane" else 1)

    after = _pane_state(model)
    for name, (pixmap, image, pixels, stretches) in before.items():
        a_pixmap, a_image, a_pixels, a_stretches = after[name]
        assert a_pixmap is pixmap, name
        assert a_image is image, name
        assert np.array_equal(a_pixels, pixels), name
        assert a_stretches is stretches, name
    assert model.app_state.get_stretches(ds.get_id(), bands) == stretches_before


def test_cancelled_stretch_change_can_be_applied_again(model, monkeypatch):
    model.load_dataset(_cube())
    model.click_stretch_linear_2_5()
    before = _pane_state(model)

    original = RenderProgress.step

    def cancel_step(self):
        self.cancel()
        original(self)

    monkeypatch.setattr(RenderProgress, "step", cancel_step)
    model.click_stretch_hist_equalize()
    assert all(rv.get_unscaled_pixmap() is before[n][0] for n, rv in _panes(model).items())

    monkeypatch.setattr(RenderProgress, "step", original)
    model.click_stretch_linear_5_0()
    for name, rv in _panes(model).items():
        assert rv.get_unscaled_pixmap() is not before[name][0], name
        assert np.array_equal(rv.get_image_data(), _whole_band_image(rv)), name


def _record_shown(monkeypatch):
    shown = []
    original = RenderProgress._finish

    def finish(self):
        shown.append(self.was_shown())
        original(self)

    monkeypatch.setattr(RenderProgress, "_finish", finish)
    return shown


def test_small_image_shows_no_dialog(model, monkeypatch):
    shown = _record_shown(monkeypatch)
    model.load_dataset(_cube())
    model.click_stretch_linear_2_5()
    model.click_stretch_hist_equalize()
    assert len(shown) >= 3
    assert not any(shown)


def test_slow_render_shows_dialog(model, monkeypatch):
    model.load_dataset(_cube())
    model.get_stretch_builder()
    shown = _record_shown(monkeypatch)
    original = RenderProgress.step

    def slow_step(self):
        import time

        time.sleep(0.03)
        original(self)

    monkeypatch.setattr(RenderProgress, "step", slow_step)
    model.click_stretch_linear_2_5()
    assert shown == [True]


def test_numba_path_renders_in_strips(model, monkeypatch):
    for mod in (sbmod, rvmod, rutils):
        monkeypatch.setattr(mod, "ARRAY_NUMBA_THRESHOLD", 0)
    model.load_dataset(_DATA_IGNORE_HDR)
    model.click_stretch_hist_equalize()
    for name, rv in _panes(model).items():
        assert isinstance(rv.get_stretches()[0].get_stretches()[0], StretchHistEqualizeUsingNumba)
        assert np.array_equal(rv.get_image_data(), _whole_band_image(rv)), name
