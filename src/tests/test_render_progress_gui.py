"""Strip-wise rendering under one cancellable progress dialog.

A stretch change re-renders the main view, context pane and zoom pane on the GUI
thread.  ``RasterView`` stretches and packs each band in row strips, reporting
each strip to one ``RenderProgress`` and checking for a cancel between strips.
These tests pin that the strips give the whole-band result exactly, that progress
runs monotonically to completion, that a cancel leaves every pane and the stretch
state as they were, and that small images never show the dialog.
"""

import time
from pathlib import Path

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

import wiser.gui.rasterview as rvmod
import wiser.gui.render_progress as rpmod
import wiser.gui.stretch_builder as sbmod
import wiser.raster.data_cache as dcmod
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
    StretchType,
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
    strips = rvmod.make_channel_image_strips(band, s1, s2, 5, lambda: None)
    assert strips.dtype == np.uint8
    assert np.array_equal(strips, whole)


@pytest.mark.parametrize("numba", [False, True])
def test_strip_callback_runs_once_per_strip(monkeypatch, numba):
    if numba:
        monkeypatch.setattr(rvmod, "ARRAY_NUMBA_THRESHOLD", 0)
    band = _band()
    calls = []
    channel = rvmod.make_channel_image_strips(band, None, None, 5, lambda: calls.append(1))
    assert len(calls) == 8  # ceil(37 / 5)

    calls.clear()
    rgb = rvmod.make_rgb_image_strips(channel, channel, channel, 8, lambda: calls.append(1))
    assert len(calls) == 5  # ceil(37 / 8)
    assert np.array_equal(rgb, rvmod.make_rgb_image(channel, channel, channel))


def test_strip_rows_is_sized_by_bytes():
    assert rvmod.strip_rows(4000) == rvmod.RENDER_STRIP_BYTES // (4 * 4000)
    assert rvmod.strip_rows(rvmod.RENDER_STRIP_BYTES) == 1


# --- application-level ------------------------------------------------------


@pytest.fixture
def model(monkeypatch):
    # Strips of 4 rows on the 40-pixel-wide cube, so a render has many steps.
    monkeypatch.setattr(rvmod, "RENDER_STRIP_BYTES", 4 * 40 * 4)
    # The dialog never shows on its own, so no test depends on how long a render
    # (or a numba compile) takes; test_slow_render_shows_dialog lowers this.
    monkeypatch.setattr(rpmod, "MINIMUM_DURATION_MS", 10**9)
    m = WiserTestModel()
    yield m
    m.close_app()


def _cube():
    return np.random.default_rng(3).random((3, 30, 40), dtype=np.float32) * 100.0


def _load(m, display):
    """
    Load a dataset shown as ``display``: "rgb" (a 3-band cube), "gray" (one band
    of the cube with a colormap) or "ignore" (a file with a data ignore value,
    so its bands are masked).
    """
    if display == "ignore":
        return m.load_dataset(_DATA_IGNORE_HDR)
    ds = m.load_dataset(_cube())
    if display == "gray":
        m.main_window._on_display_bands_change(ds.get_id(), (1,), "viridis", True)
        for rv in _panes(m).values():
            assert rv.get_display_bands() == (1,) and rv.get_colormap() == "viridis"
    return ds


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


def _cancel_at_step(monkeypatch, n):
    """
    Cancel each render progress at its ``n``-th step.  Returns, per progress,
    ``[progress, steps, dialog reported cancelled right after the cancel]``.
    """
    seen = []
    original = RenderProgress.step

    def step(self):
        if not seen or seen[-1][0] is not self:
            seen.append([self, 0, False])
        seen[-1][1] += 1
        if seen[-1][1] == n:
            self._dialog.cancel()
            seen[-1][2] = self._dialog.wasCanceled()
        original(self)

    monkeypatch.setattr(RenderProgress, "step", step)
    return seen


def _working_render_cache(monkeypatch):
    """
    Make the render cache store new entries.  ``Cache.add_cache_item`` only
    replaces existing keys (#772), so without this the render cache never holds
    anything and a cancel could not leave an entry behind.
    """

    def add_cache_item(self, key, value):
        if key in self._cache or value.nbytes > self._capacity:
            return False
        self._cache[key] = value
        self._size += value.nbytes
        return True

    monkeypatch.setattr(dcmod.Cache, "add_cache_item", add_cache_item)


@pytest.mark.parametrize("display", ["rgb", "gray", "ignore"])
@pytest.mark.parametrize(
    "click",
    [
        "click_stretch_linear_2_5",
        "click_stretch_hist_equalize",
        "click_log_conditioner",
        "click_sqrt_conditioner",
    ],
)
def test_strip_render_matches_whole_band_render(model, display, click):
    _load(model, display)
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
        name: (
            rv.get_unscaled_pixmap(),
            rv.get_image_data(),
            rv.get_image_data().copy(),
            rv.get_stretches(),
            rv._joint_render_cache,
        )
        for name, rv in _panes(m).items()
    }


def _render_cache_state(ds):
    cache = ds.get_cache().get_render_cache()
    return dict(cache._cache), cache._size


@pytest.mark.parametrize("when", ["during_first_pane", "after_first_pane"])
@pytest.mark.parametrize(
    "display, click",
    [
        ("rgb", "click_stretch_hist_equalize"),
        ("rgb", "click_stretch_decorrelation"),
        ("gray", "click_stretch_hist_equalize"),
        ("ignore", "click_stretch_hist_equalize"),
        ("ignore", "click_stretch_decorrelation"),
    ],
)
def test_cancel_keeps_every_pane_and_the_stretch_state(model, monkeypatch, when, display, click):
    _working_render_cache(monkeypatch)
    ds = _load(model, display)
    model.click_stretch_linear_2_5()
    if click == "click_stretch_decorrelation":
        # A joint render first, so each pane holds a joint result the cancel must keep.
        model.get_stretch_config()._ui.rb_stretch_decorrelation.click()
        model.click_sqrt_conditioner()
        assert any(rv._joint_render_cache is not None for rv in _panes(model).values())
    bands = model.get_main_view_rv((0, 0)).get_display_bands()
    stretches_before = model.app_state.get_stretches(ds.get_id(), bands)
    before = _pane_state(model)
    cache_before = _render_cache_state(ds)

    original = RenderProgress.step
    seen = {"steps": 0, "renders": 0, "progress": None}

    def step(self):
        seen["steps"] += 1
        seen["progress"] = self
        if when == "during_first_pane" and seen["steps"] == 3:
            self._dialog.cancel()
        if when == "after_first_pane" and seen["renders"] == 1:
            self._dialog.cancel()
        original(self)

    original_render = rvmod.RasterView._render_display_image

    def render(self, *args):
        result = original_render(self, *args)
        seen["renders"] += 1
        return result

    monkeypatch.setattr(RenderProgress, "step", step)
    monkeypatch.setattr(rvmod.RasterView, "_render_display_image", render)

    if click == "click_stretch_decorrelation":
        model.click_log_conditioner()
    else:
        getattr(model, click)()

    assert seen["progress"].was_cancelled()
    # During the first pane no render completes; after it, exactly one has and
    # its pane must have been rolled back.
    assert seen["renders"] == (0 if when == "during_first_pane" else 1)

    after = _pane_state(model)
    for name, (pixmap, image, pixels, stretches, joint) in before.items():
        a_pixmap, a_image, a_pixels, a_stretches, a_joint = after[name]
        assert a_pixmap is pixmap, name
        assert a_image is image, name
        assert np.array_equal(a_pixels, pixels), name
        assert a_stretches is stretches, name
        assert a_joint is joint, name
    assert model.app_state.get_stretches(ds.get_id(), bands) == stretches_before
    assert _render_cache_state(ds) == cache_before

    # The same change applied again renders in full.
    monkeypatch.setattr(RenderProgress, "step", original)
    if click == "click_stretch_decorrelation":
        model.click_log_conditioner()
    else:
        getattr(model, click)()
    for name, rv in _panes(model).items():
        assert rv.get_unscaled_pixmap() is not before[name][0], name
        if click != "click_stretch_decorrelation":
            assert np.array_equal(rv.get_image_data(), _whole_band_image(rv)), name


def test_cancel_of_first_stretch_removes_it(model, monkeypatch):
    ds = model.load_dataset(_cube())
    bands = model.get_main_view_rv((0, 0)).get_display_bands()
    assert model.app_state.get_all_stretches() == {}
    before = _pane_state(model)

    _cancel_at_step(monkeypatch, 3)
    model.click_stretch_hist_equalize()

    assert model.app_state.get_all_stretches() == {}
    assert model.app_state.get_stretches(ds.get_id(), bands) == [None, None, None]
    for name, rv in _panes(model).items():
        assert rv.get_unscaled_pixmap() is before[name][0], name
        assert rv.get_stretches() is before[name][3], name


def test_dataset_open_cannot_be_cancelled(model, monkeypatch):
    seen = _cancel_at_step(monkeypatch, 2)
    ds = model.load_dataset(_cube())

    assert len(seen) == 1
    progress, _, cancel_requested = seen[0]
    assert cancel_requested
    assert not progress.was_cancelled()
    for name, rv in _panes(model).items():
        assert rv.get_raster_data() is ds, name
        assert np.array_equal(rv.get_image_data(), _whole_band_image(rv)), name


def test_escape_does_not_close_a_non_cancellable_dialog(model):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    progress = RenderProgress("Opening", cancellable=False)
    progress._dialog.show()
    QTest.keyClick(progress._dialog, Qt.Key_Escape)

    assert progress._dialog.isVisible()
    assert not progress.was_cancelled()
    progress._finish()


def _record_shown(monkeypatch):
    shown = []
    original = RenderProgress._finish

    def finish(self):
        shown.append(self._dialog.isVisible())
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
    monkeypatch.setattr(rpmod, "MINIMUM_DURATION_MS", 50)
    shown = _record_shown(monkeypatch)
    original = RenderProgress.step

    def slow_step(self):
        time.sleep(0.03)
        original(self)

    # Over 30 steps of 30 ms each: far past the 50 ms minimum duration.
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


def test_cancel_reloads_stretch_builder_without_emitting(model, monkeypatch):
    ds = model.load_dataset(_cube())
    model.click_stretch_linear_2_5()
    builder = model.get_stretch_builder()
    bands = model.get_main_view_rv((0, 0)).get_display_bands()
    restored = model.app_state.get_stretches(ds.get_id(), bands)

    emitted = []
    builder.stretch_changed.connect(lambda *args: emitted.append(args))
    _cancel_at_step(monkeypatch, 1)
    model.click_stretch_hist_equalize()

    # Only the equalize request itself; the reload after the cancel emits nothing.
    assert len(emitted) == 1
    assert model.app_state.get_stretches(ds.get_id(), bands) == restored
    assert builder._stretch_config.get_stretch_type() == StretchType.LINEAR_STRETCH
    for built, expected in zip(builder.get_stretches(), restored):
        assert isinstance(built, StretchLinear)
        assert np.isclose(built.lower(), expected.lower(), atol=1e-6)
        assert np.isclose(built.upper(), expected.upper(), atol=1e-6)


def test_cancel_for_other_dataset_or_bands_leaves_builder_alone(model):
    ds = model.load_dataset(_cube())
    model.click_stretch_linear_2_5()
    builder = model.get_stretch_builder()
    bands = model.get_main_view_rv((0, 0)).get_display_bands()
    # The application state no longer holds the builder's stretches, so any
    # reload would move the builder off its linear stretch.
    for band in bands:
        model.app_state._stretches.pop((ds.get_id(), band))

    model.app_state.stretch_change_cancelled.emit(ds.get_id() + 1000, bands)
    model.app_state.stretch_change_cancelled.emit(ds.get_id(), tuple(b + 1 for b in bands))
    assert builder._stretch_config.get_stretch_type() == StretchType.LINEAR_STRETCH

    model.app_state.stretch_change_cancelled.emit(ds.get_id(), bands)
    assert builder._stretch_config.get_stretch_type() == StretchType.NO_STRETCH
