"""Tests for per-tile zoom in the main view grid and linked zoom of pixel-linked tiles.

The main view's zoom controls act on its active tile:  the tile with data last pressed
with the mouse or zoomed with Ctrl+wheel.  With linked views, the other tiles follow;
pixel-linked tiles share the active tile's scale.
"""

import numpy as np
import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

from PySide6.QtCore import QPoint

from test_utils.test_model import WiserTestModel
from wiser.gui.rasterview import GeographicLinkState
from wiser.raster.selection import SinglePixelSelection


def _array(rows=400, cols=400):
    return np.repeat(np.tile(np.linspace(0, 1, rows).reshape(rows, 1), (1, cols))[np.newaxis], 3, axis=0)


@pytest.fixture
def model():
    m = WiserTestModel()
    m.main_window.resize(640, 480)
    m.run()
    yield m
    m.close_app()


def _grid(model, layout, datasets):
    """Show ``datasets`` row by row in a main view of ``layout``; None leaves a tile empty."""
    model.set_main_view_layout(layout)
    positions = [(row, col) for row in range(layout[0]) for col in range(layout[1])]
    for pos, ds in zip(positions, datasets):
        if ds is not None:
            model.set_main_view_rv(pos, ds.get_id())


def _active_pos(model):
    return model.main_view._get_rasterview_position(model.main_view.get_active_rasterview())


def _zoom_text(model):
    return model.main_view._cbox_zoom.lineEdit().text()


def test_unlinked_tiles_zoom_independently(model):
    a, b = model.load_dataset(_array()), model.load_dataset(_array())
    _grid(model, (1, 2), [a, b])

    model.activate_main_view_rv((0, 1))
    model.click_main_view_zoom_in()
    model.click_main_view_zoom_in()

    assert model.get_main_view_rv_scale((0, 0)) == 1.0
    assert model.get_main_view_rv_scale((0, 1)) == pytest.approx(1.5625)
    assert _zoom_text(model) == "156%"

    model.activate_main_view_rv((0, 0))
    assert _zoom_text(model) == "100%"
    model.select_main_view_zoom_option(4)

    assert model.get_main_view_rv_scale((0, 0)) == 4
    assert model.get_main_view_rv_scale((0, 1)) == pytest.approx(1.5625)


def _zoom_in(model):
    model.click_main_view_zoom_in()


def _zoom_out(model):
    model.click_main_view_zoom_out()


def _option(model):
    model.select_main_view_zoom_option(2)


def _typed(model):
    model.type_main_view_zoom("300")


def _actual(model):
    model.main_view._act_zoom_to_actual.trigger()
    model.run()


def _fit(model):
    model.click_zoom_to_fit()


@pytest.mark.parametrize("zoom", [_zoom_in, _zoom_out, _option, _typed, _actual, _fit])
def test_pixel_linked_tiles_share_the_active_tiles_scale(model, zoom):
    a, b = model.load_dataset(_array()), model.load_dataset(_array())
    _grid(model, (1, 2), [a, b])
    model.activate_main_view_rv((0, 1))
    model.select_main_view_zoom_option(4)
    assert model.click_link_button()
    assert model.main_view._link_view_state == GeographicLinkState.PIXEL
    assert model.get_main_view_rv_scale((0, 0)) == 4

    zoom(model)

    leader, follower = model.get_main_view_rv((0, 1)), model.get_main_view_rv((0, 0))
    assert leader.get_scale() != 4
    assert follower.get_scale() == leader.get_scale()
    assert follower.get_scrollbar_state() == leader.get_scrollbar_state()


def test_linked_zoom_skips_an_empty_tile(model):
    a, b, c = (model.load_dataset(_array()) for _ in range(3))
    _grid(model, (2, 2), [a, b, None, c])
    assert model.click_link_button()

    model.click_main_view_zoom_in()

    scales = {pos: model.get_main_view_rv_scale(pos) for pos in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    assert scales == {(0, 0): 1.25, (0, 1): 1.25, (1, 0): 1.0, (1, 1): 1.25}


def test_press_on_an_empty_tile_keeps_the_active_tile(model):
    a = model.load_dataset(_array())
    _grid(model, (1, 2), [a, None])

    model.activate_main_view_rv((0, 1))
    model.click_main_view_zoom_in()

    assert _active_pos(model) == (0, 0)
    assert model.get_main_view_rv_scale((0, 0)) == 1.25


def test_an_empty_tile_cannot_be_made_active(model):
    a = model.load_dataset(_array())
    _grid(model, (1, 2), [a, None])

    with pytest.raises(ValueError, match="no data"):
        model.main_view._set_active_rasterview(model.get_main_view_rv((0, 1)))
    assert _active_pos(model) == (0, 0)


def test_toolbar_zooms_the_only_tile_with_data(model):
    a = model.load_dataset(_array())
    model.set_main_view_layout((1, 2))
    model.set_main_view_rv((0, 1), a.get_id())
    model.set_main_view_rv((0, 0), -1)
    assert model.get_main_view_rv((0, 0)).get_raster_data() is None

    model.click_main_view_zoom_in()

    assert _active_pos(model) == (0, 1)
    assert model.get_main_view_rv_scale((0, 1)) == 1.25


def test_single_view_zoom_is_unchanged(model):
    model.load_dataset(_array())

    model.click_main_view_zoom_in()

    assert model.get_main_view_scale() == model.get_main_view_rv_scale((0, 0)) == 1.25
    assert model.get_main_view_rv((0, 0))._scroll_area.styleSheet() == ""


def test_layout_shrink_resets_the_active_tile(model):
    a, b = model.load_dataset(_array()), model.load_dataset(_array())
    _grid(model, (1, 2), [a, b])
    model.activate_main_view_rv((0, 1))
    assert _active_pos(model) == (0, 1)

    model.set_main_view_layout((1, 1))
    model.click_main_view_zoom_in()

    assert _active_pos(model) == (0, 0)
    assert model.get_main_view_rv_scale((0, 0)) == 1.25


def test_active_frame_is_drawn_only_on_the_active_tile(model):
    a, b = model.load_dataset(_array()), model.load_dataset(_array())
    _grid(model, (1, 2), [a, b])

    def framed():
        return [
            pos for pos in [(0, 0), (0, 1)] if model.get_main_view_rv(pos)._scroll_area.styleSheet() != ""
        ]

    assert framed() == [(0, 0)]
    model.activate_main_view_rv((0, 1))
    assert framed() == [(0, 1)]

    model.set_main_view_layout((1, 1))
    assert model.get_main_view_rv((0, 0))._scroll_area.styleSheet() == ""


def test_each_tile_draws_its_crosshair_at_its_own_scale(model, monkeypatch):
    a = model.load_dataset(_array())
    _grid(model, (1, 2), [a, a])
    model.activate_main_view_rv((0, 0))
    model.select_main_view_zoom_option(4)
    model.main_view.set_pixel_highlight(SinglePixelSelection(QPoint(5, 5), a))
    scales = []
    monkeypatch.setattr(
        model.main_view, "_draw_crosshair_at_coord", lambda coord, widget, scale: scales.append(scale)
    )

    for pos in [(0, 0), (0, 1)]:
        rv = model.get_main_view_rv(pos)
        model.main_view._draw_pixel_highlight(rv, rv._image_widget, None)

    assert scales == [4, 1]
