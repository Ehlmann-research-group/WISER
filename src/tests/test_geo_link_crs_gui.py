"""Linked main-view tiles whose datasets have no transformation between their CRSs.

A spatially linked split view syncs every tile to the one that scrolled.  Showing an
Earth dataset beside a Mars dataset must leave the pair unlinked rather than raise
from inside the scroll handler.
"""

from pathlib import Path

import pytest

import tests.context  # noqa: F401 -- adds src/ to sys.path

from test_utils.test_model import WiserTestModel
from wiser.raster.dataset import GeographicLinkState

pytestmark = [pytest.mark.integration]

_DATASETS = (Path(__file__).resolve().parent / ".." / "test_utils" / "test_datasets").resolve()


@pytest.fixture
def model():
    m = WiserTestModel()
    yield m
    m.close_app()


def test_linked_split_view_switching_tile_to_mars_does_not_raise(model):
    epsg4087 = model.load_dataset(str(_DATASETS / "caltech_4_100_150_nm_epsg4087.tif"))
    utm = model.load_dataset(str(_DATASETS / "caltech_4_100_150_nm_rot_35_scale_2_linear_gt.tif"))
    mars = model.load_dataset(str(_DATASETS / "ESP_048952_1785_COLOR_crop.JP2"))

    model.set_main_view_layout((1, 2))
    model.set_main_view_rv((0, 0), epsg4087.get_id())
    model.set_main_view_rv((0, 1), utm.get_id())
    assert model.app_state.multiple_displayed_datasets_link_compatible() == (
        True,
        GeographicLinkState.SPATIAL,
    )
    assert model.click_link_button()

    model.set_main_view_rv((0, 1), mars.get_id())
    model.scroll_main_view_rv((0, 0), 10, 10)
    model.click_raster_coord_main_view_rv((0, 0), (5, 5))

    assert model.get_main_view_rv((0, 1)).get_raster_data() is mars
    assert model.app_state.multiple_displayed_datasets_link_compatible() == (
        False,
        GeographicLinkState.NO_LINK,
    )
