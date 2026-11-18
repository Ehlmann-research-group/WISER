"""Contract of ApplicationState.add_dataset for the dataset name."""

import numpy as np
import pytest

import tests.context  # noqa: F401

from wiser.gui.app_state import ApplicationState
from wiser.raster.loader import RasterDataLoader


def _numpy_dataset():
    return RasterDataLoader().dataset_from_numpy_array(np.zeros((1, 4, 4), dtype=np.float32))


def test_adding_a_dataset_without_a_name_raises_and_leaves_the_session_unchanged():
    app_state = ApplicationState(None)
    added = []
    app_state.dataset_added.connect(lambda ds_id, view: added.append(ds_id))

    with pytest.raises(ValueError, match="name"):
        app_state.add_dataset(_numpy_dataset())

    assert app_state.num_datasets() == 0
    assert added == []


def test_adding_a_named_dataset_registers_it():
    app_state = ApplicationState(None)
    ds = _numpy_dataset()
    ds.set_name("named")

    app_state.add_dataset(ds)

    assert app_state.get_dataset(ds.get_id()) is ds
