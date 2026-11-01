"""Tests for the visibility checkboxes on the spectrum plot's spectra tree (issue #806).

Each spectrum in the collected spectra and in a spectral library carries a checkbox
that shows whether it is plotted, and toggling the checkbox plots or hides it.
"""

import numpy as np
import pytest

import tests.context  # noqa: F401

from PySide6.QtCore import Qt

from test_utils.test_model import WiserTestModel
from wiser.gui.spectrum_plot import SHOW_LIBRARY_SPECTRA_THRESHOLD, SpectrumPlotGeneric
from wiser.raster.spectrum import NumPyArraySpectrum

pytestmark = [pytest.mark.integration]


@pytest.fixture
def model():
    m = WiserTestModel()
    yield m
    m.close_app()


def _spectra(count):
    return [NumPyArraySpectrum(np.arange(4, dtype=np.float32) + i, name=f"s{i}") for i in range(count)]


def _library_items(model):
    (library_item,) = model.spectrum_plot._library_treeitems.values()
    return library_item, [library_item.child(i) for i in range(library_item.childCount())]


def _shown(model, item):
    return any(shown is item.data(0, Qt.UserRole) for shown in model.get_displayed_spectra())


def test_library_checkboxes_reflect_and_toggle_the_plot(model):
    model.import_spectra(_spectra(3))
    library_item, items = _library_items(model)

    assert [item.checkState(0) for item in items] == [Qt.Checked] * 3
    assert all(_shown(model, item) for item in items)
    assert library_item.data(0, Qt.CheckStateRole) is None

    items[1].setCheckState(0, Qt.Unchecked)
    assert [_shown(model, item) for item in items] == [True, False, True]

    items[1].setCheckState(0, Qt.Checked)
    assert all(_shown(model, item) for item in items)


def test_large_library_starts_hidden_and_unchecked(model):
    model.import_spectra(_spectra(SHOW_LIBRARY_SPECTRA_THRESHOLD))
    _, items = _library_items(model)

    assert all(item.checkState(0) == Qt.Unchecked for item in items)
    assert model.get_displayed_spectra() == []

    items[4].setCheckState(0, Qt.Checked)
    assert [_shown(model, item) for item in items] == [i == 4 for i in range(len(items))]


def test_group_and_context_menu_toggles_update_the_checkboxes(model):
    model.import_spectra(_spectra(3))
    library_item, items = _library_items(model)
    plot = model.spectrum_plot

    plot._on_hide_all_spectra(library_item)
    assert all(item.checkState(0) == Qt.Unchecked for item in items)
    assert model.get_displayed_spectra() == []

    plot._on_toggle_spectrum_visible(items[0])
    assert [item.checkState(0) for item in items] == [Qt.Checked, Qt.Unchecked, Qt.Unchecked]
    assert _shown(model, items[0])

    plot._on_show_all_spectra(library_item)
    assert all(item.checkState(0) == Qt.Checked for item in items)


def test_collected_spectrum_checkbox_toggles_the_plot(model):
    (spectrum,) = _spectra(1)
    model.collect_spectrum(spectrum)
    item = model.spectrum_plot._treeitem_collected.child(0)

    assert item.checkState(0) == Qt.Checked
    assert _shown(model, item)

    item.setCheckState(0, Qt.Unchecked)
    assert not _shown(model, item)


def test_active_spectrum_has_no_checkbox(model):
    (spectrum,) = _spectra(1)
    model.set_active_spectrum(spectrum)

    assert model.spectrum_plot._treeitem_active.data(0, Qt.CheckStateRole) is None
    assert _shown(model, model.spectrum_plot._treeitem_active)


def test_generic_plot_collected_spectrum_checkbox_toggles_the_plot(model):
    plot = SpectrumPlotGeneric(app_state=model.app_state)
    (spectrum,) = _spectra(1)
    plot.add_collected_spectrum(spectrum)
    item = plot._treeitem_collected.child(0)

    assert item.checkState(0) == Qt.Checked
    assert spectrum.get_id() in plot._spectrum_display_info

    item.setCheckState(0, Qt.Unchecked)
    assert spectrum.get_id() not in plot._spectrum_display_info
    plot.close()
