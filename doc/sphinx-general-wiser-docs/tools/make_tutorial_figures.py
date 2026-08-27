#!/usr/bin/env python
"""Regenerate the screenshots used by the WISER tutorials.

Every figure in ``source/tutorials/`` is produced by a *scene* in this file: a
function that drives the real WISER application through the same steps a reader
follows, then grabs the widget it is talking about. Re-running this script
re-shoots the whole set, so the tutorials cannot drift away from the UI.

Usage
-----
    python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py --list
    python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py --only first_look
    python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py          # everything

    # on a headless Linux box
    xvfb-run -a python doc/sphinx-general-wiser-docs/tools/make_tutorial_figures.py

Figures are written to ``source/_static/tutorials/``.

Data
----
Scenes marked *bundled* use the fixtures in ``src/test_utils/``, which ship with
the repository. Scenes marked *download* use the AVIRIS-NG Caltech subset, which
is too large to commit; put it at

    src/test_utils/test_datasets/ang20171108t184227_corr_v2p13_subset_bil{,.hdr}

(see the lab page for the download link). Those scenes skip themselves with a
message if the file is absent, so the bundled scenes still run.

Prerequisites: the generated Qt modules must exist --
``make -C src/wiser/gui && make -C src generated``.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
DATA = SRC / "test_utils" / "test_datasets"
SPECTRA_DIR = SRC / "test_utils" / "test_spectra"
OUT = Path(__file__).resolve().parents[1] / "source" / "_static" / "tutorials"

sys.path.insert(0, str(SRC))
os.chdir(SRC)

# Qt must be told to render offscreen before QApplication exists. Leaving this
# unset on a desktop pops real windows up in front of whoever is running the
# script; the grab is identical either way.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMainWindow  # noqa: E402

from test_utils.test_model import WiserTestModel  # noqa: E402

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

# Bundled fixtures -- small, and present in every checkout.
CAMPUS = DATA / "caltech_4_100_150_nm.hdr"  # 150x150, 4 bands, AVIRIS over Caltech
CUBE = DATA / "caltech_425_7_7_nm.hdr"  # 7x7, 425 AVIRIS bands
SWIR = DATA / "caltech_15_20_22_bb.hdr"  # 20x22, 15 SWIR bands, has a bad-band list
BOARD = DATA / "circuit_4_100_150_um.hdr"  # 150x150, 4 bands, circuit board
USGS_LIB = SPECTRA_DIR / "usgs_resampHeadwallSWIR.hdr"  # 481 USGS mineral spectra

# Downloaded -- the full AVIRIS-NG Caltech subset, 680x500x425.
AVNG = DATA / "ang20171108t184227_corr_v2p13_subset_bil.hdr"


class SceneSkipped(Exception):
    """Raised by a scene whose input data is not present."""


def require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SceneSkipped(f"{what} not found at {path.relative_to(REPO_ROOT)}")
    return path


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _shrink(path: Path) -> None:
    """Palette-quantise a screenshot before it lands in the repository.

    A Qt window over imagery keeps ~256 colours without visible loss at the
    sizes these figures are displayed at, and roughly halves the file. Skipped
    silently if Pillow is unavailable or the result is not actually smaller.
    """
    try:
        from PIL import Image
    except ImportError:
        return
    original = path.stat().st_size
    tmp = path.with_name("_q_" + path.name)
    try:
        img = Image.open(path).convert("RGB")
        img.quantize(colors=256).save(tmp, "PNG", optimize=True, compress_level=9)
        if tmp.stat().st_size < original:
            tmp.replace(path)
        else:
            tmp.unlink()
    except Exception:
        if tmp.exists():
            tmp.unlink()


class Shoot:
    """A live WISER instance plus the helpers a scene needs."""

    def __init__(self, size=(1400, 900)):
        self.tm = WiserTestModel(use_gui=True)
        self.win = self.tm.main_window
        self.win.resize(*size)
        self.state = self.tm.app_state
        self.services = self.tm.app_services

        # WiserTestModel.run() spins the event loop and then calls
        # QApplication.quit(). In Qt 6 that closes every top-level window, and
        # DataVisualizerApp.closeEvent tears down AppServices -- including the
        # storage client every background analysis needs. A scene that ran the
        # event loop once could therefore no longer submit a task. Swap in a
        # plain QMainWindow.closeEvent for the duration of the scene: the close
        # is still accepted (so quit() completes and exec_() returns), but the
        # services stay alive.
        self._shooting = True
        self._real_close_event = type(self.win).closeEvent

        def guarded_close_event(event, _self=self):
            if _self._shooting:
                QMainWindow.closeEvent(_self.win, event)
                return
            _self._real_close_event(_self.win, event)

        self.win.closeEvent = guarded_close_event
        self.pump()

    # -- event loop --------------------------------------------------------

    def pump(self, rounds: int = 12):
        """Let Qt catch up: repaint, deliver posted events, run pending timers."""
        for _ in range(rounds):
            QApplication.processEvents()
        self.tm.run()
        for _ in range(rounds):
            QApplication.processEvents()

    def soft_pump(self, rounds: int = 25):
        """Deliver events WITHOUT spinning the event loop.

        ``WiserTestModel.run()`` ends with ``QApplication.quit()``, which closes
        every top-level window -- fatal for a modeless dialog created with
        ``WA_DeleteOnClose``. Use this while a dialog is on screen.
        """
        for _ in range(rounds):
            QApplication.processEvents()
            time.sleep(0.01)

    def wait_for_datasets(self, target: int, timeout_s: float = 240.0) -> bool:
        """Wait until *target* datasets exist, so a background task can land.

        Uses ``soft_pump`` only: ``pump`` ends in ``QApplication.quit()``, which
        closes every top-level window -- including result windows the task just
        opened, such as the PCA scree plot.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if len(self.state.get_datasets()) >= target:
                self.soft_pump()
                return True
            self.soft_pump(10)
        print(f"  !! timed out waiting for {target} datasets (have {len(self.state.get_datasets())})")
        return False

    def close(self):
        self._shooting = False
        try:
            self.tm.close_app()
        except RuntimeError:
            pass
        # WiserTestModel.__del__ calls close_app() again at collection time, by
        # which point the C++ window is gone and shiboken raises. Neutralise it
        # so the script exits without a spurious traceback.
        self.tm.close_app = lambda: None

    # -- data --------------------------------------------------------------

    def open(self, path: Path):
        self.state.open_file(str(path))
        self.pump()
        return self.state.get_datasets()[-1]

    # -- capture -----------------------------------------------------------

    def shot(self, name: str, widget=None):
        # soft_pump, not pump: pump() ends in QApplication.quit(), which closes
        # (and deleteLater()s) any result window we are about to photograph.
        OUT.mkdir(parents=True, exist_ok=True)
        self.soft_pump()
        target = widget if widget is not None else self.win
        path = OUT / f"{name}.png"
        target.grab().save(str(path))
        _shrink(path)
        print(f"  -> {path.name}")

    def dialog_shot(self, name: str, dialog: QDialog, size: Optional[tuple] = None):
        """Show a dialog modelessly, grab it, close it."""
        dialog.setModal(False)
        if size:
            dialog.resize(*size)
        dialog.show()
        self.soft_pump()
        self.shot(name, dialog)
        dialog.close()
        self.pump()

    def find_dialog(self, class_name: str):
        """Locate a modeless dialog by class name, wherever Qt parented it.

        Dialogs opened by the app often carry ``WA_DeleteOnClose``; clear it on
        the way out so a later event-loop spin cannot delete the widget we are
        about to photograph.
        """
        candidates = QApplication.topLevelWidgets() + self.win.findChildren(QDialog)
        found = [w for w in candidates if type(w).__name__ == class_name]
        found.sort(key=lambda w: not w.isVisible())
        if not found:
            return None
        widget = found[0]
        widget.setAttribute(Qt.WA_DeleteOnClose, False)
        return widget

    # -- view control ------------------------------------------------------

    def show_all_panes(self):
        self.tm.click_spectrum_plot_display_toggle()
        self.tm.click_zoom_pane_display_toggle()
        self.tm.click_dataset_info_display_toggle()
        self.pump()

    def fit(self):
        self.tm.click_zoom_to_fit()
        self.pump()

    def main_pane(self):
        return self.win._main_view

    def rv(self, pos=(0, 0)):
        return self.tm.get_main_view_rv(pos)

    def display(self, dataset, bands=None, colormap=None, pos=(0, 0)):
        """Show *dataset* in the main view, optionally setting bands/colormap."""
        self.main_pane().show_dataset(dataset, rasterview_pos=pos)
        self.pump()
        if bands is not None:
            self.main_pane().set_display_bands(dataset.get_id(), bands, colormap=colormap)
            self.pump()
        self.fit()

    def stretch_2_5(self):
        """Apply a 2.5% linear stretch through the real stretch-builder path.

        Reopening the dialog after the view's band count has changed raises
        ``IndexError`` inside ``StretchBuilderDialog.show()``, which indexes
        ``stretches[0..2]`` unconditionally in its RGB branch. Order scenes so
        that does not happen; report rather than hide it if it does.
        """
        try:
            self.main_pane()._on_stretch_builder(rasterview_pos=(0, 0))
            self.pump()
            self.tm.click_stretch_linear_2_5()
            self.pump()
            self.tm.close_stretch_builder()
            self.pump()
        except IndexError:
            print(
                "  !! stretch builder raised IndexError (band count changed on "
                "this view since the last stretch) -- figure left unstretched"
            )


SCENES: Dict[str, Callable[[], None]] = {}


def scene(name: str):
    def wrap(fn):
        SCENES[name] = fn
        return fn

    return wrap


# --------------------------------------------------------------------------
# Tutorial 1 - First Look (bundled)
# --------------------------------------------------------------------------


@scene("first_look")
def first_look():
    from wiser.gui.band_chooser import BandChooserDialog

    s = Shoot()
    s.shot("t1_empty")

    s.open(CAMPUS)
    s.fit()
    s.shot("t1_loaded")

    s.show_all_panes()
    s.fit()
    s.shot("t1_all_panes")

    rv = s.rv()
    dlg = BandChooserDialog(s.state, rv.get_raster_data(), rv.get_display_bands(), colormap=None)
    s.dialog_shot("t1_band_chooser", dlg)

    s.main_pane()._on_stretch_builder(rasterview_pos=(0, 0))
    s.pump()
    sb = s.tm.get_stretch_builder()
    s.shot("t1_stretch_default", sb)
    s.tm.click_stretch_linear_2_5()
    s.pump()
    s.shot("t1_stretch_2p5", sb)
    s.shot("t1_stretch_applied")
    s.tm.close_stretch_builder()

    ds = s.state.get_datasets()[-1]
    s.display(ds, bands=(3,), colormap="viridis")
    s.shot("t1_colormap_nir")

    s.close()


# --------------------------------------------------------------------------
# AVIRIS-NG Caltech subset (download) - the lab scene
# --------------------------------------------------------------------------


# Band indices in the AVIRIS-NG cube, resolved from its 5 nm wavelength grid.
AVNG_BANDS = {480: 21, 550: 35, 660: 57, 860: 96, 1650: 254, 2200: 364}

# Pixels chosen by spectral criteria rather than by eye -- see the lab text.
AVNG_PIXELS = [
    ("Vegetation", (330, 141), "#1a9850"),
    ("Swimming pool", (298, 428), "#2166ac"),
    ("Roof", (462, 301), "#b2182b"),
    ("Asphalt", (631, 129), "#666666"),
]


@scene("avng_overview")
def avng_overview():
    """AVIRIS-NG: true-colour and SWIR composites, and the band chooser."""
    require(AVNG, "the AVIRIS-NG Caltech subset")
    from wiser.gui.band_chooser import BandChooserDialog

    s = Shoot(size=(1500, 950))
    ds = s.open(AVNG)
    print("    dataset:", ds.get_name(), "shape", ds.get_shape())
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()
    s.shot("lab_avng_truecolour")

    rv = s.rv()
    dlg = BandChooserDialog(s.state, rv.get_raster_data(), rv.get_display_bands(), colormap=None)
    s.dialog_shot("lab_avng_band_chooser", dlg)

    b = AVNG_BANDS
    s.display(ds, bands=(b[2200], b[1650], b[860]))
    s.stretch_2_5()
    s.shot("lab_avng_swir")

    s.close()


@scene("avng_spectra")
def avng_spectra():
    """AVIRIS-NG: one 425-band spectrum per surface type."""
    require(AVNG, "the AVIRIS-NG Caltech subset")

    s = Shoot(size=(1500, 950))
    s.open(AVNG)
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()

    for label, (x, y), colour in AVNG_PIXELS:
        s.tm.click_raster_coord_main_view_rv((0, 0), (x, y))
        s.pump()
        active = s.state.get_active_spectrum()
        if active is not None:
            active.set_name(label)
            active.set_color(colour)
        s.tm.collect_active_spectrum()
        s.pump()
    s.shot("lab_avng_spectra_window")

    # Float the plot's dock so the 425-band detail is legible in the figure.
    plot = s.win._spectrum_plot
    dock = plot.parentWidget()
    while dock is not None and not hasattr(dock, "setFloating"):
        dock = dock.parentWidget()
    if dock is not None:
        dock.setFloating(True)
        dock.resize(1050, 620)
        s.soft_pump()
        s.shot("lab_avng_spectra_plot", dock)
    else:
        s.shot("lab_avng_spectra_plot", plot)

    s.close()


@scene("avng_ndvi")
def avng_ndvi():
    """AVIRIS-NG: NDVI through the real band-math pipeline, at 5 nm sampling."""
    require(AVNG, "the AVIRIS-NG Caltech subset")
    from functools import partial

    from wiser import bandmath as bm
    from wiser.bandmath.utils import bandmath_success_callback
    from wiser.gui.bandmath_dialog import BandMathDialog

    s = Shoot(size=(1500, 950))
    ds = s.open(AVNG)
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()

    dlg = BandMathDialog(s.state)
    dlg.resize(960, 660)
    dlg.show()
    s.soft_pump()
    dlg._ui.ledit_expression.setText("(nir - red) / (nir + red)")
    dlg._analyze_expr()
    s.soft_pump()

    binding = {"nir": AVNG_BANDS[860], "red": AVNG_BANDS[660]}
    tbl = dlg._ui.tbl_variables
    for row in range(tbl.rowCount()):
        name = tbl.item(row, 0).text()
        chooser = tbl.cellWidget(row, 2)
        if name in binding and hasattr(chooser, "band_chooser"):
            chooser.band_chooser.setCurrentIndex(binding[name])
    dlg._ui.ledit_result_name.setText("NDVI")
    s.soft_pump()
    s.shot("lab_avng_bandmath", dlg)

    expression = dlg.get_expression()
    expr_info = dlg.get_expression_info()
    variables = dlg.get_variable_bindings()
    dlg.close()
    s.pump()

    before = len(s.state.get_datasets())
    bm.start_bandmath_evaluation(
        bandmath_expr=expression,
        expr_info=expr_info,
        result_name="NDVI",
        cache=s.state.get_cache(),
        variables=variables,
        app_state=s.state,
        succeeded_callback=partial(
            bandmath_success_callback,
            s.win,
            s.state,
            expression=expression,
            batch_enabled=False,
            load_into_wiser=True,
        ),
    )
    if not s.wait_for_datasets(before + 1, timeout_s=300):
        s.close()
        return

    ndvi = s.state.get_datasets()[-1]
    s.display(ndvi, bands=(0,), colormap="RdYlGn")
    # Straight off the pipeline the flight-line edges drag the NDVI minimum to
    # about -3.4, so a 100% linear stretch squeezes the real -0.03..0.83 range
    # into the top third of the ramp and the whole scene reads as one green.
    # Shoot that first -- the lab tells the reader to expect it -- then fix it.
    s.shot("lab_avng_ndvi_unstretched")
    s.stretch_2_5()
    s.shot("lab_avng_ndvi")

    s.tm.set_main_view_layout((1, 2))
    s.pump()
    s.main_pane().show_dataset(ds, rasterview_pos=(0, 0))
    s.main_pane().show_dataset(ndvi, rasterview_pos=(0, 1))
    s.pump()
    s.main_pane().set_display_bands(ndvi.get_id(), (0,), colormap="RdYlGn")
    s.pump()
    s.fit()
    s.shot("lab_avng_ndvi_vs_rgb")

    s.close()


# --------------------------------------------------------------------------
# AVIRIS-NG: feature space, clustering, dimensionality reduction
# --------------------------------------------------------------------------


@scene("avng_classify")
def avng_classify():
    """AVIRIS-NG: red vs NIR feature space, and K-means over the full cube."""
    require(AVNG, "the AVIRIS-NG Caltech subset")

    s = Shoot(size=(1500, 950))
    ds = s.open(AVNG)
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()

    s.tm.open_interactive_scatter_plot_context_menu()
    s.soft_pump()
    dlg = s.main_pane()._interactive_scatter_plot_dialog
    dlg.resize(900, 820)
    s.tm.set_interactive_scatter_x_axis_dataset(ds.get_id())
    s.tm.set_interactive_scatter_y_axis_dataset(ds.get_id())
    s.tm.set_interactive_scatter_render_dataset(ds.get_id())
    s.tm.set_interactive_scatter_x_band(AVNG_BANDS[660])
    s.tm.set_interactive_scatter_y_band(AVNG_BANDS[860])
    s.soft_pump()
    s.tm.click_create_scatter_plot()
    deadline = time.time() + 240
    while dlg.get_xy() is None and time.time() < deadline:
        s.soft_pump(10)
    s.soft_pump(40)
    s.shot("lab_avng_scatter", dlg)

    try:
        s.tm.create_polygon_in_interactive_scatter_plot(
            [(0.02, 0.35), (0.10, 0.85), (0.20, 0.85), (0.10, 0.30)]
        )
        s.soft_pump(40)
        s.shot("lab_avng_scatter_selection", dlg)
        s.shot("lab_avng_scatter_highlight")
    except Exception:
        traceback.print_exc()
    dlg.close()
    s.pump()

    # K-means over the full 425-band cube. Timed, because the lab quotes it.
    s.win.show_kmeans_dialog()
    kdlg = s.win._kmeans_dialog
    kdlg.select_dataset(ds.get_id())
    kdlg._ui.ledit_k_clusters.setText("6")
    kdlg._ui.btn_advanced_options.click()
    kdlg._ui.ledit_seed.setText("42")
    s.soft_pump()
    s.shot("lab_avng_kmeans_dialog", kdlg)

    before = len(s.state.get_datasets())
    t0 = time.time()
    kdlg.perform_kmeans()
    kdlg.close()
    if s.wait_for_datasets(before + 1, timeout_s=1800):
        print(f"    K-means on {ds.get_shape()} took {time.time() - t0:.0f} s")
        labels = s.state.get_datasets()[-1]
        s.display(labels, bands=(0,), colormap="tab10")
        s.shot("lab_avng_kmeans")

    s.close()


@scene("avng_transform")
def avng_transform():
    """AVIRIS-NG: PCA over 425 bands, with the scree plot."""
    require(AVNG, "the AVIRIS-NG Caltech subset")

    s = Shoot(size=(1500, 950))
    ds = s.open(AVNG)
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()

    s.win.show_pca_dialog()
    s.soft_pump()
    dlg = s.win._pca_plugin._dialog
    s.shot("lab_avng_pca_dialog", dlg)

    before = len(s.state.get_datasets())
    t0 = time.time()
    dlg.accept()
    s.pump()
    if s.wait_for_datasets(before + 1, timeout_s=1800):
        print(f"    PCA on {ds.get_shape()} took {time.time() - t0:.0f} s")

        # Grab the scree plot FIRST. It is a separate top-level window, and
        # pump() ends in QApplication.quit(), which closes every such window.
        for widget in list(getattr(s.state, "_matplotlib_display_widgets", [])):
            if "Scree" in (widget.windowTitle() or ""):
                widget.resize(820, 600)
                s.soft_pump()
                s.shot("lab_avng_scree", widget)
                break
        else:
            print("    !! no scree window found")

        result = s.state.get_datasets()[-1]
        s.display(result, bands=(0, 1, 2))
        s.stretch_2_5()
        s.shot("lab_avng_pc_composite")
        s.display(result, bands=(0,), colormap="gray")
        s.stretch_2_5()
        s.shot("lab_avng_pc1")

    s.close()


# --------------------------------------------------------------------------
# Getting Started tutorials (bundled fixtures)
# --------------------------------------------------------------------------


@scene("spectra")
def spectra():
    """Tutorial 2: clicking pixels, collecting spectra, importing a library."""
    s = Shoot()
    s.open(CUBE)
    s.show_all_panes()
    s.fit()

    s.tm.click_raster_coord_main_view_rv((0, 0), (1, 1))
    s.pump()
    s.shot("t2_one_spectrum")

    for (x, y), colour in [((1, 1), "#1a9850"), ((3, 4), "#b2182b"), ((5, 2), "#2166ac")]:
        s.tm.click_raster_coord_main_view_rv((0, 0), (x, y))
        s.pump()
        active = s.state.get_active_spectrum()
        if active is not None:
            active.set_color(colour)
        s.tm.collect_active_spectrum()
        s.pump()
    s.shot("t2_collected")

    s.tm.import_spectral_library(str(USGS_LIB))
    s.pump()
    s.shot("t2_library")

    s.close()


@scene("rois")
def rois():
    """Tutorial 3: three land-cover ROIs and their mean spectra."""
    from wiser.raster.roi import RegionOfInterest
    from wiser.raster.selection import RectangleSelection

    s = Shoot()
    s.open(CAMPUS)
    s.show_all_panes()
    s.fit()

    plan = [
        ("Tree canopy", "#22aa22", [((104, 118), (122, 138)), ((68, 20), (80, 34))]),
        ("Building roof", "#ee3333", [((16, 26), (44, 46))]),
        ("Parking lot", "#3399ff", [((52, 40), (68, 74))]),
    ]
    made = []
    for name, colour, boxes in plan:
        roi = RegionOfInterest(name=name, color=colour)
        for (x1, y1), (x2, y2) in boxes:
            roi.add_selection(RectangleSelection(QPoint(x1, y1), QPoint(x2, y2)))
        s.state.add_roi(roi, make_name_unique=True)
        made.append(roi)
    s.pump()
    s.main_pane().update_all_rasterviews()
    s.pump()
    s.shot("t3_rois_drawn")

    rv = s.rv()
    for roi in made:
        s.main_pane()._on_show_roi_avg_spectrum(roi=roi, rasterview=rv)
        s.pump()
        s.tm.collect_active_spectrum()
        s.pump()
    s.shot("t3_roi_spectra", s.win._spectrum_plot)

    s.close()


@scene("bandmath")
def bandmath():
    """Tutorial 4: NDVI over the bundled campus scene."""
    from functools import partial

    from wiser import bandmath as bm
    from wiser.bandmath.utils import bandmath_success_callback
    from wiser.gui.bandmath_dialog import BandMathDialog

    s = Shoot()
    ds = s.open(CAMPUS)
    s.show_all_panes()
    s.fit()

    dlg = BandMathDialog(s.state)
    dlg.resize(940, 660)
    dlg.show()
    s.soft_pump()
    dlg._ui.ledit_expression.setText("(nir - red) / (nir + red)")
    dlg._analyze_expr()
    s.soft_pump()
    for row in range(dlg._ui.tbl_variables.rowCount()):
        name = dlg._ui.tbl_variables.item(row, 0).text()
        chooser = dlg._ui.tbl_variables.cellWidget(row, 2)
        if name in {"nir": 3, "red": 2} and hasattr(chooser, "band_chooser"):
            chooser.band_chooser.setCurrentIndex({"nir": 3, "red": 2}[name])
    dlg._ui.ledit_result_name.setText("NDVI")
    s.soft_pump()
    s.shot("t4_bandmath_dialog", dlg)

    expression = dlg.get_expression()
    expr_info = dlg.get_expression_info()
    variables = dlg.get_variable_bindings()
    dlg.close()
    s.pump()

    before = len(s.state.get_datasets())
    bm.start_bandmath_evaluation(
        bandmath_expr=expression,
        expr_info=expr_info,
        result_name="NDVI",
        cache=s.state.get_cache(),
        variables=variables,
        app_state=s.state,
        succeeded_callback=partial(
            bandmath_success_callback,
            s.win,
            s.state,
            expression=expression,
            batch_enabled=False,
            load_into_wiser=True,
        ),
    )
    if s.wait_for_datasets(before + 1, timeout_s=300):
        ndvi = s.state.get_datasets()[-1]
        s.display(ndvi, bands=(0,), colormap="RdYlGn")
        s.stretch_2_5()
        s.shot("t4_ndvi")

        s.tm.set_main_view_layout((1, 2))
        s.pump()
        s.main_pane().show_dataset(ds, rasterview_pos=(0, 0))
        s.main_pane().show_dataset(ndvi, rasterview_pos=(0, 1))
        s.pump()
        s.main_pane().set_display_bands(ndvi.get_id(), (0,), colormap="RdYlGn")
        s.pump()
        s.fit()
        s.shot("t4_ndvi_vs_rgb")

    s.close()


@scene("kmeans")
def kmeans():
    """Tutorial 5b: K-means over the bundled campus scene."""
    s = Shoot()
    ds = s.open(CAMPUS)
    s.show_all_panes()
    s.fit()

    s.win.show_kmeans_dialog()
    dlg = s.win._kmeans_dialog
    dlg.select_dataset(ds.get_id())
    dlg._ui.ledit_k_clusters.setText("5")
    s.soft_pump()
    s.shot("t5_kmeans_dialog", dlg)
    dlg._ui.btn_advanced_options.click()
    dlg._ui.ledit_seed.setText("42")
    s.soft_pump()
    s.shot("t5_kmeans_advanced", dlg)

    before = len(s.state.get_datasets())
    dlg.perform_kmeans()
    dlg.close()
    if s.wait_for_datasets(before + 1, timeout_s=600):
        for w in list(getattr(s.state, "_matplotlib_display_widgets", [])):
            if "entroid" in (w.windowTitle() or ""):
                s.shot("t5_kmeans_centroids", w)
                break
        labels = s.state.get_datasets()[-1]
        s.display(labels, bands=(0,), colormap="tab10")
        s.shot("t5_kmeans_labels")

    s.close()


@scene("scatter")
def scatter():
    """Tutorial 5a: feature space on the bundled campus scene."""
    s = Shoot(size=(1500, 950))
    ds = s.open(CAMPUS)
    s.show_all_panes()
    s.fit()

    s.tm.open_interactive_scatter_plot_context_menu()
    s.soft_pump()
    dlg = s.main_pane()._interactive_scatter_plot_dialog
    dlg.resize(900, 820)
    s.soft_pump()
    s.shot("t8_scatter_empty", dlg)

    s.tm.set_interactive_scatter_x_axis_dataset(ds.get_id())
    s.tm.set_interactive_scatter_y_axis_dataset(ds.get_id())
    s.tm.set_interactive_scatter_render_dataset(ds.get_id())
    s.tm.set_interactive_scatter_x_band(2)
    s.tm.set_interactive_scatter_y_band(3)
    s.soft_pump()
    s.tm.click_create_scatter_plot()
    deadline = time.time() + 180
    while dlg.get_xy() is None and time.time() < deadline:
        s.soft_pump(10)
    s.soft_pump(40)
    s.shot("t8_scatter_plot", dlg)

    try:
        s.tm.create_polygon_in_interactive_scatter_plot(
            [(0.02, 0.22), (0.14, 0.55), (0.22, 0.55), (0.10, 0.20)]
        )
        s.soft_pump(40)
        s.shot("t8_scatter_selection", dlg)
        s.shot("t8_scatter_highlight")
    except Exception:
        traceback.print_exc()
    dlg.close()
    s.close()


@scene("mnf")
def mnf():
    """Tutorial 6: the MNF dialog."""
    s = Shoot()
    s.open(CUBE)
    s.show_all_panes()
    s.fit()
    s.win.show_mnf_dialog()
    s.soft_pump()
    dlg = s.find_dialog("MinimumNoiseFractionDialog")
    if dlg is not None:
        s.shot("t6_mnf_dialog", dlg)
        dlg.close()
    s.close()


@scene("pca_dialog")
def pca_dialog():
    """Tutorial 6: the PCA dialog on a bundled cube."""
    s = Shoot()
    s.open(CUBE)
    s.show_all_panes()
    s.fit()
    s.win.show_pca_dialog()
    s.soft_pump()
    s.shot("t6_pca_dialog", s.win._pca_plugin._dialog)
    s.win._pca_plugin._dialog.close()
    s.close()


@scene("detection")
def detection():
    """Tutorial 7: the four matching and unmixing dialogs."""
    s = Shoot()
    s.open(SWIR)
    s.show_all_panes()
    s.fit()
    s.tm.import_spectral_library(str(USGS_LIB))
    s.pump()

    s.win.show_spectral_angle_mapper_dialog()
    s.soft_pump()
    dlg = s.find_dialog("SAMTool")
    if dlg is not None:
        dlg.resize(1000, 720)
        s.soft_pump()
        s.shot("t7_sam_dialog", dlg)
        dlg.close()

    s.win.show_spectral_feature_fitting_dialog()
    s.soft_pump()
    dlg = s.find_dialog("SFFTool")
    if dlg is not None:
        dlg.resize(1000, 720)
        s.soft_pump()
        s.shot("t7_sff_dialog", dlg)
        dlg.close()

    s.win.show_mtmf_dialog()
    s.soft_pump()
    s.shot("t7_mtmf_dialog", s.win._mtmf_dialog)
    s.win._mtmf_dialog.close()

    s.win.show_linear_unmixing_dialog()
    s.soft_pump()
    s.shot("t7_unmix_dialog", s.win._linear_unmixing_dialog)
    s.win._linear_unmixing_dialog.close()

    s.close()


@scene("filters")
def filters():
    """Filters reference page: the Savitzky-Golay and smoothing dialogs."""
    from wiser.bandmath.types import VariableType
    from wiser.gui.sav_golay import SavGolayDialog
    from wiser.gui.smooth_filter import SmoothingFilterDialog, SmoothingFilterKind

    s = Shoot()
    ds = s.open(CUBE)
    s.show_all_panes()
    s.fit()

    # MainView opens both with exec_(), which would block the script; build the
    # same dialogs with the same arguments and show them modelessly.
    s.dialog_shot(
        "t9_savgol_dialog",
        SavGolayDialog(
            app_state=s.state,
            app_services=s.services,
            target_type=VariableType.IMAGE_CUBE_DATASET,
            target_id=int(ds.get_id()),
        ),
    )
    s.dialog_shot(
        "t9_smooth_dialog",
        SmoothingFilterDialog(
            app_state=s.state,
            app_services=s.services,
            filter_kind=SmoothingFilterKind.GAUSSIAN,
            target_dataset_id=ds.get_id(),
        ),
    )

    s.close()


@scene("board")
def board():
    """Materials lab: the bundled circuit-board scene."""
    s = Shoot()
    s.open(BOARD)
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()
    s.shot("lab_board_rgb")

    for (x, y), colour in [((28, 38), "#1a9850"), ((104, 96), "#b2182b"), ((74, 62), "#2166ac")]:
        s.tm.click_raster_coord_main_view_rv((0, 0), (x, y))
        s.pump()
        active = s.state.get_active_spectrum()
        if active is not None:
            active.set_color(colour)
        s.tm.collect_active_spectrum()
        s.pump()
    s.shot("lab_board_spectra", s.win._spectrum_plot)

    s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", help="scene name (repeatable)")
    ap.add_argument("--list", action="store_true", help="list scene names and exit")
    args = ap.parse_args()

    if args.list:
        for name in SCENES:
            print(name)
        return

    names = args.only or list(SCENES)
    failed, skipped = [], []
    for name in names:
        print(f"[{name}]")
        try:
            SCENES[name]()
        except SceneSkipped as exc:
            print(f"  ~~ skipped: {exc}")
            skipped.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    if skipped:
        print(f"\nskipped (data not present): {', '.join(skipped)}")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    # macOS and Windows spawn worker processes by re-importing __main__; without
    # this guard WISER's scheduler pool re-runs the whole script in every child.
    multiprocessing.freeze_support()
    main()
