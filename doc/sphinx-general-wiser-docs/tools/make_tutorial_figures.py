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

# Lab B -- AVIRIS-Classic reflectance flightline over Cuprite, Nevada. The
# committed header covers lines 2400-3799 of f230918t01p00r11_rfl; see the lab
# for the byte-range recipe that produces it.
CUPRITE = DATA / "f230918t01p00r11_rfl_cuprite.hdr"

# Lab C -- CRISM MTRDR I/F cube over Jezero Crater, Mars.
CRISM = DATA / "HRL000040FF_07_IF183J_MTR3.HDR"


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

    def shot(self, name: str, widget=None, frame=None):
        # soft_pump, not pump: pump() ends in QApplication.quit(), which closes
        # (and deleteLater()s) any result window we are about to photograph.
        OUT.mkdir(parents=True, exist_ok=True)
        self.soft_pump()
        # Dock panes only claim their space once the window is about to be
        # painted, so a region framed earlier would be framed against the wrong
        # viewport. Do it here, with the geometry the grab will actually use.
        target = widget if widget is not None else self.win
        if frame is not None:
            # The first grab of a scene is what forces the docks to claim their
            # space; throw one away so the framing below sees the final layout.
            target.grab()
            self.soft_pump()
            self.frame_region(*frame)
            self.soft_pump()
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

    def shoot_spectrum_plot(self, name: str, size=(1050, 620), x_range=None, y_range=None):
        """Float the spectrum plot's dock so band detail is legible, then grab it.

        *x_range* / *y_range* pin the axes, which the plot's own settings dialog
        also exposes. Needed wherever unreliable channels at a detector edge
        would otherwise drive the autoscale (CRISM past ~2.6 um, for one).
        """
        plot = self.win._spectrum_plot
        if x_range is not None:
            plot.set_x_autorange(False)
            plot.set_x_range(x_range)
        if y_range is not None:
            plot.set_y_autorange(False)
            plot.set_y_range(y_range)
        self.soft_pump()
        dock = plot.parentWidget()
        while dock is not None and not hasattr(dock, "setFloating"):
            dock = dock.parentWidget()
        if dock is None:
            self.shot(name, plot)
            return
        dock.setFloating(True)
        dock.resize(*size)
        self.soft_pump()
        self.shot(name, dock)

    def collect_pixels(self, pixels):
        """Click each (label, (x, y), colour) pixel and collect its spectrum."""
        for label, (x, y), colour in pixels:
            self.tm.click_raster_coord_main_view_rv((0, 0), (x, y))
            self.pump()
            active = self.state.get_active_spectrum()
            if active is not None:
                active.set_name(label)
                active.set_color(colour)
            self.tm.collect_active_spectrum()
            self.pump()

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

    def frame_region(self, x0, y0, x1, y1, pos=(0, 0)):
        """Zoom and scroll the main view so the raster box (x0,y0)-(x1,y1) fills it.

        Orthocorrected flight lines carry wide no-data margins; framing the part
        that carries data beats fitting the whole array into the figure. Call it
        via ``shot(..., frame=BOX)`` so it runs against the final layout.
        """
        rv = self.rv(pos)
        viewport = rv._scroll_area.viewport()
        w, h = viewport.width(), viewport.height()
        scale = min(w / max(x1 - x0, 1), h / max(y1 - y0, 1))
        rv.scale_image(scale)
        # The scroll area only recomputes its scrollbar ranges once the rescaled
        # pixmap is in place; setting a value before that clamps it to zero.
        self.pump()
        # Offscreen, the scroll area does not get the resize event that would
        # refresh its scrollbar ranges, so they still describe the previous
        # scale and any value we set is clamped. Set the range from the pixmap
        # we just produced, then scroll.
        sa = rv._scroll_area
        sa.horizontalScrollBar().setRange(0, max(0, sa.widget().width() - w))
        sa.verticalScrollBar().setRange(0, max(0, sa.widget().height() - h))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rv.set_scrollbar_state((int(cx * scale - w / 2), int(cy * scale - h / 2)))
        self.soft_pump()
        if os.environ.get("WISER_FIGURE_DEBUG"):
            vr = rv.get_visible_region()
            print(
                f"    frame_region: viewport {w}x{h} scale {scale:.3f} "
                f"visible=({vr.x()},{vr.y()},{vr.width()},{vr.height()})"
            )

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

    def stretch_decorrelation(self):
        """Apply a decorrelation stretch, the RGB-only option in the same dialog.

        Raw SWIR composites are nearly grey because neighbouring bands are so
        strongly correlated; this rotates the display bands onto their principal
        axes and stretches those, which is what makes mineral zoning visible.
        """
        try:
            self.main_pane()._on_stretch_builder(rasterview_pos=(0, 0))
            self.pump()
            self.tm.get_stretch_config((0, 0))._ui.rb_stretch_decorrelation.click()
            self.pump()
            self.tm.close_stretch_builder()
            self.pump()
        except IndexError:
            print("  !! stretch builder raised IndexError -- figure left unstretched")


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

# The part of the flight-line window that carries data: everything outside is
# the orthocorrection's -9999 fill.
CUPRITE_DISTRICT = (220, 600, 1000, 1400)

# Cuprite: 224 bands, 378.9-2498.3 nm at ~9.5 nm sampling.
CUPRITE_BANDS = {
    480: 10,
    550: 18,
    660: 29,
    860: 53,
    1650: 136,
    2100: 183,
    2160: 189,
    2170: 190,
    2200: 193,
    2250: 198,
    2340: 207,
    2400: 213,
}

# Endmembers picked on band position, not by eye: alunite is the pixel whose
# deepest SWIR band sits at 2170 nm; kaolinite the one with a 2160 shoulder
# alongside a deeper 2200; muscovite the one with 2200 and no shoulder.
CUPRITE_PIXELS = [
    ("Alunite", (353, 993), "#d73027"),
    ("Kaolinite", (663, 745), "#4575b4"),
    ("Muscovite", (266, 1033), "#1a9850"),
    ("Calcite", (253, 364), "#8073ac"),
]

# CRISM: 489 bands, 436-3897 nm.
CRISM_BANDS = {
    440: 1,
    530: 14,
    600: 25,
    997: 75,
    1080: 83,
    1330: 121,
    1506: 148,
    1900: 208,
    2230: 258,
    2310: 270,
    2400: 283,
    2510: 300,
    2529: 303,
    2600: 314,
}

# Pixels carrying both the 2.31 and 2.51 um bands -- the carbonate pair.
CRISM_PIXELS = [
    ("Carbonate A", (123, 298), "#d73027"),
    ("Carbonate B", (182, 229), "#4575b4"),
    ("Carbonate C", (106, 371), "#1a9850"),
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

    s.collect_pixels(AVNG_PIXELS)
    s.shot("lab_avng_spectra_window")
    s.shoot_spectrum_plot("lab_avng_spectra_plot")

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


# --------------------------------------------------------------------------
# Lab B: Cuprite, Nevada -- AVIRIS-Classic reflectance
# --------------------------------------------------------------------------


def _band_depth_bandmath(s, shot_name, result_name, centre, low, high, bands):
    """Run a linear-continuum band depth through the real band-math pipeline.

    ``1 - c / ((1 - f) * a + f * b)`` with the shoulders *a* and *b* and the
    interpolation weight *f* fixed by the actual band wavelengths, so the
    expression the reader types is the one that produced the figure.
    """
    from functools import partial

    from wiser import bandmath as bm
    from wiser.bandmath.utils import bandmath_success_callback
    from wiser.gui.bandmath_dialog import BandMathDialog

    f = (centre - low) / (high - low)
    expression = f"1 - c / ({1 - f:.3f} * a + {f:.3f} * b)"

    dlg = BandMathDialog(s.state)
    dlg.resize(960, 660)
    dlg.show()
    s.soft_pump()
    dlg._ui.ledit_expression.setText(expression)
    dlg._analyze_expr()
    s.soft_pump()

    binding = {"a": bands[low], "b": bands[high], "c": bands[centre]}
    tbl = dlg._ui.tbl_variables
    for row in range(tbl.rowCount()):
        name = tbl.item(row, 0).text()
        chooser = tbl.cellWidget(row, 2)
        if name in binding and hasattr(chooser, "band_chooser"):
            chooser.band_chooser.setCurrentIndex(binding[name])
    dlg._ui.ledit_result_name.setText(result_name)
    s.soft_pump()
    s.shot(shot_name, dlg)

    expr = dlg.get_expression()
    expr_info = dlg.get_expression_info()
    variables = dlg.get_variable_bindings()
    dlg.close()
    s.pump()

    before = len(s.state.get_datasets())
    bm.start_bandmath_evaluation(
        bandmath_expr=expr,
        expr_info=expr_info,
        result_name=result_name,
        cache=s.state.get_cache(),
        variables=variables,
        app_state=s.state,
        succeeded_callback=partial(
            bandmath_success_callback,
            s.win,
            s.state,
            expression=expr,
            batch_enabled=False,
            load_into_wiser=True,
        ),
    )
    if not s.wait_for_datasets(before + 1, timeout_s=600):
        return None
    return s.state.get_datasets()[-1]


@scene("cuprite_overview")
def cuprite_overview():
    """Cuprite: true colour beside the SWIR composite that shows the alteration."""
    require(CUPRITE, "the Cuprite AVIRIS-Classic subset")

    s = Shoot(size=(1500, 950))
    ds = s.open(CUPRITE)
    print("    dataset:", ds.get_name(), "shape", ds.get_shape())
    s.show_all_panes()
    b = CUPRITE_BANDS
    s.display(ds, bands=(b[660], b[550], b[480]))
    s.stretch_2_5()
    s.shot("lab_cuprite_truecolour", frame=CUPRITE_DISTRICT)

    # Alteration composite: 2200 / 2170 / 2340 nm. Kaolinite and muscovite are
    # red, alunite green, calcite blue.
    s.display(ds, bands=(b[2200], b[2170], b[2340]))
    s.stretch_2_5()
    s.shot("lab_cuprite_swir", frame=CUPRITE_DISTRICT)

    # The same three bands, decorrelated: the alteration zoning only separates
    # into colour once the bands' mutual correlation is removed.
    s.stretch_decorrelation()
    s.shot("lab_cuprite_decorr", frame=CUPRITE_DISTRICT)

    s.close()


@scene("cuprite_spectra")
def cuprite_spectra():
    """Cuprite: one SWIR spectrum per alteration mineral."""
    require(CUPRITE, "the Cuprite AVIRIS-Classic subset")

    s = Shoot(size=(1500, 950))
    ds = s.open(CUPRITE)
    s.show_all_panes()
    b = CUPRITE_BANDS
    s.display(ds, bands=(b[2200], b[2170], b[2340]))
    s.stretch_2_5()
    s.frame_region(*CUPRITE_DISTRICT)
    s.collect_pixels(CUPRITE_PIXELS)
    s.shot("lab_cuprite_spectra_window", frame=CUPRITE_DISTRICT)
    s.shoot_spectrum_plot("lab_cuprite_spectra_plot", size=(1100, 640))
    # The diagnostic minerals all live between 2.0 and 2.5 um; the full-range
    # plot is dominated by the 1400 and 1900 nm water-vapour bands, where
    # atmospherically corrected reflectance is meaningless.
    s.shoot_spectrum_plot(
        "lab_cuprite_swir_spectra", size=(1100, 640), x_range=(2000, 2500), y_range=(0.10, 0.48)
    )
    s.close()


@scene("cuprite_continuum")
def cuprite_continuum():
    """Cuprite: what continuum removal does to the kaolinite doublet."""
    require(CUPRITE, "the Cuprite AVIRIS-Classic subset")
    from wiser.gui.permanent_plugins.continuum_removal_plugin import ContinuumRemovalPlugin

    s = Shoot(size=(1500, 950))
    ds = s.open(CUPRITE)
    s.show_all_panes()
    b = CUPRITE_BANDS
    s.display(ds, bands=(b[2200], b[2170], b[2340]))
    s.stretch_2_5()

    kaolinite = [p for p in CUPRITE_PIXELS if p[0] == "Kaolinite"]
    s.collect_pixels(kaolinite)
    collected = s.state.get_collected_spectra()
    if collected:
        ContinuumRemovalPlugin().plot_continuum_removal(collected[-1], {"wiser": s.state})
        s.pump()
    s.shoot_spectrum_plot("lab_cuprite_continuum", size=(1100, 640))
    # Zoomed onto the diagnostic window: the 2160/2200 doublet that identifies
    # kaolinite is unambiguous once the sloping continuum is divided out.
    s.shoot_spectrum_plot(
        "lab_cuprite_continuum_swir", size=(1100, 640), x_range=(2000, 2500), y_range=(0.60, 1.05)
    )
    s.close()


@scene("cuprite_bandmath")
def cuprite_bandmath():
    """Cuprite: the 2170 nm alunite band depth as a map."""
    require(CUPRITE, "the Cuprite AVIRIS-Classic subset")

    s = Shoot(size=(1500, 950))
    s.open(CUPRITE)
    s.show_all_panes()
    s.fit()

    result = _band_depth_bandmath(
        s,
        "lab_cuprite_bandmath",
        "AluniteBD2170",
        centre=2170,
        low=2100,
        high=2250,
        bands=CUPRITE_BANDS,
    )
    if result is None:
        s.close()
        return
    s.display(result, bands=(0,), colormap="inferno")
    s.stretch_2_5()
    s.shot("lab_cuprite_bd2170", frame=CUPRITE_DISTRICT)
    s.close()


# --------------------------------------------------------------------------
# Lab C: Jezero Crater, Mars -- CRISM MTRDR
# --------------------------------------------------------------------------


@scene("crism_overview")
def crism_overview():
    """CRISM: the archive's own composite over Jezero Crater."""
    require(CRISM, "the CRISM Jezero MTRDR cube")

    s = Shoot(size=(1400, 950))
    ds = s.open(CRISM)
    print("    dataset:", ds.get_name(), "shape", ds.get_shape())
    s.show_all_panes()
    s.fit()
    s.stretch_2_5()
    s.fit()
    s.shot("lab_crism_default")

    c = CRISM_BANDS
    s.display(ds, bands=(c[2529], c[1506], c[1080]))
    s.stretch_2_5()
    s.fit()
    s.shot("lab_crism_fal")
    s.close()


@scene("crism_spectra")
def crism_spectra():
    """CRISM: the paired 2.31 / 2.51 um carbonate bands."""
    require(CRISM, "the CRISM Jezero MTRDR cube")

    s = Shoot(size=(1400, 950))
    ds = s.open(CRISM)
    s.show_all_panes()
    c = CRISM_BANDS
    s.display(ds, bands=(c[2529], c[1506], c[1080]))
    s.stretch_2_5()
    s.collect_pixels(CRISM_PIXELS)
    s.shot("lab_crism_spectra_window")
    s.shoot_spectrum_plot(
        "lab_crism_spectra_plot", size=(1100, 640), x_range=(900, 2650), y_range=(0.150, 0.215)
    )
    # The carbonate pair is only a few percent deep; zoom the axes onto it so a
    # reader can see what they are meant to be looking for.
    s.shoot_spectrum_plot(
        "lab_crism_carbonate_pair", size=(1100, 640), x_range=(2150, 2600), y_range=(0.165, 0.200)
    )
    s.close()


@scene("crism_bandmath")
def crism_bandmath():
    """CRISM: the 2.51 um band depth that separates carbonate from smectite."""
    require(CRISM, "the CRISM Jezero MTRDR cube")

    s = Shoot(size=(1400, 950))
    s.open(CRISM)
    s.show_all_panes()
    s.fit()

    result = _band_depth_bandmath(
        s,
        "lab_crism_bandmath",
        "CarbonateBD2510",
        centre=2510,
        low=2400,
        high=2600,
        bands=CRISM_BANDS,
    )
    if result is None:
        s.close()
        return
    s.display(result, bands=(0,), colormap="inferno")
    s.stretch_2_5()
    s.fit()
    s.shot("lab_crism_carbonate")
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
