"""
Tests for WISER's color-scheme feature: :mod:`wiser.gui.theme` (preference
handling, the theme-aware icon loader and the dark-mode selection-color
override, issue #727), the settings dialog that drives it, and the theme-aware
startup splash.

The icon tests recolor by rendering to a pixmap and sampling pixels, which is
deterministic for an explicit LIGHT/DARK preference because the tint depends on
the preference, not on the OS palette.  (The SYSTEM branch, which does depend on
the OS, is covered by mocking ``styleHints().colorScheme()``.)
"""

import unittest
from unittest import mock
from pathlib import Path
from typing import Dict, Tuple

import tests.context  # noqa: F401  (sets up sys.path for the wiser package)

import pytest

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QToolButton

import wiser.gui.generated.resources  # noqa: F401  (registers :/icons/* resources)
from wiser.gui import startup_splash, theme
from wiser.gui.app_config import ApplicationConfig
from wiser.gui.app_config_dialog import AppConfigDialog


pytestmark = [pytest.mark.unit]


# A single QApplication is required for any QIcon/QPixmap rendering and for the
# dialog tests.  Reuse an existing instance if the test session already made one.
_app = QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opaque_colors(icon: QIcon, size: int = 32) -> Dict[Tuple[int, int, int], int]:
    """Return a dict of {(r, g, b): count} for the icon's solidly-opaque pixels."""
    image = icon.pixmap(QSize(size, size)).toImage()
    counts: Dict[Tuple[int, int, int], int] = {}
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.alpha() > 200:
                key = (color.red(), color.green(), color.blue())
                counts[key] = counts.get(key, 0) + 1
    return counts


def _dominant_opaque_color(icon: QIcon, size: int = 32) -> Tuple[int, int, int]:
    """Return the (r, g, b) of the icon's most common solidly-opaque pixel."""
    counts = _opaque_colors(icon, size)
    assert counts, "icon has no opaque pixels to sample"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _pixel_counts(pixmap) -> Dict[Tuple[int, int, int], int]:
    """Return a dict of {(r, g, b): count} over every pixel of a grabbed widget."""
    image = pixmap.toImage()
    counts: Dict[Tuple[int, int, int], int] = {}
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            key = (color.red(), color.green(), color.blue())
            counts[key] = counts.get(key, 0) + 1
    return counts


def _relative_luminance(color: QColor) -> float:
    """WCAG relative luminance of a color (0.0 black .. 1.0 white)."""
    channels = []
    for value in (color.redF(), color.greenF(), color.blueF()):
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG contrast ratio between two colors (1.0 identical .. 21.0 black/white)."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


# Purpose-built SVG fixtures, one per color-declaration style, so the icon
# recoloring tests stay stable and self-documenting regardless of which real
# WISER icons exist.  See src/test_utils/test_icons/README.md.
_ICONS_DIR = Path(__file__).resolve().parent / ".." / "test_utils" / "test_icons"


def _fixture(name: str) -> str:
    """Return the absolute path to a test-icon fixture (accepted by get_icon)."""
    return str((_ICONS_DIR / name).resolve())


class _StubAppState:
    """Minimal ApplicationState stand-in exposing just the config API the dialog uses."""

    def __init__(self):
        self._config = ApplicationConfig()

    def get_config(self, option, default=None, as_type=None):
        return self._config.get(option, default, as_type)

    def set_config(self, option, value):
        self._config.set(option, value)

    def config(self):
        return self._config


# ---------------------------------------------------------------------------
# 1-5: color-scheme preference resolution
# ---------------------------------------------------------------------------


class TestColorSchemePreference(unittest.TestCase):
    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)

    def test_scheme_roundtrip(self):
        for scheme in (theme.SYSTEM, theme.LIGHT, theme.DARK):
            theme.set_color_scheme(scheme)
            self.assertEqual(theme.get_color_scheme(), scheme)

    def test_invalid_scheme_falls_back_to_system(self):
        for bad in ("", "garbage", None, "purple"):
            theme.set_color_scheme(theme.DARK)
            theme.set_color_scheme(bad)
            self.assertEqual(theme.get_color_scheme(), theme.SYSTEM)

    def test_scheme_normalizes_case_and_whitespace(self):
        theme.set_color_scheme(" dark ")
        self.assertEqual(theme.get_color_scheme(), theme.DARK)
        theme.set_color_scheme("light")
        self.assertEqual(theme.get_color_scheme(), theme.LIGHT)

    def test_is_dark_mode_explicit(self):
        theme.set_color_scheme(theme.LIGHT)
        self.assertFalse(theme.is_dark_mode())
        theme.set_color_scheme(theme.DARK)
        self.assertTrue(theme.is_dark_mode())

    def test_is_dark_mode_system_follows_os(self):
        # under SYSTEM, is_dark_mode() follows styleHints().colorScheme().
        theme.set_color_scheme(theme.SYSTEM)
        with mock.patch.object(theme, "QGuiApplication") as mock_gui_app:
            hints = mock_gui_app.styleHints.return_value
            hints.colorScheme.return_value = Qt.ColorScheme.Dark
            self.assertTrue(theme.is_dark_mode())
            hints.colorScheme.return_value = Qt.ColorScheme.Light
            self.assertFalse(theme.is_dark_mode())
            hints.colorScheme.return_value = Qt.ColorScheme.Unknown
            self.assertFalse(theme.is_dark_mode())


# ---------------------------------------------------------------------------
# theme-aware icon loading
# ---------------------------------------------------------------------------


class TestThemedIcons(unittest.TestCase):
    """Recoloring-engine behavior, exercised with purpose-built SVG fixtures.

    The fixtures (not real WISER icons) each isolate one color-declaration
    style, so these tests describe the engine's contract and don't drift if the
    production icons change.  ``TestRealResourceIcons`` covers the real
    ``:/icons/...`` pipeline separately.
    """

    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)
        theme._icon_cache.clear()

    def test_light_mode_icon_untinted(self):
        # in light mode the icon keeps its authored (near-black) color.
        theme.set_color_scheme(theme.LIGHT)
        r, g, b = _dominant_opaque_color(theme.get_icon(_fixture("styled_stroke_black.svg")))
        self.assertLess(max(r, g, b), 40)

    def test_dark_mode_icon_tinted(self):
        # in dark mode a "stroke:#000"-in-<style> icon is tinted to a light color.
        theme.set_color_scheme(theme.DARK)
        r, g, b = _dominant_opaque_color(theme.get_icon(_fixture("styled_stroke_black.svg")))
        self.assertGreater(min(r, g, b), 200)

    def test_dark_mode_recolors_default_black_icon(self):
        # default_fill_black.svg declares NO color and relies on SVG's default
        # black fill.  A string-replace of "#000" would miss it; compositing
        # recolors it.
        theme.set_color_scheme(theme.DARK)
        r, g, b = _dominant_opaque_color(theme.get_icon(_fixture("default_fill_black.svg")))
        self.assertGreater(min(r, g, b), 200)

    def test_monochrome_false_preserves_colors(self):
        # multi-color icons opted out of tinting keep their real colors,
        # even in dark mode.
        theme.set_color_scheme(theme.DARK)
        colors = _opaque_colors(theme.get_icon(_fixture("multicolor.svg"), monochrome=False))
        self.assertGreater(len(colors), 3)
        # At least one clearly-saturated (non-gray) color survives.
        self.assertTrue(any(max(c) - min(c) > 40 for c in colors))

    def test_monochrome_false_not_cached(self):
        # an explicit opt-out also bypasses the themed cache.
        path = _fixture("styled_stroke_black.svg")
        theme.get_icon(path, monochrome=False)
        self.assertNotIn(path, theme._icon_cache)

    def test_get_icon_cached_by_path(self):
        path = _fixture("styled_stroke_black.svg")
        first = theme.get_icon(path)
        second = theme.get_icon(path)
        self.assertIs(first, second)

    def test_icon_adapts_live_without_rebuild(self):
        # THE core "live" claim -- a single icon instance re-tints itself
        # when the scheme changes; a baked-pixmap implementation would fail.
        theme.set_color_scheme(theme.LIGHT)
        icon = theme.get_icon(_fixture("styled_stroke_black.svg"))
        r, g, b = _dominant_opaque_color(icon)
        self.assertLess(max(r, g, b), 40)  # black in light mode

        theme.set_color_scheme(theme.DARK)
        r, g, b = _dominant_opaque_color(icon)  # SAME instance
        self.assertGreater(min(r, g, b), 200)  # now light


# ---------------------------------------------------------------------------
# real Qt resource pipeline (deliberately coupled to production icons)
# ---------------------------------------------------------------------------


class TestRealResourceIcons(unittest.TestCase):
    """Canary that the compiled ``:/icons/...`` resources load and recolor.

    Unlike ``TestThemedIcons``, these are intentionally coupled to real icons so
    that a broken resource bundle (missing .qrc entry, wrong prefix, un-rebuilt
    ``generated/resources.py``) is caught.
    """

    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)
        theme._icon_cache.clear()

    def test_real_resource_icon_loads_and_tints(self):
        theme.set_color_scheme(theme.DARK)
        icon = theme.get_icon(":/icons/zoom-in.svg")
        self.assertFalse(icon.isNull())
        r, g, b = _dominant_opaque_color(icon)
        self.assertGreater(min(r, g, b), 200)

    def test_non_svg_returned_raw(self):
        # non-SVG icons are not routed through the themed engine/cache.
        theme.set_color_scheme(theme.DARK)
        icon = theme.get_icon(":/icons/wiser.ico")
        self.assertFalse(icon.isNull())
        self.assertNotIn(":/icons/wiser.ico", theme._icon_cache)


# ---------------------------------------------------------------------------
# applying the scheme to the application
# ---------------------------------------------------------------------------


class TestApplyColorScheme(unittest.TestCase):
    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)

    def test_apply_color_scheme_requests_correct_scheme(self):
        fake_app = mock.Mock()
        hints = fake_app.styleHints.return_value

        theme.set_color_scheme(theme.LIGHT)
        theme.apply_color_scheme(fake_app)
        hints.setColorScheme.assert_called_with(Qt.ColorScheme.Light)

        theme.set_color_scheme(theme.DARK)
        theme.apply_color_scheme(fake_app)
        hints.setColorScheme.assert_called_with(Qt.ColorScheme.Dark)

        theme.set_color_scheme(theme.SYSTEM)
        theme.apply_color_scheme(fake_app)
        hints.setColorScheme.assert_called_with(Qt.ColorScheme.Unknown)


# ---------------------------------------------------------------------------
# dark-mode selection color
# ---------------------------------------------------------------------------


class TestDarkHighlightOverride(unittest.TestCase):
    """The dark-mode Highlight/Accent override (issue: bright blue selection).

    Qt's dark palette uses a very light blue for selection (Windows 11 style:
    Accent ``#4cc2ff``), which our near-white dark-mode icons and Qt's white
    HighlightedText sit on top of almost illegibly.  ``apply_color_scheme``
    replaces it with a darker blue -- and must *stop* doing so in light mode.

    These act on the real QApplication palette, so tearDown restores it.
    """

    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)

    def tearDown(self):
        theme.set_color_scheme(theme.SYSTEM)
        theme.apply_color_scheme(_app)
        _app.processEvents()

    def _apply(self, scheme):
        theme.set_color_scheme(scheme)
        theme.apply_color_scheme(_app)
        _app.processEvents()
        return _app.palette()

    def test_dark_mode_overrides_highlight_and_accent(self):
        palette = self._apply(theme.DARK)
        self.assertEqual(palette.color(QPalette.Highlight), theme.DARK_HIGHLIGHT_COLOR)
        self.assertEqual(palette.color(QPalette.Accent), theme.DARK_HIGHLIGHT_COLOR)

    def test_light_mode_clears_the_override(self):
        # Switching away from dark must not leave the dark blue behind.  This is
        # what pins the override to a *sparse* palette: building it from a copy
        # of the live palette instead carries the dark blue into light mode,
        # because the copy resolves Highlight/Accent from the previous apply.
        self._apply(theme.DARK)
        palette = self._apply(theme.LIGHT)
        self.assertNotEqual(palette.color(QPalette.Highlight), theme.DARK_HIGHLIGHT_COLOR)
        self.assertNotEqual(palette.color(QPalette.Accent), theme.DARK_HIGHLIGHT_COLOR)

    def test_override_readable_against_dark_mode_icons(self):
        # THE point of the override: the near-white icon tint must contrast the
        # selection fill.  4.5:1 is the WCAG AA threshold; Qt's own #4cc2ff is
        # about 1.6:1 and would fail this.
        self._apply(theme.DARK)
        ratio = _contrast_ratio(theme.DARK_HIGHLIGHT_COLOR, theme._DARK_ICON_COLOR)
        self.assertGreater(ratio, 4.5)

    def test_checked_tool_button_paints_the_darker_blue(self):
        # end-to-end: the style actually paints a checked tool button with the
        # overridden color (this is the toolbar highlight the user sees).
        self._apply(theme.DARK)
        button = QToolButton()
        button.setCheckable(True)
        button.setChecked(True)
        button.setFixedSize(40, 40)
        try:
            button.show()
            _app.processEvents()
            counts = _pixel_counts(button.grab())
        finally:
            button.hide()
            button.deleteLater()

        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
        expected = theme.DARK_HIGHLIGHT_COLOR
        self.assertEqual(dominant, (expected.red(), expected.green(), expected.blue()))


# ---------------------------------------------------------------------------
# theme-aware startup splash
# ---------------------------------------------------------------------------


class TestStartupSplashColors(unittest.TestCase):
    """The splash is styled with explicit colors, so it has to follow the scheme itself."""

    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)

    def tearDown(self):
        theme.set_color_scheme(theme.SYSTEM)

    def _splash(self, scheme):
        theme.set_color_scheme(scheme)
        splash = startup_splash.StartupSplash(QIcon().pixmap(QSize(64, 64)), QIcon())
        self.addCleanup(splash.deleteLater)
        return splash

    def test_color_set_follows_scheme(self):
        theme.set_color_scheme(theme.DARK)
        self.assertIs(startup_splash._colors_for_active_scheme(), startup_splash._DARK)
        theme.set_color_scheme(theme.LIGHT)
        self.assertIs(startup_splash._colors_for_active_scheme(), startup_splash._LIGHT)

    def test_dark_splash_is_dark(self):
        # the card background and log pane must be darker than their text, i.e.
        # the splash is not a white rectangle on a dark desktop.
        splash = self._splash(theme.DARK)
        colors = splash._colors
        self.assertLess(QColor(colors.window_bg).lightness(), 128)
        self.assertLess(QColor(colors.log_bg).lightness(), QColor(colors.log_fg).lightness())
        self.assertIn(colors.window_bg, splash.styleSheet())

    def test_light_splash_is_light(self):
        splash = self._splash(theme.LIGHT)
        colors = splash._colors
        self.assertGreater(QColor(colors.window_bg).lightness(), 128)
        self.assertGreater(QColor(colors.log_bg).lightness(), QColor(colors.log_fg).lightness())
        self.assertIn(colors.window_bg, splash.styleSheet())

    def test_failure_state_keeps_the_scheme(self):
        # set_startup_failed restyles the log pane; it must not fall back to the
        # light-mode error colors while in dark mode.
        splash = self._splash(theme.DARK)
        splash.set_startup_failed("boom")
        style = splash._log_view.styleSheet()
        self.assertIn(startup_splash._DARK.log_error_bg, style)
        self.assertNotIn(startup_splash._LIGHT.log_error_bg, style)


# ---------------------------------------------------------------------------
# settings dialog applies the scheme only on OK
# ---------------------------------------------------------------------------


class TestAppConfigDialogColorScheme(unittest.TestCase):
    def setUp(self):
        theme.set_color_scheme(theme.SYSTEM)

    def _pick(self, dialog, scheme):
        combo = dialog._ui.cbox_color_scheme
        combo.setCurrentIndex(combo.findData(scheme))

    def test_cancel_does_not_apply(self):
        # changing the combo then cancelling applies nothing and persists
        # nothing -- the scheme is untouched.
        state = _StubAppState()
        dialog = AppConfigDialog(state)

        self._pick(dialog, theme.DARK)
        dialog.reject()

        self.assertEqual(theme.get_color_scheme(), theme.SYSTEM)
        self.assertEqual(state.get_config("general.color_scheme"), theme.SYSTEM)

    def test_accept_applies_and_persists(self):
        # OK applies the selected scheme to the app AND writes it to config.
        state = _StubAppState()
        dialog = AppConfigDialog(state)

        self._pick(dialog, theme.DARK)
        dialog.accept()

        self.assertEqual(theme.get_color_scheme(), theme.DARK)
        self.assertEqual(state.get_config("general.color_scheme"), theme.DARK)
