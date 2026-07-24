"""
Tests for :mod:`wiser.gui.theme` -- WISER's color-scheme handling and the
theme-aware icon loader introduced for dark-mode support (issue #727).

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
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

import wiser.gui.generated.resources  # noqa: F401  (registers :/icons/* resources)
from wiser.gui import theme
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
