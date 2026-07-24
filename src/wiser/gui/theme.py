"""
Theme / color-scheme support for WISER's UI.

Since the PySide6 transition, WISER follows the OS light/dark theme
automatically, but its toolbar and UI icons are monochrome SVGs that were
authored in black.  In dark mode those icons render black-on-dark and become
effectively invisible.  This module centralizes:

  * the user's color-scheme preference (System / Light / Dark), backed by the
    ``general.color_scheme`` application-config option, and
  * a :func:`get_icon` helper that recolors monochrome icons on the fly to
    contrast the active scheme, caching the results in-memory.

Icons are recolored by rendering the SVG and compositing a solid tint over its
alpha (``CompositionMode_SourceIn``), so recoloring works regardless of how a
given SVG declares its color -- an explicit ``fill``/``stroke``, a ``<style>``
block, or SVG's default black fill.
"""

from __future__ import annotations

from typing import Dict, Tuple

from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QIconEngine, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QStyleOption


# Color-scheme preference values.  These match the strings stored in the
# ``general.color_scheme`` application-config option.
SYSTEM = "SYSTEM"
LIGHT = "LIGHT"
DARK = "DARK"

_VALID_SCHEMES = (SYSTEM, LIGHT, DARK)

# The color monochrome icons are tinted to when the effective scheme is dark.
# In light mode the icons are left as-authored (near-black), so no tint is
# applied and the appearance is identical to before this module existed.
_DARK_ICON_COLOR = QColor("#e6e6e6")

# The active color-scheme preference.  Seeded from config at startup via
# :func:`set_color_scheme`; defaults to following the OS.
_preference: str = SYSTEM

# In-memory cache of themed icons, keyed by resource path.  The icons are
# engine-backed and re-tint themselves at paint time, so a single instance is
# reused across scheme changes.
_icon_cache: Dict[str, QIcon] = {}


def set_color_scheme(preference: str) -> None:
    """
    Set the active color-scheme preference (``SYSTEM``, ``LIGHT`` or ``DARK``).

    Unrecognized values fall back to ``SYSTEM``.  Existing icons re-tint
    themselves at paint time, so callers should trigger a repaint (see
    :func:`refresh_icons`) for the change to become visible immediately.
    """
    global _preference

    normalized = (preference or SYSTEM).strip().upper()
    if normalized not in _VALID_SCHEMES:
        normalized = SYSTEM

    _preference = normalized


def get_color_scheme() -> str:
    """Return the active color-scheme preference (``SYSTEM``/``LIGHT``/``DARK``)."""
    return _preference


def apply_color_scheme(app=None) -> None:
    """
    Apply the active color-scheme preference to the running application, so that
    window backgrounds, text and other palette colors match the chosen scheme --
    not just the icons.

    On Qt 6.8+ this uses ``QStyleHints.setColorScheme()``, which restyles the
    whole application palette natively.  ``SYSTEM`` restores following the OS
    theme.  Older Qt versions (without ``setColorScheme``) are a no-op here, so
    the app keeps following the OS and only the icons adapt.
    """
    if app is None:
        app = QGuiApplication.instance()
    if app is None:
        return

    hints = app.styleHints()
    if hints is None or not hasattr(hints, "setColorScheme"):
        return

    if _preference == LIGHT:
        hints.setColorScheme(Qt.ColorScheme.Light)
    elif _preference == DARK:
        hints.setColorScheme(Qt.ColorScheme.Dark)
    else:
        # SYSTEM: clear any override so Qt follows the OS theme again.
        hints.setColorScheme(Qt.ColorScheme.Unknown)


def is_dark_mode() -> bool:
    """
    Return whether the *effective* color scheme is dark.

    For an explicit ``LIGHT`` or ``DARK`` preference the answer is fixed.  For
    ``SYSTEM`` it follows the OS via ``styleHints().colorScheme()`` (Qt 6.5+),
    so it can change while the app is running if the user flips their OS theme.
    """
    if _preference == LIGHT:
        return False
    if _preference == DARK:
        return True

    # SYSTEM: follow the OS.
    hints = QGuiApplication.styleHints()
    if hints is not None:
        return hints.colorScheme() == Qt.ColorScheme.Dark
    return False


class _ThemedSvgIconEngine(QIconEngine):
    """
    A :class:`QIconEngine` that renders a monochrome SVG and, in dark mode,
    recolors it to a light tint so it stays visible.

    The effective scheme is read at *paint* time via :func:`is_dark_mode`, so a
    single icon instance adapts to live color-scheme changes automatically: when
    the scheme changes and the owning widget repaints, the icon re-renders in
    the new color (see :func:`refresh_icons`).  Rendered pixmaps are cached per
    (size, mode, scheme) so repeated repaints don't re-render the SVG.
    """

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._renderer = QSvgRenderer(path)
        self._cache: Dict[Tuple[int, int, bool, int], QPixmap] = {}

    def _render(self, size: QSize, mode: QIcon.Mode) -> QPixmap:
        dark = is_dark_mode()
        key = (size.width(), size.height(), dark, mode.value)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        pixmap = QPixmap(size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        self._renderer.render(painter, QRectF(0, 0, size.width(), size.height()))
        if dark:
            # SourceIn keeps the rendered glyph's alpha but replaces its color,
            # recoloring the whole icon in one fill regardless of how the SVG
            # declared its color.
            painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
            painter.fillRect(pixmap.rect(), QColor(_DARK_ICON_COLOR))
        painter.end()

        # Let the active style apply its mode effect (e.g. graying out disabled
        # icons) so themed icons match the rest of the UI.
        if mode != QIcon.Mode.Normal:
            style = QApplication.style()
            if style is not None:
                option = QStyleOption()
                option.palette = QApplication.palette()
                pixmap = style.generatedIconPixmap(mode, pixmap, option)

        self._cache[key] = pixmap
        return pixmap

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self._render(size, mode)

    def paint(self, painter: QPainter, rect, mode: QIcon.Mode, state: QIcon.State) -> None:
        painter.drawPixmap(rect, self._render(rect.size(), mode))

    def clone(self) -> QIconEngine:
        return _ThemedSvgIconEngine(self._path)


def get_icon(path: str, monochrome: bool = True) -> QIcon:
    """
    Return a :class:`QIcon` for the given Qt resource path
    (e.g. ``":/icons/zoom-in.svg"``) that recolors itself to contrast the active
    color scheme.

    Most WISER icons are monochrome black SVGs; these are returned as
    engine-backed icons that stay black in light mode and tint to a light color
    in dark mode, adapting live if the scheme changes.  Pass
    ``monochrome=False`` for intentionally multi-color icons (e.g. the RGB
    "true color" icon) or non-SVG icons, which are returned unmodified.
    """
    if not monochrome or not path.lower().endswith(".svg"):
        return QIcon(path)

    cached = _icon_cache.get(path)
    if cached is not None:
        return cached

    icon = QIcon(_ThemedSvgIconEngine(path))
    _icon_cache[path] = icon
    return icon


def refresh_icons(app=None) -> None:
    """
    Repaint all widgets so that themed icons re-render in the active color
    scheme.  Call this after changing the scheme at runtime (together with
    :func:`apply_color_scheme`) for the change to take effect immediately.
    """
    if app is None:
        app = QApplication.instance()
    if app is None:
        return

    for widget in app.allWidgets():
        widget.update()
