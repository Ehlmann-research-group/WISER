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

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


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

# The base size icons are rendered at.  A single generously-sized pixmap is
# rendered and QIcon scales it down smoothly to whatever size a toolbar or
# button requests.
_ICON_RENDER_SIZE = 64

# The active color-scheme preference.  Seeded from config at startup via
# :func:`set_color_scheme`; defaults to following the OS.
_preference: str = SYSTEM

# In-memory cache of recolored icons, keyed by (resource path, tint-color name).
_icon_cache: Dict[Tuple[str, str], QIcon] = {}


def set_color_scheme(preference: str) -> None:
    """
    Set the active color-scheme preference (``SYSTEM``, ``LIGHT`` or ``DARK``).

    Unrecognized values fall back to ``SYSTEM``.  If the preference actually
    changes, the icon cache is cleared so that icons requested afterwards
    reflect the new scheme.
    """
    global _preference

    normalized = (preference or SYSTEM).strip().upper()
    if normalized not in _VALID_SCHEMES:
        normalized = SYSTEM

    if normalized != _preference:
        _preference = normalized
        _icon_cache.clear()


def get_color_scheme() -> str:
    """Return the active color-scheme preference (``SYSTEM``/``LIGHT``/``DARK``)."""
    return _preference


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


def get_icon(path: str, monochrome: bool = True) -> QIcon:
    """
    Return a :class:`QIcon` for the given Qt resource path
    (e.g. ``":/icons/zoom-in.svg"``), recolored to contrast the active color
    scheme.

    Most WISER icons are monochrome black SVGs; in dark mode they are tinted to
    a light color so they stay visible, and in light mode they are returned
    unmodified.  Pass ``monochrome=False`` for intentionally multi-color icons
    (e.g. the RGB "true color" icon) or non-SVG icons, which are always
    returned as-is.
    """
    if not monochrome or not is_dark_mode() or not path.lower().endswith(".svg"):
        return QIcon(path)

    color = QColor(_DARK_ICON_COLOR)
    cache_key = (path, color.name())
    cached = _icon_cache.get(cache_key)
    if cached is not None:
        return cached

    icon = QIcon(_render_tinted_svg(path, color))
    _icon_cache[cache_key] = icon
    return icon


def _render_tinted_svg(path: str, color: QColor) -> QPixmap:
    """
    Render the SVG at ``path`` to a transparent pixmap and recolor every opaque
    pixel to ``color``, preserving the icon's alpha (and thus its anti-aliased
    edges).
    """
    renderer = QSvgRenderer(path)

    pixmap = QPixmap(_ICON_RENDER_SIZE, _ICON_RENDER_SIZE)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, _ICON_RENDER_SIZE, _ICON_RENDER_SIZE))
    # SourceIn keeps the destination alpha (the rendered icon shape) but takes
    # the source color, recoloring the whole glyph in one fill.
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()

    return pixmap
