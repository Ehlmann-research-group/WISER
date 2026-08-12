"""
The raster file-format registry.

This module declares every raster format WISER knows how to open, and the rules
used to decide which one applies to a given file.  :class:`RasterDataLoader`
consumes the registry; nothing here opens a file or touches the GUI.

Adding a new format is a single :class:`FormatSpec` entry in
:data:`RASTER_FORMATS` -- see ``doc/.../adding-a-raster-format`` for the full
walkthrough.

Dispatch model
--------------

A file is matched in three steps:

1. **Override.**  If the caller names a format (``load_from_file(format=...)``),
   only that spec is tried, and a failure is raised rather than swallowed.
2. **Ordering.**  Candidates are ordered by whether the path's extension is one
   the format claims, then by descending :attr:`FormatSpec.priority`.  The
   extension is only an ordering *hint* -- a format is never excluded for
   failing to claim an extension, so files with unusual or absent extensions
   still resolve.
3. **Identification.**  Each candidate's ``identify()`` is called in order.  The
   first :attr:`Confidence.YES` wins immediately.  If no candidate is certain,
   the highest-priority :attr:`Confidence.MAYBE` wins.  Only the winner is
   actually opened.
"""

import logging
import os

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Type

from .dataset import RasterDataSet
from .dataset_impl import (
    Confidence,
    RasterDataImpl,
    ENVI_GDALRasterDataImpl,
    GTiff_GDALRasterDataImpl,
    NetCDF_GDALRasterDataImpl,
    ASC_GDALRasterDataImpl,
    JP2_GDALRasterDataImpl,
    PDS3_GDALRasterDataImpl,
    PDS4_GDALRasterDataImpl,
    FITS_GDALRasterDataImpl,
    GDALRasterDataImpl,
)

if TYPE_CHECKING:
    from .data_cache import DataCache

logger = logging.getLogger(__name__)


# Confidence is defined in dataset_impl (where RasterDataImpl.identify needs it)
# and re-exported here, since the registry is the natural place to look for it.
__all__ = [
    "Confidence",
    "FormatSpec",
    "RASTER_FORMATS",
    "candidates_for",
    "format_for_impl",
    "format_names",
    "get_format",
    "load_FITS_dataset",
    "load_normal_dataset",
]


# ---------------------------------------------------------------------------
# Dataset-construction strategies
#
# These run *after* a format has been chosen and opened.  Each takes the opened
# impl and returns the RasterDataSet objects to hand back to the caller; a list,
# because one impl may expand into several datasets.
# ---------------------------------------------------------------------------


def load_normal_dataset(impl: RasterDataImpl, data_cache: "DataCache") -> List[RasterDataSet]:
    """Wrap a single opened impl as a single dataset.  The common case."""
    return [RasterDataSet(impl, data_cache)]


def load_FITS_dataset(impl: RasterDataImpl, data_cache: "DataCache") -> List[RasterDataSet]:
    """
    FITS files may hold several images, so ask the user which to load.

    Returns an empty list if the user cancels.
    """
    # Imported lazily:  the registry is imported by non-GUI code paths (tests,
    # project restore) that must not pull in dialog classes.
    from wiser.gui.fits_loading_dialog import FitsDatasetLoadingDialog
    from PySide6.QtWidgets import QDialog

    dialog = FitsDatasetLoadingDialog(impl, data_cache)
    if dialog.exec() == QDialog.Accepted:
        return dialog.return_datasets
    return []


@dataclass(frozen=True)
class FormatSpec:
    """One raster format WISER can open."""

    name: str
    """Stable identifier.  This is the token used by the ``format=`` override and
    persisted in project files, so renaming one is a breaking change."""

    impl: Type[RasterDataImpl]
    """The :class:`RasterDataImpl` subclass that opens this format."""

    extensions: frozenset
    """Lower-case extensions this format claims, including the leading dot.
    ``""`` means "no extension".  An **ordering hint only** -- a format is never
    excluded for failing to claim the path's extension."""

    priority: int = 0
    """Higher wins.  Breaks ties between formats that are equally confident, and
    orders the search.  Explicit so that dispatch never depends on declaration
    order."""

    loader: Callable[[RasterDataImpl, "DataCache"], List[RasterDataSet]] = load_normal_dataset
    """How to turn the opened impl into datasets."""

    interactive_step: bool = False
    """True if opening this format can block on a dialog.  Informational for now;
    it marks the formats that must stay on the main thread once loading moves to
    a worker thread."""

    is_fallback: bool = False
    """True for the catch-all that runs only after every other format declines."""

    def claims_extension(self, path: str) -> bool:
        """True if this format claims the extension of ``path`` (case-insensitive)."""
        return os.path.splitext(path)[1].lower() in self.extensions


# ---------------------------------------------------------------------------
# The registry
#
# Priorities are spaced by 5 so a format can be slotted between two others
# without renumbering.  Higher = tried earlier.
# ---------------------------------------------------------------------------

RASTER_FORMATS: Tuple[FormatSpec, ...] = (
    FormatSpec(
        name="PDS4",
        impl=PDS4_GDALRasterDataImpl,
        extensions=frozenset({".xml"}),
        priority=90,
    ),
    FormatSpec(
        name="PDS3",
        impl=PDS3_GDALRasterDataImpl,
        extensions=frozenset({".lbl", ".pds", ".img"}),
        priority=85,
    ),
    FormatSpec(
        name="NetCDF",
        impl=NetCDF_GDALRasterDataImpl,
        extensions=frozenset({".nc", ".nc4", ".cdf"}),
        priority=80,
        # Prompts for a subdataset when the file holds more than one.
        interactive_step=True,
    ),
    FormatSpec(
        name="ENVI",
        impl=ENVI_GDALRasterDataImpl,
        extensions=frozenset({"", ".hdr", ".img", ".dat"}),
        priority=70,
    ),
    FormatSpec(
        name="GTiff",
        impl=GTiff_GDALRasterDataImpl,
        extensions=frozenset({".tif", ".tiff", ".tfw"}),
        priority=65,
    ),
    FormatSpec(
        name="JP2",
        impl=JP2_GDALRasterDataImpl,
        extensions=frozenset({".jp2"}),
        priority=60,
    ),
    FormatSpec(
        name="FITS",
        impl=FITS_GDALRasterDataImpl,
        extensions=frozenset({".fits", ".fit", ".fts"}),
        priority=55,
        loader=load_FITS_dataset,
        interactive_step=True,
    ),
    FormatSpec(
        name="ASCII",
        impl=ASC_GDALRasterDataImpl,
        extensions=frozenset({".asc"}),
        priority=50,
    ),
    FormatSpec(
        name="GDAL",
        impl=GDALRasterDataImpl,
        extensions=frozenset(),
        priority=-100,
        # Last resort:  let GDAL pick any driver it likes.  Keeps files with
        # unexpected extensions loadable rather than failing outright.
        is_fallback=True,
    ),
)


def get_format(name: str) -> Optional[FormatSpec]:
    """Look up a spec by :attr:`FormatSpec.name` (case-insensitive)."""
    lowered = name.lower()
    for spec in RASTER_FORMATS:
        if spec.name.lower() == lowered:
            return spec
    return None


def format_for_impl(impl: RasterDataImpl) -> Optional[str]:
    """
    The registered format name that produced ``impl``, or ``None``.

    Matched on the exact implementation type rather than by ``isinstance``, so
    that a subclass is never mistaken for its base.  ``None`` means the impl
    came from somewhere other than the registry (an in-memory NumPy dataset, or
    a format opened through the GDAL catch-all under a different class), in
    which case callers should fall back to normal detection.
    """
    impl_type = type(impl)
    for spec in RASTER_FORMATS:
        if spec.impl is impl_type:
            return spec.name
    return None


def format_names() -> List[str]:
    """Every registered format name, for error messages and UI."""
    return [spec.name for spec in RASTER_FORMATS]


def candidates_for(path: str) -> List[FormatSpec]:
    """
    Order the registry for ``path``, most likely first.

    Formats claiming the path's extension come first, then everything else, then
    the fallback.  Within each group, higher :attr:`FormatSpec.priority` wins.
    Every non-fallback format appears exactly once regardless of extension --
    the extension orders the search, it does not restrict it.
    """
    claiming: List[FormatSpec] = []
    others: List[FormatSpec] = []
    fallbacks: List[FormatSpec] = []

    for spec in RASTER_FORMATS:
        if spec.is_fallback:
            fallbacks.append(spec)
        elif spec.claims_extension(path):
            claiming.append(spec)
        else:
            others.append(spec)

    def _sort_key(s):
        return -s.priority

    return sorted(claiming, key=_sort_key) + sorted(others, key=_sort_key) + sorted(fallbacks, key=_sort_key)
