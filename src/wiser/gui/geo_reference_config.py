"""
Configuration object for driving :class:`~wiser.gui.geo_reference_dialog.GeoReferencerDialog`
programmatically (WISER#684).

A ``GeoReferencerConfig`` lets an external caller (e.g. the Seamless Mosaic pane) preset the
target dataset, the reference dataset *or* a manual reference CRS, and the output save path,
and independently lock any of those choosers so the user cannot change them. It also lets the
caller relabel the accept button (e.g. "Save to Mosaic").

Passing ``config=None`` to ``GeoReferencerDialog.show`` / ``exec_`` reproduces the classic
Tools -> Geo Reference behavior exactly.

This lives in its own module (rather than in ``geo_reference_dialog``) so callers that only
need to build a config do not import the heavy dialog module, avoiding an import cycle.
"""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.crs_model import GeneralCRS


@dataclass
class GeoReferencerConfig:
    """Presets + lock flags for :class:`GeoReferencerDialog`."""

    target_dataset: Optional["RasterDataSet"] = None
    reference_dataset: Optional["RasterDataSet"] = None
    # Used when reference_dataset is None (manual reference CRS).
    reference_crs: Optional["GeneralCRS"] = None
    save_path: Optional[str] = None

    allow_change_target: bool = True
    allow_change_reference: bool = True
    allow_change_save_path: bool = True

    # e.g. "Save to Mosaic"; None keeps the dialog's default accept-button text.
    accept_button_text: Optional[str] = None
