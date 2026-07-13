"""Region-of-interest persistence (issue #619).

ROIs are standalone ``[SOURCE]`` state with no dataset dependency, so they save
and load directly via the pyrep convention.  Each ROI is captured as its
identity plus a list of faithfully-serialized selections.
"""

from typing import TYPE_CHECKING, Any, Dict

from wiser.raster.roi import ROI_PYREP_TYPE, roi_from_pyrep, roi_to_pyrep

from ..pyrep import register_pyrep

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

register_pyrep(ROI_PYREP_TYPE, roi_from_pyrep)


def save_rois(app_state: "ApplicationState", manifest: Dict[str, Any]) -> None:
    """Write every ROI in ``app_state`` into ``manifest['rois']``."""
    manifest["rois"] = [roi_to_pyrep(roi) for roi in app_state.get_rois()]


def load_rois(manifest: Dict[str, Any], app_state: "ApplicationState") -> None:
    """Reconstruct ROIs from ``manifest['rois']`` into ``app_state``."""
    for data in manifest.get("rois", []):
        app_state.add_roi(roi_from_pyrep(data))
