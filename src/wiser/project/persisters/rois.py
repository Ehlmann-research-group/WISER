"""Region-of-interest persistence (issue #619).

ROIs are standalone ``[SOURCE]`` state with no dataset dependency, so they save
and load directly via the pyrep convention.  Each ROI is captured as its
identity plus a list of faithfully-serialized selections.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from wiser.raster.roi import ROI_PYREP_TYPE, roi_from_pyrep, roi_to_pyrep

from ..pyrep import register_pyrep

if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState

register_pyrep(ROI_PYREP_TYPE, roi_from_pyrep)


def save_rois(app_state: "ApplicationState", manifest: Dict[str, Any]) -> None:
    """Write every ROI in ``app_state`` into ``manifest['rois']``."""
    manifest["rois"] = [roi_to_pyrep(roi) for roi in app_state.get_rois()]


def load_rois(manifest: Dict[str, Any], app_state: "ApplicationState") -> List[Dict[str, Any]]:
    """Reconstruct ROIs from ``manifest['rois']`` into ``app_state``.

    Returns the entries that could not be restored (a malformed pyrep or an
    unknown type tag) so the caller can warn without aborting the load.
    """
    dropped: List[Dict[str, Any]] = []
    rois = manifest.get("rois", [])
    if not isinstance(rois, list):
        # A non-list rois section (hand-edited/corrupt manifest) has nothing to
        # restore; ignore it rather than iterating a dict's keys into add_roi.
        return dropped
    for data in rois:
        try:
            roi = roi_from_pyrep(data)
            app_state.add_roi(roi, roi_id=roi.get_id())
        except (KeyError, TypeError, ValueError, AttributeError):
            dropped.append(data)
    return dropped
