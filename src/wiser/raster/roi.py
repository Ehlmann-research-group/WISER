from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .selection import (
    Selection,
    RectangleSelection,
    PolygonSelection,
    MultiPixelSelection,
    selection_from_pyrep,
    SelectionType,
)

from PySide2.QtCore import QRect
from PySide2.QtGui import QColor

if TYPE_CHECKING:
    from .dataset import RasterDataSet


class RegionOfInterest:
    """
    Represents a Region of Interest (abbreviated "ROI") in the data being
    analyzed.  The Region of Interest may specify multiple selections of various
    types, indicating the actual area comprising the ROI.  Various other
    attributes may be specified as well, such as the color that the ROI is drawn
    in.
    """

    def __init__(self, name: Optional[str] = None, color: str = "yellow"):
        self._id: Optional[int] = None
        self._name: Optional[str] = name
        self._color: str = color
        self._description: Optional[str] = None
        self._selections: List[Selection] = []
        self._metadata: Dict[str, Any] = {}

    def get_id(self) -> Optional[int]:
        return self._id

    def set_id(self, id: int) -> None:
        self._id = id

    def __str__(self):
        return f"ROI[{self._name}, {self._selections}]"

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def get_color(self) -> str:
        """
        Returns the color of the ROI as a string.
        """
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color

    def get_description(self) -> Optional[str]:
        return self._description

    def set_description(self, description: Optional[str]) -> None:
        self._description = description

    def get_selections(self) -> List[Selection]:
        return list(self._selections)

    def add_selection(self, selection: Selection) -> None:
        if selection is None:
            raise ValueError("selection cannot be None")

        self._selections.append(selection)

    def del_selection(self, sel_index: int) -> None:
        del self._selections[sel_index]

    def get_metadata(self):
        return self._metadata

    def get_all_pixels(self) -> Set[Tuple[int, int]]:
        """
        Return a Python set containing the coordinates of all pixels that are a
        part of this Region of Interest.  Each pixel coordinate will only appear
        once, even if the pixel appears within multiple selections in the ROI.
        """
        all_pixels = set()
        for sel in self._selections:
            all_pixels.update(sel.get_all_pixels())

        return all_pixels

    def get_pixel_data(self, dataset: "RasterDataSet") -> np.ndarray:
        """
        Return a 2-D numpy array of shape ``(N, b)`` containing the spectral
        data for every deduplicated pixel in this ROI.

        ``N`` is the number of unique pixels across all selections and ``b`` is
        the number of bands in *dataset*.  Pixels are in an arbitrary but
        deterministic order (sorted by (y, x)).

        The full image cube is read once and then indexed in a vectorised
        fashion, so this is efficient even for large ROIs.

        Parameters
        ----------
        dataset:
            The raster dataset to read spectral values from.  Pixel
            coordinates in the ROI's selections are assumed to refer to this
            dataset's pixel grid.
        """
        pixels = self.get_all_pixels()  # Set[Tuple[x, y]]
        if not pixels:
            b = dataset.get_image_data().shape[0]
            return np.empty((0, b), dtype=np.float64)

        # cube shape: (b, H, W), indexed as cube[band, row, col]
        cube = dataset.get_image_data()

        # Sort for a stable, reproducible row order: primary key y, secondary x
        sorted_pixels = sorted(pixels, key=lambda p: (p[1], p[0]))
        xs = np.array([p[0] for p in sorted_pixels], dtype=np.intp)
        ys = np.array([p[1] for p in sorted_pixels], dtype=np.intp)

        # cube[:, ys, xs] → shape (b, N); transpose to (N, b)
        return np.asarray(cube[:, ys, xs]).T

    def get_bounding_box(self) -> QRect:
        all_pixels = self.get_all_pixels()
        xs = [p[0] for p in all_pixels]
        ys = [p[1] for p in all_pixels]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        return QRect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    def pprint(self):
        print(f"RegionOfInterest[{self._name}, {self._color}, {self._description}")
        for sel in self._selections:
            print(f" * {sel}")
        print("]")


def roi_to_pyrep(roi):
    data = {
        "name": roi.get_name(),
        "color": str(roi.get_color().name()),
        "metadata": roi.get_metadata(),
    }
    # TODO(donnie):  Composite/multi-selection ROIs
    data["selection"] = roi.get_selection().to_pyrep()

    return data


def roi_from_pyrep(data):
    name = data["name"]
    color = QColor(data["color"])
    metadata = data["metadata"]
    # TODO(donnie):  Composite/multi-selection ROIs
    sel = selection_from_pyrep(data["selection"])

    roi = RegionOfInterest(name, sel, color, **metadata)
    return roi
