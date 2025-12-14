from typing import Optional, TYPE_CHECKING, List, Tuple

from wiser.gui.task_delegate import TaskDelegate

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from osgeo import osr

from enum import Enum

from abc import abstractmethod, ABC

if TYPE_CHECKING:
    from .rasterpane import RasterPane
    from .geo_reference_pane import GeoReferencerPane
    from .geo_reference_dialog import GeoReferencerDialog
    from .app_state import ApplicationState

PIXEL_OFFSET = 1

ZOOMED_IN_RADIUS = 6.0
ZOOMED_OUT_RADIUS = 3.0


# This class is mainly for the developer
# to keep track of the states in this delegate
class GeoReferencerState(Enum):
    NOTHING_SELECTED = "NOTHING_SELECTED"
    FIRST_POINT_SELECTED = "FIRST_POINT_SELECTED"
    FIRST_POINT_ENTERED = "FIRST_POINT_ENTERED"
    SECOND_POINT_SELECTED = "SECOND_POINT_SELECTED"
    SECOND_POINT_ENTERED = "SECOND_POINT_ENTERED"


class PointSelectorType(Enum):
    TARGET_POINT_SELECTOR = 0
    REFERENCE_POINT_SELECTOR = 1


class PointSelector:
    """
    This class is used to get the type of view a clickable object is. Currently
    this is just meant for a view where there is a target clickable object
    and a reference clickable object like the georeferencer.
    """

    def get_point_selector_type(self) -> PointSelectorType:
        """
        Every subclass must provide the point selector type which is either
        target or reference
        """
        raise NotImplementedError("get_point_selector_type is not implemented!")


class GroundControlPoint(ABC):
    """
    This class contains the minimum amount information that a
    ground control point should have. Note that we don't require
    the ground control point to have a raster coordinate. This is
    due to the fact that a ground control point in real life is
    a spatial coordinate, so we instead subclass this to represent
    a ground control point with a raster coordinate by creating
    the class `GroundControlPointRasterPane`.
    """

    def get_spatial_reference_system(self) -> Optional[osr.SpatialReference]:
        raise NotImplementedError("get_spatial_reference_system is not implemented!")

    def get_spatial_point(self) -> Tuple[int, int]:
        raise NotImplementedError("get_spatial_point is not implemented!")

    def set_spatial_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        raise NotImplementedError("get_spatial_point is not implemented!")

    @abstractmethod
    def get_selector_type(self):
        raise NotImplementedError("get_spatial_point is not implemented!")

    @abstractmethod
    def get_selector(self):
        raise NotImplementedError("get_spatial_point is not implemented!")


class GroundControlPointRasterPane(GroundControlPoint):
    """
    This class gives the implementation for a ground control point
    created by clicking on a raster pane.
    """

    def __init__(self, point: Tuple[int, int], rasterpane: "GeoReferencerPane"):
        self._point = point  # The raster coordiante that was clicked
        self._dataset = rasterpane.get_rasterview().get_raster_data()
        self._rasterpane = rasterpane

    def set_point(self, point: Tuple[int, int]):
        self._point = point

    def get_point(self):
        return self._point

    def get_scaled_point(self) -> Tuple[int, int]:
        """
        Scales the current point by the scale of the rasterpane.

        Since this function is used to get a point to display a drawn point,
        we remove 0.5 here to recenter the rectangle.
        """
        scale = self._rasterpane.get_scale()
        return [(self._point[0]) * scale, (self._point[1]) * scale]

    def get_rasterpane(self) -> "RasterPane":
        return self._rasterpane

    def get_spatial_reference_system(self):
        return self._dataset.get_spatial_ref()

    def get_spatial_point(self):
        return self._dataset.to_geographic_coords(self._point)

    def set_spatial_point(self, spatial_point: Tuple[int, int]):
        raster_coord = self._dataset.geo_to_pixel_coords_exact(spatial_point)
        if raster_coord is None:
            return
        self._point = raster_coord

    def get_selector(self):
        return self._rasterpane

    def get_selector_type(self):
        return self._rasterpane.get_point_selector_type()


class GroundControlPointCoordinate(GroundControlPoint):
    """
    This class gives the concrete implementation for a purely spatial ground control point.
    """

    def __init__(
        self,
        spatial_coordinate: Tuple[int, int],
        selector_type: PointSelectorType,
        srs: osr.SpatialReference,
    ):
        self._selector_type = selector_type
        self._spatial_coord = spatial_coordinate
        self._srs = srs

    def set_spatial_ref(self, new_srs: osr.SpatialReference):
        self._srs = new_srs

    def set_spatial_point(self, new_spatial_coord: Tuple[int, int]):
        self._spatial_coord = new_spatial_coord

    def get_spatial_reference_system(self):
        return self._srs

    def get_spatial_point(self):
        return self._spatial_coord

    def get_selector(self):
        return self

    def get_selector_type(self):
        return self._selector_type


class GroundControlPointPair:
    """
    This class contains two GroundControlPoints from different
    rasterpanes (reference and target) that represent the same spatial
    point. Because a GroundControlPoint has a selector, we can separate
    out which of these ground control points belongs to the target pane
    and the reference pane. This class helps us do so.
    """

    def __init__(self, gcp_0: GroundControlPoint = None, gcp_1: GroundControlPoint = None):
        self._target_gcp: GroundControlPoint = None
        self._ref_gcp: GroundControlPoint = None
        if gcp_0 is not None:
            self.add_gcp(gcp_0)
        if gcp_1 is not None:
            self.add_gcp(gcp_1)

    def add_gcp(self, gcp: GroundControlPoint):
        """
        Adds the passed in GroundControlPoint by checking what its selector is
        and setting it to either the target ground control point or the reference
        ground control point.
        """
        selector_type = gcp.get_selector_type()
        if selector_type == PointSelectorType.TARGET_POINT_SELECTOR:
            self._target_gcp = gcp
        elif selector_type == PointSelectorType.REFERENCE_POINT_SELECTOR:
            self._ref_gcp = gcp
        else:
            raise ValueError(
                "The GCP passed into GroundControlPointPair "
                + f"does not have a selector type. It's type is {selector_type}"
            )

    def get_gcp(self, selector_type: PointSelectorType):
        """
        Gets the gcp that has the raster pane from the selector type
        (so either the target or reference raster pane).
        """
        if selector_type == PointSelectorType.TARGET_POINT_SELECTOR:
            return self._target_gcp
        elif selector_type == PointSelectorType.REFERENCE_POINT_SELECTOR:
            return self._ref_gcp
        else:
            raise ValueError("This GroundControlPointPair does not have a GCP" + "with a matching rasterpane")

    def remove_gcp(self, selector_type: PointSelectorType):
        """
        Gets the gcp that has the equivalent rasterpane
        """
        if selector_type == PointSelectorType.TARGET_POINT_SELECTOR:
            self._target_gcp = None
        elif selector_type == PointSelectorType.REFERENCE_POINT_SELECTOR:
            self._ref_gcp = None
        else:
            raise ValueError("This GroundControlPointPair does not have a GCP" + "with a matching rasterpane")

    def has_only_one_gcp(self):
        return (self._target_gcp is not None and self._ref_gcp is None) or (
            self._target_gcp is None and self._ref_gcp is not None
        )

    def has_both_gcps(self):
        return self._target_gcp is not None and self._ref_gcp is not None

    def get_target_gcp(self):
        return self._target_gcp

    def get_reference_gcp(self):
        return self._ref_gcp

    def get_reference_gcp_spatial_coord(self):
        return self._ref_gcp.get_spatial_point()


class GeoReferencerTaskDelegate(TaskDelegate):
    def __init__(
        self,
        target_rasterpane: "GeoReferencerPane",
        ref_rasterpane: "GeoReferencerPane",
        geo_reference_dialog: "GeoReferencerDialog",
        app_state: "ApplicationState",
    ):
        self._target_rasterpane = target_rasterpane
        self._ref_rasterpane = ref_rasterpane
        self._geo_ref_dialog = geo_reference_dialog
        self._geo_ref_dialog.gcp_add_attempt.connect(self._on_gcp_add_attempt)
        self._app_state = app_state

        self._state: GeoReferencerState = GeoReferencerState.NOTHING_SELECTED
        self._last_selector_type: Optional[PointSelectorType] = None
        self._current_selector_type: Optional[PointSelectorType] = None
        self._current_point: Optional[GroundControlPoint] = None
        self._current_point_pair: Optional[GroundControlPointPair] = None

    def draw_state(self, painter: QPainter, rasterpane: "GeoReferencerPane"):
        # There's nothing to draw
        if self._geo_ref_dialog.get_gcp_table_size() == 0 and self._current_point_pair is None:
            return

        color = self._app_state.get_config("raster.selection.edit_outline")
        painter.setPen(QPen(color))

        scale = rasterpane.get_scale()

        table_entries = self._geo_ref_dialog.get_table_entries()

        # We draw all of the previous entered gcp pairs
        for entry in table_entries:
            enabled = entry.is_enabled()
            if not enabled:
                continue
            point_pair: GroundControlPointPair = entry.get_gcp_pair()
            gcp_to_draw = point_pair.get_gcp(rasterpane.get_point_selector_type())
            if not isinstance(gcp_to_draw, GroundControlPointRasterPane):
                continue
            gcp_scaled = gcp_to_draw.get_scaled_point()

            color = QColor(entry.get_color())
            painter.setBrush(color)
            painter.setPen(QPen(color))

            assert gcp_scaled is not None, "Got a GCP scaled point as None!"

            if scale >= 6:
                painter.drawEllipse(
                    gcp_scaled[0] - ZOOMED_IN_RADIUS / 2,
                    gcp_scaled[1] - ZOOMED_IN_RADIUS / 2,
                    ZOOMED_IN_RADIUS,
                    ZOOMED_IN_RADIUS,
                )
            else:
                painter.drawEllipse(
                    gcp_scaled[0] - ZOOMED_OUT_RADIUS / 2,
                    gcp_scaled[1] - ZOOMED_OUT_RADIUS / 2,
                    ZOOMED_OUT_RADIUS,
                    ZOOMED_OUT_RADIUS,
                )

        # Now we draw the currently selected gcp pair
        if self._current_point_pair is None:
            return

        curr_gcp_to_draw = self._current_point_pair.get_gcp(rasterpane.get_point_selector_type())
        if not isinstance(curr_gcp_to_draw, GroundControlPointRasterPane):
            return

        current_point_scaled = curr_gcp_to_draw.get_scaled_point()

        # Draw different sizes depending on how zoomed in we are
        if current_point_scaled is not None:
            painter.setPen(QPen(Qt.red))
            painter.setBrush(Qt.red)
            if scale >= 6:
                painter.drawEllipse(
                    current_point_scaled[0] - ZOOMED_IN_RADIUS / 2,
                    current_point_scaled[1] - ZOOMED_IN_RADIUS / 2,
                    ZOOMED_IN_RADIUS,
                    ZOOMED_IN_RADIUS,
                )
            else:
                painter.drawEllipse(
                    current_point_scaled[0] - ZOOMED_OUT_RADIUS / 2,
                    current_point_scaled[1] - ZOOMED_OUT_RADIUS / 2,
                    ZOOMED_OUT_RADIUS,
                    ZOOMED_OUT_RADIUS,
                )

    def _on_gcp_add_attempt(self, gcp: GroundControlPoint):
        """
        This function is currently used to add a reference gcp from
        the manual way of adding reference gcps.
        """
        if (
            self._state == GeoReferencerState.NOTHING_SELECTED
            or self._state == GeoReferencerState.FIRST_POINT_SELECTED
        ):
            # We want to make sure the state is correct before we continue
            self.check_state()
            self._last_selector_type = gcp.get_selector_type()
            self._current_point = gcp
            self._current_point_pair = self.create_gcp_pair(gcp)
            self._current_selector_type = None
            self._current_point = None
            self._state = GeoReferencerState.FIRST_POINT_ENTERED
            self.check_state()
        elif (
            self._state == GeoReferencerState.FIRST_POINT_ENTERED
            or self._state == GeoReferencerState.SECOND_POINT_SELECTED
        ):
            self.check_state()
            if gcp.get_selector_type() != self._last_selector_type:
                self._current_point = gcp
                self._current_point_pair.add_gcp(gcp)
                self._geo_ref_dialog.gcp_pair_added.emit(self._current_point_pair)
                # Reset all the values
                self._reset_changeable_state()
                # This line is here just for the developers clarity
                self._state = GeoReferencerState.SECOND_POINT_ENTERED
                self._state = GeoReferencerState.NOTHING_SELECTED
                self.check_state()
            else:
                self._geo_ref_dialog.set_message_text(
                    "You already pressed enter. Please select a point in the other area. "
                )
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_key_release for the enter key")

    def on_mouse_release(self, mouse_event: QMouseEvent, rasterpane: "GeoReferencerPane"):
        """
        When the mouse releases we want to add that point as a
        ground control point and properly pain it.
        """
        if mouse_event.button() == Qt.LeftButton:
            point: QPointF = rasterpane.get_rasterview().image_coord_to_raster_coord_precise(
                mouse_event.localPos()
            )
            point = [point.x(), point.y()]
        else:
            return False

        self.handle_point_click_logic(point, rasterpane)

        return False

    def handle_point_click_logic(self, point: Tuple[int, int], rasterpane: "GeoReferencerPane"):
        """
        This function correctly updates the instances state based on what
        point was just clicked and what state the georeferencer is in.
        """
        gcp = GroundControlPointRasterPane(point, rasterpane)
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self.check_state()
            self._current_selector_type = gcp.get_selector_type()
            self._current_point = gcp
            self._current_point_pair = self.create_gcp_pair(gcp)
            self._state = GeoReferencerState.FIRST_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            self.check_state()
            # If we just clicked on the other raster pane, we have to tell
            # the user to press enter first
            if self._current_selector_type != gcp.get_selector_type():
                self._current_selector_type = gcp.get_selector_type()
            self._current_point = gcp
            self._current_point_pair = self.create_gcp_pair(self._current_point)
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self.check_state()
            # If they click on the pane that they just clicked on, we tell them
            # to press ESC to revert their choice
            if self._last_selector_type == gcp.get_selector_type():
                self._geo_ref_dialog.set_message_text(
                    "You already pressed Enter. Please press ESC to remove previous choice"
                )
            else:
                # If the panes are not equal, we want to transition to second point selected state
                self._current_selector_type = gcp.get_selector_type()
                self._current_point = gcp
                self._current_point_pair.add_gcp(self._current_point)
                self._state = GeoReferencerState.SECOND_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            # If we just clicked on the other raster pane, we have to tell the user to press enter
            # on the current raster pane to finish the current pair of gcps
            if self._current_selector_type != gcp.get_selector_type():
                self._geo_ref_dialog.set_message_text(
                    "Please press ENTER before clicking on the next raster pane."
                )
            else:
                # If the pane's are not equal, we can safely select the new point
                self._current_point = gcp
                self._current_point_pair.add_gcp(self._current_point)
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_mouse_release")

    def on_key_release(self, key_event) -> bool:
        if key_event.key() == Qt.Key_Enter or key_event.key() == Qt.Key_Return:
            self.handle_enter_key_release()
        elif key_event.key() == Qt.Key_Escape:
            self.handle_escape_key_release()
        return False

    def handle_escape_key_release(self) -> bool:
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self._geo_ref_dialog.set_message_text("Must select a point before pressing ESC again")
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            # Pressing escape here should bring us back to NOTHING_SELECTED state
            self.check_state()
            self._reset_changeable_state()
            self._state = GeoReferencerState.NOTHING_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self.check_state()
            self._current_selector_type = self._last_selector_type
            self._last_selector_type = None
            self._current_point = self._current_point_pair.get_gcp(self._current_selector_type)
            self._state = GeoReferencerState.FIRST_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            self._current_point_pair.remove_gcp(self._current_selector_type)
            self._current_selector_type = None
            self._current_point = None
            self._state = GeoReferencerState.FIRST_POINT_ENTERED
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED
            # should immediately go back to the NOTHING_SELECTED state
            raise ValueError(
                f"The state {self._state} was arrived at in on_key_release" + "for the escape key"
            )

    def handle_enter_key_release(self):
        # We have two options here:
        #   1. We have yet to select a new point, so we do nothing.
        #   2. We have selected a point, so we add the point.
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self._geo_ref_dialog.set_message_text("Must select a point before pressing ENTER again")
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            # We want to make sure the state is correct before we continue
            self.check_state()
            self._last_selector_type = self._current_selector_type
            self._current_selector_type = None
            self._current_point = None
            self._state = GeoReferencerState.FIRST_POINT_ENTERED
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self._geo_ref_dialog.set_message_text("Must select a second point before pressing ENTER again")
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            self._geo_ref_dialog.gcp_pair_added.emit(self._current_point_pair)
            # Reset all the values
            self._reset_changeable_state()
            # This line is here just for the developers clarity
            self._state = GeoReferencerState.SECOND_POINT_ENTERED
            self._state = GeoReferencerState.NOTHING_SELECTED
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_key_release for the enter key")

    def create_gcp_pair(self, gcp_0=None, gcp_1=None) -> GroundControlPointPair:
        return GroundControlPointPair(gcp_0=gcp_0, gcp_1=gcp_1)

    def get_current_point_pair(self):
        return self._current_point_pair

    def _reset_changeable_state(self):
        self._last_selector_type = None
        self._current_selector_type = None
        self._current_point = None
        self._current_point_pair = None

    def check_state(self):
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            assert (
                self._current_point is None
            ), f"self._current_point is not None when Geo Reference is in state {self._state}"
            assert (
                self._current_point_pair is None
            ), f"self._current_point_pair is not None when Geo Reference is in state {self._state}"
            assert (
                self._current_selector_type is None
            ), f"self._current_selector_type is not None when Geo Reference is in state {self._state}"
            assert (
                self._last_selector_type is None
            ), f"self._last_selector_type is not None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            assert (
                self._current_point is not None
            ), f"self._current_point is None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is not None and self._current_point_pair.has_only_one_gcp(), (
                f"self._current_point_pair is incorrect ({self._current_point_pair})"
                + f"when Geo Reference is in state {self._state}"
            )
            assert (
                self._current_selector_type is not None
            ), f"self._current_selector_type is None when Geo Reference is in state {self._state}"
            assert (
                self._last_selector_type is None
            ), f"self._last_selector_type is not None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            assert (
                self._current_point is None
            ), f"self._current_point is not None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is not None and self._current_point_pair.has_only_one_gcp(), (
                f"self._current_point_pair is incorrect ({self._current_point_pair})"
                + f"when Geo Reference is in state {self._state}"
            )
            assert (
                self._current_selector_type is None
            ), f"self._current_selector_type is not None when Geo Reference is in state {self._state}"
            assert (
                self._last_selector_type is not None
            ), f"self._last_selector_type is None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            assert (
                self._current_point is not None
            ), f"self._current_point is None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is not None and self._current_point_pair.has_both_gcps(), (
                f"self._current_point_pair is incorrect ({self._current_point_pair})"
                + f"when Geo Reference is in state {self._state}"
            )
            assert (
                self._current_selector_type is not None
            ), f"self._current_selector_type is None when Geo Reference is in state {self._state}"
            assert (
                self._last_selector_type is not None
            ), f"self._last_selector_type is None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.SECOND_POINT_ENTERED:
            # Note that this is the same as GeoReferencerState.NOTHING_SELECTED
            assert (
                self._current_point is None
            ), f"self._current_point is not None when Geo Reference is in state {self._state}"
            assert (
                self._current_point_pair is None
            ), f"self._current_point_pair is not None when Geo Reference is in state {self._state}"
            assert (
                self._current_selector_type is None
            ), f"self._current_selector_type is not None when Geo Reference is in state {self._state}"
            assert (
                self._last_selector_type is None
            ), f"self._last_selector_type is not None when Geo Reference is in state {self._state}"
