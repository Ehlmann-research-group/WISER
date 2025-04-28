from typing import Optional, TYPE_CHECKING, List, Tuple

from .rasterview import RasterView
from wiser.raster.dataset import RasterDataSet
from wiser.gui.task_delegate import TaskDelegate
from wiser.gui.util import scale_qpoint_by_float

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from osgeo import osr

from enum import Enum

from .ui_selection import CONTROL_POINT_SIZE

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

if TYPE_CHECKING:
    from .rasterpane import RasterPane

class GroundControlPoint:
    def get_spatial_reference_system(self) -> Optional[osr.SpatialReference]:
        raise NotImplementedError("get_spatial_reference_system is not implemented!")
    def get_spatial_point(self) -> Tuple[int, int]:
        raise NotImplementedError("get_spatial_point is not implemented!")

class GroundControlPointDataset(GroundControlPoint):
    def __init__(self, point: Tuple[int, int], dataset: RasterDataSet, rasterpane: 'RasterPane'):
        self._point = point  # The raster coordiante that was clicked
        self._dataset = dataset
        self._rasterpane = rasterpane

    def set_point(self, point: Tuple[int, int]):
        self._point = point

    def get_point(self):
        return self._point

    def get_scaled_point(self) -> Tuple[int, int]:
        '''
        Since this function is used to get a point to display a drawn point, 
        we remove 0.5 here to recenter the rectangle. 
        '''
        scale = self._rasterpane.get_scale()
        return [(self._point[0])*scale, (self._point[1])*scale]

    def get_rasterpane(self) -> 'RasterPane':
        return self._rasterpane

    def get_spatial_reference_system(self):
        return self._dataset.get_spatial_ref()

    def get_spatial_point(self):
        return self._dataset.to_geographic_coords(self._point)

class GroundControlPointCoordinate(GroundControlPoint):
    def __init__(self, spatial_coordinate: Tuple[int, int], srs: osr.SpatialReference):
        self._spatial_coord = spatial_coordinate
        self._srs = srs

    def set_spatial_ref(self, new_srs: osr.SpatialReference):
        self._srs = new_srs

    def set_spatial_coord(self, new_spatial_coord: Tuple[int, int]):
        self._spatial_coord = new_spatial_coord

    def get_spatial_reference_system(self):
        return self._srs
    
    def get_spatial_point(self):
        return self._spatial_coord

class GroundControlPointPair:
    '''
    Handles the logic of separating out the target gcp and reference gcp since the user
    can add either gcp first.
    '''
    def __init__(self, target_rasterpane: 'RasterPane', reference_rasterpane: 'RasterPane', \
                 gcp_0: GroundControlPoint = None, gcp_1: GroundControlPoint = None):
        self._target_rasterpane = target_rasterpane
        self._reference_rasterpane = reference_rasterpane
        self._target_gcp: GroundControlPoint = None
        self._ref_gcp: GroundControlPoint = None
        if gcp_0 is not None:
            self.add_gcp(gcp_0)
        if gcp_1 is not None:
            self.add_gcp(gcp_1) 

    def add_gcp(self, gcp: GroundControlPoint):
        if gcp.get_rasterpane() == self._target_rasterpane:
            self._target_gcp = gcp
        elif gcp.get_rasterpane() == self._reference_rasterpane:
            self._ref_gcp = gcp
        else:
            raise ValueError(f"The GCP passed into GroundControlPointPair " +
                             f"does not have a raster pane that matches any in the pair")

    def get_gcp(self, rasterpane: 'RasterPane'):
        '''
        Gets the gcp that has the equivalent rasterpane
        '''
        if rasterpane == self._target_rasterpane:
            return self._target_gcp
        elif rasterpane == self._reference_rasterpane:
            return self._ref_gcp
        else:
            raise ValueError(f"This GroundControlPointPair does not have a GCP" +
                             f"with a matching rasterpane")

    def remove_gcp(self, rasterpane: 'RasterPane'):
        '''
        Gets the gcp that has the equivalent rasterpane
        '''
        if rasterpane == self._target_rasterpane:
            self._target_gcp = None
        elif rasterpane == self._reference_rasterpane:
            self._ref_gcp = None
        else:
            raise ValueError(f"This GroundControlPointPair does not have a GCP" +
                             f"with a matching rasterpane")

    def has_only_one_gcp(self):
        return (self._target_gcp is not None and self._ref_gcp is None) or \
                (self._target_gcp is None and self._ref_gcp is not None)

    def has_both_gcps(self):
        return (self._target_gcp is not None and self._ref_gcp is not None)

    def get_target_gcp(self):
        return self._target_gcp
    
    def get_reference_gcp(self):
        return self._ref_gcp

    def get_reference_gcp_spatial_coord(self):
        ref_point = self._ref_gcp.get_point()
        ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
        ref_spatial_point = ref_ds.to_geographic_coords(ref_point)
        return ref_spatial_point

    def set_reference_gcp_spatially(self, spatial_coord: Tuple[int, int]):
        ref_ds = self._reference_rasterpane.get_rasterview().get_raster_data()
        raster_coord = ref_ds.geo_to_pixel_coords_exact(spatial_coord)
        if raster_coord is None:
            return
        self._ref_gcp.set_point(raster_coord)
        
class GeoReferencerTaskDelegate(TaskDelegate):
    def __init__(self, target_rasterpane: 'RasterPane', \
                 ref_rasterpane: 'RasterPane', \
                 geo_reference_dialog: 'GeoReferencerDialog',
                 app_state: 'ApplicationState'):
        self._target_rasterpane = target_rasterpane
        self._ref_rasterpane = ref_rasterpane
        self._geo_ref_dialog = geo_reference_dialog
        self._app_state = app_state

        self._state: GeoReferencerState = GeoReferencerState.NOTHING_SELECTED
        self._last_selected_pane: Optional['RasterPane'] = None
        self._current_selected_pane: Optional['RasterPane'] = None
        self._current_point: Optional[GroundControlPoint] = None
        self._current_point_pair: Optional[GroundControlPointPair] = None

    def draw_state(self, painter: QPainter, rasterpane: 'RasterPane'):
        if self._geo_ref_dialog.get_gcp_table_size() == 0 and self._current_point_pair is None:
            return
        
        color = self._app_state.get_config('raster.selection.edit_outline')
        painter.setPen(QPen(color))

        scale = rasterpane.get_scale()

        table_entries = self._geo_ref_dialog.get_table_entries()

        for entry in table_entries:
            point_pair: GroundControlPointPair = entry.get_gcp_pair()
            color = QColor(entry.get_color())
            painter.setBrush(color)
            enabled = entry.is_enabled()
            if not enabled:
                continue
            painter.setPen(QPen(color))
            gcp_0 = point_pair.get_target_gcp()
            gcp_1 = point_pair.get_reference_gcp()
            gcp_scaled = None
            if gcp_0.get_rasterpane() == rasterpane:
                gcp_scaled = gcp_0.get_scaled_point()
            elif gcp_1.get_rasterpane() == rasterpane:
                gcp_scaled = gcp_1.get_scaled_point()

            assert gcp_scaled is not None, "Got a GCP scaled point as None!"
    
            if scale >= 6:
                painter.drawEllipse(gcp_scaled[0] - ZOOMED_IN_RADIUS / 2,
                                 gcp_scaled[1] - ZOOMED_IN_RADIUS / 2, ZOOMED_IN_RADIUS, ZOOMED_IN_RADIUS)
            else:
                painter.drawEllipse(gcp_scaled[0] - ZOOMED_OUT_RADIUS / 2,
                                 gcp_scaled[1] - ZOOMED_OUT_RADIUS / 2, ZOOMED_OUT_RADIUS, ZOOMED_OUT_RADIUS)

        if self._current_point_pair is None:
            return
        gcp_0 = self._current_point_pair.get_target_gcp()
        gcp_1 = self._current_point_pair.get_reference_gcp()

        current_point_scaled = None
        if gcp_0 is not None:
            if gcp_0.get_rasterpane() == rasterpane:
                current_point_scaled = gcp_0.get_scaled_point()
        if gcp_1 is not None and current_point_scaled is None:
            if gcp_1.get_rasterpane() == rasterpane:
                current_point_scaled = gcp_1.get_scaled_point()
        
        if current_point_scaled is not None:
            painter.setPen(QPen(Qt.red))
            painter.setBrush(Qt.red)
            if scale >= 6:
                painter.drawEllipse(current_point_scaled[0] - ZOOMED_IN_RADIUS / 2,
                                 current_point_scaled[1] - ZOOMED_IN_RADIUS / 2, ZOOMED_IN_RADIUS, ZOOMED_IN_RADIUS)
            else:
                painter.drawEllipse(current_point_scaled[0] - ZOOMED_OUT_RADIUS / 2,
                                 current_point_scaled[1] - ZOOMED_OUT_RADIUS / 2, ZOOMED_OUT_RADIUS, ZOOMED_OUT_RADIUS)
        

    def on_mouse_release(self, mouse_event: QMouseEvent, rasterpane: 'RasterPane'):
        '''
        When the mouse releases we want to click
        Args:
        - point is the point is raster coordinates
        - rasterpane is the raster pane that was clicked in
        '''

        if mouse_event.button() == Qt.LeftButton:
            point: QPointF = rasterpane.get_rasterview().image_coord_to_raster_coord_precise(mouse_event.localPos())
            point = [point.x(), point.y()]
        else:
            return False

        self.handle_point_click_logic(point, rasterpane)

        return False

    def handle_point_click_logic(self, point: Tuple[int, int], rasterpane: 'RasterPane'):
                # We want the user to be able to press escape and clear the currently selected raster pane 
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self.check_state()
            self._current_selected_pane = rasterpane
            self._current_point = GroundControlPoint(point,
                                                     rasterpane.get_rasterview().get_raster_data(),
                                                     rasterpane)
            self._current_point_pair = self.create_gcp_pair(self._current_point)
            self._state = GeoReferencerState.FIRST_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            self.check_state()
            # If we just clickd on the other raster view, we have to tell the user to press enter first
            if self._current_selected_pane != rasterpane:
                # self._geo_ref_dialog.set_message_text('Please press ENTER before clicking'
                # 'on the next raster pane.')
                self._current_selected_pane = rasterpane
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data(),
                                                         rasterpane)
                self._current_point_pair = self.create_gcp_pair(self._current_point)
            else:
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data(),
                                                         rasterpane)
                self._current_point_pair = self.create_gcp_pair(self._current_point)
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self.check_state()
            # If they click on the pane that the just clicked on, we tell them
            # to press enter to revert their choice
            if self._last_selected_pane == rasterpane:
                self._geo_ref_dialog.set_message_text('You already pressed enter. '
                'Please press ESC to remove previous choice')
            else:
                # If the pane's are not equal, we want to transition to second point selected state
                self._current_selected_pane = rasterpane
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data(),
                                                         rasterpane)
                self._current_point_pair.add_gcp(self._current_point)
                self._state = GeoReferencerState.SECOND_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            # If we just clickd on the other raster view, we have to tell the user to press enter first
            if self._current_selected_pane != rasterpane:
                self._geo_ref_dialog.set_message_text('Please press ENTER before clicking '
                'on the next raster pane.')
            else:
                # If the pane's are not equal, we want to transition to second point selected state
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data(),
                                                         rasterpane)
                self._current_point_pair.add_gcp(self._current_point)
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_mouse_release")

    def create_gcp_pair(self, gcp_0 = None, gcp_1 = None) -> GroundControlPointPair:
        return GroundControlPointPair(self._target_rasterpane, self._ref_rasterpane, gcp_0=gcp_0, gcp_1=gcp_1)
    
    def on_key_release(self, key_event) -> bool:
        '''
        We want all of our functions to return false because this task delegate
        is permanent for the GeoReferencer Panes it operates on
        '''
        if key_event.key() == Qt.Key_Enter or key_event.key() == Qt.Key_Return:
            self.handle_enter_key_release()
        elif key_event.key() == Qt.Key_Escape:
            self.handle_escape_key_release()
        return False

    def handle_escape_key_release(self) -> bool:
        '''
        Controls the task delegate state when the ESC key is released

        '''
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self._geo_ref_dialog.set_message_text('Must select ' \
            'a point before pressing ESC again')
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            # Pressing escape here should bring us back to NOTHING_SELECTED
            # state
            self.check_state()
            self._reset_changeable_state()
            self._state = GeoReferencerState.NOTHING_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self.check_state()
            self._current_selected_pane = self._last_selected_pane
            self._last_selected_pane = None
            self._current_point = self._current_point_pair.get_gcp(self._current_selected_pane)
            self._state = GeoReferencerState.FIRST_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            self._current_point_pair.remove_gcp(self._current_selected_pane)
            self._current_selected_pane = None
            self._current_point = None
            self._state = GeoReferencerState.FIRST_POINT_ENTERED
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_key_release" + 
                                f"for the escape key")

    def get_current_point_pair(self):
        return self._current_point_pair

    def handle_enter_key_release(self):
        # We have three options here:
        #   1. We have yet to select a raster pane. This does nothing
        #   2. We have selected a point. In this case 
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self._geo_ref_dialog.set_message_text('Must select ' \
            'a point before pressing ENTER again')
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            # We want to make sure the state is correct before we continue
            self.check_state()
            self._last_selected_pane = self._current_selected_pane
            self._current_selected_pane = None
            self._current_point = None
            self._state = GeoReferencerState.FIRST_POINT_ENTERED
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self._geo_ref_dialog.set_message_text('Must select ' \
            'a second point before pressing ENTER again')
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
            raise ValueError(f"The state {self._state} was arrived at in on_key_release" + 
                                f"for the enter key")

    def _reset_changeable_state(self):
        self._last_selected_pane = None
        self._current_selected_pane = None
        self._current_point = None
        self._current_point_pair = None

    
    def check_state(self):
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            assert self._current_point is None, \
                    f"self._current_point is not None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is None, \
                    f"self._current_point_pair is not None when Geo Reference is in state {self._state}"
            assert self._current_selected_pane is None, \
                    f"self._current_selected_pane is not None when Geo Reference is in state {self._state}"
            assert self._last_selected_pane is None, \
                    f"self._last_selected_pane is not None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            assert self._current_point is not None, \
                    f"self._current_point is None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is not None and \
                    self._current_point_pair.has_only_one_gcp(), \
                    f"self._current_point_pair is incorrect ({self._current_point_pair})" + \
                    f"when Geo Reference is in state {self._state}"
            assert self._current_selected_pane is not None, \
                    f"self._current_selected_pane is None when Geo Reference is in state {self._state}"
            assert self._last_selected_pane is None, \
                    f"self._last_selected_pane is not None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            assert self._current_point is None, \
                    f"self._current_point is not None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is not None and \
                    self._current_point_pair.has_only_one_gcp(), \
                    f"self._current_point_pair is incorrect ({self._current_point_pair})" + \
                    f"when Geo Reference is in state {self._state}"
            assert self._current_selected_pane is None, \
                    f"self._current_selected_pane is not None when Geo Reference is in state {self._state}"
            assert self._last_selected_pane is not None, \
                    f"self._last_selected_pane is None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            assert self._current_point is not None, \
                    f"self._current_point is None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is not None and \
                    self._current_point_pair.has_both_gcps(), \
                    f"self._current_point_pair is incorrect ({self._current_point_pair})" + \
                    f"when Geo Reference is in state {self._state}"
            assert self._current_selected_pane is not None, \
                    f"self._current_selected_pane is None when Geo Reference is in state {self._state}"
            assert self._last_selected_pane is not None, \
                    f"self._last_selected_pane is None when Geo Reference is in state {self._state}"
        elif self._state == GeoReferencerState.SECOND_POINT_ENTERED:
            # Note that this is the same as GeoReferencerState.NOTHING_SELECTED
            assert self._current_point is None, \
                    f"self._current_point is not None when Geo Reference is in state {self._state}"
            assert self._current_point_pair is None, \
                    f"self._current_point_pair is not None when Geo Reference is in state {self._state}"
            assert self._current_selected_pane is None, \
                    f"self._current_selected_pane is not None when Geo Reference is in state {self._state}"
            assert self._last_selected_pane is None, \
                    f"self._last_selected_pane is not None when Geo Reference is in state {self._state}"
