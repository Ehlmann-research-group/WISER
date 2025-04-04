from typing import Optional, TYPE_CHECKING, List, Tuple

from .rasterview import RasterView
from wiser.raster.dataset import RasterDataSet
from wiser.gui.task_delegate import TaskDelegate
from wiser.gui.util import scale_qpoint_by_float
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from enum import Enum

PIXEL_OFFSET = 1

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
    def __init__(self, point: Tuple[int, int], dataset: RasterDataSet, rasterpane: 'RasterPane'):
        self._point = point
        self._dataset = dataset
        self._rasterpane = rasterpane
    
    def get_scaled_point(self) -> Tuple[int, int]:
        scale = self._rasterpane.get_scale()
        return [self._point[0]*scale, self._point[1]*scale]

    def get_rasterpane(self) -> 'RasterPane':
        return self._rasterpane

class GeoReferencerTaskDelegate(TaskDelegate):
    # Take in clickable_object_1 and clickable_object_2

    # When clickable object 1 receives a click, it should send it to task delegate.
    # the next click task delegate does should be in clickable object 2.

    # When 2 things are clicked and the user presses enter, it adds them to a list 
    # and emits a signal that the GeoReferenceDialog conects to to get that list 

    # It populates the GCP list with that list. If user's edit the GCP list, it updates
    # the list here
    def __init__(self, rasterpane_1: 'RasterPane', \
                 rasterpane_2: 'RasterPane', \
                 geo_reference_dialog: 'GeoReferencerDialog',
                 app_state: 'ApplicationState'):
        '''
        Technically, rasterpane_1 and rasterpane_2 can be any objects that
        can receive mouse events and when they do receive mouse events,
        will send it here.
        '''
        # So we have the last selected pane and the current selected pane. 
        # The last selected pane will remain none until the user presses enter.
        # Then the last selected raster pane updates to the currently selected raster pane
        # (the currently selected raster is always the one passed in and is filtered) by ensuring
        # its not equal to the last selected raster pane. If the user presses enter the second time and the
        # current point pair is two, we add it to point list. 

        '''
        Three states:
        1. State one. The user has yet to select anything. 
            self._last_selected_pane = None
            self._current_selected_pane = None
            self._current_point = None
            self._current_point_pair = None
        2. State two. The user has selected a pixel in a raster pane
            self._last_selected_pane = None
            self._current_selected_pane = selected_pane_1
            self._current_point = selected_point_1
            self._current_point_pair = [GCP, None]
        3. State three. The user has pressed enter
            self._last_selected_pane = selected_pane_1
            self._current_selected_pane = None
            self._current_point = None
            self._current_point_pair = [GCP, None]
        4. State four. The user has selected a pixel in other raster pane
            self._last_selected_pane = selected_pane_1
            self._current_selected_pane = pane_2
            self._current_point = selected_point_2
            self._current_point_pair = [GCP, GCP]
        5. State five. The user has pressed enter
            self._last_selected_pane = None
            self._current_selected_pane = None
            self._current_point = None
            self._current_point_pair = None

        '''
        self._rasterpane_1 = rasterpane_1
        self._rasterpane_2 = rasterpane_2
        self._geo_ref_dialog = geo_reference_dialog
        self._app_state = app_state

        self._state: GeoReferencerState = GeoReferencerState.NOTHING_SELECTED
        self._last_selected_pane: Optional['RasterPane'] = None
        self._current_selected_pane: Optional['RasterPane'] = None
        self._current_point: Optional[GroundControlPoint] = None
        self._current_point_pair: Optional[Tuple[GroundControlPoint, GroundControlPoint]] = None
        self._point_list: List[Tuple[GroundControlPoint, GroundControlPoint]] = []

    def draw_state(self, painter: QPainter, rasterpane: 'RasterPane'):
        if len(self._point_list) == 0 and self._current_point_pair is None:
            return
        
        color = self._app_state.get_config('raster.selection.edit_outline')
        painter.setPen(QPen(color))

        points_scaled = []

        scale = rasterpane.get_scale()

        for point_pair in self._point_list:
            gcp_0 = point_pair[0]
            gcp_1 = point_pair[1]
            if gcp_0.get_rasterpane() == rasterpane:
                points_scaled.append(gcp_0.get_scaled_point())
            elif gcp_1.get_rasterpane() == rasterpane:
                points_scaled.append(gcp_1.get_scaled_point())

        for p in points_scaled:
            if scale >= 6:
                painter.drawRect(p[0] + PIXEL_OFFSET,
                                 p[1] + PIXEL_OFFSET,
                                 scale - 2 * PIXEL_OFFSET,
                                 scale - 2 * PIXEL_OFFSET)
            else:
                painter.drawRect(p[0], p[1], scale, scale)

        gcp_0 = self._current_point_pair[0]
        gcp_1 = self._current_point_pair[1]

        current_point_scaled = None
        if gcp_0 is not None:
            if gcp_0.get_rasterpane() == rasterpane:
                current_point_scaled = gcp_0.get_scaled_point()
        if gcp_1 is not None and current_point_scaled is None:
            if gcp_1.get_rasterpane() == rasterpane:
                current_point_scaled = gcp_1.get_scaled_point()
        
        if current_point_scaled is not None:
            painter.setPen(QPen(Qt.red))
            if scale >= 6:
                painter.drawRect(current_point_scaled[0] + PIXEL_OFFSET,
                                 current_point_scaled[1] + PIXEL_OFFSET,
                                 scale - 2 * PIXEL_OFFSET,
                                 scale - 2 * PIXEL_OFFSET)
            else:
                painter.drawRect(current_point_scaled[0], current_point_scaled[1], scale, scale)
        

    def on_mouse_release(self, mouse_event: QMouseEvent, rasterpane: 'RasterPane'):
        '''
        When the mouse releases we want to click
        Args:
        - point is the point is raster coordinates
        - rasterpane is the raster pane that was clicked in
        '''

        if mouse_event.button() == Qt.LeftButton:
            point: QPointF = rasterpane.get_rasterview().image_coord_to_raster_coord(mouse_event.localPos())
            point = [point.x(), point.y()]
        else:
            return False
        # We want the user to be able to press escape and clear the currently selected raster pane 
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self.check_state()
            self._current_selected_pane = rasterpane
            self._current_point = GroundControlPoint(point,
                                                     rasterpane.get_rasterview().get_raster_data(),
                                                     rasterpane)
            self._current_point_pair = [self._current_point,\
                                        None]
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
                self._current_point_pair = [self._current_point,\
                                            None]
            else:
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data(),
                                                         rasterpane)
                self._current_point_pair = [self._current_point,\
                                            None]
            self.check_state()
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self.check_state()
            # If they click on the pane that the just clicked on, we tell them
            # to press enter to revert their choice
            if self._last_selected_pane == rasterpane:
                self._geo_ref_dialog.set_message_text('You already pressed enter.'
                'Please press ESC to remove previous choice')
            else:
                # If the pane's are not equal, we want to transition to second point selected state
                self._current_selected_pane = rasterpane
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data,
                                                         rasterpane)
                self._current_point_pair[1] = self._current_point
                self._state = GeoReferencerState.SECOND_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            # If we just clickd on the other raster view, we have to tell the user to press enter first
            if self._current_selected_pane != rasterpane:
                self._geo_ref_dialog.set_message_text('Please press ENTER before clicking'
                'on the next raster pane.')
            else:
                # If the pane's are not equal, we want to transition to second point selected state
                self._current_point = GroundControlPoint(point,
                                                         rasterpane.get_rasterview().get_raster_data(),
                                                         rasterpane)
                self._current_point_pair[1] = self._current_point
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_mouse_release")
        return False


    
    def on_key_release(self, key_event) -> bool:
        '''
        We want all of our functions to return false because this task delegate
        is permanent for the GeoReferencer Panes it operates on
        '''
        if key_event == Qt.Key_Enter:
            self.handle_enter_key_release()
        elif key_event == Qt.Key_Escape:
            self.handle_escape_key_release()
        return False

    def handle_escape_key_release(self) -> bool:
        '''
        Controls the task delegate state when the ESC key is released

        '''
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self._geo_ref_dialog.set_message_text('Must select' \
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
            self._current_point = self._current_point_pair[0]
            self._state = self._state == GeoReferencerState.FIRST_POINT_SELECTED
            self.check_state()
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            self._current_selected_pane = None
            self._current_point = None
            self._current_point_pair[1] = None
            self._state = self._state == GeoReferencerState.FIRST_POINT_ENTERED
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_key_release" + 
                                f"for the escape key")

    def handle_enter_key_release(self):
        # We have three options here:
        #   1. We have yet to select a raster pane. This does nothing
        #   2. We have selected a point. In this case 
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self._geo_ref_dialog.set_message_text('Must select' \
            'a point before pressing ENTER again')
        elif self._state == GeoReferencerState.FIRST_POINT_SELECTED:
            # We want to make sure the state is correct before we continue
            self.check_state()
            self._last_selected_pane = self._current_selected_pane
            self._current_selected_pane = None
            self._current_point = None
            self._state = GeoReferencerState.FIRST_POINT_ENTERED
        elif self._state == GeoReferencerState.FIRST_POINT_ENTERED:
            self._geo_ref_dialog.set_message_text('Must select' \
            'a second point before pressing ENTER again')
        elif self._state == GeoReferencerState.SECOND_POINT_SELECTED:
            self.check_state()
            self._point_list.append(self._current_selected_pair)
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
        self._current_selected_pair = None

    
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
                    self._current_point_pair[0] is not None and \
                    self._current_point_pair[1] is None, \
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
                    self._current_point_pair[0] is not None and \
                    self._current_point_pair[1] is None, \
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
                    self._current_point_pair[0] is not None and \
                    self._current_point_pair[1] is not None, \
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

