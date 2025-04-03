from typing import Optional, TYPE_CHECKING, List, Tuple

from .rasterview import RasterView
from wiser.raster.dataset import RasterDataSet
from wiser.gui.geo_reference_dialog import GeoReferencerDialog

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from enum import Enum

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
    def __init__(self, point: List[int, int], dataset: RasterDataSet):
        self._point = point
        self._dataset = dataset

class GeoReferencerTaskDelegate:
    # Take in clickable_object_1 and clickable_object_2

    # When clickable object 1 receives a click, it should send it to task delegate.
    # the next click task delegate does should be in clickable object 2.

    # When 2 things are clicked and the user presses enter, it adds them to a list 
    # and emits a signal that the GeoReferenceDialog conects to to get that list 

    # It populates the GCP list with that list. If user's edit the GCP list, it updates
    # the list here
    def __init__(self, rasterpane_1: RasterPane, \
                 rasterpane_2: RasterPane, \
                 geo_reference_dialog: GeoReferencerDialog):
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
        self._state: GeoReferencerState = GeoReferencerState.NOTHING_SELECTED
        self._last_selected_pane: Optional[RasterPane] = None
        self._current_selected_pane: Optional[RasterPane] = None
        self._current_point: Optional[GroundControlPoint] = None
        self._current_point_pair = Optional[List[GroundControlPoint, GroundControlPoint]] = None
        self._point_list: List[List[GroundControlPoint, GroundControlPoint]] = []
    
    def on_mouse_release(self, point: List[int, int], rasterpane: RasterPane):
        # We want the user to be able to press escape and clear the currently selected raster pane 
        if self._state == GeoReferencerState.NOTHING_SELECTED:
            self.check_state()
            self._current_selected_pane = rasterpane
            self._current_point = GroundControlPoint(point, rasterpane.get_rasterview().get_raster_data())
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
                self._current_point = GroundControlPoint(point, rasterpane.get_rasterview().get_raster_data())
                self._current_point_pair = [self._current_point,\
                                            None]
            else:
                self._current_point = GroundControlPoint(point, rasterpane.get_rasterview().get_raster_data())
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
                self._current_point = GroundControlPoint(point, rasterpane.get_rasterview().get_raster_data())
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
                self._current_point = GroundControlPoint(point, rasterpane.get_rasterview().get_raster_data())
                self._current_point_pair[1] = self._current_point
            self.check_state()
        else:
            # We should never reach this point because the state SECOND_POINT_ENTERED should
            # immediately go back to the NOTHING_SELECTED state
            raise ValueError(f"The state {self._state} was arrived at in on_mouse_release")


    
    def on_key_release(self, key_event):
        if key_event == Qt.Key_Enter:
            self.handle_enter_key_release()
        elif key_event == Qt.Key_Escape:
            self.handle_escape_key_release()

    def handle_escape_key_release(self):
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

