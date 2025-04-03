from typing import Optional, TYPE_CHECKING, List, Tuple

from .rasterview import RasterView
from wiser.raster.dataset import RasterDataSet

if TYPE_CHECKING:
    from .rasterpane import RasterPane

class GroundControlPoint:
    def __init__(self, point: List[int, int], dataset: RasterDataSet):
        self._point = point
        self._dataset = dataset

class GeoReferenceTaskDelegate:
    # Take in clickable_object_1 and clickable_object_2

    # When clickable object 1 receives a click, it should send it to task delegate.
    # the next click task delegate does should be in clickable object 2.

    # When 2 things are clicked and the user presses enter, it adds them to a list 
    # and emits a signal that the GeoReferenceDialog conects to to get that list 

    # It populates the GCP list with that list. If user's edit the GCP list, it updates
    # the list here
    def __init__(self, rasterpane_1, rasterpane_2):
        '''
        Technically, rasterpane_1 and rasterpane_2 can be any objects that
        can receive mouse events and when they do receive mouse events,
        will send it here.
        '''
        # So we have the last selected pane and the current selected pane. 
        # The last selected pane will remain none until the user presses enter 
        self._rasterpane_1: RasterPane = rasterpane_1
        self._rasterpane_2: RasterPane = rasterpane_2
        self._last_selected_pane: Optional[RasterPane] = None
        self._current_selected_pane: Optional[RasterPane] = None
        self._current_point: Optional[GroundControlPoint] = None
        self._current_point_pair = Optional[List[GroundControlPoint, GroundControlPoint]] = None
        self._point_list: List[List[GroundControlPoint, GroundControlPoint]] = []
    
    def on_mouse_release(self, point: List[int, int], rasterpane: RasterPane):
        # We want the user to be able to press escape and clear the currently selected raster pane 
        if self._last_selected_pane is None:
            self._last_selected_pane = rasterpane
            self._current_point = GroundControlPoint(point, rasterpane.get_rasterview().get_raster_data())
    
