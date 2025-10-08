from typing import Optional, TYPE_CHECKING, List

from .rasterview import RasterView

if TYPE_CHECKING:
    from .rasterpane import RasterPane


class TaskDelegate:
    """
    This interface specifies the operations that task-delegates of the raster
    pane need to implement, in order to handle events from the raster pane's
    components.

    Each of the event-handler functions must return a Boolean value indicating
    whether the task is now completed, with True meaning that the task is done.
    The default implementations return False, so that subclasses can implement
    only the functions that are needed.

    After every event-handling operation is called, the raster pane will refresh
    its UI.  (TODO:  THIS IS SLOW, and the delegate should be the one who
    requests UI refreshes, so it can specify the region that needs updating.)

    Once the task is completed, the raster pane will call the delegate's
    finish() implementation, in order to perform any final operations associated
    with the task.
    """

    def __init__(self, rasterpane: "RasterPane", rasterview: Optional[RasterView] = None):
        """
        Initialize the task delegate.  An optional raster-view may be specified,
        but in multi-view contexts, the set_rasterview() function will be the
        preferred option.
        """
        if rasterpane is None:
            raise ValueError("rasterpane cannot be None")

        self._rasterpane: "RasterPane" = rasterpane
        self._app_state = rasterpane.get_app_state()
        self._rasterview: Optional[RasterView] = rasterview

    def set_rasterview(self, rasterview: RasterView):
        """
        Specify the raster-view that this task delegate is operating on behalf
        of.
        """
        self._rasterview = rasterview

    def get_rasterview(self) -> RasterView:
        """
        Returns the raster-view for this task delegate, or None if a raster-view
        hasn't yet been assigned.
        """
        return self._rasterview

    def on_mouse_press(self, mouse_event) -> bool:
        return False

    def on_mouse_release(self, mouse_event) -> bool:
        return False

    def on_mouse_move(self, mouse_event) -> bool:
        return False

    def on_key_press(self, key_event) -> bool:
        return False

    def on_key_release(self, key_event) -> bool:
        return False

    def draw_state(self, painter) -> bool:
        pass

    def finish(self) -> None:
        pass
