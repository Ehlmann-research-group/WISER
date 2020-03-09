class TaskDelegate:
    '''
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
    '''

    def on_mouse_press(self, widget, mouse_event):
        return False

    def on_mouse_release(self, widget, mouse_event):
        return False

    def on_mouse_move(self, widget, mouse_event):
        return False

    def on_key_press(self, widget, key_event):
        return False

    def on_key_release(self, widget, key_event):
        return False

    def draw_state(self, painter):
        pass

    def finish(self):
        pass
