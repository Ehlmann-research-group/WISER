from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wiser.gui.app_state import ApplicationState


class WISERControl:
    '''
    This class exposes certain parts of the WISER application's state, and
    provides operations that can be performed on the WISER application,
    allowing plugins to interact with the WISER application programmatically.
    '''

    def __init__(self, app_state: 'ApplicationState'):
        # Try to keep this private, so that plugin writers and others know they
        # really shouldn't muck around with it!
        self.__app_state: 'ApplicationState' = app_state
