import abc
import enum
import importlib

from typing import Any, Callable, Dict, List, Optional, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gui.app_state import ApplicationState

from PySide6.QtWidgets import QMenu

from wiser.bandmath import BandMathValue, BandMathFunction


class ContextMenuType(enum.Enum):
    '''
    This enumeration specifies the kind of context-menu event that occurred,
    so that plugins know what items to add to the menu.
    '''

    # Context-menu display in a raster-view, which probably is showing a
    # dataset.  The current dataset is passed to the plugin.
    RASTER_VIEW = 1

    # Context-menu display in the spectrum-plot window.
    SPECTRUM_PLOT = 2

    # A specific dataset was picked.  This may not be in the context of a
    # raster-view window, e.g. if the user right-clicks on a dataset in the info
    # viewer.
    DATASET_PICK = 10

    # A specific spectrum was picked.  The spectrum is passed to the plugin.
    SPECTRUM_PICK = 11

    # A specific ROI was picked.  The ROI is passed, along with the current
    # dataset (if available).
    ROI_PICK = 12


class Plugin(abc.ABC):
    ''' The base type for all WISER plugins. '''
    pass


class ToolsMenuPlugin(Plugin):
    '''
    This is the base type for plugins that integrate into the WISER "Tools"
    application-menu.
    '''

    def __init__(self):
        super().__init__()


    def add_tool_menu_items(self, tool_menu: QMenu, wiser: 'ApplicationState') -> None:
        '''
        This method is called by WISER to allow plugins to add menu actions or
        submenus into the Tools application menu.

        If a plugin provides multiple actions, the developer has several
        choices.  If all actions are useful and expected to be invoked
        regularly, the actions can be added directly to the Tools menu.  If
        some actions are used much less frequently, it is recommended that
        these actions be put into a submenu, to keep the Tools menu from
        becoming too cluttered.

        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        '''
        pass


class ContextMenuPlugin(Plugin):
    '''
    This is the base type for plugins that integrate into WISER pop-up context
    menus.
    '''

    def __init__(self):
        super().__init__()

    def add_context_menu_items(self, context_type: ContextMenuType,
            context_menu: QMenu, context: Dict[str, Any]) -> None:
        '''
        This method is called by WISER when it is constructing a context menu,
        so that the plugin can add any menu-actions relevant to the context.

        The type of context is indicated by the ``context_type`` argument/enum.
        Plugins should examine this value and only add menu entries relevant to
        the context type, to avoid cluttering up the WISER context menus.  This
        is particularly important, as this method may be called multiple times
        (with different ``context_type`` values), in the population of a single
        context-menu before it is displayed.  For example, it is possible for a
        plugin to see calls to this method with ``RASTER_VIEW``, then
        ``DATASET_PICK``, then ``ROI_PICK``, in the construction of a single
        context menu.

        Based on the type of context, the ``context`` dictionary will contain
        specific key/value pairs relevant to the context, that the plugin may
        need for its operation.  The details are specified below.  Besides these
        values, the ``context`` dictionary will also always contain a ``wiser``
        key that references a :class:`wiser.gui.app_state.ApplicationState` object for
        accessing and manipulating WISER's internal state in specific ways.

        ``RASTER_VIEW``
          Indicates a *general* operation on a dataset within a raster display
          window - that is, an operation not related to the cursor location.
          The ``context`` dictionary includes these keys:

          *   ``dataset`` - a reference to the :class:`wiser.raster.RasterDataSet`
              object currently being displayed.
          *   ``display_bands`` - a tuple of integers specifying the bands
              currently being displayed in the raster-view.  This will either
              hold 1 element if the display is grayscale, or 3 elements if the
              display is red/green/blue.

        ``SPECTRUM_PLOT``
          Indicates a *general* operation within a spectrum-plot window - that
          is, not related to a specific spectrum or the cursor location.  The
          ``context`` dictionary will not have any additional keys.

        ``DATASET_PICK``
          Indicates a *location-specific* operation on a dataset within a raster
          display window - that is, an operation that requires the cursor
          location.  The ``context`` dictionary includes these additional keys:

          *   ``dataset`` - a reference to the :class:`wiser.raster.RasterDataSet`
              object currently being displayed.
          *   ``display_bands`` - a tuple of integers specifying the bands
              currently being displayed in the raster-view.  This will either
              hold 1 element if the display is grayscale, or 3 elements if the
              display is red/green/blue.
          *   ``ds_coord`` - an ``(int, int)`` tuple of the pixel in the dataset
              that was picked by the user.

        ``SPECTRUM_PICK``
          Indicates a *spectrum-specific* operation within a spectrum-plot
          window.  The ``context`` dictionary will have this additional key:

          *   ``spectrum`` - a reference to the :class:`wiser.raster.Spectrum`
              object that was picked by the user.

        ``ROI_PICK``
          Indicates a *region-of-interest-specific* operation within a raster
          display window.  The ``context`` dictionary includes these additional
          keys:

          *   ``dataset`` - a reference to the :class:`wiser.raster.RasterDataSet`
              object currently being displayed.
          *   ``display_bands`` - a tuple of integers specifying the bands
              currently being displayed in the raster-view.  This will either
              hold 1 element if the display is grayscale, or 3 elements if the
              display is red/green/blue.
          *   ``roi`` - a reference to the :class:`wiser.raster.RegionOfInterest`
              object that was picked by the user.
          *   ``ds_coord`` - an ``(int, int)`` tuple of the pixel in the dataset
              that was picked by the user.

        Plugins should be careful not to hold onto any context references for
        too long, as it will generate resource leaks within WISER.  A
        recommended pattern for adding menu actions is as follows:

        .. code-block:: python

            # Construct a lambda that is called when the QAction is clicked;
            # it traps the context dictionary and passes it to the relevant
            # handler.  The context is reclaimed when the QAction goes away.
            act = context_menu.addAction(context_menu.tr('Some task...'))
            act.triggered.connect(lambda checked=False: self.on_some_task(context=context))

        '''
        pass


class BandMathPlugin(Plugin):
    '''
    This is the base type for plugins that provide custom band-math functions.
    '''

    def __init__(self):
        super().__init__()

    def get_bandmath_functions(self) -> Dict[str, BandMathFunction]:
        '''
        This method returns a dictionary of all band-math functions provided by
        the plugin.

        The keys are the function names, and must satisfy the parsing
        requirements of the band-math parser:  names must start with an
        alphabetical character (a-z), and must include only alphanumeric
        characters and underscores (a-z, 0-9, _).

        The values are instances of classes that extend the
        :class:`BandMathFunction` type, to provide the various operations
        required by band-math functions.

        Band-math expressions are *case-insensitive*.  Therefore, all function
        names specified by a plugin are converted to lowercase when loaded into
        the band-math evaluator.
        '''
        pass
