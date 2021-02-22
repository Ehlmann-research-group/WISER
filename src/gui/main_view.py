import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

import gui.generated.resources

from .bandmath_dialog import BandMathDialog
from .export_image import ExportImageDialog
from .toolbarmenu import ToolbarMenu
from .rasterpane import RasterPane
from .rasterview import RasterView
from .split_pane_dialog import SplitPaneDialog
from .stretch_builder import StretchBuilderDialog
from .util import add_toolbar_action

import bandmath


class MainViewWidget(RasterPane):
    '''
    This widget provides the main raster-data view in the user interface.
    '''


    def __init__(self, app_state, parent=None):
        super().__init__(app_state=app_state, parent=parent, embed_toolbar=False,
            max_zoom_scale=16, zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16],
            initial_zoom=1)

        self._stretch_builder = StretchBuilderDialog(parent=self)
        self._export_image = ExportImageDialog(parent=self)
        self._link_view_scrolling = False

        if self._app_state.get_config('feature-flags.linked-multi-view', default=True, as_type=bool):
            self._set_link_views_button_state()

        self._set_dataset_tools_button_state()


    def _init_toolbar(self):
        '''
        The main view initializes dataset tools, view tools, zoom tools, and
        selection tools.
        '''
        self._init_dataset_tools()

        if self._app_state.get_config('feature-flags.linked-multi-view', default=True, as_type=bool):
            self._toolbar.addSeparator()
            self._init_view_tools()

        self._toolbar.addSeparator()
        self._init_zoom_tools()
        self._toolbar.addSeparator()
        self._init_select_tools()


    def _init_dataset_tools(self):
        '''
        Override the default dataset-tools function to add a stretch-builder
        button to the dataset-tools part of the toolbar.
        '''
        super()._init_dataset_tools()

        self._act_stretch_builder = add_toolbar_action(self._toolbar,
            ':/icons/stretch-builder.svg', self.tr('Stretch builder'), self)
        self._act_stretch_builder.triggered.connect(self._on_stretch_builder)


    def _init_view_tools(self):
        '''
        Initialize view management tools.  This subclass has the ability to show
        multiple sub-views, all at the same zoom level, and possibly with linked
        scrolling.
        '''

        # The split-views chooser is a drop down menu of options the user can
        # choose from.  First populate the menu, then create the chooser button.

        chooser_items = [
            (self.tr('1 row x 1 column'  ), (1, 1)),
            (self.tr('1 row x 2 columns' ), (1, 2)),
            (self.tr('2 rows x 1 column' ), (2, 1)),
            (self.tr('2 rows x 2 columns'), (2, 2)),
            (self.tr('Other layout...'   ), (-1, -1)),
        ]
        self._view_chooser = ToolbarMenu(icon=QIcon(':/icons/split-view.svg'),
            items=chooser_items)
        self._view_chooser.setToolTip(self.tr('Split/unsplit the main view'))
        self._toolbar.addWidget(self._view_chooser)
        self._view_chooser.triggered.connect(self._on_split_views)

        self._act_link_view_scroll = add_toolbar_action(self._toolbar,
            ':/icons/link-scroll.svg', self.tr('Link view scrolling'), self)
        self._act_link_view_scroll.setCheckable(True)
        self._act_link_view_scroll.triggered.connect(self._on_link_view_scroll)


    def _init_zoom_tools(self):
        '''
        Initialize zoom toolbar buttons.  This subclass adds a few extra zoom
        buttons to the zoom tools.
        '''
        super()._init_zoom_tools()

        # Zoom to Actual Size
        self._act_zoom_to_actual = add_toolbar_action(self._toolbar,
            ':/icons/zoom-to-actual.svg', self.tr('Zoom to actual size'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_actual.triggered.connect(self._on_zoom_to_actual)

        # Zoom to Fit
        self._act_zoom_to_fit = add_toolbar_action(self._toolbar,
            ':/icons/zoom-to-fit.svg', self.tr('Zoom to fit'), self,
            before=self._act_cbox_zoom)
        self._act_zoom_to_fit.triggered.connect(self._on_zoom_to_fit)


    def _context_menu_add_global_items(self, menu, rasterview):
        '''
        This helper function adds "global" items to the context menu, that is,
        items that aren't specific to the location clicked in the window.
        '''

        # When listening for these actions, pass in the rasterview that
        # generated the event.

        # Band math

        act = menu.addAction(self.tr('Band math...'))
        act.triggered.connect(lambda checked=False, rv=rasterview, **kwargs :
                              self._on_dataset_band_math(rv))

        # Submenu for RGB image export

        submenu = menu.addMenu(self.tr('Export RGB image'))

        act = submenu.addAction(self.tr('Export visible image area to RGB image...'))
        act.triggered.connect(lambda checked=False, rv=rasterview, **kwargs :
                              self._on_export_image_visible_area(rv))

        act = submenu.addAction(self.tr('Export full image extent to RGB image...'))
        act.triggered.connect(lambda checked=False, rv=rasterview, **kwargs :
                              self._on_export_image_full(rv))


    def _set_link_views_button_state(self):
        '''
        Sets the enabled/disabled state of the "link view scrolling" button.
        '''
        self._act_link_view_scroll.setEnabled(self._num_views != (1, 1))

    def _set_dataset_tools_button_state(self):
        '''
        Sets the enabled/disabled state of the "stretch builder" button.
        '''

        enabled = (self._app_state.num_datasets() > 0 and self._num_views == (1, 1))

        self._dataset_chooser.setEnabled(enabled)
        self._act_band_chooser.setEnabled(enabled)
        self._act_stretch_builder.setEnabled(enabled)


    def _on_dataset_added(self, ds_id):
        '''
        Override the base-class implementation so we can also update the
        stretch-builder button state.
        '''
        super()._on_dataset_added(ds_id)
        self._set_dataset_tools_button_state()


    def _on_dataset_band_math(self, rasterview):
        dataset = rasterview.get_raster_data()
        dialog = BandMathDialog(self._app_state, rasterview=rasterview)
        if dialog.exec() == QDialog.Accepted:
            expression = dialog.get_expression()
            print(f'Evaluate band math:  {expression}')

            variables = dialog.get_variable_bindings()
            (result_type, result) = bandmath.eval_bandmath_expr(expression,
                variables, {})

            print(f'Result of band-math evaluation is type {result_type}')
            print(f'Result is:\n{result}')

            print('TODO:  Display result of band-math')

            if result_type == bandmath.VariableType.IMAGE_CUBE:
                loader = self._app_state.get_loader()
                new_dataset = loader.from_numpy_array(result)
                self._app_state.add_dataset(new_dataset)

            elif result_type == bandmath.VariableType.IMAGE_BAND:
                # Convert the image band into a 1-band image cube
                result = result[np.newaxis, :]
                loader = self._app_state.get_loader()
                new_dataset = loader.from_numpy_array(result)
                self._app_state.add_dataset(new_dataset)

            elif result_type == bandmath.VariableType.SPECTRUM:
                # new_spectrum = bandmath.result_to_spectrum(result_type, result)
                # self._app_state.set_active_spectrum(new_spectrum)
                print('TODO:  create new spectrum')


    def _on_export_image_visible_area(self, rasterview):
        visible = rasterview.get_visible_region()
        self._export_image.configure(rasterview, visible.x(), visible.y(),
            visible.width(), visible.height(), rasterview.get_scale())

        # The dialog will save the image to disk, if the user completes the
        # operation.
        self._export_image.exec()


    def _on_export_image_full(self, rasterview):
        dataset = rasterview.get_raster_data()
        self._export_image.configure(rasterview, 0, 0,
            dataset.get_width(), dataset.get_height(), 1.0)

        self._export_image.exec()


    def get_stretch_builder(self):
        return self._stretch_builder


    def _on_split_views(self, action):
        '''
        This function handles when the user requests a split-view layout of some
        structure.
        '''
        new_dim = action.data()

        if new_dim == (-1, -1):
            # User wants a general MxN layout.
            dialog = SplitPaneDialog(initial=self._num_views)
            if dialog.exec() != QDialog.Accepted:
                return

            new_dim = dialog.get_dimensions()

        if new_dim != self._num_views:
            msg = self.tr('Switching main view to {rows}x{cols} display')
            msg = msg.format(rows=new_dim[0], cols=new_dim[1])
            self._app_state.show_status_text(msg, 10)

        self._init_rasterviews(new_dim)

        self._set_dataset_tools_button_state()
        self._set_link_views_button_state()


    def is_scrolling_linked(self):
        return self._link_view_scrolling


    def _on_link_view_scroll(self, checked):
        '''
        This function handles when the user links or unlinks raster-view
        scrolling.  When raster-view scrolling is linked, all raster views must
        be updated to show the same coordinates as the top left raster view.
        '''

        if checked and not self._app_state.multiple_datasets_same_size():
            # TODO(donnie):  Not sure if it's better to tell the user why the
            #     view scrolling can't be linked, or to disable it.  For now,
            #     we tell the user why it isn't possible.
            QMessageBox.information(self, self.tr('Cannot Link Views'),
                self.tr('Cannot link views unless multiple datasets\n' +
                        'are loaded, and are all the same dimensions.'))

            self._act_link_view_scroll.setChecked(False)
            return

        # If we got here, we can link or unlink view scrolling.
        self._link_view_scrolling = checked
        if checked:
            self._app_state.show_status_text(self.tr('Linked view scrolling is ON'), 5)

            # Sync scroll state of all raster views to the one in the top left.
            self._sync_scroll_state(self.get_rasterview())
        else:
            self._app_state.show_status_text(self.tr('Linked view scrolling is OFF'), 5)


    def _afterRasterScroll(self, rasterview, dx, dy):
        '''
        This function is called when the raster-view's scrollbars are moved.

        It fires an event that the visible region of the raster-view has
        changed.

        If multiple raster-views are active, and scrolling is linked, this also
        propagates the scroll changes to the other raster-views.
        '''
        # Invoke the superclass version of this operation to emit the
        # viewport-changed event.
        super()._afterRasterScroll(rasterview, dx, dy)
        self._sync_scroll_state(rasterview)


    def _sync_scroll_state(self, rasterview: RasterView) -> None:
        '''
        This helper function synchronizes the scroll state of all raster views
        in this pane.  The source rasterview to take the scroll state from is
        specified as the argument.
        '''
        sb_state = rasterview.get_scrollbar_state()
        if len(self._rasterviews) > 1 and self._link_view_scrolling:
            for rv in self._rasterviews.values():
                # Skip the rasterview that generated the scroll event
                if rv is rasterview:
                    continue

                rv.set_scrollbar_state(sb_state)


    def _on_zoom_to_actual(self, evt):
        ''' Zoom the view to 100% scale. '''

        self.set_scale(1.0)
        self._update_zoom_widgets()


    def _on_zoom_to_fit(self):
        ''' Zoom the view such that the entire image fits in the view. '''

        # Use the rasterview at (0, 0) to compute the scale for the image to fit
        rasterview = self.get_rasterview()
        rasterview.scale_image_to_fit()

        # If we are in a multi-view mode, propagate that scale to all views
        if self.is_multi_view():
            self.set_scale(rasterview.get_scale())

        self._update_zoom_widgets()
