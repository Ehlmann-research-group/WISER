from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .dataset_chooser import DatasetChooser
from .rasterview import RasterView, ScaleToFitMode


class OverviewRasterView(RasterView):
    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self._app_state = app_state


    def _afterRasterPaint(self, widget, paint_event):
        visible_area = self._app_state.get_view_attribute('image.visible_area')
        if visible_area is None:
            return

        # Draw the visible area on the summary view.
        painter = QPainter(widget)
        painter.setPen(QPen(Qt.yellow))

        scaled = QRect(visible_area.x() * self._scale_factor,
                       visible_area.y() * self._scale_factor,
                       visible_area.width() * self._scale_factor,
                       visible_area.height() * self._scale_factor)

        if scaled.width() >= widget.width():
            scaled.setWidth(widget.width() - 1)

        if scaled.height() >= widget.height():
            scaled.setHeight(widget.height() - 1)

        painter.drawRect(scaled)

        painter.end()


class OverviewPane(QDockWidget):
    '''
    This widget provides a dockable overview pane in the user interface.  The
    pane always shows the raster image zoomed out, with highlights such as the
    area visible in the main image window, annotation locations, etc.
    '''

    def __init__(self, app_state, parent=None):
        super().__init__('Overview', parent=parent)

        # Initialize widget's internal state

        self._app_state = app_state
        self._dataset_index = None

        # Initialize contents of the widget

        self._init_ui()

        # Initialize docking details

        # TODO(donnie):  Overview pane always stays on top!  See if we can
        #     make it go underneath other windows.
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)

        self.visibilityChanged.connect(self._on_visibility_changed)

        # Register for events from the application state

        self._app_state.dataset_added.connect(self._on_dataset_added)
        self._app_state.dataset_removed.connect(self._on_dataset_removed)
        self._app_state.view_attr_changed.connect(self._on_view_attr_changed)

        # Register for events from the UI

        # self.summary_view.rasterview().mouse_click.connect(self.summaryview_mouse_click)


    def _init_ui(self):
        ''' Initialize the contents of this widget '''

        toolbar = QToolBar(self.tr('Toolbar'), parent=self)
        toolbar.setIconSize(QSize(20, 20))

        self._dataset_chooser = DatasetChooser(self._app_state)
        toolbar.addWidget(self._dataset_chooser)
        self._dataset_chooser.triggered.connect(self._on_dataset_changed)

        self._act_fit_to_window = toolbar.addAction(
            QIcon('resources/zoom-to-fit.svg'),
            self.tr('Fit image to window'))
        self._act_fit_to_window.setCheckable(True)
        self._act_fit_to_window.setChecked(True)

        self._act_fit_to_window.triggered.connect(self._on_toggle_fit_to_window)

        # Raster image view widget

        self._rasterview = OverviewRasterView(self._app_state, parent=self)

        # Widget layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.setMenuBar(toolbar)
        layout.addWidget(self._rasterview)

        widget = QWidget(parent=self)
        widget.setLayout(layout)

        self.setWidget(widget)


    def toggleViewAction(self):
        '''
        Returns a QAction object that can be used to toggle the visibility of
        this dockable pane.  This class overrides the QDockWidget implementation
        to specify a nice icon and tooltip on the action.
        '''
        act = super().toggleViewAction()
        act.setIcon(QIcon('resources/overview-pane.svg'))
        act.setToolTip(self.tr('Show/hide overview pane'))
        return act


    def sizeHint(self):
        ''' The default size of the overview-pane widget is 200x200. '''
        return QSize(200, 200)


    def resizeEvent(self, event):
        ''' Update the image scaling when this widget is resized. '''
        self._update_image()


    def _on_visibility_changed(self, visible):
        self._app_state.set_view_attribute('overview.visible', visible)

        # Work around a known Qt bug:  if a dockable window is floating, and is
        # closed while floating, it can't be redocked unless we toggle its
        # floating state.
        if self.isFloating() and not visible:
            self.setFloating(False)
            self.setFloating(True)


    def _on_toggle_fit_to_window(self):
        '''
        Update the raster-view image when the "fit to window" button is toggled.
        '''
        self._update_image()


    def _on_dataset_added(self, index):
        if self._app_state.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self._update_image()


    def _on_dataset_removed(self, index):
        num = self._app_state.num_datasets()
        if num == 0 or self._dataset_index == index:
            self._dataset_index = min(self._dataset_index, num - 1)
            if self._dataset_index == -1:
                self._dataset_index = None

            self._update_image()


    def _on_dataset_changed(self, act):
        self._dataset_index = act.data()
        self._update_image()


    def _on_view_attr_changed(self, attr_name):
        if attr_name == 'image.visible_area':
            self._rasterview.update()


    def _update_image(self):
        '''
        Scale the raster-view image based on the image size, and the state of
        the "fit to window" button.
        '''

        dataset = None
        if self._app_state.num_datasets() > 0:
            dataset = self._app_state.get_dataset(self._dataset_index)

        # TODO(donnie):  Only do this when the raster dataset actually changes,
        #     or the displayed bands change, etc.
        self._rasterview.set_raster_data(dataset)

        if dataset is None:
            return

        # Handle window-scaling changes
        if self._act_fit_to_window.isChecked():
            # The entire image needs to fit in the summary view.
            self._rasterview.scale_image_to_fit(
                mode=ScaleToFitMode.FIT_BOTH_DIMENSIONS)
        else:
            # Just zoom such that one of the dimensions fits.
            self._rasterview.scale_image_to_fit(
                mode=ScaleToFitMode.FIT_ONE_DIMENSION)
