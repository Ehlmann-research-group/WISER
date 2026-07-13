from typing import TYPE_CHECKING

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .band_chooser import BandChooserDialog
from .rasterview import RasterView
from .rasterpane import RasterPane
from .util import get_painter, add_toolbar_action
from .geo_reference_task_delegate import PointSelectorType, PointSelector

if TYPE_CHECKING:
    from .geo_reference_task_delegate import GeoReferencerTaskDelegate


class GeoReferencerPane(RasterPane, PointSelector):
    """
    This class represents one of the geo referencer panes. It is
    a regular rasterpane except that it doesn't need a lot of the
    capabilities like adding datasets or having ROIs. It is just
    needed for displaying the datasets visually and selecting
    target and reference points.
    """

    def __init__(self, app_state, pane_type: PointSelectorType, parent=None):
        super().__init__(
            app_state=app_state,
            parent=parent,
            max_zoom_scale=64,
            zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16, 24, 32],
            initial_zoom=1,
        )
        self._pane_type = pane_type

        # Whether a dataset has been loaded but not yet auto-fitted.  The fit is
        # deferred until the pane is actually laid out (see _run_pending_fit).
        self._pending_fit = False

    def get_point_selector_type(self):
        return self._pane_type

    def set_task_delegate(self, task_delegate: "GeoReferencerTaskDelegate"):
        self._task_delegate = task_delegate

    def _init_dataset_tools(self):
        self._dataset_chooser = None

        self._act_band_chooser = add_toolbar_action(
            self._toolbar, ":/icons/choose-bands.svg", self.tr("Band chooser"), self
        )
        self._act_band_chooser.triggered.connect(self._on_band_chooser)

        self._act_band_chooser.setEnabled(False)

    def _init_select_tools(self):
        """
        We don't want this or the parent class to initialize any of
        the select tools. The select tools currently are just the ROI
        tools
        """
        return

    def _init_zoom_tools(self):
        """
        Add a "Zoom to fit" button to the base zoom tools, so the user can
        quickly frame the whole image while placing control points.
        """
        super()._init_zoom_tools()

        self._act_zoom_to_fit = add_toolbar_action(
            self._toolbar,
            ":/icons/zoom-to-fit.svg",
            self.tr("Zoom to fit"),
            self,
            before=self._act_cbox_zoom,
        )
        self._act_zoom_to_fit.triggered.connect(self._on_zoom_to_fit)

    def _on_zoom_to_fit(self):
        """Zoom the view such that the entire image fits in the view."""
        # Use the rasterview at (0, 0) to compute the scale for the image to fit
        rasterview = self.get_rasterview()
        rasterview.scale_image_to_fit()

        # If we are in a multi-view mode, propagate that scale to all views
        if self.is_multi_view():
            self.set_scale(rasterview.get_scale())

        self._update_zoom_widgets()

    def show_dataset(self, dataset, rasterview_pos=(0, 0)):
        """
        Show a dataset, and auto-fit it the first time it is loaded so large
        images are immediately usable.  The fit itself is deferred until the
        pane is laid out (see _run_pending_fit).
        """
        super().show_dataset(dataset, rasterview_pos=rasterview_pos)

        if dataset is not None:
            self._pending_fit = True
            # Attempt the fit on the next event-loop turn (handles switching
            # datasets while the dialog is already open).  If the pane isn't
            # laid out yet -- e.g. the dataset was set via GeoReferencerConfig
            # before the dialog was shown -- _run_pending_fit bails and showEvent
            # schedules another attempt once the pane is visible.
            QTimer.singleShot(0, self._run_pending_fit)

    def showEvent(self, event):
        super().showEvent(event)
        # Datasets are commonly set (via GeoReferencerConfig) before the dialog
        # is shown, when the viewport is not yet laid out.  Run any pending fit
        # now that the pane is visible.
        if self._pending_fit:
            QTimer.singleShot(0, self._run_pending_fit)

    def _run_pending_fit(self):
        """Perform a deferred auto-fit once the pane's viewport is laid out."""
        if not self._pending_fit:
            return

        rasterview = self.get_rasterview()
        # Bail if there is no data or the pane isn't laid out yet; showEvent /
        # show_dataset will schedule another attempt.
        if rasterview.get_raster_data() is None:
            return
        if rasterview.width() <= 1:
            return

        self._on_zoom_to_fit()
        self._pending_fit = False

    def _on_band_chooser(self, checked=False, rasterview_pos=(0, 0)):
        super()._on_band_chooser(
            checked=checked,
            rasterview_pos=rasterview_pos,
            singular_update=True,
        )

    def _on_dataset_added(self, ds_id):
        return

    def _onRasterMousePress(self, rasterview, mouse_event):
        self._task_delegate.on_mouse_press(mouse_event)
        self.update_all_rasterviews()

    def _onRasterMouseMove(self, rasterview, mouse_event):
        self._task_delegate.on_mouse_move(mouse_event)

    def _onRasterMouseRelease(self, rasterview, mouse_event):
        """
        When the display image is clicked on, this method gets invoked, and it
        translates the click event's coordinates into the location on the
        raster data set.
        """
        if not isinstance(mouse_event, QMouseEvent):
            return

        self._task_delegate.on_mouse_release(mouse_event, self)

        self.update_all_rasterviews()

    def _afterRasterPaint(self, rasterview, widget, paint_event):
        # Draw the pixel highlight, if there is one
        self._draw_pixel_highlight(rasterview, widget, paint_event)

        # Let the task-delegate draw any state it needs to draw.
        with get_painter(widget) as painter:
            self._task_delegate.draw_state(painter, self)

    def _onRasterKeyPress(self, rasterview, key_event):
        self._task_delegate.on_key_press(key_event)
        self.update_all_rasterviews()

    def _onRasterKeyRelease(self, rasterview, key_event):
        self._task_delegate.on_key_release(key_event)
        self.update_all_rasterviews()

    def _has_delegate_for_rasterview(self, rasterview: RasterView, user_input: bool = True) -> bool:
        return

    def _update_delegate(self, done: bool) -> None:
        return
