from typing import TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .band_chooser import BandChooserDialog
from .rasterview import RasterView
from .rasterpane import RasterPane
from .util import get_painter, add_toolbar_action
from .geo_reference_task_delegate import PointSelectorType, PointSelector

if TYPE_CHECKING:
    from .geo_reference_task_delegate import GeoReferencerTaskDelegate


class GeoReferencerPane(RasterPane, PointSelector):
    def __init__(self, app_state, pane_type: PointSelectorType, parent=None):
        super().__init__(
            app_state=app_state,
            parent=parent,
            max_zoom_scale=64,
            zoom_options=[0.25, 0.5, 0.75, 1, 2, 4, 8, 16, 24, 32],
            initial_zoom=1,
        )
        self._pane_type = pane_type

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
        We don't want this to initialize any of the select tools.
        The select tools currently are just the ROI tools
        """
        return

    def _on_band_chooser(self, checked=False, rasterview_pos=(0, 0)):
        rasterview = self.get_rasterview(rasterview_pos)
        dataset = rasterview.get_raster_data()
        display_bands = rasterview.get_display_bands()
        colormap = rasterview.get_colormap()

        dialog = BandChooserDialog(
            self._app_state,
            dataset,
            display_bands,
            colormap=colormap,
            can_apply_global=False,
            parent=self,
        )
        dialog.setModal(True)

        if dialog.exec_() == QDialog.Accepted:
            bands = dialog.get_display_bands()
            colormap = dialog.get_colormap_name()

            # For the geo referencer, the change shouldn't be global
            self.set_display_bands(dataset.get_id(), bands, colormap=colormap)

    def _on_dataset_added(self, ds_id):
        """
        We don't want to do anything here
        """
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

        # print(f'MouseEvent at pos={mouse_event.pos()}, localPos={mouse_event.localPos()}')

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

    def _has_delegate_for_rasterview(
        self, rasterview: RasterView, user_input: bool = True
    ) -> bool:
        """
        We do not want to call this function in the parent class
        """
        return

    def _update_delegate(self, done: bool) -> None:
        """
        We do not want to call this function in the parent class
        """
        return
