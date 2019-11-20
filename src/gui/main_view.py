import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .rasterview import RasterView
from .util import add_toolbar_action


class MainViewWidget(QWidget):
    '''
    This widget provides the main raster-data view in the user interface.
    '''

    # Signal:  the displayed region has changed
    display_area_changed = Signal( (int, int, int, int) )


    def __init__(self, model, parent=None):
        super().__init__(parent=parent)

        self._model = model
        self._dataset_index = None

        self._model.dataset_added.connect(self.add_dataset)
        self._model.dataset_removed.connect(self.remove_dataset)

        # Raster image view widget

        self._rasterview = RasterView(parent=self)

        # Toolbar

        toolbar = QToolBar(self.tr('Toolbar'), parent=self)

        self._cbox_dataset = QComboBox(parent=self)
        self._cbox_dataset.setEditable(False)
        toolbar.addWidget(self._cbox_dataset)

        # TODO(donnie):
        # self._cbox_dataset.activated.connect(self.change_dataset)

        # Zoom In
        self._act_zoom_in = add_toolbar_action(toolbar,
            'resources/zoom-in.svg', self.tr('Zoom in'), self, QKeySequence.ZoomIn)
        self._act_zoom_in.triggered.connect(self.zoom_in)

        # Zoom Out
        self._act_zoom_out = add_toolbar_action(toolbar,
            'resources/zoom-out.svg', self.tr('Zoom out'), self, QKeySequence.ZoomOut)
        self._act_zoom_out.triggered.connect(self.zoom_out)

        # Zoom to Actual Size
        self._act_zoom_to_actual = add_toolbar_action(toolbar,
            'resources/zoom-to-actual.svg', self.tr('Zoom to actual size'), self, None)
        self._act_zoom_to_actual.triggered.connect(self.zoom_to_actual)

        # Zoom to Fit
        self._act_zoom_to_fit = add_toolbar_action(toolbar,
            'resources/zoom-to-fit.svg', self.tr('Zoom to fit'), self, None)
        self._act_zoom_to_fit.triggered.connect(self.zoom_to_fit)

        # self.image_toolbar.addSeparator()

        # Choose RGB Bands
        # self.act_choose_rgb_bands = add_toolbar_action(self.image_toolbar,
        #     'resources/choose-colors.svg', self.tr('Choose RGB Bands'), self, None)
        # self.act_choose_rgb_bands.triggered.connect(self.choose_colors)

        # self.image_toolbar.addSeparator()

        # Show spectral data for a specific band
        # self.act_show_spectrum = add_toolbar_action(self.image_toolbar,
        #     'resources/target.svg', self.tr('Show Pixel Spectrum'), self, None)
        # self.act_show_spectrum.triggered.connect(self.show_spectrum)

        # Widget layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.setMenuBar(toolbar)
        layout.addWidget(self._rasterview)

        self.setLayout(layout)


    def add_dataset(self, index):
        dataset = self._model.get_dataset(index)
        file_path = dataset.get_filepath()

        self._cbox_dataset.insertItem(index, os.path.basename(file_path))

        if self._model.num_datasets() == 1:
            # We finally have a dataset!
            self._dataset_index = 0
            self.update_image()


    def remove_dataset(self, index):
        self._cbox_dataset.removeItem(index)

        num = self._model.num_datasets()

        if num == 0 or self._dataset_index == index:
            self._dataset_index = min(self._dataset_index, num - 1)
            if self._dataset_index == -1:
                self._dataset_index = None

            self.update_image()


    def rasterview(self):
        return self._rasterview


    def update_image(self):
        '''
        Scale the raster-view image based on the image size, and the state of
        the "fit to window" button.
        '''

        dataset = None
        if self._model.num_datasets() > 0:
            dataset = self._model.get_dataset(self._dataset_index)

        # TODO(donnie):  Only do this when the raster dataset actually changes,
        #     or the displayed bands change, etc.
        self._rasterview.set_raster_data(dataset)

    @Slot()
    def zoom_in(self, evt):
        ''' Zoom in the view by 20%. '''

        scale = self._rasterview.get_scale()
        self._rasterview.scale_image(scale * 1.25)

        # TODO:  Disable zoom-in if too zoomed

    @Slot()
    def zoom_out(self, evt):
        ''' Zoom out the view by 20%. '''

        scale = self._rasterview.get_scale()
        self._rasterview.scale_image(scale * 0.8)

        # TODO:  Disable zoom-out if too zoomed out

    @Slot()
    def zoom_to_actual(self, evt):
        ''' Zoom the view to 100% scale. '''

        self._rasterview.scale_image(1.0)

    @Slot()
    def zoom_to_fit(self):
        ''' Zoom the view such that the entire image fits in the view. '''
        self._rasterview.scale_image_to_fit()
