import os

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .rasterview import RasterView


class DetailViewWidget(QWidget):
    '''
    This widget provides the detail view in the user interface.  The detail
    view always shows the raster image zoomed in at a large magnification.
    '''

    def __init__(self, model, parent=None):
        super().__init__(parent=parent)

        self._model = model
        self._dataset_index = None

        self._model.dataset_added.connect(self.add_dataset)
        self._model.dataset_removed.connect(self.remove_dataset)

        # Toolbar

        toolbar = QToolBar(self.tr('Toolbar'), parent=self)

        self._cbox_dataset = QComboBox(parent=self)
        self._cbox_dataset.setEditable(False)
        self._cbox_dataset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        toolbar.addWidget(self._cbox_dataset)

        self._cbox_dataset.activated.connect(self.change_dataset)

        # Raster image view widget

        self._rasterview = RasterView(parent=self)

        # Widget layout

        layout = QVBoxLayout()
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        layout.setMenuBar(toolbar)
        layout.addWidget(self._rasterview)

        self.setLayout(layout)


    def sizeHint(self):
        ''' The default size of the detail-view widget is 200x200. '''
        return QSize(200, 200)


    def resizeEvent(self, event):
        ''' Update the raster-view image when this widget is resized. '''
        self.update_image()


    def rasterview(self):
        return self._rasterview


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


    def change_dataset(self, index):
        self._dataset_index = index
        self.update_image()


    def update_image(self):
        dataset = None
        if self._model.num_datasets() > 0:
            dataset = self._model.get_dataset(self._dataset_index)

        # TODO(donnie):  Only do this when the raster dataset actually changes,
        #     or the displayed bands change, etc.
        self._rasterview.set_raster_data(dataset)

        if dataset is None:
            return

        self._rasterview.scale_image(4)
