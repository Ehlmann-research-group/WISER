from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from raster.dataset import RasterDataSet


class AppModel(QObject):

    # Signal:  a data-set was added at the specified index
    dataset_added = Signal( (int) )

    # Signal:  the data-set at the specified index was removed
    dataset_removed = Signal( (int) )


    def __init__(self):
        super().__init__()
        self._datasets = []


    def add_dataset(self, dataset):
        if not isinstance(dataset, RasterDataSet):
            raise ValueError('dataset must be a RasterDataSet')

        index = len(self._datasets)
        self._datasets.append(dataset)

        self.dataset_added.emit(index)

    def get_dataset(self, index):
        return self._datasets[index]

    def num_datasets(self):
        return len(self._datasets)

    def get_datasets(self):
        return list(self._datasets)
