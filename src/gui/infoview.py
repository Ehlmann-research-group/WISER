import os, sys
from enum import Enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from .constants import ImageColors

from raster.dataset import RasterDataSet, find_display_bands


class DatasetInfoView(QTreeWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent=parent)

        self._model = model
        self._model.dataset_added.connect(self.add_dataset)
        self._model.dataset_removed.connect(self.remove_dataset)

        self.setColumnCount(3)
        self.setHeaderLabels(
            [self.tr('Description'), self.tr('Detail'), self.tr('Extra')])

        top = QTreeWidgetItem(self)
        top.setText(0, self.tr('No datasets loaded'))
        self.addTopLevelItem(top)


    def add_dataset(self, index):

        if self._model.num_datasets() == 1:
            self.clear()

        dataset = self._model.get_dataset(index)

        top = QTreeWidgetItem(self)
        top.setText(0, self.tr(f'Dataset {index+1}'))

        # General info subsection

        info = QTreeWidgetItem(top)
        info.setText(0, self.tr('General'))

        item = QTreeWidgetItem(info)
        item.setText(0, self.tr('Description'))
        item.setText(1, f'{dataset.get_description()}')

        item = QTreeWidgetItem(info)
        item.setText(0, self.tr('File Type'))
        item.setText(1, dataset.get_filetype())

        item = QTreeWidgetItem(info)
        item.setText(0, self.tr('Width'))
        item.setText(1, f'{dataset.get_width()}')

        item = QTreeWidgetItem(info)
        item.setText(0, self.tr('Height'))
        item.setText(1, f'{dataset.get_height()}')

        item = QTreeWidgetItem(info)
        item.setText(0, self.tr('Bands'))
        item.setText(1, f'{dataset.num_bands()}')

        # Files subsection

        files = QTreeWidgetItem(top)
        files.setText(0, self.tr('Files'))

        filepaths = sorted(dataset.get_filepaths())
        for fp in filepaths:
            file_item = QTreeWidgetItem(files)
            file_item.setFirstColumnSpanned(True)
            file_item.setText(0, os.path.basename(fp))

        # Bands section

        bands = QTreeWidgetItem(top)
        bands.setText(0, self.tr('Bands'))
        bands.setText(1, f'({dataset.num_bands()})')

        band_list = dataset.band_list()
        bad_bands = dataset.get_bad_bands()

        default_bands = dataset.default_display_bands()
        if default_bands is None:
            default_bands = []

        for i in range(dataset.num_bands()):
            band_info = band_list[i]
            bad = bad_bands[i]

            band_item = QTreeWidgetItem(bands)

            band_item.setText(0, band_info.get('description', self.tr('(no description)')))

            s = ''
            if 'wavelength' in band_info:
                v = band_info['wavelength']
                s = '{0:0.02f}'.format(v)
            band_item.setText(1, s)

            s = []
            if bad == 0:
                s.append(self.tr('(bad)'))

            if i in default_bands:
                if len(default_bands) == 1:
                    s.append(self.tr('(default - grayscale)'))
                else:
                    if len(default_bands) != 3:
                        raise ValueError(f'Expected 3 default bands, got {default_bands}')

                    if default_bands[0] == i:
                        s.append(self.tr('(default - red)'))
                    elif default_bands[1] == i:
                        s.append(self.tr('(default - green)'))
                    elif default_bands[2] == i:
                        s.append(self.tr('(default - blue)'))

            band_item.setText(2, ' '.join(s))

        # All done!

        self.insertTopLevelItem(index, top)


    def remove_dataset(self, index):
        # Remove the information for the dataset at the specified index
        self.takeTopLevelItem(index)

        if self.topLevelItemCount() == 0:
            top = QTreeWidgetItem(self)
            top.setText(0, self.tr('No datasets loaded'))
            self.addTopLevelItem(top)
