import os, sys
from enum import Enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from raster.dataset import RasterDataSet, find_display_bands


class DatasetInfoView(QTreeWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent=parent)

        self._model = model
        self._model.dataset_added.connect(self._on_dataset_added)
        self._model.dataset_removed.connect(self._on_dataset_removed)

        self.setColumnCount(1)
        self.setHeaderLabels([self.tr('Description')])

        top = QTreeWidgetItem(self)
        top.setText(0, self.tr('No datasets loaded'))
        self.addTopLevelItem(top)

        # Update the header info so that columns will resize based on their
        # contents.  This is useful for filenames of data files, which are long.
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(False)


    def _on_dataset_added(self, index):

        if self._model.num_datasets() == 1:
            self.clear()

        dataset = self._model.get_dataset(index)

        top = QTreeWidgetItem(self)
        top.setText(0, self.tr(f'Dataset {index+1}'))

        # General info subsection

        info = QTreeWidgetItem(top)
        info.setText(0, self.tr('General'))

        item = QTreeWidgetItem(info)
        item.setText(0, f'Description:  {dataset.get_description()}')

        item = QTreeWidgetItem(info)
        item.setText(0, f'File Type:  {dataset.get_filetype()}')

        item = QTreeWidgetItem(info)
        item.setText(0, f'Size:  {dataset.get_width()}x{dataset.get_height()}')

        item = QTreeWidgetItem(info)
        item.setText(0, f'Bands:  {dataset.num_bands()}')

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
        bands.setText(0, f'Bands ({dataset.num_bands()})')

        band_list = dataset.band_list()
        bad_bands = dataset.get_bad_bands()

        default_bands = dataset.default_display_bands()
        if default_bands is None:
            default_bands = []

        for i in range(dataset.num_bands()):
            band_info = band_list[i]
            bad = bad_bands[i]

            band_item = QTreeWidgetItem(bands)

            s = []

            if 'wavelength' in band_info:
                v = band_info['wavelength']
                s.append('{0:0.02f}'.format(v))
            else:
                s.append(band_info.get('description', '(no description)'))

            if bad == 0:
                s.append('(bad)')

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

            band_item.setText(0, ' '.join(s))

        # All done!

        self.insertTopLevelItem(index, top)


    def _on_dataset_removed(self, index):
        # Remove the information for the dataset at the specified index
        self.takeTopLevelItem(index)

        if self.topLevelItemCount() == 0:
            top = QTreeWidgetItem(self)
            top.setText(0, self.tr('No datasets loaded'))
            self.addTopLevelItem(top)
