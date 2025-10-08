import os
import sys
from enum import Enum
from typing import Optional, Tuple

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import numpy as np

from wiser.raster.dataset import RasterDataSet, find_display_bands


class DatasetInfoView(QTreeWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent=parent)

        self._model = model
        self._model.dataset_added.connect(self._on_dataset_added)
        self._model.dataset_removed.connect(self._on_dataset_removed)

        self.setColumnCount(1)
        self.setHeaderLabels([self.tr("Description")])

        top = QTreeWidgetItem(self)
        top.setText(0, self.tr("No datasets loaded"))
        self.addTopLevelItem(top)

        # Update the header info so that columns will resize based on their
        # contents.  This is useful for filenames of data files, which are long.
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(False)

    def _on_dataset_added(self, ds_id: int, view_dataset: bool = True):
        """
        When a data set is added to the application state, this method populates
        the data info view with information about the data set.
        """

        if self._model.num_datasets() == 1:
            self.clear()

        dataset = self._model.get_dataset(ds_id)

        top = QTreeWidgetItem(self)
        top.setText(0, self.tr(f"Dataset {ds_id}"))
        top.setData(0, Qt.UserRole, ds_id)

        # General info subsection

        # info = QTreeWidgetItem(top)
        # info.setText(0, self.tr('General'))

        item = QTreeWidgetItem(top)
        item.setText(0, f"Description:  {dataset.get_description()}")

        item = QTreeWidgetItem(top)
        item.setText(0, f"Format:  {dataset.get_format()}")

        item = QTreeWidgetItem(top)
        item.setText(0, f"Size:  {dataset.get_width()}x{dataset.get_height()}")

        # TODO(donnie):  Number of bands is specified in the "bands" group item,
        #     so we probably don't need this.
        # item = QTreeWidgetItem(info)
        # item.setText(0, f'Bands:  {dataset.num_bands()}')

        data_ignore = dataset.get_data_ignore_value()
        item = QTreeWidgetItem(top)
        if data_ignore is not None:
            item.setText(0, f"Data-ignore value:  {data_ignore}")
        else:
            item.setText(0, "No data-ignore value specified")

        # Files subsection

        filepaths = [os.path.basename(fp) for fp in dataset.get_filepaths()]
        filepaths.sort()

        files = QTreeWidgetItem(top)
        files.setText(0, self.tr(f"Files ({len(filepaths)})"))

        for fp in filepaths:
            file_item = QTreeWidgetItem(files)
            # file_item.setFirstColumnSpanned(True)
            file_item.setText(0, fp)

        # Bands section

        bands = QTreeWidgetItem(top)
        bands.setText(0, f"Bands ({dataset.num_bands()})")

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

            s.append(band_info.get("description", "(no description)"))

            if bad == 0:
                s.append("(bad)")

            if i in default_bands:
                if len(default_bands) == 1:
                    s.append(self.tr("(default - grayscale)"))
                else:
                    if len(default_bands) != 3:
                        raise ValueError(
                            f"Expected 3 default bands, got {default_bands}"
                        )

                    if default_bands[0] == i:
                        s.append(self.tr("(default - red)"))
                    elif default_bands[1] == i:
                        s.append(self.tr("(default - green)"))
                    elif default_bands[2] == i:
                        s.append(self.tr("(default - blue)"))

            band_item.setText(0, " ".join(s))

        # All done!
        self.addTopLevelItem(top)

    def _find_dataset_entry(self, ds_id: int) -> Optional[Tuple[int, QTreeWidgetItem]]:
        """
        This helper function finds the tree entry for the dataset with the
        specified ID.  If found, the function returns the tuple
        (index, QTreeWidgetItem) indicating both the top-level tree entry that
        represents the dataset, as well as the index of the entry in the tree.

        If it can't be found, the function returns None.
        """

        for i in range(self.topLevelItemCount()):
            entry = self.topLevelItem(i)
            if entry.data(0, Qt.UserRole) == ds_id:
                return (i, entry)

        return None

    def _on_dataset_removed(self, ds_id: int):
        result = self._find_dataset_entry(ds_id)
        if result is None:
            print(f"WARNING:  Info-view encountered unrecognized dataset {ds_id}")
            return

        # Unpack the result.
        (index, entry) = result

        # Remove the information for the dataset at the specified index
        self.takeTopLevelItem(index)

        if self.topLevelItemCount() == 0:
            top = QTreeWidgetItem(self)
            top.setText(0, self.tr("No datasets loaded"))
            self.addTopLevelItem(top)
