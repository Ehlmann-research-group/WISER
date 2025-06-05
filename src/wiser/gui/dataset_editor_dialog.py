from typing import List, Optional, Dict, Tuple, Union, TYPE_CHECKING

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


from wiser.gui.generated.dataset_editor_ui import Ui_DatasetEditor

from osgeo import gdal, osr

import numpy as np

from astropy import units as u

Number = Union[int, float]

if TYPE_CHECKING:
    from wiser.raster.dataset import RasterDataSet
    from wiser.raster.dataset_impl import RasterDataImpl
    from wiser.gui.app import DataVisualizerApp


class DatasetEditorDialog(QDialog):

    def __init__(self, dataset: 'RasterDataSet', app: 'DataVisualizerApp', parent = None):
        super().__init__(parent=parent)

        self._ui = Ui_DatasetEditor()
        self._ui.setupUi(self)

        self._raster_dataset = dataset
        self._app = app

        # Add a QDoubleValidator that allows positive and negative values
        self._init_validators()

    def _init_validators(self) -> None:
        """
        Creates and assigns a QDoubleValidator on the `ledit_data_ignore` field
        so that it only accepts valid floating-point numbers (both positive and negative).
        """
        validator = QDoubleValidator(-1e308, 1e308, 10, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self._ui.ledit_data_ignore.setValidator(validator)
        self._ui.ledit_data_ignore.setText(str(self._raster_dataset.get_data_ignore_value()))

    def _get_data_ignore_value(self) -> Optional[float]:
        """
        Reads the text from the line edit and returns it as a float.
        Returns None if the field is empty or cannot be parsed.
        """
        text = self._ui.ledit_data_ignore.text().strip()
        if not text:
            return None

        try:
            return float(text)
        except ValueError:
            return None

    def accept(self) -> None:
        """
        Overrides QDialog.accept(). Retrieves the data-ignore value,
        applies it to the raster dataset, refreshes all raster views,
        and then closes the dialog.
        """
        ignore_value: Optional[Number] = self._get_data_ignore_value()
        self._raster_dataset.set_data_ignore_value(ignore_value)
        self._app.update_all_rasterpane_displays()

        super().accept()

        # init function should call a helper function that adds a DoubleValidator to ledit_ 
        # that allows positive and negative numbers 

    # There should be a function called _get_data_ignore_value that extracts the float out of the ledit field

    # You should override the accept function for the QDialog and then calls def set_data_ignore_value(self, ignore_value: Optional[Number]) -> None:
    # on self._raster_dataset. It should also call self._app.refresh_all_rasterviews()
