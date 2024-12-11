import enum

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.fits_dialog_ui import Ui_FitsDialog

from astropy.io import fits

class DataType(enum.Enum):
    SPECTRA = 0
    IMAGE_BAND = 1
    IMAGE_CUBE = 2

class FitsDatasetLoadingDialog(QDialog):

    # Takes in a fits file
    def __init__(self, fits_gdal_dataset_impl, data_cache, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_FitsDialog()
        self._ui.setupUi(self)

        self._dataset = fits_gdal_dataset_impl
        self._data_cache = data_cache

        filepath = fits_gdal_dataset_impl.get_filepaths()[0]
        # Initialize the naxis label
        # Initialize the list of naxis lengths
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                self._naxis = header['NAXIS'] 
                self._axis_lengths = []
                for i in range(self._naxis):
                    self._axis_lengths.append(header[f'NAXIS{i+1}'])
        except BaseException as e:
            raise TypeError(f'Could not open {filepath} as fits file.')

        self._possible_datatypes = []
        # Base on the number of axis, initialize what to show the user for axis interpretation
        if self._naxis == 1:
            self._possible_datatypes.append(DataType.SPECTRA)
        elif self._naxis == 2:
            self._possible_datatypes.append(DataType.SPECTRA)
            self._possible_datatypes.append(DataType.IMAGE_BAND)
        elif self._naxis == 3:
            self._possible_datatypes.append(DataType.IMAGE_BAND)
            self._possible_datatypes.append(DataType.IMAGE_CUBE)
        else:
            raise Exception(f'WISER does not support naxis number greater than 3. Yours is {self._naxis}')
        
        self._ui.naxis_field.setText(str(self._naxis))
        group_box = self._ui.axis_lengths_group_box
        holding_box = QVBoxLayout()
        for i in range(len(self._axis_lengths)):
            axis_length = self._axis_lengths[i]
            hbox = QHBoxLayout()
            axis_index_label = QLabel(f'Dimension {i}')
            axis_length_label = QLabel(f'{axis_length}')
            hbox.addWidget(axis_index_label)
            hbox.addWidget(axis_length_label)
            holding_box.addLayout(hbox)
        group_box.setLayout(holding_box)
        # When done we should either return an RasterDataset that is an image cube or image band
        # or we should return a spectral library 


