import enum

import numpy as np
from typing import List

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import RasterDataImpl, NumPyRasterDataImpl
from wiser.raster.spectral_library import ListSpectralLibrary

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from .generated.fits_dataset_dialog_ui import Ui_FitsDialog
from .generated.fits_spectra_dialog_ui import Ui_FitsSpectraDialog

from astropy.io import fits

class DataType(enum.Enum):
    SPECTRUM = 'Spectrum'
    SPECTRA = 'Spectra'
    SINGLE_IMAGE_BAND = 'Image Band'
    MANY_IMAGE_BAND = 'Image Bands'
    IMAGE_CUBE = 'Image Cube'

class FitsDatasetLoadingDialog(QDialog):

    # Takes in a fits file
    def __init__(self, fits_gdal_dataset_impl: RasterDataImpl, data_cache, parent=None):
        super().__init__(parent=parent)

        # Set up the UI state
        self._ui = Ui_FitsDialog()
        self._ui.setupUi(self)

        self._dataset_impl = fits_gdal_dataset_impl
        self._data_cache = data_cache

        filepath = fits_gdal_dataset_impl.get_filepaths()[0]
        # Initialize the naxis label
        # Initialize the list of naxis lengths
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                self._naxis = header['NAXIS'] 
                self._axis_lengths = []
                if self._naxis == 1:
                    self._axis_lengths.append(header[f'NAXIS{1}'])
                elif self._naxis == 2:
                    self._axis_lengths.append(fits_gdal_dataset_impl.get_height())
                    self._axis_lengths.append(fits_gdal_dataset_impl.get_width())
                elif self._naxis == 3:
                    self._axis_lengths.append(fits_gdal_dataset_impl.num_bands())
                    self._axis_lengths.append(fits_gdal_dataset_impl.get_height())
                    self._axis_lengths.append(fits_gdal_dataset_impl.get_width())
        except BaseException as e:
            raise TypeError(f'Could not open {filepath} as fits file.')
        
        # Set the naxis information
        self._ui.naxis_field.setText(str(self._naxis))

        self._init_axis_lengths()

        self._init_interpretation_options()

        self._init_data_varying_options()
        # When done we should either return an RasterDataset that is an image cube or image band
        # or we should return a spectral library 

    def _init_axis_lengths(self):
        # Set the axis length information
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

    def _init_interpretation_options(self):
        self._possible_datatypes: List[DataType] = []
        # Base on the number of axis, initialize what to show the user for axis interpretation
        if self._naxis == 2:
            self._possible_datatypes.append(DataType.SINGLE_IMAGE_BAND)
        elif self._naxis == 3:
            self._possible_datatypes.append(DataType.MANY_IMAGE_BAND)
            self._possible_datatypes.append(DataType.IMAGE_CUBE)
        else:
            raise Exception(f"WISER does not support naxis numberother than 2 or 3 for datasets" +
                            f"Yours is {self._naxis}. If your naxis is 1, then load it in as a spectra")

        # Set the combo box information
        interpretation_options = self._ui.interp_opt_combo

        for datatype in self._possible_datatypes:
            interpretation_options.addItem(datatype.value, datatype)
        
        if interpretation_options.count() <= 1:
            interpretation_options.setEnabled(False)
        
        interpretation_options.currentIndexChanged.connect(self._on_interp_opt_changed)
    
    def _init_data_varying_options(self):
        '''
        Sets up the data varying axis. An example of this is if you had a 2D array that could
        be either a collection of spectra or an image band. The data varying axis is the axis
        along which are individual spectra.
        '''
        data_varying_options = self._ui.data_vary_combo
        for i in range(len(self._axis_lengths)):
            data_varying_options.addItem(f'Dimension {i}', i)
        
        # We want to grey this out if image cube is selected. 
        if data_varying_options.count() <= 1:
            data_varying_options.setEnabled(False)
    
    def _on_interp_opt_changed(self):
        interpretation_options = self._ui.interp_opt_combo
        data_varying_options = self._ui.data_vary_combo
        if interpretation_options.currentData() == DataType.IMAGE_CUBE or \
            interpretation_options.currentData() == DataType.SINGLE_IMAGE_BAND:
            data_varying_options.setEnabled(False)
        else:
            data_varying_options.setEnabled(True)

    def accept(self):
        self.return_datasets: List = []
        
        axis_interpretation = self._ui.interp_opt_combo.currentData()
        data_varying_axis = self._ui.data_vary_combo.currentData()

        if axis_interpretation == DataType.IMAGE_CUBE or axis_interpretation == DataType.SINGLE_IMAGE_BAND:
            self._dataset_impl.get_image_data()
            self.return_datasets = [RasterDataSet(self._dataset_impl, self._data_cache)]
        elif axis_interpretation == DataType.MANY_IMAGE_BAND:
            numpy_raster_impl_list = []
            if data_varying_axis == 0:
                num_bands = self._dataset_impl.num_bands()
                for i in range(num_bands):
                    arr = self._dataset_impl.get_band_data(i)
                    arr = arr[np.newaxis,:,:]
                    numpy_impl = NumPyRasterDataImpl(arr)
                    numpy_raster_impl_list.append(numpy_impl)
            elif data_varying_axis == 1:
                # Means it varies along the height, so we want to span the width
                height = self._dataset_impl.get_height()
                raster_x_size = self._dataset_impl.get_width()
                for i in range(height):
                    arr = self._dataset_impl.get_all_bands_at_rect(0, i, raster_x_size, 1)
                    arr = arr.reshape((1, arr.shape[0], -1))  # 0 is the band index, I only move this so width is in same spot
                    numpy_impl = NumPyRasterDataImpl(arr)
                    numpy_raster_impl_list.append(numpy_impl)
            elif data_varying_axis == 2:
                # Means it varies along the width, so we want to span the height
                width = self._dataset_impl.get_width()
                raster_y_size = self._dataset_impl.get_height()
                for i in range(width):
                    arr = self._dataset_impl.get_all_bands_at_rect(i, 0, 1, raster_y_size)
                    arr = arr.reshape((1, -1, arr.shape[0]))  # 0 is the band index, I only move this so the height stays the same
                    numpy_impl = NumPyRasterDataImpl(arr)
                    numpy_raster_impl_list.append(numpy_impl)
            else:
                raise Exception(f'Data varying axis is somehow 3. Should be between 0 and 2')
            
            for i in range(len(numpy_raster_impl_list)):
                numpy_impl = numpy_raster_impl_list[i]
                ds = RasterDataSet(numpy_impl, self._data_cache)
                ds.set_name(f'{ds.get_name()}_{i}')
                self.return_datasets.append(ds)

        super().accept()

class FitsSpectraLoadingDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent=parent)
        
        # Set up the UI state
        self._ui = Ui_FitsSpectraDialog()
        self._ui.setupUi(self)

        # Expects the fits file to have dimension 2
        self._filepath = filepath

        try:
            with fits.open(self._filepath) as hdul:
                self._header = hdul[0].header
                self._data = hdul[0].data
                self._naxis = self._header['NAXIS']
                self._axis_lengths = []
                for i in range(self._naxis):
                    self._axis_lengths.append(self._header[f'NAXIS{i+1}'])
        except BaseException as e:
            raise TypeError(f'Error while loading in fits file at filepath: {filepath}.\nError: {e}')
    
        # Set the naxis information
        self._ui.naxis_field.setText(str(self._naxis))

        self._init_axis_lengths()

        self._init_data_varying_options()
    
        self._possible_datatypes: List[DataType] = []
        if self._naxis == 2:
            self._possible_datatypes.append(DataType.SPECTRA)
        elif self._naxis == 1:
            self._possible_datatypes.append(DataType.SPECTRUM)
        else:
            raise TypeError(f'Fits spectra file expected to have NAXIS=2 or 1, but NAXIS={self._naxis}')

    def _init_axis_lengths(self):
        # Set the axis length information
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
        
    def _init_data_varying_options(self):
        '''
        Sets up the data varying axis. An example of this is if you had a 2D array that could
        be either a collection of spectra or an image band. The data varying axis is the axis
        along which are individual spectra.
        '''
        data_varying_options = self._ui.data_vary_combo
        for i in range(len(self._axis_lengths)):
            data_varying_options.addItem(f'Dimension {i}', i)

        # We want to grey this out if image cube is selected. 
        if data_varying_options.count() <= 1:
            data_varying_options.setEnabled(False)
    
    def accept(self):
        self.return_datasets: List = []

        data_varying_axis = self._ui.data_vary_combo.currentData()

        spectra_list  = []
        if data_varying_axis == 0:
            # This will be the width
            width = self._data.shape[1]
            for i in range(width):
                arr = self._data[:,i:i+1]
                print(f"data_varying axis is 1, shape: {arr.shape}")
                spectra_list.append(arr)
        elif data_varying_axis == 1:
            # This will be the height
            height = self._data.shape[0]
            for i in range(height):
                arr = self._data[i:i+1,:]
                print(f"data_varying axis is 0, shape: {arr.shape}")
                spectra_list.append(arr)
        # Next we must make spectral_list into a list of Spectrum objects instead of arrays
        library = ListSpectralLibrary(spectra_list)

        super().accept()


