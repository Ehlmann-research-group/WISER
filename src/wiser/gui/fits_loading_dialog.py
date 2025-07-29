import os

import enum

import numpy as np
from typing import List

import re

from wiser.raster.dataset import RasterDataSet
from wiser.raster.dataset_impl import RasterDataImpl, NumPyRasterDataImpl
from wiser.raster.spectral_library import ListSpectralLibrary
from wiser.raster.utils import KNOWN_SPECTRAL_UNITS
from wiser.raster.spectrum import NumPyArraySpectrum

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from .generated.fits_dataset_dialog_ui import Ui_FitsDialog
from .generated.fits_spectra_dialog_ui import Ui_FitsSpectraDialog

from astropy.io import fits
from astropy import units as u

DEFAULT_SPECTRAL_AXIS_NUMBER = 0

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
            raise Exception(f"WISER does not support naxis number other than 2 or 3 for datasets. " +
                            f"Yours is {self._naxis}. If your naxis number is 1, then load it in as a spectra.")

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
                self._units: List[u.Unit] = []
                for i in range(self._naxis):
                    self._axis_lengths.append(self._header[f'NAXIS{i+1}'])
                pattern = re.compile(r'unit', re.IGNORECASE)
                # Go through each key in the fits file and parse it for unit. Make the units into astropy units
                for key in self._header:
                    if pattern.search(key):
                        print(f"fits spectra Key: {key}")
                        unit_str = self._header[key]
                        print(f"fits spectra Value: {unit_str}, type: {type(unit_str)}")
                        unit_value = None
                        try:
                            unit_value = u.Unit(unit_str)
                            print(f"Type of unit value on startup: {type(unit_value)}")
                            print(f"Unit_value: {unit_value}")
                        except BaseException as e:
                            continue
                        self._units.append(unit_value)
    
        except BaseException as e:
            raise TypeError(f'Error while loading in fits file at filepath: {filepath}.\nError: {e}')
    
        # Set the naxis information
        self._ui.naxis_field.setText(str(self._naxis))

        self._init_axis_lengths()

        self._init_data_varying_options()

        self._init_wavelength_units()

        self._init_spectrum_suffix()

        self._setup_line_edits()

    def _init_wavelength_units(self):
        append = self.tr('Yours: ')
        if len(self._units) > 0:
            for unit in self._units:
                self._ui.wavelength_units_combo.addItem(append + unit.to_string(), unit)
            append = self.tr('Other: ')
        for key, value in KNOWN_SPECTRAL_UNITS.items():
            self._ui.wavelength_units_combo.addItem(append + key, value)

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
        
        data_varying_options.currentIndexChanged.connect(self._setup_line_edits)
    
    def _init_spectrum_suffix(self):
        line_edit = self._ui.spectrum_suffix_line_edit

        regex = QRegularExpression(r'^[a-zA-Z0-9]*$')  # Matches only letters and digits
        validator = QRegularExpressionValidator(regex, line_edit)

        # Apply the validator to the QLineEdit
        line_edit.setValidator(validator)

    def _setup_line_edits(self):
        # Get the line edit 
        x_ax_line_edit = self._ui.x_axis_line_edit
        x_ax_line_edit.setText(str(DEFAULT_SPECTRAL_AXIS_NUMBER))

        # Go through the currently selected data varying axis and get the min and max. Set that as the bounds for the line edit
        max_line_edit_value = self._data.shape[1]-1 if self._ui.data_vary_combo.currentData() == 0 else self._data.shape[0]-1
        min_line_edit_value = 0

        x_validator = QIntValidator(min_line_edit_value, max_line_edit_value, x_ax_line_edit)

        x_ax_line_edit.setValidator(x_validator)

    def accept(self):
        self.return_datasets: List = []

        filename_suffix = self._ui.spectrum_suffix_line_edit.text()
        basename = os.path.basename(self._filepath)
        filename, _ = os.path.splitext(basename) 
        spectrum_name = filename if filename_suffix == "" else f'{filename}_{filename_suffix}'

        data_varying_axis = self._ui.data_vary_combo.currentData()
        unit = self._ui.wavelength_units_combo.currentData()

        x_axis = int(self._ui.x_axis_line_edit.text())

        x_arr  = []
        y_arrays = []
        if data_varying_axis == 0:
            for i in range(self._data.shape[1]):
                if i == x_axis:
                    x_arr = self._data[:,i]
                    x_arr = np.squeeze(x_arr)
                else:
                    y_arr = self._data[:,i]
                    y_arr = np.squeeze(y_arr)
                    y_arrays.append(y_arr)
        elif data_varying_axis == 1:
            for i in range(self._data.shape[0]):
                if i == x_axis:
                    x_arr = self._data[i,:]
                    x_arr = np.squeeze(x_arr)
                else:
                    y_arr = self._data[i,:]
                    y_arr = np.squeeze(y_arr)
                    y_arrays.append(y_arr)
        
        numpy_spectrum_list = []
        wavelengths = x_arr*unit
        for i in range(len(y_arrays)):
            numpy_spectrum = NumPyArraySpectrum(arr=y_arrays[i], name=f'{spectrum_name}_{i}', wavelengths=wavelengths, editable=False)
            numpy_spectrum_list.append(numpy_spectrum)
            
        # Next we must make spectral_list into a list of Spectrum objects instead of arrays
        self.spectral_library = ListSpectralLibrary(numpy_spectrum_list, \
                                                    name=spectrum_name, \
                                                    path=self._filepath)

        super().accept()
