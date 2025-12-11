import logging

from wiser.plugins import ToolsMenuPlugin

from PySide2.QtWidgets import QMenu, QMessageBox

from wiser.raster.dataset import RasterDataSet

from wiser.raster.roi import RegionOfInterest

from wiser.raster.spectrum import Spectrum

import numpy

from typing import List

from astropy import units as u

from osgeo import gdal

logger = logging.getLogger(__name__)


class HelloToolPlugin(ToolsMenuPlugin):
    """
    A simple "Hello world!" example of a Tools plugin.
    """

    def __init__(self):
        super().__init__()

    def add_tool_menu_items(self, tool_menu: QMenu, wiser) -> None:
        """
        Use QMenu.addAction() to add individual actions, or QMenu.addMenu() to
        add sub-menus to the Tools menu.
        """
        logger.info("HelloToolPlugin is adding tool-menu items")
        act = tool_menu.addAction("Say hello...")
        act.triggered.connect(self.say_hello)

    def say_hello(self, checked: bool = False):
        logger.info("HelloToolPlugin.say_hello() was called!")
        QMessageBox.information(None, "Hello-Tool Plugin", "Hello from the toolbar!")

        # dataset: RasterDataSet = None

        # # Gets all of the array data as a 3D array with dimensions [b][y][x]
        # dataset_array: numpy.ndarray = dataset.get_image_data()
        # # Gets the band at the specified index. Note that this indexes by 0,
        # # so the first band is at index 0
        # dataset_band_array: numpy.ndarray = dataset.get_band_data(0)
        # # Essentially gets the spectra at the location in the dataset
        # # Bad bands and data ignore bands will be set to NaN
        # dataset_spectrum_array: numpy.ndarray = dataset.get_all_bands_at(0, 0)
        # # A mask that will mask out the bad bands. 0s are bad bands, 1s are good bands
        # dataset_bad_bands: List[int] = dataset.get_bad_bands()
        # # The wavelengths of each band in the array. A u.Quantity is a
        # # astropy.units.Quantity object and it contains a numeric value and units
        # dataset_wavelengths: List[u.Quantity] = dataset.get_spectral_metadata().get_wavelengths()

        # region_of_interest: RegionOfInterest = None

        # # Get all of the pixel coordinates in the region of interest
        # roi_all_pixels = region_of_interest.get_all_pixels()
        # # Get the name of the region of interest
        # roi_name = region_of_interest.get_name()

        # spectrum: Spectrum = None

        # # Get the array for the spectrum. Shape (b,)
        # spectrum_arr = spectrum.get_spectrum()
        # # Get the wavelengths for the spectrum
        # spectrum_wvls = spectrum.get_wavelengths()
