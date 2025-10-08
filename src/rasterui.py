import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from units import RED_WAVELENGTH, GREEN_WAVELENGTH, BLUE_WAVELENGTH
from gui.rasterview import RasterView


def find_band_near_wavelength(raster_data, wavelength, max_distance=None):
    bands = raster_data.band_list()
    best_band = None
    best_distance = None

    # TODO(donnie):  assert that wavelength units is nanometers

    for band in bands:
        band_wavelength = band.get("wavelength")
        if band_wavelength is None:
            continue

        # TODO(donnie):  assert that band_wavelength units is nanometers
        distance = abs(band_wavelength - wavelength)
        if max_distance is not None and distance > max_distance:
            continue

        if best_band is None or distance < best_distance:
            best_band = band
            best_distance = distance

    index = None
    if best_band is not None:
        index = best_band["index"]

    return index


def find_rgb_bands(raster_data):
    # Try to find bands based on what is close to visible spectral bands
    red_band = find_band_near_wavelength(raster_data, RED_WAVELENGTH)
    green_band = find_band_near_wavelength(raster_data, GREEN_WAVELENGTH)
    blue_band = find_band_near_wavelength(raster_data, BLUE_WAVELENGTH)

    # If that didn't work, just choose first, middle and last bands
    if red_band is None or green_band is None or blue_band is None:
        red_band = 0
        green_band = max(0, raster_data.num_bands() // 2 - 1)
        blue_band = max(0, raster_data.num_bands() - 1)

    return (red_band, green_band, blue_band)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    ui = RasterView()
    ui.show()

    raster_data = None
    rgb_bands = (0, 0, 0)
    if len(sys.argv) == 2:
        from gdal_dataset import GDALRasterDataLoader

        loader = GDALRasterDataLoader()
        raster_data = loader.load(sys.argv[1])
        rgb_bands = find_rgb_bands(raster_data)

    elif len(sys.argv) > 2:
        print(
            "This program takes one optional argument, the name of a raster data file."
        )
        sys.exit(1)

    ui.set_raster_data(raster_data, rgb_bands)

    sys.exit(app.exec_())
