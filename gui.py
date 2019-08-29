import sys
import numpy

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import envi
import util
import constants
from spectra import NumPySpectralData


class ImageViewer(QMainWindow):

    def __init__(self):
        super().__init__(None)

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.setCentralWidget(self.scrollArea)

    def set_data(self, spectral_data):
        print("Preparing image.")
        newImage = self.get_display_image(spectral_data)

        print("Displaying image.")
        self.imageLabel.setPixmap(QPixmap.fromImage(newImage))
        self.imageLabel.adjustSize()
        self.scrollArea.setVisible(True)

    def get_display_image(self, spectral_data):
        print("Identifying RGB bands")
        wavelengths = spectral_data.get_wavelengths()
        red_band = util.closest_value(wavelengths, constants.RED_WAVELENGTH)[0]
        grn_band = util.closest_value(wavelengths, constants.GREEN_WAVELENGTH)[0]
        blu_band = util.closest_value(wavelengths, constants.BLUE_WAVELENGTH)[0]

        print("Extracting raw band data")
        red_data = spectral_data.get_spectral_band(red_band)
        grn_data = spectral_data.get_spectral_band(grn_band)
        blu_data = spectral_data.get_spectral_band(blu_band)

        print("Converting raw data to RGB color data")

        '''
        # TODO(donnie):  This is a slow way to generate an image.  Generate a blob
        #     of data using numpy, and stuff that into the QImage.

        red_data = (red_data * 255 + 30).clip(0, 255).astype(numpy.uint8)
        grn_data = (grn_data * 255 + 30).clip(0, 255).astype(numpy.uint8)
        blu_data = (blu_data * 255 + 30).clip(0, 255).astype(numpy.uint8)

        print("Making QImage")
        image = QImage(spectral_data.get_width(), spectral_data.get_height(),
                       QImage.Format_RGB32)

        for y in range(spectral_data.get_height()):
            for x in range(spectral_data.get_width()):
                image.setPixel(x, y, qRgb(red_data[y][x], grn_data[y][x], blu_data[y][x]))
        '''

        ''' '''
        # This is a very fast way of generating an image.  It could probably be
        # made even faster by combining some of these steps, although the
        # present formulation could also be parallelized.

        red_data = (red_data * 255 + 30).clip(0, 255).astype(numpy.uint32)
        grn_data = (grn_data * 255 + 30).clip(0, 255).astype(numpy.uint32)
        blu_data = (blu_data * 255 + 30).clip(0, 255).astype(numpy.uint32)
        img_data = (red_data << 16 | grn_data << 8 | blu_data) | 0xff000000

        print("Making QImage")
        image = QImage(img_data,
            spectral_data.get_width(), spectral_data.get_height(),
            QImage.Format_RGB32)
        ''' '''

        print("QImage is complete")

        self.img_data = img_data
        return image



if __name__ == '__main__':
    app = QApplication(sys.argv)

    filename = sys.argv[1]
    header_filename = sys.argv[2]
    print(f'Opening {filename} with header {header_filename}')

    df = envi.load_envi_data(filename, header_filename)
    sd = NumPySpectralData(df[1], df[0])

    v = ImageViewer()
    v.set_data(sd)
    v.show()

    sys.exit(app.exec_())
